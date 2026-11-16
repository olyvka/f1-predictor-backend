"""
Replay минулої сесії OpenF1 "як наживо": щоб перевірити live-режим (карта,
телеметрія, таблиця позицій) без справжньої гонки. Також годиться як окрема
функція "переглянути гонку ще раз".

Чому так можна: live-фід OpenF1 (MQTT/WebSocket, платний) шле ті самі
JSON-записи, що й REST, лише з додатковими _id/_key. Історичний REST
безкоштовний, тож ми програємо записи минулої сесії за їхнім полем date у
реальному темпі (або прискорено), і фронт отримує той самий потік, який
потім дасть live-джерело. Протокол повідомлень описаний у README
("Replay сесії через WebSocket").

Обсяги (гонка в Бразилії 2026): location і car_data дають ~5 тис.
записів на кожні 60 с (22 пілоти × ~3.7 Гц), тому їх тягнемо вікнами по
60 с і заздалегідь вантажимо наступне вікно. Вікна вирівняні від старту
сесії й кешуються в пам'яті: кілька глядачів однієї гонки ділять ті самі
запити, а хвилинний ліміт OpenF1 (30 запитів без підписки) не вичерпується. Решту топіків (кола, позиції,
інтервали, race control, піт-стопи, стінти, погода, разом ~30 тис.
записів) тягнемо одним запитом на старті.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

from fastapi import WebSocket, WebSocketDisconnect

from app import openf1_client
from app.i18n import tr
from app.openf1_client import parse_date

# Великі топіки (~3.7 Гц на пілота), які тягнемо вікнами.
STREAM_TOPICS = ("location", "car_data")
# Топіки з полем date, які вантажимо цілком на старті.
DATED_TOPICS = ("position", "intervals", "race_control", "pit", "weather", "team_radio")
# Топіки-"стан". Для старту з середини сесії досить останнього запису на
# пілота: історія інтервалів до цього моменту фронту не потрібна.
STATE_TOPICS = frozenset({"position", "intervals", "weather"})

WINDOW = timedelta(seconds=60)
TICK_SECONDS = 0.25
MIN_SPEED, MAX_SPEED = 0.25, 20.0
# Сесія може затягнутись за запланований date_end (червоний прапор), але
# не більше ніж на стільки.
MAX_OVERRUN = timedelta(hours=3)

_DROP_FIELDS = ("session_key", "meeting_key")

# Подія в timeline і вікнах: (час, порядковий номер для стабільного
# сортування, топік, запис).
FetchWindow = Callable[[str, datetime, datetime], "list[dict]"]


def _clean(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _DROP_FIELDS}


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat()


def clamp_speed(value) -> float:
    return max(MIN_SPEED, min(MAX_SPEED, float(value)))


def build_timeline(small: dict, session_start: datetime) -> list:
    """Усі невеликі топіки, впорядковані за часом, коли live-фід їх надіслав би.

    Записи з date просто беремо за date. Колам і стінтам власної дати бракує,
    тому:
      - коло "приходить", коли завершилось: date_start + lap_duration, а
        якщо тривалості немає (перше коло, сходження), то на старті
        наступного кола цього пілота або, за браком його, на date_start;
      - стінт приходить на старті свого першого кола, а якщо кола немає -
        на старті сесії.
    """
    events: list = []

    def add(t: Optional[datetime], topic: str, record: dict) -> None:
        if t is not None:
            events.append((t, len(events), topic, _clean(record)))

    for topic in DATED_TOPICS:
        for r in small.get(topic, []):
            add(parse_date(r.get("date")), topic, r)

    laps = small.get("laps", [])
    lap_starts = {(r.get("driver_number"), r.get("lap_number")): parse_date(r.get("date_start")) for r in laps}
    for r in laps:
        key = (r.get("driver_number"), r.get("lap_number"))
        begin = lap_starts.get(key)
        duration = r.get("lap_duration")
        if begin is not None and duration:
            done = begin + timedelta(seconds=float(duration))
        else:
            done = lap_starts.get((key[0], (key[1] or 0) + 1)) or begin
        add(done, "laps", r)

    for r in small.get("stints", []):
        begin = lap_starts.get((r.get("driver_number"), r.get("lap_start")))
        add(begin or session_start, "stints", r)

    events.sort(key=lambda e: (e[0], e[1]))
    return events


def replay_end(session: dict, timeline: list) -> datetime:
    """Кінець replay: запланований date_end або останній запис, якщо сесія
    затягнулась (але не далі ніж на MAX_OVERRUN)."""
    end = parse_date(session.get("date_end")) or parse_date(session["date_start"]) + timedelta(hours=2)
    if timeline:
        end = max(end, min(timeline[-1][0], end + MAX_OVERRUN))
    return end


def _wire(events: list) -> list[dict]:
    return [{"topic": e[2], "data": e[3]} for e in events]


class ReplayPlayer:
    """Програє timeline і вікна STREAM_TOPICS за "годинником replay".

    Годинник: replay_time = anchor_replay + (wall - anchor_wall) * speed.
    Пауза і зміна швидкості перекладають якір, тож час не стрибає. Якщо
    наступне вікно location/car_data ще вантажиться, годинник чекає разом із
    ним (буферизація), щоб позиції на карті не приходили пачкою із запізненням.
    """

    def __init__(
        self,
        timeline: list,
        fetch_window: FetchWindow,
        start: datetime,
        end: datetime,
        speed: float = 1.0,
        *,
        tick: float = TICK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        window: timedelta = WINDOW,
        align_base: Optional[datetime] = None,
    ):
        self.start, self.end = start, end
        self.speed = clamp_speed(speed)
        self.paused = False
        self._timeline = timeline
        self._fetch_window = fetch_window
        self._tick = tick
        self._window = window
        # Межі вікон рахуються від align_base (старт сесії), а не від start:
        # тоді в різних глядачів вікна однакові й кешуються спільно.
        self._base = align_base or start
        self._clock = clock
        self._anchor_replay = start
        self._anchor_wall = clock()
        self._pos = 0  # перший ще не надісланий запис timeline
        self._chunk: list = []
        self._chunk_pos = 0
        self._loaded_until = start  # кінець останнього взятого вікна
        self._prefetch: Optional[asyncio.Task] = None

    # --- годинник -----------------------------------------------------------

    def now(self) -> datetime:
        if self.paused:
            return self._anchor_replay
        elapsed = (self._clock() - self._anchor_wall) * self.speed
        return min(self._anchor_replay + timedelta(seconds=elapsed), self.end)

    def _reanchor(self) -> None:
        self._anchor_replay = self.now()
        self._anchor_wall = self._clock()

    def set_speed(self, value) -> None:
        self._reanchor()
        self.speed = clamp_speed(value)

    def pause(self) -> None:
        self._reanchor()
        self.paused = True

    def resume(self) -> None:
        if self.paused:
            self._anchor_wall = self._clock()
            self.paused = False

    def state(self) -> dict:
        return {"type": "state", "paused": self.paused, "speed": self.speed, "clock": _iso(self.now())}

    # --- дані ---------------------------------------------------------------

    def catch_up(self) -> list:
        """Усе з timeline до моменту start. Для STATE_TOPICS лише останній
        запис на пілота (на погоду один), щоб фронт одразу мав стан сесії."""
        latest: dict = {}
        rest: list = []
        while self._pos < len(self._timeline) and self._timeline[self._pos][0] <= self.start:
            e = self._timeline[self._pos]
            self._pos += 1
            if e[2] in STATE_TOPICS:
                latest[(e[2], e[3].get("driver_number"))] = e
            else:
                rest.append(e)
        return sorted(rest + list(latest.values()), key=lambda e: (e[0], e[1]))

    def _take_timeline(self, now: datetime) -> list:
        out = []
        while self._pos < len(self._timeline) and self._timeline[self._pos][0] <= now:
            out.append(self._timeline[self._pos])
            self._pos += 1
        return out

    def _load_window(self, since: datetime) -> tuple:
        """Синхронно (у потоці): усі STREAM_TOPICS від since до кінця вирівняного
        вікна, що його містить. Повертає (записи, кінець вікна)."""
        k = (since - self._base) // self._window
        window_start = self._base + k * self._window
        until = min(window_start + self._window, self.end)
        out = []
        for topic in STREAM_TOPICS:
            for r in self._fetch_window(topic, window_start, until):
                t = parse_date(r.get("date"))
                # Межі запиту включні з обох боків: відсікаємо праву, щоб
                # запис на стику вікон не прийшов двічі (крім самого кінця).
                if t is not None and since <= t and (t < until or until == self.end):
                    out.append((t, 0, topic, _clean(r)))
        out.sort(key=lambda e: e[0])
        return out, until

    def _start_prefetch(self) -> None:
        if self._prefetch is None and self._loaded_until < self.end:
            self._prefetch = asyncio.ensure_future(asyncio.to_thread(self._load_window, self._loaded_until))

    async def _take_stream(self, now: datetime) -> list:
        out = []
        while True:
            while self._chunk_pos < len(self._chunk) and self._chunk[self._chunk_pos][0] <= now:
                out.append(self._chunk[self._chunk_pos])
                self._chunk_pos += 1
            if self._chunk_pos < len(self._chunk) or self._loaded_until > now or self._loaded_until >= self.end:
                return out
            # Поточне вікно вичерпане, а годинник уже в наступному: беремо
            # передзавантажене. Час очікування не рахуємо в годинник replay.
            self._start_prefetch()
            waited_from = self._clock()
            task, self._prefetch = self._prefetch, None
            self._chunk, until = await task
            self._chunk_pos = 0
            if not self.paused:
                self._anchor_wall += self._clock() - waited_from
            self._loaded_until = until
            self._start_prefetch()

    def finished(self, now: datetime) -> bool:
        return (
            now >= self.end
            and self._pos >= len(self._timeline)
            and self._chunk_pos >= len(self._chunk)
            and self._loaded_until >= self.end
        )

    # --- цикл ---------------------------------------------------------------

    async def run(self, send: Callable[[dict], Awaitable[None]]) -> None:
        self._start_prefetch()
        try:
            await send({"type": "batch", "catchup": True, "clock": _iso(self.start), "events": _wire(self.catch_up())})
            self._anchor_wall = self._clock()  # час на catch-up у replay не йде
            while True:
                if not self.paused:
                    now = self.now()
                    events = self._take_timeline(now) + await self._take_stream(now)
                    if events:
                        events.sort(key=lambda e: e[0])
                        await send({"type": "batch", "clock": _iso(now), "events": _wire(events)})
                    if self.finished(now):
                        await send({"type": "end", "clock": _iso(now)})
                        return
                await asyncio.sleep(self._tick)
        finally:
            if self._prefetch is not None:
                self._prefetch.cancel()


# --- збирання replay з OpenF1 ------------------------------------------------

_INTRO_DRIVER_FIELDS = (
    "driver_number", "name_acronym", "full_name", "broadcast_name",
    "team_name", "team_colour", "headshot_url",
)
_INTRO_SESSION_FIELDS = (
    "session_key", "session_name", "session_type", "date_start", "date_end",
    "circuit_short_name", "country_name", "location", "year",
    "gmt_offset",  # "-03:00:00": місцевий час траси (як у повідомленнях FIA "TIMED AT")
)


def track_outline(session: dict, laps: list[dict]) -> Optional[list[list[int]]]:
    """Контур траси для карти: точки location найшвидшого чистого кола
    (не виїзд з піт-лейну). None, якщо такого кола немає."""
    clean = [
        r for r in laps
        if r.get("lap_duration") and r.get("date_start") and not r.get("is_pit_out_lap") and (r.get("lap_number") or 0) > 1
    ]
    if not clean:
        return None
    best = min(clean, key=lambda r: r["lap_duration"])
    begin = parse_date(best["date_start"])
    rows = openf1_client.session_window(
        session, "location", begin, begin + timedelta(seconds=float(best["lap_duration"])),
        driver_number=best["driver_number"],
    )
    rows.sort(key=lambda r: r.get("date") or "")
    return [[r["x"], r["y"]] for r in rows if r.get("x") is not None and r.get("y") is not None] or None


def starting_grid(session: dict, positions: list[dict]) -> Optional[dict]:
    """Стартова позиція кожного пілота: перший запис position у сесії (OpenF1
    публікує порядок решітки ще до старту). Лише для гонок і спринтів (у
    OpenF1 обидва мають session_type "Race"). Після перемотування catch-up
    містить лише останні позиції, тож фронт бере "▲▼ від старту" звідси."""
    if session.get("session_type") != "Race":
        return None
    grid: dict = {}
    for r in sorted(positions, key=lambda r: r.get("date") or ""):
        n, p = r.get("driver_number"), r.get("position")
        if n is not None and p is not None:
            grid.setdefault(n, p)
    return grid or None


# Спільний кеш вікон у пам'яті. Не на диску: location + car_data важать
# ~1.5 МБ на хвилину, тобто ~180 МБ на гонку. 16 вікон = 8 хвилин сесії на
# двох топіках - вистачає, щоб кілька глядачів поруч у часі ділили запити.
_WINDOW_CACHE: "OrderedDict[tuple, list]" = OrderedDict()
_WINDOW_CACHE_SIZE = 16
_window_lock = threading.Lock()


def _cached_window(session: dict, topic: str, since: datetime, until: datetime) -> list[dict]:
    key = (session["session_key"], topic, since.isoformat(), until.isoformat())
    with _window_lock:
        if key in _WINDOW_CACHE:
            _WINDOW_CACHE.move_to_end(key)
            return _WINDOW_CACHE[key]
    rows = openf1_client.session_window(session, topic, since, until, disk_cache=False)
    with _window_lock:
        _WINDOW_CACHE[key] = rows
        while len(_WINDOW_CACHE) > _WINDOW_CACHE_SIZE:
            _WINDOW_CACHE.popitem(last=False)
    return rows


def _fetch_window(session: dict) -> FetchWindow:
    return lambda topic, since, until: _cached_window(session, topic, since, until)


def prepare_replay(session: dict, offset: float = 0.0, speed: float = 1.0) -> tuple[dict, ReplayPlayer]:
    """Синхронно (у потоці): тягне невеликі топіки сесії, контур траси і
    повертає (вступне повідомлення "session", гравець). offset - секунди від
    запланованого старту сесії, від яких грати (можна від'ємні)."""
    small = {topic: openf1_client.session_data(session, topic) for topic in ("drivers", "laps", "stints", *DATED_TOPICS)}
    session_start = parse_date(session["date_start"])
    timeline = build_timeline(small, session_start)
    end = replay_end(session, timeline)
    start = min(max(session_start + timedelta(seconds=offset), session_start - timedelta(hours=1)), end)

    try:
        outline = track_outline(session, small["laps"])
    except Exception:
        outline = None  # карта без контуру гірша, але replay від цього не ламається

    player = ReplayPlayer(timeline, _fetch_window(session), start, end, speed, align_base=session_start)
    intro = {
        "type": "session",
        "mode": "replay",
        "session": {k: session.get(k) for k in _INTRO_SESSION_FIELDS},
        "drivers": [{k: d.get(k) for k in _INTRO_DRIVER_FIELDS} for d in small["drivers"]],
        "track": {"points": outline} if outline else None,
        "grid": starting_grid(session, small["position"]),
        "replay": {"start": _iso(start), "end": _iso(end), "speed": player.speed},
    }
    return intro, player


# --- WebSocket ---------------------------------------------------------------

def apply_command(player: ReplayPlayer, message) -> Optional[str]:
    """Команда від клієнта. Повертає None, якщо виконано, або текст помилки."""
    action = message.get("action") if isinstance(message, dict) else None
    try:
        if action == "pause":
            player.pause()
        elif action == "resume":
            player.resume()
        elif action == "speed":
            player.set_speed(message["value"])
        else:
            return tr("replay_bad_command", command=action)
    except (KeyError, TypeError, ValueError):
        return tr("replay_bad_command", command=f"{action} {message.get('value')!r}")
    return None


async def stream(websocket: WebSocket, player: ReplayPlayer) -> None:
    """Програє replay у вже прийнятий WebSocket і паралельно слухає команди
    клієнта (pause / resume / speed). Завершується з кінцем replay або з
    відключенням клієнта."""

    async def receive_commands() -> None:
        while True:
            text = await websocket.receive_text()
            try:
                message = json.loads(text)
            except ValueError:
                message = None
            error = apply_command(player, message)
            await websocket.send_json({"type": "error", "detail": error} if error else player.state())

    play = asyncio.ensure_future(player.run(websocket.send_json))
    listen = asyncio.ensure_future(receive_commands())
    try:
        done, _ = await asyncio.wait({play, listen}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                await websocket.send_json({"type": "error", "detail": tr("replay_failed", error=exc)})
        if play in done:
            await websocket.close()
    except (WebSocketDisconnect, RuntimeError):
        pass  # клієнт уже пішов - надсилати нікому
    finally:
        for task in (play, listen):
            task.cancel()

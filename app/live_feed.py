"""
Live-сесія через платну підписку OpenF1 (Sponsor) - той самий протокол
WebSocket, що й replay (app/live_replay.py), але з mode "live".

    OpenF1 MQTT (mqtt.openf1.org:8883, TLS; пароль = OAuth2-токен)
        │  одне підключення на весь сервер (MqttFeed, потік paho)
        ▼
    LiveHub (asyncio): LiveState згортає записи → batch кожні 0.25 с
        │  розсилка всім глядачам (asyncio.Queue на клієнта)
        ▼
    WS /ws/live → session (mode "live") → batch catchup → batch … → end

Чому так:
  - OpenF1 дозволяє до 10 MQTT-підключень на підписку, тож кожному глядачеві
    власне підключення не дати. Сервер тримає одне й роздає всім.
  - MQTT шле лише нові записи. Новому глядачеві потрібен стан "на зараз",
    тому хаб на старті тягне сесію через REST (з токеном) і далі веде стан
    сам (LiveState.catchup - як catch-up у replay).
  - Live-фід шле коло на його старті й потім оновлює, а replay - коли коло
    завершене. Фронт чекає другого, тож LiveState пересилає коло, коли
    з'явився lap_duration (або почалось наступне коло цього пілота).

OpenF1 віддає live-дані від 30 хв до старту сесії до 30 хв після кінця.
Поза цим вікном /ws/live відповідає повідомленням "waiting" з часом
відкриття, і фронт показує таймер.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from fastapi import WebSocket, WebSocketDisconnect

from app import fastf1_client, live_replay, openf1_client
from app.i18n import tr
from app.openf1_client import parse_date

log = logging.getLogger(__name__)

MQTT_HOST = os.getenv("OPENF1_MQTT_HOST", "mqtt.openf1.org")
MQTT_PORT = int(os.getenv("OPENF1_MQTT_PORT", "8883"))
TOPICS = ("location", "car_data", "position", "intervals", "laps", "stints", "pit", "race_control", "weather",
          "team_radio")
STREAM_TOPICS = frozenset(live_replay.STREAM_TOPICS)
STATE_TOPICS = live_replay.STATE_TOPICS
LIVE_WINDOW = timedelta(minutes=30)
BATCH_SECONDS = 0.25
IDLE_STOP_SECONDS = 60       # скільки тримати MQTT після відходу останнього глядача
CLIENT_QUEUE = 240           # батчів (~1 хв): повільнішого глядача відключаємо, він перепідключиться
_DROP_FIELDS = ("_id", "_key", "session_key", "meeting_key")

RecordCallback = Callable[[str, dict], None]


def _iso(t: Optional[datetime]) -> Optional[str]:
    return t.astimezone(timezone.utc).isoformat() if t else None


def _clean(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _DROP_FIELDS}


def _event(topic: str, data: dict) -> dict:
    return {"topic": topic, "data": data}


def _event_time(e: dict) -> str:
    d = e["data"]
    return d.get("date") or d.get("date_start") or ""


# --- згортка потоку ------------------------------------------------------------

class LiveState:
    """Стан live-сесії для нових глядачів і відбір подій для розсилки.

    ingest() повертає події, які треба розіслати (можливо, порожній список:
    повтор того самого запису або коло, що ще не завершилось). catchup() -
    знімок, рівноцінний catch-up у replay: усі кола, стінти, піт-стопи й
    race control, а для position / intervals / weather - останній запис.
    """

    def __init__(self) -> None:
        self.latest: dict = {}        # (topic, driver_number) -> запис
        self.laps: dict = {}          # (n, lap_number) -> завершене коло
        self.pending_laps: dict = {}  # (n, lap_number) -> коло без lap_duration
        self.stints: dict = {}        # (n, stint_number)
        self.pits: dict = {}          # (n, date)
        self.race_control: dict = {}  # (date, message)
        self.team_radio: dict = {}    # recording_url
        self.clock: Optional[datetime] = None

    def ingest(self, topic: str, record: dict) -> list[dict]:
        data = _clean(record)
        when = parse_date(data.get("date"))
        if when and (self.clock is None or when > self.clock):
            self.clock = when
        n = data.get("driver_number")
        if topic in STREAM_TOPICS:
            return [_event(topic, data)]
        if topic in STATE_TOPICS:
            return self._keep(self.latest, (topic, n), topic, data)
        if topic == "laps":
            return self._lap(data)
        if topic == "stints":
            return self._keep(self.stints, (n, data.get("stint_number")), topic, data)
        if topic == "pit":
            return self._keep(self.pits, (n, data.get("date")), topic, data)
        if topic == "team_radio":
            return self._keep(self.team_radio, (data.get("recording_url"),), topic, data)
        if topic == "race_control":
            return self._keep(self.race_control, (data.get("date"), data.get("message")), topic, data)
        return [_event(topic, data)]

    @staticmethod
    def _keep(store: dict, key: tuple, topic: str, data: dict) -> list[dict]:
        if store.get(key) == data:
            return []  # повтор (той самий _key, ті самі дані)
        store[key] = data
        return [_event(topic, data)]

    def _lap(self, data: dict) -> list[dict]:
        n, lap = data.get("driver_number"), data.get("lap_number")
        if n is None or lap is None:
            return []
        out = []
        # Почалось нове коло: попереднє, що так і не отримало тривалості
        # (перше коло, сход), пересилаємо як є.
        for key in sorted(k for k in self.pending_laps if k[0] == n and k[1] < lap):
            out += self._keep(self.laps, key, "laps", self.pending_laps.pop(key))
        if data.get("lap_duration") is None:
            if (n, lap) not in self.laps:
                self.pending_laps[(n, lap)] = data
            return out
        self.pending_laps.pop((n, lap), None)
        return out + self._keep(self.laps, (n, lap), "laps", data)

    def catchup(self) -> list[dict]:
        events = [_event("laps", d) for d in self.laps.values()]
        events += [_event("stints", d) for d in self.stints.values()]
        events += [_event("pit", d) for d in self.pits.values()]
        events += [_event("race_control", d) for d in self.race_control.values()]
        events += [_event("team_radio", d) for d in self.team_radio.values()]
        events += [_event(topic, d) for (topic, _n), d in self.latest.items()]
        return sorted(events, key=_event_time)


# --- сесія ---------------------------------------------------------------------

def live_window(session: dict) -> tuple[Optional[datetime], Optional[datetime]]:
    start, end = parse_date(session.get("date_start")), parse_date(session.get("date_end"))
    return (start - LIVE_WINDOW if start else None, end + LIVE_WINDOW if end else None)


def find_live_session(now: datetime) -> tuple[str, Optional[dict]]:
    """("live", сесія), якщо зараз відкрите live-вікно; ("upcoming", найближча
    майбутня); ("none", None), якщо в календарі нічого не лишилось."""
    upcoming = None
    for year in (now.year, now.year + 1):
        try:
            sessions = openf1_client._season_sessions(year)
        except Exception:
            continue
        for s in sessions:
            if s.get("is_cancelled"):
                continue
            opens, closes = live_window(s)
            if opens and closes and opens <= now <= closes:
                return "live", s
            if opens and opens > now and (upcoming is None or opens < live_window(upcoming)[0]):
                upcoming = s
        if upcoming:
            break
    return ("upcoming", upcoming) if upcoming else ("none", None)


def _outline_from_earlier_session(session: dict) -> Optional[list]:
    """Контур траси з попередньої сесії на тій самій трасі - на початку першої
    практики власних кіл ще немає."""
    start = parse_date(session.get("date_start"))
    year = start.year if start else datetime.now(timezone.utc).year
    candidates = []
    for y in (year, year - 1):
        try:
            candidates += [s for s in openf1_client._season_sessions(y)
                           if s.get("circuit_key") == session.get("circuit_key")
                           and s.get("session_key") != session.get("session_key")
                           and parse_date(s.get("date_end")) and parse_date(s["date_end"]) < (start or datetime.now(timezone.utc))]
        except Exception:
            continue
    for s in sorted(candidates, key=lambda s: s["date_start"], reverse=True)[:3]:
        try:
            outline = live_replay.track_outline(s, openf1_client.session_data(s, "laps"))
        except Exception:
            continue
        if outline:
            return outline
    return None


def calendar_ref(session: Optional[dict]) -> Optional[dict]:
    """{year, round, session} сесії OpenF1 за календарем FastF1 (номерів етапів
    в OpenF1 немає, а фронт за ними тягне прогноз і факти траси). Зіставляємо
    за часом старту, як openf1_client.find_session, але у зворотний бік."""
    start = parse_date(session.get("date_start")) if session else None
    if start is None:
        return None
    try:
        schedule = fastf1_client.get_season_schedule(start.year)
    except Exception:
        return None
    for event in schedule:
        for i in range(1, 6):
            name, when = event.get(f"Session{i}"), event.get(f"Session{i}DateUtc")
            if not name or not when:
                continue
            t = datetime.fromisoformat(str(when)).replace(tzinfo=timezone.utc)
            if abs(t - start) < timedelta(hours=2):
                code = fastf1_client.SESSION_NAME_TO_CODE.get(name, name)
                return {"year": start.year, "round": event.get("RoundNumber"), "session": code,
                        "event": event.get("EventName")}
    return None


INITIAL_TOPICS = ("laps", "stints", "pit", "race_control", "position", "intervals", "weather", "team_radio")


def prepare_live(session: dict) -> tuple[dict, list[tuple[str, dict]]]:
    """Синхронно (у потоці): вступне повідомлення session і записи сесії "на
    зараз" через REST (з токеном - у live-вікні OpenF1 вимагає підписку)."""
    small = {t: openf1_client.session_data(session, t) for t in ("drivers", *INITIAL_TOPICS)}
    try:
        outline = live_replay.track_outline(session, small["laps"]) or _outline_from_earlier_session(session)
    except Exception:
        outline = None  # фронт збере контур сам із точок location
    intro = {
        "type": "session",
        "mode": "live",
        "session": {k: session.get(k) for k in live_replay._INTRO_SESSION_FIELDS},
        "drivers": [{k: d.get(k) for k in live_replay._INTRO_DRIVER_FIELDS} for d in small["drivers"]],
        "track": {"points": outline} if outline else None,
        "grid": live_replay.starting_grid(session, small["position"]),
        "replay": None,
        "calendar": calendar_ref(session),
    }
    initial = [(t, r) for t in INITIAL_TOPICS for r in small[t]]
    initial.sort(key=lambda tr_: tr_[1].get("date") or tr_[1].get("date_start") or "")
    return intro, initial


# --- MQTT ------------------------------------------------------------------------

class MqttFeed:
    """Одне MQTT-підключення до OpenF1 у потоці paho. on_record викликається з
    цього потоку - хаб перекидає записи в event loop сам."""

    def __init__(self, on_record: RecordCallback, host: str = MQTT_HOST, port: int = MQTT_PORT):
        self._on_record = on_record
        self._host, self._port = host, port
        self._client = None

    def start(self) -> None:
        import paho.mqtt.client as mqtt  # лише коли live справді вмикається

        user, _ = openf1_client.credentials()
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"f1-predictor-{uuid.uuid4().hex[:8]}")
        client.tls_set()
        client.username_pw_set(user, openf1_client.get_token())
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.connect_async(self._host, self._port, keepalive=30)
        client.loop_start()
        self._client = client

    def stop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.disconnect()
            client.loop_stop()

    def _on_connect(self, client, _userdata, _flags, reason_code, _properties) -> None:
        if reason_code.is_failure:
            log.warning("OpenF1 MQTT: підключення відхилено (%s)", reason_code)
            return
        client.subscribe([(f"v1/{t}", 0) for t in TOPICS])
        log.info("OpenF1 MQTT: підключено, підписка на %d топіків", len(TOPICS))

    def _on_disconnect(self, client, _userdata, _flags, reason_code, _properties) -> None:
        # paho перепідключиться сам із тими ж обліковими даними. Токен живе
        # годину, тож перед повтором кладемо свіжий (а після відмови в доступі -
        # примусово новий).
        log.warning("OpenF1 MQTT: розрив (%s)", reason_code)
        if self._client is None:
            return
        try:
            denied = "authori" in str(reason_code).lower() or "password" in str(reason_code).lower()
            client.username_pw_set(openf1_client.credentials()[0], openf1_client.get_token(force=denied))
        except Exception as e:  # мережа до /token теж може лежати - paho спробує ще раз
            log.warning("OpenF1: не вдалося оновити токен: %s", e)

    def _on_message(self, _client, _userdata, msg) -> None:
        import json

        try:
            record = json.loads(msg.payload)
        except ValueError:
            return
        if isinstance(record, dict):
            self._on_record(msg.topic.rsplit("/", 1)[-1], record)


# --- хаб -------------------------------------------------------------------------

class LiveHub:
    """Одна live-сесія на процес: MQTT + стан + розсилка глядачам."""

    def __init__(self, feed_factory: Callable[[RecordCallback], object] = MqttFeed,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 batch_seconds: float = BATCH_SECONDS, idle_stop_seconds: float = IDLE_STOP_SECONDS):
        self._feed_factory = feed_factory
        self._now = now
        self._batch_seconds = batch_seconds
        self._idle_stop_seconds = idle_stop_seconds
        self._lock: Optional[asyncio.Lock] = None  # створюємо в event loop, не під час імпорту
        self._clients: set = set()
        self.session: Optional[dict] = None
        self._intro: Optional[dict] = None
        self._state: Optional[LiveState] = None
        self._feed = None
        self._pending: list = []
        self._ticker: Optional[asyncio.Task] = None
        self._stopper: Optional[asyncio.TimerHandle] = None

    async def join(self, session: dict) -> tuple[dict, dict, asyncio.Queue]:
        """Підписує глядача. Повертає (session, batch catch-up, черга батчів)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._stopper is not None:
                self._stopper.cancel()
                self._stopper = None
            if self.session is None or self.session.get("session_key") != session.get("session_key"):
                await self._start(session)
            queue: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_QUEUE)
            self._clients.add(queue)
            catchup = {"type": "batch", "catchup": True, "clock": _iso(self._state.clock), "events": self._state.catchup()}
            return self._intro, catchup, queue

    def leave(self, queue: asyncio.Queue) -> None:
        self._clients.discard(queue)
        if not self._clients and self.session is not None and self._stopper is None:
            loop = asyncio.get_running_loop()
            self._stopper = loop.call_later(self._idle_stop_seconds, lambda: asyncio.ensure_future(self.stop()))

    @property
    def viewers(self) -> int:
        return len(self._clients)

    async def _start(self, session: dict) -> None:
        await self.stop(notify=True)
        loop = asyncio.get_running_loop()
        intro, initial = await asyncio.to_thread(prepare_live, session)
        state = LiveState()
        for topic, record in initial:
            state.ingest(topic, record)
        self.session, self._intro, self._state, self._pending = session, intro, state, []
        key = session.get("session_key")

        def on_record(topic: str, record: dict) -> None:  # потік paho
            if record.get("session_key") in (None, key):
                loop.call_soon_threadsafe(self._on_record, topic, record)

        self._feed = self._feed_factory(on_record)
        await asyncio.to_thread(self._feed.start)
        self._ticker = asyncio.ensure_future(self._tick())

    def _on_record(self, topic: str, record: dict) -> None:
        if self._state is not None:
            self._pending.extend(self._state.ingest(topic, record))

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(self._batch_seconds)
            if self._pending:
                events, self._pending = self._pending, []
                events.sort(key=_event_time)
                self._broadcast({"type": "batch", "clock": _iso(self._state.clock), "events": events})
            closes = live_window(self.session)[1]
            if closes and self._now() > closes:
                self._broadcast({"type": "end", "clock": _iso(self._state.clock)})
                await self.stop(from_ticker=True)
                return

    def _broadcast(self, message: dict) -> None:
        for queue in list(self._clients):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # Глядач не встигає (повільна мережа) - відключаємо, щоб не
                # тримати пам'ять. Фронт перепідключиться й отримає свіжий стан.
                self._clients.discard(queue)
                queue.dropped = True

    async def stop(self, notify: bool = False, from_ticker: bool = False) -> None:
        if self._stopper is not None:
            self._stopper.cancel()
            self._stopper = None
        if self._ticker is not None and not from_ticker:
            self._ticker.cancel()
        self._ticker = None
        feed, self._feed = self._feed, None
        if feed is not None:
            await asyncio.to_thread(feed.stop)
        if notify and self.session is not None:
            self._broadcast({"type": "end", "clock": _iso(self._state.clock) if self._state else None})
        for queue in list(self._clients):
            queue.dropped = True
        self._clients.clear()
        self.session = self._intro = self._state = None
        self._pending = []


hub = LiveHub()


# --- WebSocket -------------------------------------------------------------------

def waiting_message(status: str, session: Optional[dict]) -> dict:
    opens = live_window(session)[0] if session else None
    return {
        "type": "waiting",
        "status": status,
        "session": {k: session.get(k) for k in live_replay._INTRO_SESSION_FIELDS} if session else None,
        "opens_at": _iso(opens),
        "calendar": calendar_ref(session),
    }


def latest_finished_race(now: datetime) -> Optional[dict]:
    """Остання гонка, чиє live-вікно вже закрилось (для replay): сесія OpenF1."""
    best = None
    for year in (now.year, now.year - 1):
        try:
            sessions = openf1_client._season_sessions(year)
        except Exception:
            continue
        for s in sessions:
            closes = live_window(s)[1]
            if s.get("session_type") == "Race" and not s.get("is_cancelled") and closes and closes < now:
                if best is None or s["date_start"] > best["date_start"]:
                    best = s
        if best:
            break
    return best


def live_status(now: Optional[datetime] = None) -> dict:
    """Для REST /api/live/status. Календар OpenF1 доступний і без підписки, тож
    найближчу сесію (і таймер до неї) показуємо завжди; configured каже, чи
    сервер зможе віддати саму трансляцію. latest_replay - остання завершена
    гонка (тимчасова кнопка "Тестовий replay" на фронті)."""
    now = now or datetime.now(timezone.utc)
    configured = openf1_client.configured()
    status, session = find_live_session(now)
    body = waiting_message(status, session)
    return {
        "configured": configured,
        "status": status,
        "session": body["session"],
        "opens_at": body["opens_at"],
        "calendar": body["calendar"],
        "viewers": hub.viewers if status == "live" else 0,
        "latest_replay": calendar_ref(latest_finished_race(now)),
    }


async def serve(websocket: WebSocket, live_hub: LiveHub = None, now: Callable[[], datetime] = None) -> None:
    """WS /ws/live: session → batch catch-up → батчі наживо → end."""
    live_hub = live_hub or hub
    now = now or (lambda: datetime.now(timezone.utc))
    await websocket.accept()
    if not openf1_client.configured():
        await websocket.send_json({"type": "error", "detail": tr("live_not_configured")})
        await websocket.close(code=1011)
        return
    try:
        status, session = await asyncio.to_thread(find_live_session, now())
        if status != "live":
            await websocket.send_json(waiting_message(status, session))
            await websocket.close(code=1000)
            return
        intro, catchup, queue = await live_hub.join(session)
    except Exception as e:
        log.exception("live: не вдалося підключити глядача")
        await websocket.send_json({"type": "error", "detail": tr("live_unavailable", error=e)})
        await websocket.close(code=1011)
        return

    async def drain_client() -> None:
        # Команд у live немає (ні паузи, ні швидкості), але читати треба, щоб
        # помітити відключення.
        while True:
            await websocket.receive_text()

    listen = asyncio.ensure_future(drain_client())
    try:
        await websocket.send_json(intro)
        await websocket.send_json(catchup)
        while not listen.done():
            try:
                message = await asyncio.wait_for(queue.get(), timeout=5)
            except asyncio.TimeoutError:
                if getattr(queue, "dropped", False):
                    break
                continue
            await websocket.send_json(message)
            if message.get("type") == "end":
                await websocket.close()
                break
        if getattr(queue, "dropped", False) and not listen.done():
            await websocket.close(code=1013)  # "спробуйте ще": фронт перепідключиться
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        listen.cancel()
        live_hub.leave(queue)

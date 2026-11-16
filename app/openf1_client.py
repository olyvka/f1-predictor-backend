"""
Клієнт OpenF1 (api.openf1.org) - запасне джерело кіл/погоди для сесій,
яких FastF1 більше не може завантажити.

Чому: з жовтня 2026 livetiming.formula1.com/static (звідки FastF1 бере кола,
телеметрію й погоду) відповідає 401 на запити без авторизації. Сесії, що вже
лежать у кеші FastF1 (cache/), працюють і далі; для решти - OpenF1. Він має
дані з сезону 2023 і віддає їх у власному форматі, тож app/fallback.py
перекладає їх у ті самі поля, що й FastF1.

OpenF1 обмежує частоту: 3 запити/с і 30 запитів/хв без підписки (відповідь
429 з Retry-After, який буває до хвилини). Тримаємо паузу між запитами і
рахуємо запити за останню хвилину, щоб не впиратися в 429; після 429
все одно повторюємо. Історичні дані не змінюються,
тож відповіді кешуємо на диск (cache/openf1/) - повторне відкриття сесії не
ходить у мережу взагалі.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from app import fastf1_client
from app.i18n import tr

log = logging.getLogger(__name__)

API = "https://api.openf1.org/v1"
TOKEN_URL = "https://api.openf1.org/token"
FIRST_YEAR = 2023  # раніших сезонів в OpenF1 немає

_CACHE_DIR = os.path.join(fastf1_client.CACHE_DIR, "openf1")
os.makedirs(_CACHE_DIR, exist_ok=True)

_MIN_INTERVAL = 0.4  # с між запитами: трохи повільніше за ліміт 3/с
# Хвилинний ліміт: 30 без підписки, 60 з підпискою Sponsor. Беремо з
# невеликим запасом, бо лічильник OpenF1 і наш годинник не збігаються.
_PER_MINUTE = int(os.getenv("OPENF1_PER_MINUTE", "28"))
_lock = threading.Lock()
_last_request = 0.0
_recent: collections.deque = collections.deque()  # monotonic-час останніх запитів

# Список сесій сезону змінюється (додаються нові етапи), тож його не кладемо
# на диск, а тримаємо в пам'яті годину.
_SESSIONS_TTL = 3600
_sessions_by_year: dict[int, tuple[float, list[dict]]] = {}


# --- платний доступ (live) -----------------------------------------------
# Логін і пароль підписки OpenF1 (Sponsor) - у .env. Без них усе працює як
# раніше: історичні дані без авторизації, live недоступний.
_TOKEN_MARGIN = 300  # с: оновлюємо токен за 5 хв до кінця (живе годину)
_token_lock = threading.Lock()
_token: Optional[str] = None
_token_expires = 0.0
_token_failed_until = 0.0  # після невдачі не смикаємо /token на кожен запит
_TOKEN_RETRY = 60


def credentials() -> tuple[Optional[str], Optional[str]]:
    return os.getenv("OPENF1_USERNAME") or None, os.getenv("OPENF1_PASSWORD") or None


def configured() -> bool:
    user, password = credentials()
    return bool(user and password)


def get_token(force: bool = False) -> Optional[str]:
    """OAuth2-токен OpenF1 або None, якщо підписку не налаштовано. Обмін
    логіна/пароля на токен - лише тут, на бекенді (вимога OpenF1)."""
    global _token, _token_expires, _token_failed_until
    user, password = credentials()
    if not (user and password):
        return None
    with _token_lock:
        if not force and _token and time.time() < _token_expires - _TOKEN_MARGIN:
            return _token
        if not force and time.time() < _token_failed_until:
            raise RuntimeError("OpenF1 token unavailable (recent failure)")
        try:
            res = requests.post(TOKEN_URL, data={"username": user, "password": password}, timeout=(10, 30))
            res.raise_for_status()
        except Exception:
            _token_failed_until = time.time() + _TOKEN_RETRY
            raise
        body = res.json()
        _token = body["access_token"]
        _token_expires = time.time() + float(body.get("expires_in", 3600))
        return _token


class OpenF1Unavailable(Exception):
    """Для сесії немає даних в OpenF1 (сезон до 2023, сесію не знайдено тощо)."""


def _respect_limits() -> None:
    """Чекає, доки новий запит не порушить ні 3/с, ні хвилинний ліміт.
    Викликається під _lock."""
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    while _recent and time.monotonic() - _recent[0] >= 60:
        _recent.popleft()
    if len(_recent) >= _PER_MINUTE:
        time.sleep(max(0.0, 60 - (time.monotonic() - _recent[0])) + 0.05)
        while _recent and time.monotonic() - _recent[0] >= 60:
            _recent.popleft()


def _get(endpoint: str, params: dict, *, disk_cache: bool = True) -> list[dict]:
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    path = f"{endpoint}?{query}"
    cache_file = os.path.join(_CACHE_DIR, hashlib.sha1(path.encode()).hexdigest() + ".json")
    if disk_cache and os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    global _last_request
    refreshed = False
    for attempt in range(5):
        try:
            token = get_token()
        except Exception as e:
            # Неправильний пароль чи недоступний /token не мають ламати історичні
            # дані - вони доступні й без авторизації. Live REST тоді отримає 401.
            log.warning("OpenF1: не вдалося отримати токен, запит без авторизації: %s", e)
            token = None
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        # Лок серіалізує запити з різних потоків Starlette, щоб разом вони
        # не перевищили ліміт OpenF1.
        with _lock:
            _respect_limits()
            res = requests.get(f"{API}/{path}", headers=headers, timeout=(10, 30))
            _last_request = time.monotonic()
            _recent.append(_last_request)
        if res.status_code == 401 and token and not refreshed:
            get_token(force=True)  # токен відкликали або сплив раніше
            refreshed = True
            continue
        if res.status_code == 429:
            time.sleep(float(res.headers.get("retry-after", 1)) + attempt * 0.5)
            continue
        # "No results found" OpenF1 віддає як 404 - для нас це порожній список.
        if res.status_code == 404:
            data = []
            break
        res.raise_for_status()
        data = res.json()
        break
    else:
        raise OpenF1Unavailable(tr("openf1_rate_limited", path=path))

    if disk_cache:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _season_sessions(year: int) -> list[dict]:
    cached = _sessions_by_year.get(year)
    if cached and time.monotonic() - cached[0] < _SESSIONS_TTL:
        return cached[1]
    sessions = _get("sessions", {"year": year}, disk_cache=False)
    _sessions_by_year[year] = (time.monotonic(), sessions)
    return sessions


def find_session(year: int, round_number: int, session: str) -> dict:
    """Сесія OpenF1 для (рік, етап, код сесії FastF1). Номерів етапів в
    OpenF1 немає, тож зіставляємо за часом старту сесії з календаря FastF1
    (календар FastF1 приходить не з живого таймінгу й працює далі)."""
    if year < FIRST_YEAR:
        raise OpenF1Unavailable(tr("openf1_since", year=FIRST_YEAR))

    code = fastf1_client.SESSION_NAME_TO_CODE.get(session, session).upper()
    wanted = next(
        (s for s in fastf1_client.get_event_sessions(year, round_number) if s["code"].upper() == code),
        None,
    )
    if not wanted or not wanted.get("date_utc"):
        raise OpenF1Unavailable(tr("openf1_session_not_in_calendar", session=session, round=round_number))

    start = datetime.fromisoformat(wanted["date_utc"]).replace(tzinfo=timezone.utc)
    for s in _season_sessions(year):
        begins = parse_date(s.get("date_start"))
        if begins and abs(begins - start) < timedelta(hours=2):
            return s
    raise OpenF1Unavailable(tr("openf1_session_not_found", session=session, round=round_number, year=year))


def _settled(session: dict) -> bool:
    """Сесію, що закінчилась менше 6 годин тому, OpenF1 ще може доповнювати -
    її відповіді на диск не кладемо."""
    ended = parse_date(session.get("date_end"))
    return ended is None or datetime.now(timezone.utc) - ended > timedelta(hours=6)


def session_data(session: dict, endpoint: str) -> list[dict]:
    """Усі записи ендпоінта для сесії (laps, stints, pit, position, weather,
    race_control, drivers) - одним запитом, без фільтра за пілотом."""
    return _get(endpoint, {"session_key": session["session_key"]}, disk_cache=_settled(session))


def _query_time(t: datetime) -> str:
    # Без зсуву: OpenF1 читає такий час як UTC, а "+00:00" у неекранованому
    # URL перетворився б на пробіл.
    return t.astimezone(timezone.utc).replace(tzinfo=None).isoformat()


def session_window(
    session: dict, endpoint: str, since: datetime, until: datetime, *, disk_cache: bool = True, **filters,
) -> list[dict]:
    """Записи ендпоінта сесії з date у [since, until] (обидві межі включно -
    _get склеює ключ "date>" і значення в "date>=..."). Для location /
    car_data, які на всю сесію займають сотні тисяч записів."""
    params = {
        "session_key": session["session_key"],
        "date>": _query_time(since),
        "date<": _query_time(until),
        **filters,
    }
    return _get(endpoint, params, disk_cache=disk_cache and _settled(session))

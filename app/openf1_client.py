"""
Клієнт OpenF1 (api.openf1.org) - запасне джерело кіл/погоди для сесій,
яких FastF1 більше не може завантажити.

Чому: з жовтня 2026 livetiming.formula1.com/static (звідки FastF1 бере кола,
телеметрію й погоду) відповідає 401 на запити без авторизації. Сесії, що вже
лежать у кеші FastF1 (cache/), працюють і далі; для решти - OpenF1. Він має
дані з сезону 2023 і віддає їх у власному форматі, тож app/fallback.py
перекладає їх у ті самі поля, що й FastF1.

OpenF1 обмежує частоту: 3 запити/с (відповідь 429 з Retry-After). Тримаємо
паузу між запитами й повторюємо після 429. Історичні дані не змінюються,
тож відповіді кешуємо на диск (cache/openf1/) - повторне відкриття сесії не
ходить у мережу взагалі.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

from app import fastf1_client
from app.i18n import tr

API = "https://api.openf1.org/v1"
FIRST_YEAR = 2023  # раніших сезонів в OpenF1 немає

_CACHE_DIR = os.path.join(fastf1_client.CACHE_DIR, "openf1")
os.makedirs(_CACHE_DIR, exist_ok=True)

_MIN_INTERVAL = 0.4  # с між запитами: трохи повільніше за ліміт 3/с
_lock = threading.Lock()
_last_request = 0.0

# Список сесій сезону змінюється (додаються нові етапи), тож його не кладемо
# на диск, а тримаємо в пам'яті годину.
_SESSIONS_TTL = 3600
_sessions_by_year: dict[int, tuple[float, list[dict]]] = {}


class OpenF1Unavailable(Exception):
    """Для сесії немає даних в OpenF1 (сезон до 2023, сесію не знайдено тощо)."""


def _get(endpoint: str, params: dict, *, disk_cache: bool = True) -> list[dict]:
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    path = f"{endpoint}?{query}"
    cache_file = os.path.join(_CACHE_DIR, hashlib.sha1(path.encode()).hexdigest() + ".json")
    if disk_cache and os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    global _last_request
    for attempt in range(5):
        # Лок серіалізує запити з різних потоків Starlette, щоб разом вони
        # не перевищили ліміт OpenF1.
        with _lock:
            wait = _MIN_INTERVAL - (time.monotonic() - _last_request)
            if wait > 0:
                time.sleep(wait)
            res = requests.get(f"{API}/{path}", timeout=(10, 30))
            _last_request = time.monotonic()
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


def session_data(session: dict, endpoint: str) -> list[dict]:
    """Усі записи ендпоінта для сесії (laps, stints, pit, position, weather,
    race_control, drivers) - одним запитом, без фільтра за пілотом. Сесію,
    що закінчилась менше 6 годин тому, OpenF1 ще може доповнювати - її
    відповіді на диск не кладемо."""
    ended = parse_date(session.get("date_end"))
    settled = ended is None or datetime.now(timezone.utc) - ended > timedelta(hours=6)
    return _get(endpoint, {"session_key": session["session_key"]}, disk_cache=settled)

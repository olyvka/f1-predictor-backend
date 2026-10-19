"""
Обгортка над бібліотекою FastF1.
Тримаємо всю роботу з FastF1 в одному місці, щоб легко було
підмінити джерело даних (наприклад, на власну модель передбачення).
"""
# X | None у типах нижче - синтаксис PEP 604, який без цього імпорту
# потребує Python 3.10+; з ним анотації обчислюються лениво і працюють на 3.9.
from __future__ import annotations

import os
import threading
import fastf1
import pandas as pd
import requests
from app.i18n import tr

# Скільки чекати мережеву відповідь від джерел даних FastF1:
# (timeout з'єднання, timeout читання відповіді), секунди.
_HTTP_TIMEOUT = (10, 30)

# FastF1 не потокобезпечний: його SQLite HTTP-кеш і pickle-кеш (stage 1/2)
# не розраховані на одночасний доступ з кількох потоків. Ендпоінти в
# app/main.py - звичайні синхронні def-функції, тому Starlette виконує
# кожен запит у ОКРЕМОМУ потоці зі свого threadpool. Якщо фронт одночасно
# шле кілька запитів (наприклад, одна сторінка одразу тягне results +
# weather + laps), кілька потоків можуть одночасно звертатись до того
# самого SQLite-конекшна кешу - це реально валить процес по SIGSEGV
# усередині libsqlite3 (пошкодження пам'яті при паралельному доступі).
# Цей лок серіалізує будь-яку взаємодію з FastF1 (мережа/кеш) в один
# потік за раз - ціна цього: запити до FastF1-даних виконуються по черзі
# замість паралельно, але це прийнятно і набагато краще, ніж падіння
# сервера.
_FASTF1_LOCK = threading.Lock()


def _patch_requests_default_timeout() -> None:
    """FastF1 (fastf1/req.py: Cache.requests_get/requests_post) робить усі
    мережеві виклики через requests.Session(), і ЖОДЕН виклик усередині
    бібліотеки не передає timeout. У requests це означає "чекати відповідь
    необмежено довго" (timeout=None за замовчуванням). Якщо джерело даних
    підвисає чи мовчить після встановлення з'єднання (наприклад, запит
    календаря сезону, якого ще фактично не існує, як 2026-го задовго до
    старту), потік FastAPI/Starlette-threadpool, що обробляє цей запит,
    блокується НАЗАВЖДИ: клієнтський таймаут/AbortController скасовує лише
    HTTP-запит у браузері, серверного потоку це не стосується. У Starlette
    спільний пул потоків фіксованого розміру для всіх синхронних
    def-ендпоінтів — кілька таких зависань поспіль вичерпують пул цілком,
    і тоді підвисають навіть тривіальні ендпоінти на кшталт "/" чи "/docs",
    бо їм просто нема на чому виконуватись.

    Патчимо requests.Session.request так, щоб timeout був завжди заданий,
    якщо викликач (тут - fastf1) сам його не задав. Це найнижчий рівень,
    де можна перехопити виклик: fastf1 не дає власного конфіга для timeout.
    """
    if getattr(requests.Session.request, "_f1_timeout_patched", False):
        return  # вже пропатчено (напр. повторний імпорт модуля) - не дублюємо

    original_request = requests.Session.request

    def _request_with_default_timeout(self, method, url, **kwargs):
        kwargs.setdefault("timeout", _HTTP_TIMEOUT)
        return original_request(self, method, url, **kwargs)

    _request_with_default_timeout._f1_timeout_patched = True
    requests.Session.request = _request_with_default_timeout


_patch_requests_default_timeout()

CACHE_DIR = os.getenv("FASTF1_CACHE_DIR", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Короткі коди сесій, які приймає fastf1.get_session(), і їх людські назви.
# FP1-3 - практики, Q - кваліфікація, S - спринт, SQ - спринт-кваліфікація
# (на вікендах спринту заміняє звичайну практику), R - гонка.
SESSION_NAME_TO_CODE = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Sprint": "S",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",
    "Race": "R",
}


def _clean_value(value):
    """Одне значення FastF1/pandas -> JSON-сумісне значення.

    NaN/NaT -> None; Timedelta (час кола, сектори, відставання) -> секунди
    (float, зручно для сортування/порівняння на фронті); Timestamp -> ISO-рядок.
    """
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _clean(df: pd.DataFrame) -> list[dict]:
    """FastF1 повертає NaN/NaT/Timedelta, які не серіалізуються в JSON напряму.

    df.where(pd.notnull(df), None) тут не підходить:
    - на float64-колонках (Position, Points тощо) pandas все одно
      мовчки повертає NaN назад замість None, бо None не вміщується
      в float64-масив, а FastAPI/Starlette серіалізує з allow_nan=False
      і падає з "Out of range float values are not JSON compliant";
    - якщо серед колонок є datetime, конвертований в рядок через
      astype(str), pd.notnull(df) на рівні всього DataFrame може впасти
      з "bad operand type for unary ~: 'DatetimeArray'" через те, як
      pandas консолідує внутрішні блоки.

    Тому чистимо значення поскалярно вже після to_dict().
    """
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            record[key] = _clean_value(value)
    return records


def _clean_series(series: pd.Series) -> dict:
    """Те саме, що _clean(), але для одного рядка (pd.Series/fastf1.core.Lap тощо)."""
    return {key: _clean_value(value) for key, value in series.to_dict().items()}


def _load_session(year: int, round_number: int, session: str, **load_kwargs):
    """Завантажує сесію FastF1. session приймає короткі коди (R, Q, S, SQ,
    FP1, FP2, FP3) або повні назви (Race, Qualifying, Practice 1, ...) -
    саме так, як їх розуміє сама бібліотека FastF1.

    Під локом (_FASTF1_LOCK): і get_session(), і load() можуть звертатись
    до мережі/кешу FastF1, який не є потокобезпечним (див. коментар біля
    _FASTF1_LOCK)."""
    with _FASTF1_LOCK:
        fsession = fastf1.get_session(year, round_number, session)
        fsession.load(**load_kwargs)
    return fsession


def get_season_schedule(year: int) -> list[dict]:
    """Календар сезону: етапи разом з датами всіх сесій вікенду
    (практики/кваліфікація/спринт/гонка), форматом вікенду і офіційною назвою."""
    with _FASTF1_LOCK:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    cols = [
        "RoundNumber",
        "EventName",
        "OfficialEventName",
        "Country",
        "Location",
        "EventDate",
        "EventFormat",
        "Session1", "Session1DateUtc",
        "Session2", "Session2DateUtc",
        "Session3", "Session3DateUtc",
        "Session4", "Session4DateUtc",
        "Session5", "Session5DateUtc",
    ]
    df = schedule[[c for c in cols if c in schedule.columns]].copy()
    return _clean(df)


def get_event_sessions(year: int, round_number: int) -> list[dict]:
    """Список сесій конкретного етапу (FP1/FP2/FP3/Q/S/SQ/R) з датами й
    коротким кодом, який можна одразу підставляти в інші ендпоінти."""
    with _FASTF1_LOCK:
        event = fastf1.get_event(year, round_number)
    sessions = []
    for i in range(1, 6):
        name = event.get(f"Session{i}")
        if not name or pd.isna(name):
            continue
        sessions.append({
            "number": i,
            "name": name,
            "code": SESSION_NAME_TO_CODE.get(name, name),
            "date": _clean_value(event.get(f"Session{i}Date")),
            "date_utc": _clean_value(event.get(f"Session{i}DateUtc")),
        })
    return sessions


def get_session_results(year: int, round_number: int, session: str) -> list[dict]:
    """Повні результати довільної сесії: гонка, кваліфікація, спринт,
    спринт-кваліфікація, практика. Набір колонок віддає сама FastF1 (біо
    пілота, команда, колір команди, стартова решітка, Q1/Q2/Q3, час/статус,
    очки) - заповненість колонок залежить від типу сесії (наприклад, Q1-Q3
    мають сенс лише для кваліфікацій, Points/Status - для гонки/спринту)."""
    fsession = _load_session(
        year, round_number, session,
        laps=False, telemetry=False, weather=False, messages=False,
    )
    return _clean(fsession.results)


def get_race_results(year: int, round_number: int) -> list[dict]:
    """Сумісність зі старим API: результати гонки = get_session_results(..., "R")."""
    return get_session_results(year, round_number, "R")


def get_session_drivers(year: int, round_number: int, session: str) -> list[dict]:
    """Короткий список пілотів сесії - зручно для вибору перед запитом
    /api/driver/... або /api/compare/..."""
    fsession = _load_session(
        year, round_number, session,
        laps=False, telemetry=False, weather=False, messages=False,
    )
    cols = [
        "DriverNumber", "Abbreviation", "FullName", "TeamName", "TeamColor",
        "Position", "GridPosition", "Status", "Points",
    ]
    df = fsession.results[[c for c in cols if c in fsession.results.columns]]
    return _clean(df)


def get_session_laps(
    year: int, round_number: int, session: str, drivers: list[str] | None = None
) -> list[dict]:
    """Усі кола сесії (час кола, сектори, швидкості на пастках, шини, стінт,
    позиція на колі тощо), опційно відфільтровані по списку пілотів."""
    fsession = _load_session(
        year, round_number, session,
        laps=True, telemetry=False, weather=False, messages=False,
    )
    laps = fsession.laps
    if drivers:
        laps = laps.pick_drivers(drivers)
    return _clean(laps)


def get_session_weather(year: int, round_number: int, session: str) -> list[dict]:
    """Погодні дані сесії протягом усього заїзду: температура повітря й
    траси, вологість, тиск, вітер, наявність дощу."""
    fsession = _load_session(
        year, round_number, session,
        laps=False, telemetry=False, weather=True, messages=False,
    )
    return _clean(fsession.weather_data)


def get_driver_summary(year: int, round_number: int, session: str, driver: str) -> dict:
    """Детальна інформація по одному пілоту в конкретній сесії: біографічні
    дані, підсумковий результат, розбивка по стінтах (шини/кількість кіл),
    найшвидше коло та повний список кіл."""
    fsession = _load_session(
        year, round_number, session,
        laps=True, telemetry=False, weather=False, messages=False,
    )
    try:
        driver_info = _clean_series(fsession.get_driver(driver))
    except ValueError as e:
        raise ValueError(f"{tr('driver_not_in_session', driver=driver)}: {e}")

    abbreviation = driver_info.get("Abbreviation", driver)
    laps = fsession.laps.pick_drivers([abbreviation])

    result_rows = _clean(
        fsession.results[fsession.results["Abbreviation"] == abbreviation]
    )
    result = result_rows[0] if result_rows else None

    stints = []
    for stint_no, stint_laps in laps.groupby("Stint"):
        stints.append({
            "stint": _clean_value(stint_no),
            "compound": stint_laps["Compound"].iloc[0],
            "laps": int(len(stint_laps)),
            "first_lap": _clean_value(stint_laps["LapNumber"].min()),
            "last_lap": _clean_value(stint_laps["LapNumber"].max()),
        })

    fastest_lap = None
    if not laps.empty:
        fl = laps.pick_fastest()
        if fl is not None and not fl.empty:
            fastest_lap = _clean_series(fl)

    return {
        "driver": driver_info,
        "result": result,
        "stints": stints,
        "fastest_lap": fastest_lap,
        "laps": _clean(laps),
    }


def compare_drivers_laps(
    year: int, round_number: int, session: str, drivers: list[str]
) -> dict:
    """Порівняння кількох пілотів по колах однієї сесії: для кожного пілота -
    час кола, сектори, шини, позиція на колі й найшвидше коло. Зручно для
    побудови графіків темпу/відставання на фронті."""
    fsession = _load_session(
        year, round_number, session,
        laps=True, telemetry=False, weather=False, messages=False,
    )
    lap_cols = [
        "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "Compound", "TyreLife", "Stint", "Position", "IsPersonalBest",
    ]
    comparison = {}
    for drv in drivers:
        drv_laps = fsession.laps.pick_drivers([drv])
        if drv_laps.empty:
            comparison[drv] = {"laps": [], "fastest_lap": None}
            continue
        fl = drv_laps.pick_fastest()
        comparison[drv] = {
            "laps": _clean(drv_laps[[c for c in lap_cols if c in drv_laps.columns]]),
            "fastest_lap": _clean_series(fl) if fl is not None and not fl.empty else None,
        }
    return comparison


def get_driver_standings_snapshot(year: int, round_number: int) -> list[dict]:
    """Проста агрегація очок команди/пілота після конкретного етапу.
    (Для повноцінного заліку сезону краще підсумовувати по всіх етапах —
    це буде наступний крок у твоїй логіці.)
    """
    results = get_race_results(year, round_number)
    return sorted(results, key=lambda r: (r["Position"] is None, r["Position"]))

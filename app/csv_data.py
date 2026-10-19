"""
Обгортка над локальним статичним датасетом у data/ (Ergast-схема: сезони,
гонки, пілоти, конструктори, підсумкові залікові очки по кожному етапу).

Цей датасет використовуємо для сезонних зведень - календар року, залік
пілотів/конструкторів: ці дані історичні, не змінюються між запитами, і
не мають сенсу тягнути щоразу через мережу/FastF1.

Деталі конкретної сесії (кола, погода, телеметрія, порівняння пілотів по
колах) і далі йдуть через app/fastf1_client.py - у CSV-датасеті цього
просто немає.
"""
from __future__ import annotations

import os

import pandas as pd
from app.i18n import tr

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _read_csv(name: str) -> pd.DataFrame:
    # Ergast позначає відсутні значення як літеральний рядок "\N"
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"), na_values=["\\N"])


# Датасет статичний і невеликий - читаємо один раз при імпорті модуля,
# а не на кожен запит.
_seasons = _read_csv("seasons")
_races = _read_csv("races")
_drivers = _read_csv("drivers")
_constructors = _read_csv("constructors")
_driver_standings = _read_csv("driver_standings")
_constructor_standings = _read_csv("constructor_standings")
_results = _read_csv("results")
_status = _read_csv("status")
_qualifying = _read_csv("qualifying")
_sprint_results = _read_csv("sprint_results")
_pit_stops = _read_csv("pit_stops")

# Заздалегідь відібрані й перейменовані колонки для мерджів нижче - щоб не
# тягнути зайві стовпці (url, dob тощо) і уникнути конфлікту імен колонок
# "name"/"number" (є і в drivers.csv, і в constructors.csv).
_drivers_slim = _drivers[["driverId", "number", "code", "forename", "surname", "nationality"]].rename(
    columns={"number": "driver_number"}
)
_constructors_slim = _constructors[["constructorId", "name"]].rename(columns={"name": "team"})
_status_slim = _status[["statusId", "status"]]

# Короткі коди сесій -> які з них взагалі мають "результати" в Ergast-датасеті.
# Практики (FP1-3) тут не мають сенсу: Ergast не веде результатів практик,
# лише кола гонки (лежать не в цьому модулі); для практик - тільки FastF1.
_SESSION_ALIASES = {
    "R": "R", "RACE": "R",
    "Q": "Q", "QUALIFYING": "Q",
    "S": "S", "SPRINT": "S",
}


def _clean_records(df: pd.DataFrame) -> list[dict]:
    """NaN (з "\\N" у вихідному CSV) -> None, щоб коректно серіалізувалось у JSON."""
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _time_to_seconds(value) -> float | None:
    """Рядок часу з CSV ("1:34:50.616" / "1:26.572" / "+5.478" / "49.111")
    -> кількість секунд (float). Узгоджено з наскрізною вимогою фронта
    (FRONTEND_REQUIREMENTS.md, п.0.7): час скрізь віддається як число
    секунд, а не рядок."""
    if value is None or not isinstance(value, str):
        return None
    try:
        parts = [float(p) for p in value.split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return round(seconds, 3)


def _race_id(year: int, round_number: int) -> int:
    race = _races[(_races["year"] == year) & (_races["round"] == round_number)]
    if race.empty:
        raise ValueError(tr("round_not_found", round=round_number, year=year))
    return int(race.iloc[0]["raceId"])


def _last_race_id_of_season(year: int) -> int:
    """raceId останнього за розкладом (найбільший round) етапу сезону -
    саме на ньому driver_standings/constructor_standings містять підсумкові
    очки за весь сезон (це кумулятивний залік по кожній гонці)."""
    season_races = _races[_races["year"] == year]
    if season_races.empty:
        raise ValueError(tr("season_not_found", year=year))
    return season_races.loc[season_races["round"].idxmax(), "raceId"]


def list_seasons() -> list[int]:
    """Усі сезони, наявні в датасеті - для випадаючого списку на фронті."""
    return sorted(int(y) for y in _seasons["year"].tolist())


def get_season_races(year: int) -> list[dict]:
    """Календар сезону з датасету: раунд, назва Гран-прі, дата й час гонки
    та (де відомо) дати практик/кваліфікації/спринту."""
    races = _races[_races["year"] == year].sort_values("round")
    if races.empty:
        raise ValueError(tr("season_not_found", year=year))
    cols = [
        "raceId", "round", "name", "date", "time",
        "fp1_date", "fp1_time", "fp2_date", "fp2_time", "fp3_date", "fp3_time",
        "quali_date", "quali_time", "sprint_date", "sprint_time", "url",
    ]
    return _clean_records(races[[c for c in cols if c in races.columns]])


def get_driver_standings(year: int) -> list[dict]:
    """Підсумковий залік пілотів сезону: кумулятивні очки/позиція/перемоги
    з driver_standings.csv станом на останній етап сезону, доєднані з
    іменем і кодом пілота з drivers.csv."""
    last_race_id = _last_race_id_of_season(year)
    standings = _driver_standings[_driver_standings["raceId"] == last_race_id]
    merged = standings.merge(_drivers, on="driverId", how="left").sort_values("position")
    merged = merged.astype({"position": "Int64", "wins": "Int64"})
    cols = ["position", "points", "wins", "driverId", "code", "forename", "surname", "nationality"]
    return _clean_records(merged[[c for c in cols if c in merged.columns]])


def get_constructor_standings(year: int) -> list[dict]:
    """Підсумковий залік конструкторів сезону, аналогічно get_driver_standings()."""
    last_race_id = _last_race_id_of_season(year)
    standings = _constructor_standings[_constructor_standings["raceId"] == last_race_id]
    merged = standings.merge(_constructors, on="constructorId", how="left").sort_values("position")
    merged = merged.astype({"position": "Int64", "wins": "Int64"})
    cols = ["position", "points", "wins", "constructorId", "name", "nationality"]
    return _clean_records(merged[[c for c in cols if c in merged.columns]])


# ---------------------------------------------------------------------------
# Результати сесій (R/Q/S) з CSV.
#
# УВАГА: це НЕ заміна /api/results (FastF1) - контракт для фронта
# (FRONTEND_REQUIREMENTS.md, розділ 3) вимагає полів на кшталт TeamColor,
# HeadshotUrl, BroadcastName, ClassifiedPosition, CountryCode - їх у
# Ergast-датасеті просто немає. Це самостійні, простіші за формою
# ендпоінти (/api/csv/results/...) - легкий, офлайн, історичний доступ
# до результатів без мережі й без FastF1-кешу.
#
# Практики (FP1-3) сюди не входять: Ergast не веде результатів практик
# узагалі (тільки кола гонки, яких тут теж немає) - для практик і надалі
# лише FastF1 (/api/results, /api/laps).
# ---------------------------------------------------------------------------

def get_race_results(year: int, round_number: int) -> list[dict]:
    """Результати гонки з results.csv: підсумкова позиція, очки, кола,
    загальний час, найшвидше коло, статус (Finished/Retired/...)."""
    race_id = _race_id(year, round_number)
    results = _results[_results["raceId"] == race_id]
    if results.empty:
        raise ValueError(tr("race_results_missing", round=round_number, year=year))

    merged = (
        results.merge(_drivers_slim, on="driverId", how="left")
        .merge(_constructors_slim, on="constructorId", how="left")
        .merge(_status_slim, on="statusId", how="left")
        .sort_values("positionOrder")
    )
    return [
        {
            "position": None if pd.isna(row.position) else int(row.position),
            "position_text": row.positionText,
            "grid": None if pd.isna(row.grid) else int(row.grid),
            "points": float(row.points),
            "laps": int(row.laps),
            "total_time": None if pd.isna(row.milliseconds) else round(row.milliseconds / 1000, 3),
            "time_display": None if pd.isna(row.time) else row.time,
            "fastest_lap": None if pd.isna(row.fastestLap) else int(row.fastestLap),
            "fastest_lap_time": _time_to_seconds(row.fastestLapTime),
            "fastest_lap_rank": None if pd.isna(row.rank) else int(row.rank),
            "fastest_lap_speed": None if pd.isna(row.fastestLapSpeed) else float(row.fastestLapSpeed),
            "status": row.status,
            "driver_number": None if pd.isna(row.driver_number) else int(row.driver_number),
            "code": None if pd.isna(row.code) else row.code,
            "full_name": f"{row.forename} {row.surname}",
            "nationality": row.nationality,
            "team": row.team,
        }
        for row in merged.itertuples()
    ]


def get_qualifying_results(year: int, round_number: int) -> list[dict]:
    """Результати кваліфікації з qualifying.csv: позиція і час Q1/Q2/Q3
    (у секундах - узгоджено з наскрізною вимогою фронта про час)."""
    race_id = _race_id(year, round_number)
    quali = _qualifying[_qualifying["raceId"] == race_id]
    if quali.empty:
        raise ValueError(tr("quali_results_missing", round=round_number, year=year))

    merged = (
        quali.merge(_drivers_slim, on="driverId", how="left")
        .merge(_constructors_slim, on="constructorId", how="left")
        .sort_values("position")
    )
    return [
        {
            "position": None if pd.isna(row.position) else int(row.position),
            "q1": _time_to_seconds(row.q1),
            "q2": _time_to_seconds(row.q2),
            "q3": _time_to_seconds(row.q3),
            "driver_number": None if pd.isna(row.driver_number) else int(row.driver_number),
            "code": None if pd.isna(row.code) else row.code,
            "full_name": f"{row.forename} {row.surname}",
            "nationality": row.nationality,
            "team": row.team,
        }
        for row in merged.itertuples()
    ]


def get_sprint_results(year: int, round_number: int) -> list[dict]:
    """Результати спринту з sprint_results.csv. Спринтів немає до 2021 і
    не на кожному етапі - для решти повертає ValueError (як і відсутній рік)."""
    race_id = _race_id(year, round_number)
    sprint = _sprint_results[_sprint_results["raceId"] == race_id]
    if sprint.empty:
        raise ValueError(tr("sprint_results_missing", round=round_number, year=year))

    merged = (
        sprint.merge(_drivers_slim, on="driverId", how="left")
        .merge(_constructors_slim, on="constructorId", how="left")
        .merge(_status_slim, on="statusId", how="left")
        .sort_values("positionOrder")
    )
    return [
        {
            "position": None if pd.isna(row.position) else int(row.position),
            "position_text": row.positionText,
            "grid": None if pd.isna(row.grid) else int(row.grid),
            "points": float(row.points),
            "laps": int(row.laps),
            "total_time": None if pd.isna(row.milliseconds) else round(row.milliseconds / 1000, 3),
            "time_display": None if pd.isna(row.time) else row.time,
            "fastest_lap": None if pd.isna(row.fastestLap) else int(row.fastestLap),
            "fastest_lap_time": _time_to_seconds(row.fastestLapTime),
            "status": row.status,
            "driver_number": None if pd.isna(row.driver_number) else int(row.driver_number),
            "code": None if pd.isna(row.code) else row.code,
            "full_name": f"{row.forename} {row.surname}",
            "nationality": row.nationality,
            "team": row.team,
        }
        for row in merged.itertuples()
    ]


def get_session_results(year: int, round_number: int, session: str) -> list[dict]:
    """Диспетчер за коротким кодом/назвою сесії (R/Race, Q/Qualifying,
    S/Sprint - регістр не важливий). Для FP1-3 навмисно кидає ValueError:
    Ergast-датасет результатів практик не має."""
    code = _SESSION_ALIASES.get(session.strip().upper())
    if code is None:
        raise ValueError(tr("csv_session_unsupported", session=session))
    if code == "R":
        return get_race_results(year, round_number)
    if code == "Q":
        return get_qualifying_results(year, round_number)
    return get_sprint_results(year, round_number)


def get_session_drivers(year: int, round_number: int, session: str) -> list[dict]:
    """Короткий список пілотів сесії (R/Q/S) - витяг ключових полів з
    get_session_results(), як FastF1-варіант /api/drivers."""
    results = get_session_results(year, round_number, session)
    keys = ["driver_number", "code", "full_name", "team", "position", "points"]
    return [{k: r.get(k) for k in keys} for r in results]


def get_pit_stops(year: int, round_number: int) -> list[dict]:
    """Піт-стопи гонки з pit_stops.csv: коло, номер стопу, час доби і
    тривалість (секунди). Функціонал, якого зараз немає в FastF1-частині API."""
    race_id = _race_id(year, round_number)
    stops = _pit_stops[_pit_stops["raceId"] == race_id]
    if stops.empty:
        raise ValueError(tr("pitstops_missing", round=round_number, year=year))

    merged = stops.merge(_drivers_slim, on="driverId", how="left").sort_values(["lap", "stop"])
    return [
        {
            "driver_number": None if pd.isna(row.driver_number) else int(row.driver_number),
            "code": None if pd.isna(row.code) else row.code,
            "full_name": f"{row.forename} {row.surname}",
            "stop": int(row.stop),
            "lap": int(row.lap),
            "time_of_day": row.time,  # годинник доби ("14:01:34"), не інтервал
            "duration": None if pd.isna(row.milliseconds) else round(row.milliseconds / 1000, 3),
        }
        for row in merged.itertuples()
    ]

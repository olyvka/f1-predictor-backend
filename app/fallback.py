"""
Запасне джерело кіл, погоди, порівняння й профілю пілота для сесій, яких
FastF1 не може завантажити (див. app/openf1_client.py: живий таймінг F1
закритий з жовтня 2026, працює лише вже закешоване).

- Сезони 2023+ - OpenF1: усі сесії (практики, кваліфікації, гонки), сектори,
  швидкості на пастках, стінти з шинами, піт-стопи, позиції, погода.
- Старші сезони - CSV-датасет: лише гонка (lap_times.csv + pit_stops.csv) -
  час і позиція на кожному колі та піт-стопи, без шин, секторів і погоди.

Кожна функція повертає ті самі поля, що й відповідна в app/fastf1_client.py,
тож фронт не розрізняє джерела. Чого джерело не має - None.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from app import csv_data, event_info, fastf1_client, openf1_client
from app.openf1_client import OpenF1Unavailable, parse_date

# Колонки кола в тому ж порядку, що й у FastF1 (Session.laps).
LAP_COLUMNS = [
    "Time", "Driver", "DriverNumber", "LapTime", "LapNumber", "Stint", "PitOutTime", "PitInTime",
    "Sector1Time", "Sector2Time", "Sector3Time",
    "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "IsPersonalBest", "Compound", "TyreLife",
    "FreshTyre", "Team", "LapStartTime", "LapStartDate", "TrackStatus", "Position",
    "Deleted", "DeletedReason", "FastF1Generated", "IsAccurate",
]
# Підмножина для /api/compare - як у fastf1_client.compare_drivers_laps.
COMPARE_COLUMNS = [
    "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "Compound", "TyreLife", "Stint", "Position", "IsPersonalBest",
]

RACE_SESSIONS = {"R", "S"}  # лише в них позиція на колі має сенс


class FallbackUnavailable(Exception):
    """Жодне запасне джерело не має цих даних."""


def _code(session: str) -> str:
    return fastf1_client.SESSION_NAME_TO_CODE.get(session, session).upper()


def _seconds(delta) -> float | None:
    return round(delta.total_seconds(), 3) if delta is not None else None


def _new_lap(**values) -> dict:
    lap = dict.fromkeys(LAP_COLUMNS)
    lap.update(DeletedReason="", FastF1Generated=False, **values)
    return lap


def _mark_personal_bests(laps: list[dict]) -> None:
    """IsPersonalBest - найкраще коло пілота на момент проходження (як у FastF1),
    лише серед чистих кіл."""
    best: dict[str, float] = {}
    for lap in sorted(laps, key=lambda l: l["LapNumber"]):
        t = lap["LapTime"]
        ok = t is not None and lap["IsAccurate"]
        lap["IsPersonalBest"] = bool(ok and t < best.get(lap["Driver"], float("inf")))
        if lap["IsPersonalBest"]:
            best[lap["Driver"]] = t


# --------------------------------------------------------------------------
# OpenF1 (2023+)
# --------------------------------------------------------------------------

def _track_status_by_lap(race_control: list[dict]) -> dict[int, str]:
    """Коди стану траси FastF1 по колах з повідомлень дирекції гонки:
    4 - Safety Car, 6 - VSC, 5 - червоні прапори. Жовті прапори в OpenF1 -
    посекторні, на рівень кола їх не зводимо (як і "1" - чисто)."""
    status: dict[int, str] = {}
    sc_from = vsc_from = None
    for m in sorted(race_control, key=lambda m: m.get("date") or ""):
        # До 2026 дирекція писала "VIRTUAL SAFETY CAR DEPLOYED", з 2026 - "VSC DEPLOYED".
        lap, msg = m.get("lap_number"), (m.get("message") or "").upper().replace("VIRTUAL SAFETY CAR", "VSC")
        if lap is None:
            continue
        if m.get("category") == "SafetyCar":
            if msg.startswith("VSC DEPLOYED"):
                vsc_from = lap
            elif msg.startswith("VSC ENDING") and vsc_from is not None:
                for n in range(vsc_from, lap + 1):
                    status[n] = status.get(n, "") + "6"
                vsc_from = None
            elif msg.startswith("SAFETY CAR DEPLOYED"):
                sc_from = lap
            elif msg.startswith("SAFETY CAR IN THIS LAP") and sc_from is not None:
                for n in range(sc_from, lap + 1):
                    status[n] = status.get(n, "") + "4"
                sc_from = None
        elif m.get("flag") == "RED" and m.get("scope") == "Track":
            status[lap] = status.get(lap, "") + "5"
    return status


def _origin(session: dict) -> datetime:
    """Нуль осі Time. FastF1 рахує час сесії від початку потоку даних -
    приблизно за годину до старту; беремо так само, щоб кола й погода мали
    спільну вісь і час не був від'ємним (погода пишеться й до старту)."""
    return parse_date(session["date_start"]) - pd.Timedelta(hours=1)


def _openf1_laps(year: int, round_number: int, session: str) -> list[dict]:
    s = openf1_client.find_session(year, round_number, session)
    session_start = _origin(s)
    raw_laps = openf1_client.session_data(s, "laps")
    if not raw_laps:
        raise OpenF1Unavailable("OpenF1 не має кіл для цієї сесії")
    # Коло повернення в бокси після фінішу OpenF1 віддає без часу - FastF1
    # його не показує. Відкидаємо кола без часу після останнього залікового.
    last_timed: dict[int, int] = {}
    for r in raw_laps:
        if r.get("lap_duration"):
            last_timed[r["driver_number"]] = max(last_timed.get(r["driver_number"], 0), r["lap_number"])
    raw_laps = [
        r for r in raw_laps
        if r.get("lap_duration") or r["lap_number"] <= last_timed.get(r["driver_number"], 0)
    ]

    drivers = {d["driver_number"]: d for d in openf1_client.session_data(s, "drivers")}
    stints = openf1_client.session_data(s, "stints")
    pit_laps = {(p["driver_number"], p["lap_number"]) for p in openf1_client.session_data(s, "pit")}
    is_race = _code(session) in RACE_SESSIONS
    positions = openf1_client.session_data(s, "position") if is_race else []
    track_status = _track_status_by_lap(openf1_client.session_data(s, "race_control")) if is_race else {}

    # Позиція на кінець кола = остання зміна позиції до цього моменту.
    pos_by_driver: dict[int, list[tuple[datetime, int]]] = {}
    for p in positions:
        pos_by_driver.setdefault(p["driver_number"], []).append((parse_date(p["date"]), p["position"]))
    for seq in pos_by_driver.values():
        seq.sort()

    def position_at(num: int, when: datetime | None):
        if when is None:
            return None
        current = None
        for date, pos in pos_by_driver.get(num, []):
            if date > when:
                break
            current = pos
        return current

    def stint_for(num: int, lap_number: int):
        # Після червоного прапора OpenF1 може дати два стінти з тим самим
        # першим колом - коло належить ранішому.
        for st in sorted((x for x in stints if x["driver_number"] == num), key=lambda x: x["stint_number"]):
            if (st.get("lap_start") or 0) <= lap_number <= (st.get("lap_end") or 10**6):
                return st
        return None

    laps = []
    for r in raw_laps:
        num, n = r["driver_number"], r["lap_number"]
        info = drivers.get(num, {})
        start = parse_date(r.get("date_start"))
        duration = r.get("lap_duration")
        end = start + pd.Timedelta(seconds=duration) if start and duration else None
        start_t = _seconds(start - session_start) if start else None
        end_t = _seconds(end - session_start) if end else None
        st = stint_for(num, n)
        pit_in = (num, n) in pit_laps
        pit_out = bool(r.get("is_pit_out_lap"))
        position = position_at(num, end) if is_race else None

        laps.append(_new_lap(
            Time=end_t,
            Driver=info.get("name_acronym") or str(num),
            DriverNumber=str(num),
            LapTime=duration,
            LapNumber=float(n),
            Stint=float(st["stint_number"]) if st else None,
            PitOutTime=start_t if pit_out else None,
            PitInTime=end_t if pit_in else None,
            Sector1Time=r.get("duration_sector_1"),
            Sector2Time=r.get("duration_sector_2"),
            Sector3Time=r.get("duration_sector_3"),
            SpeedI1=r.get("i1_speed"),
            SpeedI2=r.get("i2_speed"),
            SpeedST=r.get("st_speed"),
            Compound=st.get("compound") if st else None,
            TyreLife=float((st.get("tyre_age_at_start") or 0) + n - (st.get("lap_start") or 1) + 1) if st else None,
            FreshTyre=(st.get("tyre_age_at_start") == 0) if st else None,
            Team=info.get("team_name"),
            LapStartTime=start_t,
            LapStartDate=start.replace(tzinfo=None).isoformat() if start else None,
            TrackStatus=track_status.get(n, "1") if is_race else None,
            Position=float(position) if position is not None else None,
            # Як і FastF1: кола з піт-лейном і перше коло гонки - не показові для темпу.
            IsAccurate=bool(duration) and not pit_in and not pit_out and not (is_race and n == 1),
        ))
    _mark_personal_bests(laps)
    return laps


def _openf1_weather(year: int, round_number: int, session: str) -> list[dict]:
    s = openf1_client.find_session(year, round_number, session)
    session_start = _origin(s)
    rows = openf1_client.session_data(s, "weather")
    if not rows:
        raise OpenF1Unavailable("OpenF1 не має погоди для цієї сесії")
    return [
        {
            "Time": _seconds(parse_date(w["date"]) - session_start),
            "AirTemp": w.get("air_temperature"),
            "Humidity": w.get("humidity"),
            "Pressure": w.get("pressure"),
            "Rainfall": bool(w.get("rainfall")),
            "TrackTemp": w.get("track_temperature"),
            "WindDirection": w.get("wind_direction"),
            "WindSpeed": w.get("wind_speed"),
        }
        for w in sorted(rows, key=lambda w: w["date"])
    ]


# --------------------------------------------------------------------------
# CSV (лише гонки; для сезонів до 2023 або якщо OpenF1 недоступний)
# --------------------------------------------------------------------------

_lap_times: pd.DataFrame | None = None  # ~600 тис. рядків - вантажимо лише за потреби


def _csv_laps(year: int, round_number: int, session: str) -> list[dict]:
    if _code(session) != "R":
        raise FallbackUnavailable("У датасеті є кола лише гонок, без практик, кваліфікацій і спринтів")

    global _lap_times
    if _lap_times is None:
        _lap_times = csv_data._read_csv("lap_times")

    race_id = csv_data._race_id(year, round_number)
    rows = _lap_times[_lap_times["raceId"] == race_id]
    if rows.empty:
        raise FallbackUnavailable("У датасеті немає кіл цієї гонки")

    people = csv_data._drivers_slim.set_index("driverId")
    results = csv_data._results[csv_data._results["raceId"] == race_id]
    team_of = results.merge(csv_data._constructors_slim, on="constructorId").set_index("driverId")["team"]
    pit = csv_data._pit_stops[csv_data._pit_stops["raceId"] == race_id]
    pit_laps = {(r.driverId, int(r.lap)) for r in pit.itertuples()}

    laps = []
    for driver_id, drv in rows.sort_values("lap").groupby("driverId"):
        person = people.loc[driver_id]
        code = person["code"] if pd.notna(person["code"]) else person["surname"][:3].upper()
        number = person["driver_number"]
        elapsed, stint, prev_end = 0.0, 1, None
        for r in drv.itertuples():
            lap_time = r.milliseconds / 1000
            start, elapsed = elapsed, elapsed + lap_time
            n = int(r.lap)
            pit_in = (driver_id, n) in pit_laps
            pit_out = (driver_id, n - 1) in pit_laps
            if pit_out:
                stint += 1
            laps.append(_new_lap(
                # Час від старту гонки (а не від початку сесії, як у FastF1) -
                # для графіків, що будують вісь по Time, різниці немає.
                Time=round(elapsed, 3),
                Driver=code,
                DriverNumber=str(int(number)) if pd.notna(number) else None,
                LapTime=lap_time,
                LapNumber=float(n),
                Stint=float(stint),
                PitOutTime=prev_end if pit_out else None,
                PitInTime=round(elapsed, 3) if pit_in else None,
                Team=team_of.get(driver_id),
                LapStartTime=round(start, 3),
                Position=float(r.position),
                IsAccurate=not pit_in and not pit_out and n > 1,
            ))
            prev_end = round(elapsed, 3)
    _mark_personal_bests(laps)
    return laps


# --------------------------------------------------------------------------
# Публічні функції - ті самі підписи й поля, що в fastf1_client
# --------------------------------------------------------------------------

def get_session_laps(year: int, round_number: int, session: str, drivers: list[str] | None = None) -> list[dict]:
    reasons = []
    for source in (_openf1_laps, _csv_laps):
        try:
            laps = source(year, round_number, session)
            break
        except (OpenF1Unavailable, FallbackUnavailable, ValueError) as e:
            reasons.append(str(e))
    else:
        raise FallbackUnavailable("; ".join(reasons))
    if drivers:
        wanted = {d.upper() for d in drivers}
        laps = [l for l in laps if l["Driver"] in wanted or l["DriverNumber"] in wanted]
    return sorted(laps, key=lambda l: (l["Driver"], l["LapNumber"]))


def get_session_weather(year: int, round_number: int, session: str) -> list[dict]:
    try:
        return _openf1_weather(year, round_number, session)
    except OpenF1Unavailable as e:
        raise FallbackUnavailable(f"{e}; у датасеті погоди немає") from e


def _fastest(laps: list[dict]) -> dict | None:
    timed = [l for l in laps if l["LapTime"] is not None and l["IsAccurate"]]
    return min(timed, key=lambda l: l["LapTime"]) if timed else None


def compare_drivers_laps(year: int, round_number: int, session: str, drivers: list[str]) -> dict:
    laps = get_session_laps(year, round_number, session, drivers)
    comparison = {}
    for drv in drivers:
        own = [l for l in laps if drv.upper() in (l["Driver"], l["DriverNumber"])]
        comparison[drv] = {
            "laps": [{k: l[k] for k in COMPARE_COLUMNS} for l in own],
            "fastest_lap": _fastest(own),
        }
    return comparison


def get_driver_summary(year: int, round_number: int, session: str, driver: str) -> dict:
    # Результати сесії FastF1 бере з Ergast, тож вони працюють і без живого
    # таймінгу - лише без кольору команди й фото, які доповнюємо довідником.
    results = fastf1_client.get_session_results(year, round_number, session)
    wanted = driver.upper()
    row = next((r for r in results if wanted in (str(r.get("Abbreviation")).upper(), str(r.get("DriverNumber")))), None)
    if row is None:
        raise FallbackUnavailable(f"Пілот '{driver}' не знайдений у цій сесії")

    code = row.get("Abbreviation") or wanted
    media = event_info.driver_media(year, code)
    if not row.get("TeamColor") and media["team_color"]:
        row["TeamColor"] = media["team_color"].lstrip("#")  # FastF1 віддає колір без "#"
    if not row.get("HeadshotUrl") and media["headshot_url"]:
        row["HeadshotUrl"] = media["headshot_url"]

    laps = get_session_laps(year, round_number, session, [code])
    stints = []
    for stint_no in sorted({l["Stint"] for l in laps if l["Stint"] is not None}):
        own = [l for l in laps if l["Stint"] == stint_no]
        stints.append({
            "stint": stint_no,
            "compound": own[0]["Compound"],
            "laps": len(own),
            "first_lap": min(l["LapNumber"] for l in own),
            "last_lap": max(l["LapNumber"] for l in own),
        })

    return {"driver": row, "result": row, "stints": stints, "fastest_lap": _fastest(laps), "laps": laps}

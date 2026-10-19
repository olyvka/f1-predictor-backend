"""
Дані для вкладки «Огляд» етапу: підсумок гонки, чемпіонат після етапу,
історія й факти траси. Усе рахується з CSV у пам'яті (ті самі таблиці,
що в app/csv_data.py), тож відповідь миттєва й не залежить ні від мережі,
ні від FastF1/Live Timing.

Траса етапу - races.circuitId. Групуємо саме за ним, а не за назвою
гран-прі: на одній трасі бувають етапи з різними назвами (Styrian /
Austrian на Red Bull Ring, 70th Anniversary / British у Сільверстоуні).

Історія траси рахується станом ДО цього етапу (гонки з датою раніше), щоб
сторінка етапу 2024 показувала те, що було відомо перед ним, а майбутні
етапи - всю відому історію.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from app import csv_data
from app.csv_data import _time_to_seconds
from app.i18n import tr

_races = csv_data._races
_results = csv_data._results
_qualifying = csv_data._qualifying
_sprint_results = csv_data._sprint_results
_pit_stops = csv_data._pit_stops
_driver_standings = csv_data._driver_standings
_constructor_standings = csv_data._constructor_standings
_drivers_slim = csv_data._drivers_slim
_constructors_slim = csv_data._constructors_slim
_status_slim = csv_data._status_slim
_circuits = csv_data._read_csv("circuits")

_races_dated = _races.assign(date=pd.to_datetime(_races["date"], errors="coerce"))

_LAPPED = re.compile(r"^\+\d+ Laps?$")

# Довідник фото/кольорів пілотів (scripts/build_driver_media.py, OpenF1,
# сезони 2023+). Без нього поля headshot_url/team_color просто None.
_MEDIA_FILE = os.path.join(csv_data.DATA_DIR, "driver_media.json")


def _load_media() -> dict:
    if not os.path.exists(_MEDIA_FILE):
        return {}
    with open(_MEDIA_FILE, encoding="utf-8") as f:
        return json.load(f).get("seasons", {})


_media = _load_media()


def driver_media(year: int, code: str | None) -> dict:
    """{headshot_url, team_color} пілота в сезоні або None-поля."""
    entry = _media.get(str(year), {}).get(str(code).upper(), {}) if code else {}
    return {"headshot_url": entry.get("headshot_url"), "team_color": entry.get("team_color")}


def season_media(year: int) -> dict:
    """Увесь довідник сезону: {код пілота: {headshot_url, team_color, team}}."""
    return _media.get(str(year), {})


def is_dnf(status) -> bool:
    """Схід - будь-який статус, крім Finished і "+N Lap(s)" (фінішував з
    відставанням на коло й більше). Узгоджено з app/ml/dataset.py."""
    if status is None or pd.isna(status):
        return False
    s = str(status).strip()
    return not (s == "Finished" or _LAPPED.match(s))


def _race_row(year: int, round_number: int) -> pd.Series:
    race = _races_dated[(_races_dated["year"] == year) & (_races_dated["round"] == round_number)]
    if race.empty:
        raise ValueError(tr("round_not_found", round=round_number, year=year))
    return race.iloc[0]


def _int(v):
    return None if v is None or pd.isna(v) else int(v)


def _float(v, digits: int = 3):
    return None if v is None or pd.isna(v) else round(float(v), digits)


def _driver(row, year: int) -> dict:
    return {
        "code": None if pd.isna(row.code) else row.code,
        "full_name": f"{row.forename} {row.surname}",
        "team": row.team,
        **driver_media(year, None if pd.isna(row.code) else row.code),
    }


# ---------------------------------------------------------------------------
# Задача 1: підсумок гонки
# ---------------------------------------------------------------------------

def _session_summary(rows: pd.DataFrame, year: int, pole_row=None) -> dict:
    """Спільне для гонки й спринту: rows - results/sprint_results одного
    етапу, змерджені з пілотами/командами/статусами."""
    rows = rows.sort_values("positionOrder")
    winner = rows.iloc[0]
    winner_ms = winner["milliseconds"]

    podium = []
    for r in rows[rows["position"].notna()].head(3).itertuples():
        gap = None
        if r.position != 1 and pd.notna(r.milliseconds) and pd.notna(winner_ms):
            gap = round((r.milliseconds - winner_ms) / 1000, 3)
        podium.append({
            "position": int(r.position),
            **_driver(r, year),
            "total_time": _float(r.milliseconds / 1000 if pd.notna(r.milliseconds) else None),
            "gap": gap,  # секунди до переможця; None у переможця і в тих, хто відстав на коло
            "time_display": None if pd.isna(r.time) else r.time,
            "status": r.status,
            "points": float(r.points),
        })

    # Найшвидше коло: rank == 1 у results.csv; у sprint_results rank порожній -
    # тоді мінімальний час кола.
    fl = rows[rows["rank"] == 1] if "rank" in rows and rows["rank"].notna().any() else pd.DataFrame()
    if fl.empty:
        times = rows["fastestLapTime"].map(_time_to_seconds)
        fl = rows.loc[[times.idxmin()]] if times.notna().any() else fl
    fastest_lap = None
    if not fl.empty:
        r = next(fl.itertuples())
        fastest_lap = {
            **_driver(r, year),
            "time": _time_to_seconds(r.fastestLapTime),
            "lap": _int(r.fastestLap),
            "avg_speed": _float(getattr(r, "fastestLapSpeed", None)),  # км/год, лише для гонки
        }

    dnf_rows = rows[rows["status"].map(is_dnf)]
    dnfs = [{**_driver(r, year), "status": r.status, "laps": _int(r.laps)} for r in dnf_rows.itertuples()]

    # Найбільший приріст: grid - position серед класифікованих; старт з
    # піт-лейну (grid == 0) - як з останньої позиції.
    classified = rows[rows["position"].notna()].copy()
    biggest_gainer = None
    if not classified.empty:
        classified["grid_eff"] = classified["grid"].where(classified["grid"] > 0, len(rows))
        classified["gained"] = classified["grid_eff"] - classified["position"]
        r = classified.loc[classified["gained"].idxmax()]
        if r["gained"] > 0:
            biggest_gainer = {
                "code": None if pd.isna(r["code"]) else r["code"],
                "full_name": f"{r['forename']} {r['surname']}",
                "team": r["team"],
                **driver_media(year, None if pd.isna(r["code"]) else r["code"]),
                "grid": int(r["grid"]),
                "pit_lane_start": bool(r["grid"] == 0),
                "position": int(r["position"]),
                "gained": int(r["gained"]),
            }

    pole = None
    if pole_row is not None:
        pole = {**_driver(pole_row, year), "time": pole_row.best_time}

    return {
        "podium": podium,
        "pole": pole,
        "fastest_lap": fastest_lap,
        "dnf_count": len(dnfs),
        "dnfs": dnfs,
        "biggest_gainer": biggest_gainer,
        "laps": _int(winner["laps"]),
    }


def _merge_people(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.merge(_drivers_slim, on="driverId", how="left")
             .merge(_constructors_slim, on="constructorId", how="left"))
    if "statusId" in out:
        out = out.merge(_status_slim, on="statusId", how="left")
    return out


def get_event_summary(year: int, round_number: int) -> dict | None:
    """Підсумок етапу або None, якщо гонка ще не відбулась (немає в results.csv)."""
    race_id = int(_race_row(year, round_number)["raceId"])
    results = _results[_results["raceId"] == race_id]
    if results.empty:
        return None

    # Поул - найкращий з Q1/Q2/Q3 у пілота з позицією 1 у кваліфікації.
    quali = _merge_people(_qualifying[(_qualifying["raceId"] == race_id) & (_qualifying["position"] == 1)])
    pole_row = None
    if not quali.empty:
        q = quali.iloc[0]
        times = [t for t in (_time_to_seconds(q.get(c)) for c in ("q1", "q2", "q3")) if t is not None]
        pole_row = next(quali.assign(best_time=min(times) if times else None).itertuples())

    summary = {"race_id": race_id, **_session_summary(_merge_people(results), year, pole_row)}

    sprint = _sprint_results[_sprint_results["raceId"] == race_id]
    if sprint.empty:
        summary["sprint"] = None
    else:
        sprint = _merge_people(sprint)
        # Кваліфікації спринту в CSV немає - "поул" спринту = старт з першого місця.
        sp = sprint[sprint["grid"] == 1]
        sp_pole = next(sp.assign(best_time=None).itertuples()) if not sp.empty else None
        summary["sprint"] = _session_summary(sprint, year, sp_pole)
    return summary


# ---------------------------------------------------------------------------
# Задача 2: чемпіонат після етапу
# ---------------------------------------------------------------------------

def _standings_table(table: pd.DataFrame, id_col: str, race_id: int, prev_race_id: int | None,
                     enrich) -> list[dict]:
    cur = table[table["raceId"] == race_id]
    prev = (table[table["raceId"] == prev_race_id][[id_col, "position", "points"]]
            .rename(columns={"position": "prev_position", "points": "prev_points"})
            if prev_race_id is not None else None)
    if prev is not None:
        cur = cur.merge(prev, on=id_col, how="left")
    cur = cur.sort_values("position")
    leader_points = float(cur["points"].max()) if not cur.empty else 0.0

    rows = []
    for r in cur.itertuples():
        delta_pos = delta_pts = None
        if prev is not None:
            if pd.notna(r.prev_position):
                delta_pos = int(r.prev_position - r.position)  # > 0 - піднявся
                delta_pts = float(r.points - r.prev_points)
            else:
                delta_pts = float(r.points)  # перший етап у сезоні для цього учасника
        rows.append({
            "position": int(r.position),
            **enrich(r),
            "points": float(r.points),
            "wins": int(r.wins),
            "delta_position": delta_pos,
            "delta_points": delta_pts,
            "gap_to_leader": float(leader_points - r.points),
        })
    return rows


def get_event_standings(year: int, round_number: int) -> dict:
    """Чемпіонат пілотів і конструкторів після етапу, з дельтами до
    попереднього етапу. Для етапу, що ще не відбувся, - таблиця після
    останнього завершеного етапу сезону (as_of_round)."""
    _race_row(year, round_number)  # ValueError для невідомого етапу
    season = _races[_races["year"] == year].sort_values("round")
    done = season[season["raceId"].isin(_driver_standings["raceId"])
                  & (season["round"] <= round_number)]
    if done.empty:
        return {"as_of_round": None, "requested_round": round_number, "drivers": [], "constructors": []}

    as_of = done.iloc[-1]
    race_id = int(as_of["raceId"])
    prev_race_id = int(done.iloc[-2]["raceId"]) if len(done) > 1 else None

    # Команда пілота - з його результату на цьому етапі (або останнього до нього).
    season_results = _results[_results["raceId"].isin(done["raceId"])].merge(
        season[["raceId", "round"]], on="raceId").sort_values("round")
    team_of = (season_results.drop_duplicates("driverId", keep="last")
               .merge(_constructors_slim, on="constructorId")
               .set_index("driverId")["team"].to_dict())

    drivers = _driver_standings.merge(_drivers_slim, on="driverId", how="left")

    def enrich_driver(r):
        code = None if pd.isna(r.code) else r.code
        return {"driver_id": int(r.driverId), "code": code, "full_name": f"{r.forename} {r.surname}",
                "team": team_of.get(r.driverId), **driver_media(year, code)}

    constructors = _constructor_standings.merge(_constructors_slim, on="constructorId", how="left")

    def enrich_constructor(r):
        return {"constructor_id": int(r.constructorId), "team": r.team}

    return {
        "as_of_round": int(as_of["round"]),
        "requested_round": round_number,
        "drivers": _standings_table(drivers, "driverId", race_id, prev_race_id, enrich_driver),
        "constructors": _standings_table(constructors, "constructorId", race_id, prev_race_id,
                                         enrich_constructor),
    }


# ---------------------------------------------------------------------------
# Задача 3: історія траси
# ---------------------------------------------------------------------------

def _circuit_past_winners(year: int, round_number: int) -> tuple[pd.Series, pd.DataFrame]:
    race = _race_row(year, round_number)
    past = _races_dated[(_races_dated["circuitId"] == race["circuitId"])
                        & (_races_dated["date"] < race["date"])]
    winners = (_results[_results["raceId"].isin(past["raceId"]) & (_results["positionOrder"] == 1)]
               .merge(past[["raceId", "year", "round", "name", "date"]], on="raceId")
               .pipe(_merge_people)
               .sort_values("date"))
    return race, winners


def _pole_conversion(winners: pd.DataFrame) -> dict:
    n = len(winners)
    return {"rate": round(float((winners["grid"] == 1).mean()), 3) if n else None, "races": n}


def get_circuit_history(year: int, round_number: int) -> dict:
    """Історія траси станом до цього етапу. pole_conversion - частка гонок,
    де переміг пілот, що стартував з першого місця (grid == 1: з
    урахуванням штрафів, і кваліфікацій у CSV немає до 1994)."""
    race, winners = _circuit_past_winners(year, round_number)
    circuit = _circuits[_circuits["circuitId"] == race["circuitId"]].iloc[0]
    last10 = winners.tail(10)

    most = (winners.groupby(["driverId"])
            .agg(wins=("raceId", "size"), last_win=("year", "max"),
                 code=("code", "first"), forename=("forename", "first"), surname=("surname", "first"))
            .reset_index()
            .sort_values(["wins", "last_win"], ascending=[False, False])
            .head(3))

    all_races = _races_dated[(_races_dated["circuitId"] == race["circuitId"])
                             & (_races_dated["date"] < race["date"])]
    return {
        "circuit_id": int(race["circuitId"]),
        "circuit": circuit["name"],
        "location": circuit["location"],
        "country": circuit["country"],
        "first_gp_year": int(all_races["year"].min()) if not all_races.empty else None,
        "races_held": int(len(all_races)),
        "winners": [{
            "year": int(r.year), "round": int(r.round), "race_name": r.name,
            "code": None if pd.isna(r.code) else r.code,
            "full_name": f"{r.forename} {r.surname}", "team": r.team,
            "grid": _int(r.grid),
        } for r in last10.iloc[::-1].itertuples()],
        "most_wins": [{
            "code": None if pd.isna(r.code) else r.code,
            "full_name": f"{r.forename} {r.surname}", "wins": int(r.wins), "last_win": int(r.last_win),
        } for r in most.itertuples()],
        "pole_conversion": {"last_10": _pole_conversion(last10), "all_time": _pole_conversion(winners)},
    }


# ---------------------------------------------------------------------------
# Задача 4: факти траси
# ---------------------------------------------------------------------------

# Довжина кола, км. Цього немає в жодному джерелі датасету - заповнено
# вручну за офіційними даними F1 для трас календаря 2015+. Для трас, що
# змінювали конфігурацію: [(з якого року, км), ...].
CIRCUIT_LENGTH_KM: dict[str, list[tuple[int, float]]] = {
    "albert_park": [(0, 5.303), (2022, 5.278)],
    "americas": [(0, 5.513)],
    "bahrain": [(0, 5.412)],
    "baku": [(0, 6.003)],
    "catalunya": [(0, 4.655), (2021, 4.675), (2023, 4.657)],
    "hockenheimring": [(0, 4.574)],
    "hungaroring": [(0, 4.381)],
    "imola": [(0, 4.909)],
    "interlagos": [(0, 4.309)],
    "istanbul": [(0, 5.338)],
    "jeddah": [(0, 6.174)],
    "losail": [(0, 5.380), (2023, 5.419)],
    "madring": [(0, 5.474)],
    "marina_bay": [(0, 5.065), (2018, 5.063), (2023, 4.940)],
    "miami": [(0, 5.412)],
    "monaco": [(0, 3.337)],
    "monza": [(0, 5.793)],
    "mugello": [(0, 5.245)],
    "nurburgring": [(0, 5.148)],
    "portimao": [(0, 4.653)],
    "red_bull_ring": [(0, 4.318)],
    "ricard": [(0, 5.842)],
    "rodriguez": [(0, 4.304)],
    "sepang": [(0, 5.543)],
    "shanghai": [(0, 5.451)],
    "silverstone": [(0, 5.891)],
    "sochi": [(0, 5.848)],
    "spa": [(0, 7.004)],
    "suzuka": [(0, 5.807)],
    "vegas": [(0, 6.201)],
    "villeneuve": [(0, 4.361)],
    "yas_marina": [(0, 5.554), (2021, 5.281)],
    "zandvoort": [(0, 4.259)],
}


def circuit_length_km(circuit_ref: str, year: int) -> float | None:
    versions = CIRCUIT_LENGTH_KM.get(circuit_ref)
    if not versions:
        return None
    return [km for since, km in versions if since <= year][-1]


PIT_WINDOW = 10


def get_circuit_facts(year: int, round_number: int) -> dict:
    """Факти траси станом до етапу. Піт-стопи - з pit_stops.csv за останні
    PIT_WINDOW гонок на трасі, де вони є (у цьому датасеті - з 1994, але
    епоха дозаправок і інші ліміти швидкості в піт-лейні робили б середнє
    й "найшвидший" стоп нерелевантними). duration - час проїзду піт-лейну,
    а не стоянки. Втрата часу на піт-стоп (normal/SC/VSC) у CSV немає -
    вона в статичних JSON трас фронтенду (scripts/build_circuits.py,
    pit_loss з MultiViewer)."""
    race, winners = _circuit_past_winners(year, round_number)
    circuit = _circuits[_circuits["circuitId"] == race["circuitId"]].iloc[0]
    past_ids = winners["raceId"]

    last5 = winners.tail(5)
    typical_laps = int(last5["laps"].median()) if not last5.empty else None

    stops = _pit_stops[_pit_stops["raceId"].isin(past_ids)]
    recent = winners[winners["raceId"].isin(stops["raceId"])].tail(PIT_WINDOW)["raceId"]
    stops = stops[stops["raceId"].isin(recent)]
    avg_stops = fastest = None
    pit_races = 0
    if not stops.empty:
        per_race = stops.groupby("raceId").size()
        starters = _results[_results["raceId"].isin(per_race.index)].groupby("raceId").size()
        avg_stops = round(float((per_race / starters).mean()), 2)
        pit_races = int(len(per_race))
        # Проїзд піт-лейну без зупинки (drive-through) - помітно швидший за
        # будь-який реальний стоп; відсіюємо все, що на > 15% швидше медіани гонки.
        race_median = stops.groupby("raceId")["milliseconds"].transform("median")
        real = stops[stops["milliseconds"] >= race_median * 0.85]
        f = real.loc[real["milliseconds"].idxmin()]
        who = _drivers_slim[_drivers_slim["driverId"] == f["driverId"]].iloc[0]
        f_race = _races[_races["raceId"] == f["raceId"]].iloc[0]
        fastest = {
            "duration": round(float(f["milliseconds"]) / 1000, 3),
            "code": who["code"] if pd.notna(who["code"]) else None,
            "full_name": f"{who['forename']} {who['surname']}",
            "year": int(f_race["year"]),
            "lap": int(f["lap"]),
        }

    return {
        "circuit_id": int(race["circuitId"]),
        "circuit": circuit["name"],
        "length_km": circuit_length_km(circuit["circuitRef"], year),
        "typical_laps": typical_laps,
        "avg_pit_stops_per_driver": avg_stops,
        "fastest_pit_stop": fastest,
        "pit_data_races": pit_races,
        "lat": _float(circuit["lat"], 5),
        "lng": _float(circuit["lng"], 5),
    }

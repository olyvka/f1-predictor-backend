"""
Статистика пілотів поза межами одного етапу: динаміка чемпіонату сезону,
профіль пілота за сезон (з порівнянням з напарником і оцінкою), кар'єра і
контекст конкретного гран-прі. Усе з CSV у пам'яті (app/csv_data.py) -
миттєво й без мережі.

Пілота ідентифікуємо за driverRef з датасету ("max_verstappen"), а не за
трибуквеним кодом: у пілотів до 2000-х коду немає, а деякі коди
повторювались у різні роки. FastF1 віддає той самий ідентифікатор у полі
DriverId, тож сторінка етапу може послатися на профіль напряму.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from app import csv_data
from app.csv_data import _time_to_seconds
from app.event_info import driver_media, is_dnf

# Сходи через аварію чи зіткнення (на відміну від технічних). Хто винен,
# датасет не розрізняє - "Collision" однаковий для обох учасників.
INCIDENT = {"Accident", "Collision", "Collision damage", "Spun off", "Damage"}
FASTEST_LAP_DATA_FROM = 2004  # rank у results.csv заповнений лише з цього сезону
MIN_RACES_FOR_RATING = 5

_people = csv_data._drivers[
    ["driverId", "driverRef", "number", "code", "forename", "surname", "dob", "nationality"]
]
_teams = csv_data._constructors[["constructorId", "constructorRef", "name"]].rename(columns={"name": "team"})
_races = csv_data._races[["raceId", "year", "round", "circuitId", "name", "date", "sprint_date"]].copy()
_races["date"] = pd.to_datetime(_races["date"], errors="coerce")


def _merge(df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(_races, on="raceId").merge(_teams, on="constructorId", how="left")
    if "statusId" in out:
        out = out.merge(csv_data._status_slim, on="statusId", how="left")
    return out


_results = _merge(csv_data._results)
_sprints = _merge(csv_data._sprint_results)
_quali = _merge(csv_data._qualifying)
for _q in ("q1", "q2", "q3"):
    _quali[f"{_q}_s"] = _quali[_q].map(_time_to_seconds)


def _none(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        # Позиції/номери в CSV читаються як float через пропуски - 1.0 -> 1.
        return int(v) if float(v).is_integer() else float(v)
    return v


def _media(driver_id: int, code: str | None, year: int | None) -> dict:
    """Фото й колір команди пілота в сезоні. Довідник є лише з 2023 - для
    старших сезонів беремо фото з найближчого пізнішого сезону ЦЬОГО Ж
    пілота (за його гонками, а не лише за кодом: коди повторювались, напр.
    MSC у Міхаеля й Міка Шумахерів). Колір команди не переносимо - команда
    в іншому сезоні могла бути інша."""
    media = driver_media(year, code) if year else {"headshot_url": None, "team_color": None}
    if media["headshot_url"] or not code:
        return media
    own_years = sorted(int(y) for y in _results.loc[_results["driverId"] == driver_id, "year"].unique())
    for y in [y for y in own_years if not year or y > year] + [y for y in reversed(own_years) if year and y < year]:
        photo = driver_media(y, code)["headshot_url"]
        if photo:
            return {**media, "headshot_url": photo}
    return media


def _person(driver_id: int, year: int | None = None) -> dict:
    p = _people[_people["driverId"] == driver_id].iloc[0]
    code = _none(p["code"])
    return {
        "driver_id": int(driver_id),
        "driver_ref": p["driverRef"],
        "code": code,
        "full_name": f"{p['forename']} {p['surname']}",
        "number": _none(p["number"]),
        "nationality": _none(p["nationality"]),
        "dob": _none(p["dob"]),
        **_media(int(driver_id), code, year),
    }


def driver_id_by_ref(driver_ref: str) -> int:
    row = _people[_people["driverRef"] == driver_ref]
    if row.empty:
        raise ValueError(f"Пілот '{driver_ref}' відсутній у датасеті")
    return int(row.iloc[0]["driverId"])


def _result_label(r) -> str:
    """Коротка позначка результату для "форми": "1", "12", "DNF", "DSQ"."""
    if r["status"] == "Disqualified":
        return "DSQ"
    if is_dnf(r["status"]):
        return "DNF"
    return str(int(r["positionOrder"]))


def _season_races(year: int) -> pd.DataFrame:
    season = _races[_races["year"] == year].sort_values("round")
    if season.empty:
        raise ValueError(f"Сезон {year} відсутній у датасеті")
    return season


def _poles(results: pd.DataFrame) -> pd.Series:
    """Поули по пілотах: перша позиція кваліфікації; для гонок без даних
    кваліфікації (до 1994) - старт з першого місця."""
    race_ids = results["raceId"].unique()
    q = _quali[_quali["raceId"].isin(race_ids) & (_quali["position"] == 1)][["raceId", "driverId"]]
    without_q = set(race_ids) - set(_quali[_quali["raceId"].isin(race_ids)]["raceId"])
    g = results[results["raceId"].isin(without_q) & (results["grid"] == 1)][["raceId", "driverId"]]
    return pd.concat([q, g]).groupby("driverId").size()


def _max_points_per_race(year: int) -> tuple[int, int]:
    """(гонка, спринт) - максимум очок одного пілота за вікенд за правилами року."""
    race = 25 + (1 if 2019 <= year <= 2024 else 0)  # очко за найшвидше коло 2019-2024
    sprint = 8 if year >= 2022 else 3
    return race, sprint


# ---------------------------------------------------------------------------
# Динаміка чемпіонату сезону (вкладки "Пілоти" / "Команди")
# ---------------------------------------------------------------------------

def get_season_progress(year: int) -> dict:
    season = _season_races(year)
    results = _results[_results["year"] == year]
    sprints = _sprints[_sprints["year"] == year]
    done = season[season["raceId"].isin(results["raceId"])]
    remaining = season[~season["raceId"].isin(results["raceId"])]
    done_ids = list(done["raceId"])
    # Переможець кожного завершеного етапу - для календаря на головній.
    winners = results[results["positionOrder"] == 1].set_index("raceId")
    rounds = []
    for _, r in done.iterrows():
        w = winners.loc[r["raceId"]] if r["raceId"] in winners.index else None
        winner = None
        if w is not None:
            person = _person(int(w["driverId"]), year)
            winner = {key: person[key] for key in ("driver_ref", "code", "full_name", "headshot_url", "team_color")}
            winner["team"] = w["team"]
        rounds.append({"round": int(r["round"]), "name": r["name"], "winner": winner})

    race_max, sprint_max = _max_points_per_race(year)
    max_remaining = len(remaining) * race_max + int(remaining["sprint_date"].notna().sum()) * sprint_max

    if not done_ids:
        return {"year": year, "total_rounds": len(season), "completed_rounds": 0, "final": False,
                "max_remaining_points": max_remaining, "rounds": [], "drivers": [], "constructors": []}

    # --- пілоти ---
    standings = csv_data._driver_standings[csv_data._driver_standings["raceId"].isin(done_ids)]
    pts = standings.pivot(index="driverId", columns="raceId", values="points").reindex(columns=done_ids)
    pos = standings.pivot(index="driverId", columns="raceId", values="position").reindex(columns=done_ids)
    final = standings[standings["raceId"] == done_ids[-1]].set_index("driverId")
    leader_points = float(final["points"].max())

    classified = results.assign(dnf=results["status"].map(is_dnf))
    wins = classified[classified["position"] == 1].groupby("driverId").size()
    podiums = classified[classified["position"] <= 3].groupby("driverId").size()
    dnfs = classified[classified["dnf"]].groupby("driverId").size()
    starts = classified.groupby("driverId").size()
    poles = _poles(results)
    sprint_points = sprints.groupby("driverId")["points"].sum()

    drivers = []
    for driver_id, f in final.sort_values("position").iterrows():
        own = classified[classified["driverId"] == driver_id].sort_values("round")
        teams = list(dict.fromkeys(own["team"]))
        points = float(f["points"])
        drivers.append({
            **_person(driver_id, year),
            "team": teams[-1] if teams else None,
            "teams": teams,
            "position": int(f["position"]),
            "points": points,
            "gap_to_leader": leader_points - points,
            "wins": int(wins.get(driver_id, 0)),
            "podiums": int(podiums.get(driver_id, 0)),
            "poles": int(poles.get(driver_id, 0)),
            "dnfs": int(dnfs.get(driver_id, 0)),
            "starts": int(starts.get(driver_id, 0)),
            "sprint_points": float(sprint_points.get(driver_id, 0)),
            # Пілот міг дебютувати посеред сезону - до дебюту 0 очок.
            "points_by_round": [float(v) if pd.notna(v) else 0.0 for v in pts.loc[driver_id]],
            "position_by_round": [_none(v) for v in pos.loc[driver_id]],
            "form": [
                {"round": int(r["round"]), "label": _result_label(r), "points": float(r["points"])}
                for _, r in own.tail(5).iterrows()
            ],
            "can_win_title": points + max_remaining >= leader_points,
        })

    # --- команди ---
    c_stand = csv_data._constructor_standings[csv_data._constructor_standings["raceId"].isin(done_ids)]
    c_pts = c_stand.pivot(index="constructorId", columns="raceId", values="points").reindex(columns=done_ids)
    c_final = c_stand[c_stand["raceId"] == done_ids[-1]].set_index("constructorId")
    c_leader = float(c_final["points"].max())
    # Внесок кожного пілота в очки команди - гонки + спринти.
    contrib = pd.concat([results[["constructorId", "driverId", "points"]],
                         sprints[["constructorId", "driverId", "points"]]])
    contrib = contrib.groupby(["constructorId", "driverId"])["points"].sum()
    c_wins = classified[classified["position"] == 1].groupby("constructorId").size()
    c_podiums = classified[classified["position"] <= 3].groupby("constructorId").size()
    color_by_team = {d["team"]: d["team_color"] for d in drivers if d["team_color"]}

    constructors = []
    for cid, f in c_final.sort_values("position").iterrows():
        team = _teams.loc[_teams["constructorId"] == cid, "team"].iloc[0]
        split = contrib.get(cid, pd.Series(dtype=float)).sort_values(ascending=False)
        constructors.append({
            "constructor_id": int(cid),
            "team": team,
            "team_color": color_by_team.get(team),
            "position": int(f["position"]),
            "points": float(f["points"]),
            "gap_to_leader": c_leader - float(f["points"]),
            "wins": int(c_wins.get(cid, 0)),
            "podiums": int(c_podiums.get(cid, 0)),
            "points_by_round": [float(v) if pd.notna(v) else 0.0 for v in c_pts.loc[cid]],
            "drivers": [
                {"driver_ref": _person(did)["driver_ref"], "code": _person(did)["code"],
                 "full_name": _person(did)["full_name"], "points": float(p)}
                for did, p in split.items()
            ],
            "can_win_title": float(f["points"]) + 2 * max_remaining >= c_leader,
        })

    return {
        "year": year,
        "total_rounds": len(season),
        "completed_rounds": len(done),
        "final": remaining.empty,
        "max_remaining_points": max_remaining,
        "rounds": rounds,
        "drivers": drivers,
        "constructors": constructors,
    }


# ---------------------------------------------------------------------------
# Порівняння з напарником і оцінка сезону
# ---------------------------------------------------------------------------

def _best_common_quali_gap(mine, theirs) -> float | None:
    """Різниця (%) у найпізнішому сегменті кваліфікації, де є час в обох:
    Q3, інакше Q2, інакше Q1. > 0 - пілот повільніший за напарника."""
    for seg in ("q3_s", "q2_s", "q1_s"):
        a, b = mine.get(seg), theirs.get(seg)
        if a is not None and b is not None and not pd.isna(a) and not pd.isna(b):
            return (a - b) / b * 100
    return None


def _teammate_rows(year: int, driver_id: int) -> list[dict]:
    """По кожній гонці сезону: пілот і напарник (той самий constructorId)."""
    season = _results[_results["year"] == year]
    quali = _quali[_quali["year"] == year].set_index(["raceId", "driverId"])
    rows = []
    for _, mine in season[season["driverId"] == driver_id].sort_values("round").iterrows():
        mates = season[(season["raceId"] == mine["raceId"]) & (season["constructorId"] == mine["constructorId"])
                       & (season["driverId"] != driver_id)]
        for _, mate in mates.iterrows():
            qm = quali.loc[(mine["raceId"], driver_id)] if (mine["raceId"], driver_id) in quali.index else None
            qt = quali.loc[(mine["raceId"], mate["driverId"])] if (mine["raceId"], mate["driverId"]) in quali.index else None
            both_finished = not is_dnf(mine["status"]) and not is_dnf(mate["status"])
            rows.append({
                "round": int(mine["round"]),
                "mate_id": int(mate["driverId"]),
                "quali_mine": _none(qm["position"]) if qm is not None else None,
                "quali_mate": _none(qt["position"]) if qt is not None else None,
                "quali_gap_pct": _best_common_quali_gap(qm, qt) if qm is not None and qt is not None else None,
                "race_mine": int(mine["positionOrder"]) if both_finished else None,
                "race_mate": int(mate["positionOrder"]) if both_finished else None,
                "points_mine": float(mine["points"]),
                "points_mate": float(mate["points"]),
            })
    return rows


def _teammate_summary(year: int, driver_id: int) -> list[dict]:
    rows = _teammate_rows(year, driver_id)
    out = []
    for mate_id in dict.fromkeys(r["mate_id"] for r in rows):
        own = [r for r in rows if r["mate_id"] == mate_id]
        q = [r for r in own if r["quali_mine"] is not None and r["quali_mate"] is not None]
        rc = [r for r in own if r["race_mine"] is not None]
        gaps = [r["quali_gap_pct"] for r in own if r["quali_gap_pct"] is not None]
        mate = _person(mate_id, year)
        out.append({
            "driver_ref": mate["driver_ref"], "code": mate["code"], "full_name": mate["full_name"],
            "headshot_url": mate["headshot_url"], "team_color": mate["team_color"],
            "races": len(own),
            "quali": {"won": sum(r["quali_mine"] < r["quali_mate"] for r in q),
                      "lost": sum(r["quali_mine"] > r["quali_mate"] for r in q)},
            "quali_gap_pct_median": round(float(np.median(gaps)), 3) if gaps else None,
            "race": {"won": sum(r["race_mine"] < r["race_mate"] for r in rc),
                     "lost": sum(r["race_mine"] > r["race_mate"] for r in rc)},
            "points": {"mine": sum(r["points_mine"] for r in own), "mate": sum(r["points_mate"] for r in own)},
        })
    return out


RATING_COMPONENTS = [
    ("quali_vs_teammate", "Кваліфікація проти напарника", "частка виграних дуелей"),
    ("race_vs_teammate", "Гонка проти напарника", "частка виграних дуелей, коли обидва фінішували"),
    ("positions_gained", "Позиції в гонці", "середньо виграних позицій від старту до фінішу"),
    ("consistency", "Стабільність", "розкид фінішних позицій (менше - краще)"),
    ("incident_free", "Без аварій", "частка гонок без сходу через аварію чи зіткнення (вину датасет не розрізняє)"),
    ("results", "Результати", "середньо очок за гонку"),
]


@lru_cache(maxsize=64)
def _season_rating_table(year: int) -> pd.DataFrame:
    """Сирі значення складових оцінки для всіх пілотів сезону з
    щонайменше MIN_RACES_FOR_RATING гонками. Кешується - датасет у пам'яті
    не змінюється до перезапуску."""
    season = _results[_results["year"] == year]
    counts = season.groupby("driverId").size()
    field_size = season.groupby("raceId").size()
    rows = {}
    for driver_id in counts[counts >= MIN_RACES_FOR_RATING].index:
        own = season[season["driverId"] == driver_id]
        finished = own[~own["status"].map(is_dnf)]
        # Старт з піт-лейну (grid == 0) - як з останнього місця.
        grid_eff = finished["grid"].where(finished["grid"] > 0, finished["raceId"].map(field_size))
        tm = _teammate_summary(year, int(driver_id))
        q_w = sum(t["quali"]["won"] for t in tm); q_l = sum(t["quali"]["lost"] for t in tm)
        r_w = sum(t["race"]["won"] for t in tm); r_l = sum(t["race"]["lost"] for t in tm)
        rows[int(driver_id)] = {
            "quali_vs_teammate": q_w / (q_w + q_l) if q_w + q_l >= 3 else np.nan,
            "race_vs_teammate": r_w / (r_w + r_l) if r_w + r_l >= 3 else np.nan,
            "positions_gained": float((grid_eff - finished["positionOrder"]).mean()) if len(finished) else np.nan,
            # Розкид зі знаком мінус, щоб "більше - краще" для всіх складових.
            "consistency": -float(finished["positionOrder"].std()) if len(finished) >= 3 else np.nan,
            "incident_free": 1 - float(own["status"].isin(INCIDENT).mean()),
            "results": float(own["points"].mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def get_season_rating(year: int, driver_id: int) -> dict | None:
    """Оцінка сезону 0-100: середнє перцентилів складових серед пілотів
    сезону. Перцентиль 80 - кращий за 80% пілотів у цій складовій."""
    table = _season_rating_table(year)
    if driver_id not in table.index:
        return None
    pct = table.rank(pct=True, method="average") * 100
    components = []
    for key, label, hint in RATING_COMPONENTS:
        value = table.at[driver_id, key]
        components.append({
            "key": key, "label": label, "hint": hint,
            "value": None if pd.isna(value) else round(float(abs(value) if key == "consistency" else value), 3),
            "percentile": None if pd.isna(value) else round(float(pct.at[driver_id, key])),
        })
    totals = pct.mean(axis=1, skipna=True)
    return {
        "score": round(float(totals.at[driver_id])),
        "rank": int((totals > totals.at[driver_id]).sum()) + 1,
        "of": len(totals),
        "components": components,
    }


# ---------------------------------------------------------------------------
# Профіль пілота за сезон
# ---------------------------------------------------------------------------

def get_driver_season(year: int, driver_ref: str) -> dict:
    driver_id = driver_id_by_ref(driver_ref)
    season = _season_races(year)
    own = _results[(_results["year"] == year) & (_results["driverId"] == driver_id)].sort_values("round")
    if own.empty:
        raise ValueError(f"Пілот '{driver_ref}' не їхав у сезоні {year}")
    sprints = _sprints[(_sprints["year"] == year) & (_sprints["driverId"] == driver_id)].set_index("round")
    quali = _quali[(_quali["year"] == year) & (_quali["driverId"] == driver_id)].set_index("round")
    standings = (csv_data._driver_standings[csv_data._driver_standings["driverId"] == driver_id]
                 .merge(season[["raceId", "round"]], on="raceId").set_index("round"))

    races = []
    for _, r in own.iterrows():
        rnd = int(r["round"])
        sp = sprints.loc[rnd] if rnd in sprints.index else None
        races.append({
            "round": rnd,
            "name": r["name"],
            "team": r["team"],
            "quali_position": _none(quali.loc[rnd, "position"]) if rnd in quali.index else None,
            "grid": _none(r["grid"]),
            "position": _none(r["position"]),
            "label": _result_label(r),
            "status": r["status"],
            "points": float(r["points"]),
            "sprint_position": _none(sp["position"]) if sp is not None else None,
            "sprint_points": float(sp["points"]) if sp is not None else 0.0,
            "championship_position": _none(standings.loc[rnd, "position"]) if rnd in standings.index else None,
        })

    finished = own[~own["status"].map(is_dnf)]
    dnf = own[own["status"].map(is_dnf)]
    last = standings.sort_index().iloc[-1] if not standings.empty else None
    teams = list(dict.fromkeys(own["team"]))
    person = _person(driver_id, year)
    born = pd.to_datetime(person["dob"], errors="coerce")

    return {
        **person,
        "year": year,
        "age": int((pd.Timestamp.now() - born).days // 365.25) if pd.notna(born) else None,
        "teams": teams,
        "team": teams[-1],
        "championship": {
            "position": _none(last["position"]) if last is not None else None,
            "points": float(last["points"]) if last is not None else 0.0,
            "after_round": int(standings.index.max()) if not standings.empty else None,
        },
        "stats": {
            "starts": len(own),
            "wins": int((own["position"] == 1).sum()),
            "podiums": int((own["position"] <= 3).sum()),
            "poles": int(_poles(own).get(driver_id, 0)),
            "dnfs": len(dnf),
            "dnfs_incident": int(dnf["status"].isin(INCIDENT).sum()),
            "best_finish": _none(own["position"].min()),
            "avg_grid": round(float(own.loc[own["grid"] > 0, "grid"].mean()), 1) if (own["grid"] > 0).any() else None,
            "avg_finish": round(float(finished["positionOrder"].mean()), 1) if len(finished) else None,
            "points_finishes": int((own["points"] > 0).sum()),
            "sprint_points": float(sprints["points"].sum()) if not sprints.empty else 0.0,
        },
        "races": races,
        "teammates": _teammate_summary(year, driver_id),
        "rating": get_season_rating(year, driver_id),
        "seasons": sorted(int(y) for y in _results.loc[_results["driverId"] == driver_id, "year"].unique()),
    }


# ---------------------------------------------------------------------------
# Кар'єра
# ---------------------------------------------------------------------------

def get_driver_career(driver_ref: str) -> dict:
    driver_id = driver_id_by_ref(driver_ref)
    own = _results[_results["driverId"] == driver_id].sort_values(["year", "round"])
    if own.empty:
        raise ValueError(f"У датасеті немає гонок пілота '{driver_ref}'")
    poles = _poles(own)
    # Найшвидші кола відомі лише з 2004 - для старших кар'єр число було б
    # заниженим, тож не показуємо його зовсім.
    fastest = int((own["rank"] == 1).sum()) if own["year"].min() >= FASTEST_LAP_DATA_FROM else None

    seasons = []
    for year, rows in own.groupby("year"):
        season_ids = _races.loc[_races["year"] == year, "raceId"]
        standing = csv_data._driver_standings[
            csv_data._driver_standings["raceId"].isin(season_ids) & (csv_data._driver_standings["driverId"] == driver_id)
        ].merge(_races[["raceId", "round"]], on="raceId").sort_values("round")
        last = standing.iloc[-1] if not standing.empty else None
        season_done = not _races[(_races["year"] == year) & ~_races["raceId"].isin(_results["raceId"])].shape[0]
        seasons.append({
            "year": int(year),
            "teams": list(dict.fromkeys(rows["team"])),
            "starts": len(rows),
            "position": _none(last["position"]) if last is not None else None,
            "points": float(last["points"]) if last is not None else float(rows["points"].sum()),
            "wins": int((rows["position"] == 1).sum()),
            "podiums": int((rows["position"] <= 3).sum()),
            "poles": int(_poles(rows).get(driver_id, 0)),
            "final": bool(season_done),
        })

    finished_seasons = [s for s in seasons if s["final"] and s["position"]]
    best = min(finished_seasons, key=lambda s: (s["position"], -s["points"])) if finished_seasons else None
    first, last = own.iloc[0], own.iloc[-1]
    return {
        **_person(driver_id, int(last["year"])),
        "totals": {
            "seasons": len(seasons),
            "starts": len(own),
            "titles": sum(1 for s in finished_seasons if s["position"] == 1),
            "wins": int((own["position"] == 1).sum()),
            "podiums": int((own["position"] <= 3).sum()),
            "poles": int(poles.get(driver_id, 0)),
            "fastest_laps": fastest,
            "points": float(sum(s["points"] for s in seasons)),
            "dnfs": int(own["status"].map(is_dnf).sum()),
        },
        "teams": list(dict.fromkeys(own["team"])),
        "debut": {"year": int(first["year"]), "round": int(first["round"]), "name": first["name"]},
        "last_race": {"year": int(last["year"]), "round": int(last["round"]), "name": last["name"]},
        "best_season": best,
        "seasons": seasons,
    }


# ---------------------------------------------------------------------------
# Контекст пілота на конкретному гран-прі
# ---------------------------------------------------------------------------

def get_driver_event(year: int, round_number: int, driver_ref: str) -> dict:
    driver_id = driver_id_by_ref(driver_ref)
    race = _races[(_races["year"] == year) & (_races["round"] == round_number)]
    if race.empty:
        raise ValueError(f"Етап {round_number} сезону {year} відсутній у датасеті")
    race = race.iloc[0]
    season = _season_races(year)
    st = csv_data._driver_standings
    prev_race = season[season["round"] < round_number].tail(1)

    def standing(race_id):
        row = st[(st["raceId"] == race_id) & (st["driverId"] == driver_id)]
        return None if row.empty else {"position": int(row.iloc[0]["position"]), "points": float(row.iloc[0]["points"])}

    before = standing(int(prev_race.iloc[0]["raceId"])) if not prev_race.empty else None
    after = standing(int(race["raceId"]))

    teammate = None
    tm_rows = [r for r in _teammate_rows(year, driver_id) if r["round"] == round_number]
    if tm_rows:
        r = tm_rows[0]
        mate = _person(r["mate_id"], year)
        mine_res = _results[(_results["raceId"] == race["raceId"]) & (_results["driverId"] == driver_id)].iloc[0]
        mate_res = _results[(_results["raceId"] == race["raceId"]) & (_results["driverId"] == r["mate_id"])].iloc[0]
        teammate = {
            "driver_ref": mate["driver_ref"], "code": mate["code"], "full_name": mate["full_name"],
            "headshot_url": mate["headshot_url"], "team_color": mate["team_color"],
            "quali": {"mine": r["quali_mine"], "mate": r["quali_mate"], "gap_pct": _none(r["quali_gap_pct"])},
            "race": {"mine": _result_label(mine_res), "mate": _result_label(mate_res)},
            "points": {"mine": r["points_mine"], "mate": r["points_mate"]},
        }

    history = _results[(_results["driverId"] == driver_id) & (_results["circuitId"] == race["circuitId"])
                       & (_results["date"] < race["date"])].sort_values("date", ascending=False)

    return {
        "driver_ref": driver_ref,
        # FastF1 віддає CountryCode порожнім - сторінка етапу бере країну звідси.
        "nationality": _person(driver_id)["nationality"],
        "championship": {"before": before, "after": after},
        "teammate": teammate,
        "circuit_history": [
            {"year": int(h["year"]), "team": h["team"], "grid": _none(h["grid"]), "label": _result_label(h),
             "status": h["status"], "points": float(h["points"])}
            for _, h in history.iterrows()
        ],
    }

"""
Побудова рядка фіч для гонки, якої ще НЕМАЄ в df_model - тобто для
поточного/майбутнього етапу сезону, який ще не встиг потрапити у
статичний Ergast-датасет (data/*.csv оновлюється періодично, не в
реальному часі - див. приклад: у серпні 2026 голландський етап (раунд
12) вже відбувся, а кваліфікації/результатів на нього в CSV ще нема).

Цього немає в оригінальному f1_pipeline_v2.py - там працювали виключно
з готовим історичним датасетом. Тут ті самі формули фіч (shift+rolling/
expanding з dataset.py), але пораховані "на льоту" для конкретної дати
гонки замість векторизовано по всій історії.

Обмеження (навмисні спрощення, задокументовані також у відповіді
app/analysis.py через notes):
  - grid = позиція кваліфікації (реальні грід-пенальті після кваліфікації
    тут не враховуються - для гонки, що ще не проїхана, їх узагалі не
    видно наперед);
  - справжньої практики немає: early_pace_* - це темп на колах 2-6
    попередніх гонок пілота (як і в тренуванні, див. dataset.py);
  - constructorId пілота визначається його останньою відомою командою,
    з спробою уточнити за назвою команди з FastF1 (на випадок трансферу).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app import fastf1_client
from app.i18n import canonical, tr
from app.ml.dataset import CATEGORICAL, CON_FORM, Dataset, constructor_race_table
from app.ml.fastf1_features import (WEEKEND_FEATURES, add_pace_vs_grid, load_session_aggregates,
                                    weekend_features)

warnings.filterwarnings("ignore")


@dataclass
class LiveDriverEntry:
    driver_id: int
    driver_name: str
    constructor_id: int | None
    constructor_name: str
    grid: float | None = None
    quali_position: float | None = None
    best_quali_sec: float | None = None
    q1_sec: float | None = None
    q2_sec: float | None = None
    q3_sec: float | None = None


def _tail_agg(series: pd.Series, w: int, fn: str, min_p: int = 3) -> float:
    """Еквівалент shift_roll(w, fn, min_p) з dataset.py, пораховане для
    "наступного" (ще не існуючого) рядка: останні w значень СТРОГО до
    цільової дати - те саме, що дає shift(1).rolling(w). Як і pandas
    rolling, вікно - це останні w РЯДКІВ, а min_p рахується по не-NaN
    значеннях усередині нього."""
    s = series.tail(w).dropna()
    if len(s) < min_p:
        return np.nan
    if fn == "mean":
        return float(s.mean())
    if fn == "std":
        return float(s.std()) if len(s) > 1 else np.nan
    if fn == "sum":
        return float(s.sum())
    raise ValueError(fn)


def _expand_agg(series: pd.Series, fn: str = "mean", min_p: int = 3) -> float:
    series = series.dropna()  # як pandas expanding(min_periods): рахуються лише не-NaN
    if len(series) < min_p:
        return np.nan
    if fn == "mean":
        return float(series.mean())
    if fn == "std":
        return float(series.std()) if len(series) > 1 else np.nan
    raise ValueError(fn)


def _driver_history(dataset: Dataset, driver_id: int, before_date: pd.Timestamp) -> pd.DataFrame:
    d = dataset.raw[(dataset.raw["driverId"] == driver_id) & (dataset.raw["date"] < before_date)]
    return d.sort_values("date")


def _resolve_driver_id(dataset: Dataset, code: str | None, full_name: str | None) -> int | None:
    """Зіставляє пілота FastF1 (Abbreviation/FullName) з driverId
    локального датасету. 3-літерний код НЕ унікальний за всю історію F1
    (наприклад "VER" - і Jean-Éric Vergne, і Max Verstappen) - тому
    спершу пробуємо повний збіг за іменем (майже завжди унікальний), а
    коли лишається кілька кандидатів (за іменем чи за кодом) - беремо
    того, хто реально їздив НАЙПІЗНІШЕ (в history), а не першого-ліпшого
    за driverId."""
    candidates = pd.DataFrame()

    if full_name:
        full = dataset.drivers["forename"].fillna("") + " " + dataset.drivers["surname"].fillna("")
        match = dataset.drivers[full.str.lower() == full_name.strip().lower()]
        if len(match) == 1:
            return int(match.iloc[0]["driverId"])
        if len(match) > 1:
            candidates = match

    if code:
        match = dataset.drivers[dataset.drivers["code"].astype(str).str.upper() == code.upper()]
        if len(match) == 1:
            return int(match.iloc[0]["driverId"])
        if len(match) > 1 and candidates.empty:
            candidates = match

    if not candidates.empty:
        ids = candidates["driverId"].tolist()
        recency = dataset.raw[dataset.raw["driverId"].isin(ids)].groupby("driverId")["date"].max()
        if not recency.empty:
            return int(recency.idxmax())
        return int(candidates.iloc[0]["driverId"])

    return None


def _normalize_team(name: str) -> str:
    n = name.lower().strip()
    for junk in ("f1 team", "formula 1 team", "racing team", "racing", " team", "-"):
        n = n.replace(junk, "")
    return "".join(n.split())


def _resolve_constructor_id(dataset: Dataset, team_name: str | None,
                             fallback_constructor_id: int | None,
                             candidate_ids: set[int]) -> tuple[int | None, str]:
    """Пробуємо впізнати команду за назвою з FastF1 серед конструкторів,
    що реально виступали на найсвіжішому відомому етапі (candidate_ids).
    Якщо однозначного збігу нема - лишаємось на останній відомій команді
    пілота (fallback_constructor_id)."""
    if team_name:
        target = _normalize_team(team_name)
        best_id, best_len = None, 0
        for cid in candidate_ids:
            row = dataset.constructors[dataset.constructors["constructorId"] == cid]
            if row.empty:
                continue
            cname = _normalize_team(str(row.iloc[0]["name"]))
            if target == cname:
                best_id, best_len = cid, len(cname)
                break
            if target in cname or cname in target:
                if len(cname) > best_len:
                    best_id, best_len = cid, len(cname)
        if best_id is not None:
            name = dataset.constructors.loc[
                dataset.constructors["constructorId"] == best_id, "name"
            ].iloc[0]
            return best_id, str(name)

    if fallback_constructor_id is not None:
        row = dataset.constructors[dataset.constructors["constructorId"] == fallback_constructor_id]
        if not row.empty:
            return fallback_constructor_id, str(row.iloc[0]["name"])

    return None, team_name or "Unknown"


def _fetch_live_quali(year: int, round_number: int) -> list[dict] | None:
    try:
        results = fastf1_client.get_session_results(year, round_number, "Q")
    except Exception:
        return None
    return results or None


def _fetch_live_weekend(year: int, round_number: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Практики й погода кваліфікації вікенду напряму з FastF1 - для етапу,
    якого ще нема в data/fastf1_*.csv. Сесії, що не відбулись/не існують
    (спринт-формат, майбутнє), просто пропускаються."""
    practice, weather = [], []
    for code, with_laps in (("FP1", True), ("FP2", True), ("FP3", True), ("Q", False)):
        try:
            agg, wx, _, _ = load_session_aggregates(year, round_number, code, with_laps)
        except Exception:
            continue
        weather.append({"session": code, **wx})
        if agg is not None:
            practice.append(agg.assign(session=code))
    return (pd.concat(practice, ignore_index=True) if practice else pd.DataFrame(),
            pd.DataFrame(weather))


def _weekend_rows(dataset: Dataset, year: int, round_number: int, race_id: int,
                  entries: list[LiveDriverEntry], notes: list[str]) -> pd.DataFrame:
    """Фічі вікенду (pr_*/wx_*) для пілотів entries - та сама
    weekend_features(), що й у dataset.py: з уже завантажених
    data/fastf1_*.csv, а якщо там цього етапу нема - наживо з FastF1."""
    practice = (dataset.practice[dataset.practice["raceId"] == race_id].copy()
                if not dataset.practice.empty else pd.DataFrame())
    weather = (dataset.weather[dataset.weather["raceId"] == race_id].copy()
               if not dataset.weather.empty else pd.DataFrame())
    if practice.empty:
        practice, weather = _fetch_live_weekend(year, round_number)
        if practice.empty:
            notes.append(canonical("note_practice_missing"))
            return pd.DataFrame(columns=["driverId", *WEEKEND_FEATURES])
        practice["driverId"] = np.nan
        practice["raceId"] = race_id
        weather["raceId"] = race_id

    # Пілоти, яких ще нема в results.csv (етап поза датасетом), зіставляємо
    # як і кваліфікацію; лишаємо лише тих, хто в складі гонки (без резервістів
    # FP1) - як dataset.py, де зіставлення йде по учасниках гонки.
    missing = practice["driverId"].isna()
    practice.loc[missing, "driverId"] = [
        _resolve_driver_id(dataset, r.driver_code, r.full_name) for r in practice[missing].itertuples()
    ]
    entry_ids = {e.driver_id for e in entries}
    practice = practice[practice["driverId"].isin(entry_ids)]
    if practice.empty:
        return pd.DataFrame(columns=["driverId", *WEEKEND_FEATURES])
    return weekend_features(practice, weather).drop(columns="raceId")


def build_entry_list(dataset: Dataset, year: int, round_number: int,
                      race_date: pd.Timestamp) -> tuple[list[LiveDriverEntry], bool, list[str]]:
    """Повертає (entries, quali_available, notes)."""
    notes: list[str] = []

    # Пілоти/команди останнього відомого (в df.raw) етапу - базовий
    # склад "за замовчуванням" і джерело fallback-constructorId.
    past = dataset.raw[dataset.raw["date"] < race_date]
    if past.empty:
        raise ValueError(tr("note_no_history"))
    last_race_date = past["date"].max()
    last_lineup = past[past["date"] == last_race_date][["driverId", "constructor_name", "constructorId"]]
    last_constructor_of_driver = dict(zip(last_lineup["driverId"], last_lineup["constructorId"]))
    candidate_constructor_ids = set(last_lineup["constructorId"].dropna().astype(int).tolist())

    live_quali = _fetch_live_quali(year, round_number)

    if live_quali:
        entries = []
        for row in live_quali:
            code = row.get("Abbreviation")
            full_name = row.get("FullName")
            driver_id = _resolve_driver_id(dataset, code, full_name)
            if driver_id is None:
                notes.append(canonical("note_driver_unmatched", name=full_name or code))
                continue
            fallback_cid = last_constructor_of_driver.get(driver_id)
            cid, cname = _resolve_constructor_id(
                dataset, row.get("TeamName"), fallback_cid, candidate_constructor_ids
            )
            q1 = row.get("Q1")
            q2 = row.get("Q2")
            q3 = row.get("Q3")
            best = min([v for v in (q1, q2, q3) if v is not None], default=None)
            quali_pos = row.get("Position")
            entries.append(LiveDriverEntry(
                driver_id=driver_id,
                driver_name=full_name or code or str(driver_id),
                constructor_id=cid,
                constructor_name=cname,
                grid=quali_pos,          # проксі: реальні грід-пенальті тут не відомі наперед
                quali_position=quali_pos,
                best_quali_sec=best,
                q1_sec=q1, q2_sec=q2, q3_sec=q3,
            ))
        if entries:
            return entries, True, notes
        notes.append(canonical("note_quali_unmatched"))

    # Fallback: кваліфікації ще нема (або не вдалось завантажити) -
    # беремо склад пілотів/команд з останнього відомого етапу.
    notes.append(canonical("note_quali_missing"))
    entries = [
        LiveDriverEntry(
            driver_id=int(r.driverId),
            driver_name=str(dataset.drivers.loc[dataset.drivers["driverId"] == r.driverId, "forename"].iloc[0])
            + " " + str(dataset.drivers.loc[dataset.drivers["driverId"] == r.driverId, "surname"].iloc[0]),
            constructor_id=int(r.constructorId) if pd.notna(r.constructorId) else None,
            constructor_name=str(r.constructor_name),
        )
        for r in last_lineup.itertuples()
    ]
    return entries, False, notes


def build_feature_rows(dataset: Dataset, year: int, round_number: int) -> tuple[pd.DataFrame, bool, list[str]]:
    """Формує DataFrame з тими самими колонками, що dataset.df_model
    (features + raceId/driverId/driver_name/constructor_name/name/year/
    positionOrder/dnf), готовий для app.ml.predict.predict_two_stage().

    positionOrder/dnf у виводі - завжди NaN (результату гонки ще/навмисно
    не існує для моделі - вона його не бачить)."""
    race_row = dataset.races[(dataset.races["year"] == year) & (dataset.races["round"] == round_number)]
    if race_row.empty:
        raise ValueError(tr("round_not_in_calendar", round=round_number, year=year))
    race_row = race_row.iloc[0]
    race_id = int(race_row["raceId"])
    race_date = race_row["date"]
    circuit_id = int(race_row["circuitId"])
    race_name = str(race_row["name"])

    entries, quali_available, notes = build_entry_list(dataset, year, round_number, race_date)

    poles_sec = None
    if quali_available:
        best_vals = [e.best_quali_sec for e in entries if e.best_quali_sec is not None]
        poles_sec = min(best_vals) if best_vals else None

    rows = []
    for e in entries:
        drv_hist = _driver_history(dataset, e.driver_id, race_date)
        con_hist = (dataset.raw[(dataset.raw["constructorId"] == e.constructor_id) &
                                 (dataset.raw["date"] < race_date)].sort_values("date")
                    if e.constructor_id is not None else drv_hist.iloc[0:0])
        track_hist = drv_hist[drv_hist["circuitId"] == circuit_id]
        con_track_hist = con_hist[con_hist["circuitId"] == circuit_id]
        con_races = constructor_race_table(con_hist)
        circuit_hist = dataset.raw[(dataset.raw["circuitId"] == circuit_id) &
                                    (dataset.raw["date"] < race_date)]

        dob_row = dataset.drivers.loc[dataset.drivers["driverId"] == e.driver_id, "dob"]
        dob = pd.to_datetime(dob_row.iloc[0], errors="coerce") if not dob_row.empty else pd.NaT
        driver_age = round((race_date - dob).days / 365.25, 1) if pd.notna(dob) else np.nan

        delta_to_pole_sec = (e.best_quali_sec - poles_sec) if (e.best_quali_sec is not None and poles_sec is not None) else np.nan
        delta_to_pole_pct = (delta_to_pole_sec / poles_sec * 100) if (pd.notna(delta_to_pole_sec) and poles_sec) else np.nan
        q1_q2_delta = (e.q1_sec - e.q2_sec) if (e.q1_sec is not None and e.q2_sec is not None) else np.nan
        q2_q3_delta = (e.q2_sec - e.q3_sec) if (e.q2_sec is not None and e.q3_sec is not None) else np.nan
        progressed_to_q3 = int(e.q3_sec is not None)
        grid_penalty = 0.0  # невідомо наперед без офіційного грід-листа

        career_races = len(drv_hist)

        # championship standings: останній відомий рядок для цього пілота
        # з raceId, чия дата строго до цільової гонки.
        ds = dataset.driver_standings.merge(
            dataset.races[["raceId", "date"]], on="raceId", how="left"
        )
        ds_driver = ds[(ds["driverId"] == e.driver_id) & (ds["date"] < race_date)].sort_values("date")
        if not ds_driver.empty:
            champ_points = float(ds_driver.iloc[-1]["points"])
            champ_position = float(ds_driver.iloc[-1]["position"])
            champ_wins = float(ds_driver.iloc[-1]["wins"])
        else:
            champ_points = champ_position = champ_wins = np.nan

        # races_with_team: скільки ПОСПІЛЬ останніх гонок пілот провів у e.constructor_id
        races_with_team = 0
        for _, r in drv_hist.sort_values("date", ascending=False).iterrows():
            if e.constructor_id is not None and int(r["constructorId"]) == e.constructor_id:
                races_with_team += 1
            else:
                break

        row = {
            "raceId": race_id, "driverId": e.driver_id, "constructorId": e.constructor_id,
            "circuitId": circuit_id, "year": year, "round": int(round_number),
            "name": race_name, "driver_name": e.driver_name, "constructor_name": e.constructor_name,
            "positionOrder": np.nan, "dnf": np.nan,

            "grid": e.grid, "quali_position": e.quali_position, "best_quali_sec": e.best_quali_sec,
            "delta_to_pole_sec": delta_to_pole_sec, "delta_to_pole_pct": delta_to_pole_pct,
            "q1_q2_delta": q1_q2_delta, "q2_q3_delta": q2_q3_delta,
            "progressed_to_q3": progressed_to_q3, "grid_penalty": grid_penalty,

            "driver_avg_pos_5": _tail_agg(drv_hist["positionOrder"], 5, "mean"),
            "driver_avg_pos_10": _tail_agg(drv_hist["positionOrder"], 10, "mean"),
            "driver_avg_quali_5": _tail_agg(drv_hist["quali_position"], 5, "mean"),
            "driver_avg_quali_10": _tail_agg(drv_hist["quali_position"], 10, "mean"),
            "driver_pts_5": _tail_agg(drv_hist["points"], 5, "sum"),
            "driver_pts_10": _tail_agg(drv_hist["points"], 10, "sum"),
            "driver_consistency_5": _tail_agg(drv_hist["positionOrder"], 5, "std"),
            "driver_quali_std_10": _tail_agg(drv_hist["quali_position"], 10, "std"),
            "driver_dnf_rate_5": _tail_agg(drv_hist["dnf"], 5, "mean"),
            "driver_dnf_rate_10": _tail_agg(drv_hist["dnf"], 10, "mean"),
            "driver_gain_5": _tail_agg(drv_hist["start_finish_delta"], 5, "mean"),
            "driver_gain_10": _tail_agg(drv_hist["start_finish_delta"], 10, "mean"),
            "driver_overtakes_5": _tail_agg(drv_hist["overtakes"], 5, "mean"),
            "driver_lap1_gain_5": _tail_agg(drv_hist["lap1_gain"], 5, "mean"),
            "driver_cum_avg_pos": _expand_agg(drv_hist["positionOrder"], "mean"),
            "driver_career_races": career_races,
            "is_rookie": int(career_races < 10),

            "driver_quali_pace_3": _tail_agg(drv_hist["delta_to_pole_pct"], 3, "mean", min_p=2),
            "driver_quali_pace_5": _tail_agg(drv_hist["delta_to_pole_pct"], 5, "mean"),

            **{feat: _tail_agg(con_races[col], w, fn, min_p)
               for feat, (col, w, fn, min_p) in CON_FORM.items()},

            "track_avg_pos": _expand_agg(track_hist["positionOrder"], "mean", min_p=2),
            "track_avg_quali": _expand_agg(track_hist["quali_position"], "mean", min_p=2),
            "track_races": len(track_hist),
            "circuit_avg_dnf": _expand_agg(circuit_hist["dnf"], "mean", min_p=10),
            "circuit_avg_finish": _expand_agg(circuit_hist["positionOrder"], "mean", min_p=10),

            "teammate_quali_delta_5": _tail_agg(drv_hist["quali_vs_team"], 5, "mean"),
            "teammate_finish_delta_5": _tail_agg(drv_hist["finish_vs_team"], 5, "mean"),
            "new_team_period": int(races_with_team < 3),

            # Темп на ранніх колах за ПОПЕРЕДНІ гонки пілота - та сама
            # формула shift+rolling(5), що й у dataset.py.
            "early_pace_gap_5": _tail_agg(drv_hist["fp_gap_to_leader_pct"], 5, "mean"),
            "early_pace_rank_5": _tail_agg(drv_hist["fp_rank"], 5, "mean"),
            "early_pace_std_5": _tail_agg(drv_hist["fp_pace_std"], 5, "mean"),

            "driver_age": driver_age,
            "wet_skill": (_expand_agg(drv_hist["positionOrder"].where(drv_hist["wet"] == 0), min_p=2)
                          - _expand_agg(drv_hist["positionOrder"].where(drv_hist["wet"] == 1), min_p=2)),
            "comeback_score": _expand_agg(drv_hist["start_finish_delta"].where(drv_hist["grid"] >= 10)),

            "champ_points": champ_points, "champ_position": champ_position, "champ_wins": champ_wins,
        }

        con_track_avg = _expand_agg(con_track_hist["positionOrder"], "mean", min_p=2)
        track_avg_pos = row["track_avg_pos"]
        row["track_affinity"] = (con_track_avg - track_avg_pos) if (pd.notna(con_track_avg) and pd.notna(track_avg_pos)) else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    weekend = _weekend_rows(dataset, year, round_number, race_id, entries, notes)
    out = out.merge(weekend.astype({"driverId": "int64"}), on="driverId", how="left")
    add_pace_vs_grid(out)

    # Категоріальні колонки - ті самі категорії, що використані під час
    # тренування (інакше LightGBM/XGB розкодують інший driverId в інший індекс).
    for col, cat_col in (("driverId", "driverId_cat"), ("constructorId", "constructorId_cat"),
                          ("circuitId", "circuitId_cat")):
        known_categories = dataset.raw[cat_col].cat.categories
        out[cat_col] = pd.Categorical(out[col], categories=known_categories)

    # NaN лишаються NaN, як і в тренуванні (dataset.py) - моделі
    # обробляють їх нативно. to_numeric - бо None з LiveDriverEntry
    # інакше дає object-колонку.
    for col in dataset.num_features:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out, quali_available, notes

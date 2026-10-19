"""
Завантаження локального Ergast-датасету (data/*.csv) + feature
engineering. Перенесено з CELL 2-17 f1_pipeline_v2.py, з виправленим
leakage (див. FEATURES_VERSION): fp_*/wet поточної гонки не є фічами, а
командні/трасові агрегати не бачать інших рядків тієї самої гонки.
Перевіряється tests/test_ml_leakage.py.

build_dataset() важка (~20с на всьому датасеті через lap_times.csv -
878К рядків) і чиста (не залежить від конкретного запиту), тому
результат кешується на диск (joblib) і в пам'яті процесу - див.
get_dataset() унизу файлу.
"""
from __future__ import annotations

import hashlib
import os
import threading
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from app.ml import config
from app.ml.fastf1_features import WEEKEND_FEATURES, add_pace_vs_grid, weekend_features

warnings.filterwarnings("ignore")

CACHE_FILE = os.path.join(config.MODELS_DIR, "dataset.joblib")

# Ті самі CSV, що й у CELL 2. lap_times - опційний (без нього просто
# не буде wet/overtakes/practice-proxy фіч), решта обов'язкові.
_REQUIRED_FILES = [
    "races.csv", "results.csv", "drivers.csv", "constructors.csv",
    "qualifying.csv", "circuits.csv", "driver_standings.csv", "status.csv",
]
# fastf1_*.csv - практики й погода з FastF1 (scripts/fetch_fastf1.py), 2018+;
# без них фічі вікенду (pr_*/wx_*) просто NaN.
_OPTIONAL_FILES = ["lap_times.csv", "fastf1_practice.csv", "fastf1_weather.csv"]

# Версія формул фіч - входить у fingerprint, тож зміна коду feature
# engineering інвалідує кеш датасету й моделей так само, як зміна CSV.
# v2: fp_* -> early_pace_* (shift+rolling), без wet, _prior_races_only.
# v3: без імпутації NaN медіанами.
# v4: форма команди по гонках (CON_FORM), races_with_team, без дублів рядків.
# v5: фічі вікенду з FastF1 (pr_*/wx_*).
FEATURES_VERSION = 5

CATEGORICAL = ["driverId_cat", "constructorId_cat", "circuitId_cat"]

FEATURE_ORDER = [
    # Quali (head)
    "grid", "quali_position", "best_quali_sec",
    "delta_to_pole_sec", "delta_to_pole_pct",
    "q1_q2_delta", "q2_q3_delta", "progressed_to_q3",
    "grid_penalty",
    # Driver form
    "driver_avg_pos_5", "driver_avg_pos_10",
    "driver_avg_quali_5", "driver_avg_quali_10",
    "driver_pts_5", "driver_pts_10",
    "driver_consistency_5", "driver_quali_std_10",
    "driver_dnf_rate_5", "driver_dnf_rate_10",
    "driver_gain_5", "driver_gain_10",
    "driver_overtakes_5", "driver_lap1_gain_5",
    "driver_cum_avg_pos", "driver_career_races", "is_rookie",
    # Driver pace
    "driver_quali_pace_3", "driver_quali_pace_5",
    # Constructor
    "con_avg_pos_5", "con_avg_pos_10", "con_avg_quali_5",
    "con_pts_5", "con_dnf_rate_5",
    "con_quali_pace_3", "con_quali_pace_5",
    # Track
    "track_avg_pos", "track_avg_quali", "track_races", "track_affinity",
    "circuit_avg_dnf", "circuit_avg_finish",
    # Teammate
    "teammate_quali_delta_5", "teammate_finish_delta_5", "new_team_period",
    # Weekend: практики й погода цього вікенду (FastF1, 2018+)
    *WEEKEND_FEATURES,
    # Early-lap pace (форма за попередні гонки, не поточна гонка)
    "early_pace_gap_5", "early_pace_rank_5", "early_pace_std_5",
    # Context (wet поточної гонки навмисно не фіча - наперед невідомий)
    "round", "driver_age",
    "wet_skill", "comeback_score",
    # Standings
    "champ_points", "champ_position", "champ_wins",
    # Categorical
    *CATEGORICAL,
]


@dataclass
class Dataset:
    """Результат feature engineering + сирі таблиці, потрібні пізніше
    (app/ml/live.py) для побудови рядка фіч гонки поза df_model."""
    raw: pd.DataFrame          # base+фічі, ДО фільтра dropna/MIN_YEAR - повна історія
    df_model: pd.DataFrame     # raw, відфільтрований під моделювання (те, на чому тренуємось)
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    num_features: list[str]
    categorical: list[str]
    races: pd.DataFrame
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    circuits: pd.DataFrame
    qualifying: pd.DataFrame
    driver_standings: pd.DataFrame
    # Агрегати FastF1 з raceId; driverId = NaN, якщо гонки ще нема в
    # results.csv (live.py зіставляє таких пілотів сам).
    practice: pd.DataFrame = field(default_factory=pd.DataFrame)
    weather: pd.DataFrame = field(default_factory=pd.DataFrame)
    fingerprint: str = field(default="")


def _load_csv(name: str, required: bool = True) -> pd.DataFrame | None:
    path = os.path.join(config.DATA_DIR, name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Обов'язковий файл датасету відсутній: {path}")
        return None
    return pd.read_csv(path, na_values=["\\N", "N", ""])


def _fingerprint() -> str:
    """Хеш з mtime+розміру CSV, які впливають на датасет - щоб не
    перебудовувати engineered dataset/моделі на кожен рестарт, але
    підхопити зміну даних без ручного видалення кешу."""
    h = hashlib.sha256()
    h.update(f"features:v{FEATURES_VERSION}".encode())
    for name in _REQUIRED_FILES + _OPTIONAL_FILES:
        path = os.path.join(config.DATA_DIR, name)
        if os.path.exists(path):
            st = os.stat(path)
            h.update(f"{name}:{st.st_mtime_ns}:{st.st_size}".encode())
        else:
            h.update(f"{name}:missing".encode())
    return h.hexdigest()


def _lap_to_sec(t):
    if pd.isna(t):
        return np.nan
    t = str(t).strip()
    if not t or t in ("\\N", "N"):
        return np.nan
    try:
        parts = t.split(":")
        return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(t)
    except Exception:
        return np.nan


def _classify_dnf(status_str) -> int:
    if pd.isna(status_str):
        return 0
    s = str(status_str).strip()
    if s == "Finished":
        return 0
    if s.startswith("+") and "Lap" in s:  # фінішував, але з відставанням
        return 0
    return 1


def _shift_roll(s: pd.Series, w: int, fn: str = "mean", min_p: int = 3) -> pd.Series:
    """Rolling з shift(1) - гарантовано без leakage: рядок raceId=X
    бачить лише гонки СТРОГО до X."""
    s = s.shift(1)
    if fn == "mean":
        return s.rolling(w, min_periods=min_p).mean()
    if fn == "std":
        return s.rolling(w, min_periods=min_p).std()
    if fn == "sum":
        return s.rolling(w, min_periods=min_p).sum()
    if fn == "median":
        return s.rolling(w, min_periods=min_p).median()
    raise ValueError(fn)


def _shift_expand(s: pd.Series, fn: str = "mean", min_p: int = 3) -> pd.Series:
    s = s.shift(1)
    if fn == "mean":
        return s.expanding(min_periods=min_p).mean()
    if fn == "std":
        return s.expanding(min_periods=min_p).std()
    if fn == "count":
        return s.expanding(min_periods=min_p).count()
    raise ValueError(fn)


def _prior_races_only(df: pd.DataFrame, group_cols: list[str], cols: list[str]) -> None:
    """Для груп, де в одній гонці кілька рядків (команда - 2 пілоти,
    траса - ~20), shift(1) прибирає лише власний рядок: другий рядок
    гонки вже бачить результат першого (напарника) у ТІЙ САМІЙ гонці.
    Лише перший рядок гонки в групі бачить виключно попередні гонки -
    його значення й розносимо на всю гонку. df має бути відсортований
    за group_cols + date. Змінює df на місці."""
    keys = group_cols + ["raceId"]
    first = df.groupby(keys).cumcount() == 0
    for c in cols:
        df[c] = df[c].where(first).groupby([df[k] for k in keys]).transform("max")


# Форма команди рахується по ГОНКАХ, а не по рядках пілотів: вікно "5
# рядків" = 2.5 гонки, і в неповну гонку потрапляв випадковий з двох
# пілотів (різний у dataset.py і live.py). feature -> (колонка
# constructor_race_table, вікно в гонках, fn, min_periods).
CON_FORM = {
    "con_avg_pos_5": ("con_pos", 5, "mean", 2),
    "con_avg_pos_10": ("con_pos", 10, "mean", 2),
    "con_avg_quali_5": ("con_quali", 5, "mean", 2),
    "con_pts_5": ("con_pts", 5, "sum", 2),
    "con_dnf_rate_5": ("con_dnf", 5, "mean", 2),
    "con_quali_pace_3": ("con_pace", 3, "mean", 2),
    "con_quali_pace_5": ("con_pace", 5, "mean", 2),
}


def constructor_race_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Рядки пілотів -> один рядок на (constructorId, raceId), по даті.
    Спільна для dataset.py і app/ml/live.py, щоб форма команди в
    тренуванні й у live-прогнозі рахувалась однаково."""
    return (rows.groupby(["constructorId", "raceId"], as_index=False)
            .agg(date=("date", "first"),
                 con_pos=("positionOrder", "mean"),
                 con_quali=("quali_position", "mean"),
                 con_pts=("points", "sum"),
                 con_dnf=("dnf", "mean"),
                 con_pace=("delta_to_pole_pct", "mean"))
            .sort_values(["constructorId", "date"])
            .reset_index(drop=True))


def _load_fastf1(races: pd.DataFrame, results: pd.DataFrame,
                 drivers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """data/fastf1_*.csv -> (practice, weather) з raceId (по year+round) і
    driverId (по номеру боліда в results.csv цієї гонки, запасний варіант -
    3-літерний код; в межах однієї гонки обидва унікальні). Резервні пілоти
    FP1, яких нема в гонці, лишаються з driverId = NaN."""
    practice = _load_csv("fastf1_practice.csv", required=False)
    weather = _load_csv("fastf1_weather.csv", required=False)
    if practice is None or weather is None:
        return pd.DataFrame(), pd.DataFrame()

    race_ids = races[["raceId", "year", "round"]]
    practice = practice.merge(race_ids, on=["year", "round"], how="inner")
    weather = weather.merge(race_ids, on=["year", "round"], how="inner")

    entrants = results[["raceId", "driverId", "number"]].merge(
        drivers[["driverId", "code"]], on="driverId", how="left")
    entrants["number"] = pd.to_numeric(entrants["number"], errors="coerce").astype("Int64").astype(str)
    practice["driver_number"] = (pd.to_numeric(practice["driver_number"], errors="coerce")
                                 .astype("Int64").astype(str))

    by_number = entrants.drop_duplicates(["raceId", "number"]).rename(columns={"number": "driver_number"})
    by_code = entrants.drop_duplicates(["raceId", "code"]).rename(columns={"code": "driver_code",
                                                                           "driverId": "_id_code"})
    practice = (practice
                .merge(by_number[["raceId", "driver_number", "driverId"]], on=["raceId", "driver_number"], how="left")
                .merge(by_code[["raceId", "driver_code", "_id_code"]], on=["raceId", "driver_code"], how="left"))
    practice["driverId"] = practice["driverId"].fillna(practice.pop("_id_code"))
    # На випадок повторного завантаження тієї самої сесії - лишаємо останнє.
    practice = practice.drop_duplicates(["raceId", "session", "driver_code"], keep="last")
    weather = weather.drop_duplicates(["raceId", "session"], keep="last")
    return practice, weather


def _build_uncached() -> Dataset:
    dataset_fp = _fingerprint()

    races = _load_csv("races.csv")
    results = _load_csv("results.csv")
    drivers = _load_csv("drivers.csv")
    constructors = _load_csv("constructors.csv")
    qualifying = _load_csv("qualifying.csv")
    circuits = _load_csv("circuits.csv")
    drv_stand = _load_csv("driver_standings.csv")
    status = _load_csv("status.csv")
    lap_times = _load_csv("lap_times.csv", required=False)
    practice, weather = _load_fastf1(races, results, drivers)

    races = races.copy()
    races["date"] = pd.to_datetime(races["date"], errors="coerce")
    races = races[races["date"].notna()].copy()

    results = results.copy()
    for col in ["grid", "position", "positionOrder", "points", "laps",
                "milliseconds", "fastestLap", "rank"]:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    qualifying = qualifying.copy()
    for col in ["q1", "q2", "q3"]:
        if col in qualifying.columns:
            qualifying[f"{col}_sec"] = qualifying[col].apply(_lap_to_sec)
    qualifying["best_quali_sec"] = qualifying[["q1_sec", "q2_sec", "q3_sec"]].min(axis=1)

    base = (
        results
        .merge(races[["raceId", "year", "round", "circuitId", "name", "date"]],
               on="raceId", how="left")
        .merge(drivers[["driverId", "driverRef", "forename", "surname", "dob"]],
               on="driverId", how="left")
        .merge(constructors[["constructorId", "constructorRef", "name"]]
               .rename(columns={"name": "constructor_name"}),
               on="constructorId", how="left")
        .merge(circuits[["circuitId", "circuitRef", "country", "lat", "lng", "alt"]],
               on="circuitId", how="left")
        .merge(status[["statusId", "status"]], on="statusId", how="left")
        .merge(qualifying[["raceId", "driverId", "position", "best_quali_sec",
                            "q1_sec", "q2_sec", "q3_sec"]]
               .rename(columns={"position": "quali_position"}),
               on=["raceId", "driverId"], how="left")
    )

    base = base.sort_values(["driverId", "date"]).reset_index(drop=True)
    base["driver_name"] = base["forename"] + " " + base["surname"]
    base["dnf"] = base["status"].apply(_classify_dnf)
    base["dob"] = pd.to_datetime(base["dob"], errors="coerce")
    base["driver_age"] = ((base["date"] - base["dob"]).dt.days / 365.25).round(1)

    # --- wet labels (leakage-safe: expanding медіана траси по МИНУЛИХ гонках) ---
    if lap_times is None or lap_times.empty:
        race_weather = pd.DataFrame(columns=["raceId", "wet"])
    else:
        lt = lap_times.copy()
        lt["milliseconds"] = pd.to_numeric(lt["milliseconds"], errors="coerce")
        lt = lt.dropna(subset=["milliseconds"])
        q05, q95 = lt["milliseconds"].quantile([0.05, 0.95])
        lt_clean = lt[(lt["milliseconds"] >= q05) & (lt["milliseconds"] <= q95)]

        race_pace = (lt_clean.groupby("raceId")["milliseconds"]
                     .median().reset_index().rename(columns={"milliseconds": "race_med_ms"}))
        race_pace = race_pace.merge(races[["raceId", "circuitId", "date", "year"]], on="raceId")
        race_pace = race_pace.sort_values(["circuitId", "date"])

        race_pace["track_med_ms"] = (
            race_pace.groupby("circuitId")["race_med_ms"]
            .transform(lambda x: x.shift(1).expanding(min_periods=2).median())
        )
        track_global = race_pace.groupby("circuitId")["race_med_ms"].median()
        race_pace["track_med_ms"] = race_pace["track_med_ms"].fillna(
            race_pace["circuitId"].map(track_global)
        )
        race_pace["pace_ratio"] = race_pace["race_med_ms"] / race_pace["track_med_ms"]
        race_pace["wet"] = (race_pace["pace_ratio"] >= 1.05).astype(int)
        race_weather = race_pace[["raceId", "wet", "pace_ratio"]].copy()

    # --- overtakes з lap_times ---
    if lap_times is None or lap_times.empty:
        overtakes = pd.DataFrame(columns=["raceId", "driverId", "overtakes",
                                           "positions_lost", "lap1_position"])
    else:
        lt = lap_times.copy()
        lt["position"] = pd.to_numeric(lt["position"], errors="coerce")
        lt = lt.dropna(subset=["position"]).sort_values(["raceId", "driverId", "lap"])
        lt["pos_prev"] = lt.groupby(["raceId", "driverId"])["position"].shift(1)
        lt["delta"] = lt["pos_prev"] - lt["position"]
        overtakes = (lt.groupby(["raceId", "driverId"])
                     .agg(overtakes=("delta", lambda x: x[x > 0].sum()),
                          positions_lost=("delta", lambda x: -x[x < 0].sum()),
                          lap1_position=("position", "first"))
                     .reset_index())

    df = base.copy()
    df = df.merge(race_weather[["raceId", "wet"]], on="raceId", how="left")
    df["wet"] = df["wet"].fillna(0).astype(int)

    if not overtakes.empty:
        df = df.merge(overtakes, on=["raceId", "driverId"], how="left")
        for c in ["overtakes", "positions_lost"]:
            df[c] = df[c].fillna(0)
        df["lap1_position"] = df["lap1_position"].fillna(df["grid"])
    else:
        df["overtakes"] = 0
        df["positions_lost"] = 0
        df["lap1_position"] = df["grid"]

    df["start_finish_delta"] = df["grid"] - df["positionOrder"]
    df["lap1_gain"] = df["grid"] - df["lap1_position"]
    df = df.sort_values(["driverId", "date"]).reset_index(drop=True)

    # --- driver features ---
    g = df.groupby("driverId")
    df["driver_avg_pos_5"] = g["positionOrder"].transform(lambda x: _shift_roll(x, 5))
    df["driver_avg_pos_10"] = g["positionOrder"].transform(lambda x: _shift_roll(x, 10))
    df["driver_avg_quali_5"] = g["quali_position"].transform(lambda x: _shift_roll(x, 5))
    df["driver_avg_quali_10"] = g["quali_position"].transform(lambda x: _shift_roll(x, 10))
    df["driver_pts_5"] = g["points"].transform(lambda x: _shift_roll(x, 5, "sum"))
    df["driver_pts_10"] = g["points"].transform(lambda x: _shift_roll(x, 10, "sum"))
    df["driver_consistency_5"] = g["positionOrder"].transform(lambda x: _shift_roll(x, 5, "std"))
    df["driver_dnf_rate_5"] = g["dnf"].transform(lambda x: _shift_roll(x, 5))
    df["driver_dnf_rate_10"] = g["dnf"].transform(lambda x: _shift_roll(x, 10))
    df["driver_gain_5"] = g["start_finish_delta"].transform(lambda x: _shift_roll(x, 5))
    df["driver_gain_10"] = g["start_finish_delta"].transform(lambda x: _shift_roll(x, 10))
    df["driver_overtakes_5"] = g["overtakes"].transform(lambda x: _shift_roll(x, 5))
    df["driver_lap1_gain_5"] = g["lap1_gain"].transform(lambda x: _shift_roll(x, 5))
    df["driver_quali_std_10"] = g["quali_position"].transform(lambda x: _shift_roll(x, 10, "std"))
    df["driver_career_races"] = g.cumcount()
    df["driver_cum_avg_pos"] = g["positionOrder"].transform(lambda x: _shift_expand(x))
    df["is_rookie"] = (df["driver_career_races"] < 10).astype(int)

    df["_pos_wet"] = df["positionOrder"].where(df["wet"] == 1)
    df["_pos_dry"] = df["positionOrder"].where(df["wet"] == 0)
    df["wet_avg_pos"] = g["_pos_wet"].transform(lambda x: _shift_expand(x, min_p=2))
    df["dry_avg_pos"] = g["_pos_dry"].transform(lambda x: _shift_expand(x, min_p=2))
    df["wet_skill"] = df["dry_avg_pos"] - df["wet_avg_pos"]
    df.drop(columns=["_pos_wet", "_pos_dry"], inplace=True)

    df["_gain_bad"] = df["start_finish_delta"].where(df["grid"] >= 10)
    df["comeback_score"] = g["_gain_bad"].transform(lambda x: _shift_expand(x, min_p=3))
    df.drop(columns="_gain_bad", inplace=True)

    # --- track features ---
    df = df.sort_values(["driverId", "circuitId", "date"])
    dc = df.groupby(["driverId", "circuitId"])
    df["track_avg_pos"] = dc["positionOrder"].transform(lambda x: _shift_expand(x, min_p=2))
    df["track_avg_quali"] = dc["quali_position"].transform(lambda x: _shift_expand(x, min_p=2))
    df["track_races"] = dc.cumcount()

    df = df.sort_values(["constructorId", "circuitId", "date"])
    df["con_track_avg"] = (df.groupby(["constructorId", "circuitId"])["positionOrder"]
                            .transform(lambda x: _shift_expand(x, min_p=2)))
    _prior_races_only(df, ["constructorId", "circuitId"], ["con_track_avg"])
    df["track_affinity"] = df["con_track_avg"] - df["track_avg_pos"]

    df = df.sort_values(["circuitId", "date"])
    df["circuit_avg_dnf"] = df.groupby("circuitId")["dnf"].transform(
        lambda x: _shift_expand(x, min_p=10))
    df["circuit_avg_finish"] = df.groupby("circuitId")["positionOrder"].transform(
        lambda x: _shift_expand(x, min_p=10))
    _prior_races_only(df, ["circuitId"], ["circuit_avg_dnf", "circuit_avg_finish"])

    # --- pace features ---
    poles = (qualifying.groupby("raceId")["best_quali_sec"]
             .min().reset_index().rename(columns={"best_quali_sec": "pole_sec"}))
    df = df.merge(poles, on="raceId", how="left")
    df["delta_to_pole_sec"] = df["best_quali_sec"] - df["pole_sec"]
    df["delta_to_pole_pct"] = (df["delta_to_pole_sec"] / df["pole_sec"]) * 100

    df["q1_q2_delta"] = df["q1_sec"] - df["q2_sec"]
    df["q2_q3_delta"] = df["q2_sec"] - df["q3_sec"]
    df["progressed_to_q3"] = df["q3_sec"].notna().astype(int)

    df = df.sort_values(["driverId", "date"])
    g = df.groupby("driverId")
    df["driver_quali_pace_3"] = g["delta_to_pole_pct"].transform(lambda x: _shift_roll(x, 3, min_p=2))
    df["driver_quali_pace_5"] = g["delta_to_pole_pct"].transform(lambda x: _shift_roll(x, 5))

    # --- constructor features (вікна по ГОНКАХ команди, див. CON_FORM) ---
    ct = constructor_race_table(df)
    cg = ct.groupby("constructorId")
    for feat, (col, w, fn, min_p) in CON_FORM.items():
        ct[feat] = cg[col].transform(lambda x: _shift_roll(x, w, fn, min_p))
    df = df.merge(ct[["constructorId", "raceId", *CON_FORM]],
                  on=["constructorId", "raceId"], how="left", validate="many_to_one")

    # --- teammate comparison + team change ---
    # transform, а не merge по (raceId, driverId): у старих гонках пілот
    # буває двічі в одній гонці (спільні болiди), і merge розмножував би
    # ці рядки вже ПІСЛЯ підрахунку трасових агрегатів.
    team = df.groupby(["raceId", "constructorId"])
    df["quali_vs_team"] = df["quali_position"] - team["quali_position"].transform("mean")
    df["finish_vs_team"] = df["positionOrder"] - team["positionOrder"].transform("mean")

    df = df.sort_values(["driverId", "date"])
    g = df.groupby("driverId")
    df["teammate_quali_delta_5"] = g["quali_vs_team"].transform(lambda x: _shift_roll(x, 5))
    df["teammate_finish_delta_5"] = g["finish_vs_team"].transform(lambda x: _shift_roll(x, 5))

    df["prev_con"] = g["constructorId"].shift(1)
    df["team_changed"] = (df["constructorId"] != df["prev_con"]) & df["prev_con"].notna()
    # Скільки гонок ПОСПІЛЬ до цієї пілот провів у поточній команді:
    # номер рядка всередині "стінту" (відрізка між змінами команди).
    df["_stint"] = g["team_changed"].cumsum()
    df["races_with_team"] = df.groupby(["driverId", "_stint"]).cumcount()
    df["new_team_period"] = (df["races_with_team"] < 3).astype(int)
    df.drop(columns=["prev_con", "_stint"], inplace=True)

    # --- practice proxy (ранні кола гонки як proxy практики) ---
    if lap_times is not None and not lap_times.empty:
        lt = lap_times.copy()
        lt["milliseconds"] = pd.to_numeric(lt["milliseconds"], errors="coerce")
        lt["lap"] = pd.to_numeric(lt["lap"], errors="coerce")
        lt = lt.dropna(subset=["milliseconds", "lap"])
        early = lt[(lt["lap"] >= 2) & (lt["lap"] <= 6)].copy()

        def _trim_outliers(gr):
            if len(gr) < 3:
                return gr
            q10, q90 = gr["milliseconds"].quantile([0.1, 0.9])
            return gr[(gr["milliseconds"] >= q10) & (gr["milliseconds"] <= q90)]

        early = early.groupby(["raceId", "driverId"], group_keys=False).apply(_trim_outliers)

        fp = (early.groupby(["raceId", "driverId"])
              .agg(fp_median_ms=("milliseconds", "median"),
                   fp_pace_std=("milliseconds", "std"))
              .reset_index())
        leader = fp.groupby("raceId")["fp_median_ms"].min().reset_index().rename(
            columns={"fp_median_ms": "fp_leader_ms"})
        fp = fp.merge(leader, on="raceId")
        fp["fp_gap_to_leader_pct"] = ((fp["fp_median_ms"] - fp["fp_leader_ms"]) /
                                       fp["fp_leader_ms"]) * 100
        fp["fp_rank"] = fp.groupby("raceId")["fp_median_ms"].rank(method="first").astype(int)
        fp = fp[["raceId", "driverId", "fp_median_ms", "fp_pace_std",
                 "fp_gap_to_leader_pct", "fp_rank"]]
        df = df.merge(fp, on=["raceId", "driverId"], how="left")
    else:
        for c in ["fp_median_ms", "fp_pace_std", "fp_gap_to_leader_pct", "fp_rank"]:
            df[c] = np.nan

    # Сирі fp_* - це кола 2-6 САМОЇ гонки, тобто результат, якого до
    # старту не існує (NaN там майже завжди = "зійшов на перших колах",
    # і DNF-класифікатор вчився саме на цьому). У фічі йде лише форма за
    # ПОПЕРЕДНІ гонки (shift+rolling) - так само її рахує app/ml/live.py.
    # fp_median_ms у фічі не йде: абсолютний час кола між різними трасами
    # не порівнюваний, усереднювати його немає сенсу.
    df = df.sort_values(["driverId", "date"])
    g = df.groupby("driverId")
    df["early_pace_gap_5"] = g["fp_gap_to_leader_pct"].transform(lambda x: _shift_roll(x, 5))
    df["early_pace_rank_5"] = g["fp_rank"].transform(lambda x: _shift_roll(x, 5))
    df["early_pace_std_5"] = g["fp_pace_std"].transform(lambda x: _shift_roll(x, 5))

    df["grid_penalty"] = (df["grid"] - df["quali_position"]).fillna(0)

    # --- standings (shift: "чинний" залік станом ПЕРЕД цим етапом) ---
    ds = drv_stand.rename(columns={"points": "champ_points",
                                    "position": "champ_position",
                                    "wins": "champ_wins"})
    df = df.merge(ds[["raceId", "driverId", "champ_points", "champ_position", "champ_wins"]],
                  on=["raceId", "driverId"], how="left")
    df = df.sort_values(["driverId", "date"])
    for c in ["champ_points", "champ_position", "champ_wins"]:
        df[c] = df.groupby("driverId")[c].shift(1)

    # --- weekend: практики й погода (відбуваються ДО гонки - не leakage) ---
    if not practice.empty:
        wk = weekend_features(practice.dropna(subset=["driverId"]), weather)
        df = df.merge(wk, on=["raceId", "driverId"], how="left", validate="many_to_one")
    else:
        for c in WEEKEND_FEATURES:
            df[c] = np.nan
    add_pace_vs_grid(df)

    # --- categorical ---
    for c in ["driverId", "constructorId", "circuitId"]:
        df[c + "_cat"] = df[c].astype("category")

    features = [f for f in FEATURE_ORDER if f in df.columns]
    num_features = [f for f in features if f not in CATEGORICAL]

    df_model = df[df["year"] >= config.MIN_YEAR].copy()
    df_model = df_model.dropna(subset=["positionOrder", "quali_position", "grid"])
    df_model = df_model.sort_values(["date", "raceId"]).reset_index(drop=True)

    train = df_model[df_model["year"] <= config.TRAIN_END].copy()
    val = df_model[(df_model["year"] > config.TRAIN_END) & (df_model["year"] <= config.VAL_END)].copy()
    test = df_model[df_model["year"] >= config.TEST_START].copy()

    # NaN навмисно не імпутуються: LightGBM/XGBoost обробляють їх нативно,
    # а "NaN = фічу не порахувати" (новачок, мало історії) - сам по собі
    # сигнал, який медіана затирала. Так і df_model (історичні гонки в
    # app/analysis.py), і live-рядки йдуть у модель у тому самому вигляді.

    return Dataset(
        raw=df,
        df_model=df_model,
        train=train,
        val=val,
        test=test,
        features=features,
        num_features=num_features,
        categorical=CATEGORICAL,
        races=races,
        drivers=drivers,
        constructors=constructors,
        circuits=circuits,
        qualifying=qualifying,
        driver_standings=drv_stand,
        practice=practice,
        weather=weather,
        fingerprint=dataset_fp,
    )


_lock = threading.Lock()
_cached: Dataset | None = None


def get_dataset(force_rebuild: bool = False) -> Dataset:
    """Лінива побудова датасету: один раз за життя процесу, з диск-кешем
    (joblib) між рестартами процесу, доки не зміняться CSV у data/."""
    global _cached
    with _lock:
        if _cached is not None and not force_rebuild:
            return _cached

        current_fp = _fingerprint()
        if not force_rebuild and os.path.exists(CACHE_FILE):
            try:
                loaded: Dataset = joblib.load(CACHE_FILE)
                if loaded.fingerprint == current_fp:
                    _cached = loaded
                    return _cached
            except Exception:
                pass  # пошкоджений/несумісний кеш - перебудуємо

        dataset = _build_uncached()
        try:
            joblib.dump(dataset, CACHE_FILE)
        except Exception:
            pass  # кеш - оптимізація, не критично якщо диск недоступний
        _cached = dataset
        return _cached

"""
Агрегати практик і погоди з FastF1 - сирі дані для фіч вікенду.

Одна й та сама функція рахує агрегати і для офлайн-завантаження історії
(scripts/fetch_fastf1.py -> data/fastf1_*.csv, на чому вчиться модель),
і для поточного вікенду в live-прогнозі - щоб "практика" в тренуванні
й у проді рахувалась однаково (див. tests/test_ml_parity.py).

Практика відбувається ДО гонки, тож на відміну від старих fp_* (кола
2-6 самої гонки) тут немає leakage.

На пілота за сесію:
  best_lap_s          - найшвидше чисте коло (зазвичай квалі-симуляція);
  long_run_*          - довгі серії: стінти з >= LONG_RUN_MIN_LAPS чистих
                        кіл у темпі на одному комплекті - гоночний темп;
  long_run_deg_s      - деградація: нахил часу кола від віку шини (с/коло)
                        в межах довгих серій, середнє зважене по колах.
На сесію: погода (температури, частка часу з дощем).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LONG_RUN_MIN_LAPS = 5
# Коло, повільніше за найкраще коло пілота в сесії на > 10%, - охолоджувальне
# коло / трафік / помилка, а не гоночний темп (він зазвичай на 3-7% повільніший
# за квалі-симуляцію). Поріг від найкращого кола, а не від медіани стінту:
# у "рваних" стінтах (червоні прапори) медіана сама складається з повільних кіл.
LONG_RUN_OUTLIER = 1.10

PRACTICE_COLUMNS = [
    "driver_code", "driver_number", "full_name", "team_name",
    "best_lap_s", "valid_laps",
    "long_run_laps", "long_run_median_s", "long_run_compound", "long_run_deg_s",
]
WEATHER_COLUMNS = ["air_temp", "track_temp", "humidity", "rain_fraction"]


def _clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Чисті кола: є час, точний тайминг, не видалене, не in/out lap,
    повністю під зеленим прапором (TrackStatus == "1")."""
    l = laps.copy()
    l["lap_s"] = l["LapTime"].dt.total_seconds()
    ok = (
        l["lap_s"].notna()
        & l["IsAccurate"].fillna(False).astype(bool)
        & ~l["Deleted"].fillna(False).astype(bool)
        & l["PitInTime"].isna()
        & l["PitOutTime"].isna()
        & (l["TrackStatus"].astype(str) == "1")
    )
    return l[ok]


def _long_runs(clean: pd.DataFrame) -> pd.DataFrame:
    """Кола довгих серій: чисті кола в темпі (<= LONG_RUN_OUTLIER від
    найкращого кола пілота), стінти з >= LONG_RUN_MIN_LAPS таких кіл."""
    best = clean.groupby("Driver")["lap_s"].transform("min")
    paced = clean[clean["lap_s"] <= best * LONG_RUN_OUTLIER]
    parts = []
    for (_, _), stint in paced.groupby(["Driver", "Stint"]):
        if len(stint) >= LONG_RUN_MIN_LAPS:
            parts.append(stint)
    return pd.concat(parts) if parts else clean.iloc[0:0]


def _deg_slope(runs: pd.DataFrame) -> float:
    slopes, weights = [], []
    for _, stint in runs.groupby("Stint"):
        life = stint["TyreLife"].astype(float)
        if life.nunique() < 3:
            continue
        slopes.append(np.polyfit(life, stint["lap_s"], 1)[0])
        weights.append(len(stint))
    return float(np.average(slopes, weights=weights)) if slopes else np.nan


def practice_aggregates(laps: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """laps = session.laps, results = session.results (FastF1) ->
    рядок на пілота з колонками PRACTICE_COLUMNS."""
    clean = _clean_laps(laps)
    runs = _long_runs(clean)

    rows = []
    for r in results.itertuples():
        code = r.Abbreviation
        c = clean[clean["Driver"] == code]
        lr = runs[runs["Driver"] == code] if len(runs) else runs
        rows.append({
            "driver_code": code,
            "driver_number": str(r.DriverNumber),
            "full_name": r.FullName,
            "team_name": r.TeamName,
            "best_lap_s": float(c["lap_s"].min()) if len(c) else np.nan,
            "valid_laps": int(len(c)),
            "long_run_laps": int(len(lr)),
            "long_run_median_s": float(lr["lap_s"].median()) if len(lr) else np.nan,
            "long_run_compound": (str(lr["Compound"].mode().iloc[0])
                                  if len(lr) and lr["Compound"].notna().any() else None),
            "long_run_deg_s": _deg_slope(lr) if len(lr) else np.nan,
        })
    return pd.DataFrame(rows, columns=PRACTICE_COLUMNS)


def weather_aggregates(weather: pd.DataFrame | None) -> dict:
    if weather is None or len(weather) == 0:
        return {c: np.nan for c in WEATHER_COLUMNS}
    return {
        "air_temp": float(weather["AirTemp"].mean()),
        "track_temp": float(weather["TrackTemp"].mean()),
        "humidity": float(weather["Humidity"].mean()),
        "rain_fraction": float(weather["Rainfall"].astype(bool).mean()),
    }


def load_session_aggregates(year: int, round_number: int, session: str,
                            with_laps: bool) -> tuple[pd.DataFrame | None, dict, str, str]:
    """Завантажує сесію через FastF1 і повертає (practice_rows | None,
    weather, status, message). status: "ok" | "unavailable" (тайминг
    сесії не опубліковано). Спільне для scripts/fetch_fastf1.py і live.

    Кола, які FastF1 не зміг завантажити, - це RuntimeError, а не
    "unavailable": FastF1 ковтає мережеві помилки (у т.ч. обмеження частоти
    запитів з боку livetiming) і лише пише їх у лог, тож "нема кіл" майже
    завжди тимчасове і має повторюватись."""
    from app import fastf1_client  # тут, а не згори: модуль потрібен і без мережі/FastF1

    fsession = fastf1_client._load_session(year, round_number, session, laps=with_laps,
                                           telemetry=False, weather=True, messages=False)
    try:
        weather = weather_aggregates(fsession.weather_data)
    except Exception:
        weather = weather_aggregates(None)

    if not with_laps:
        return None, weather, "ok", ""
    try:
        laps = fsession.laps
    except Exception as e:  # DataNotLoadedError
        raise RuntimeError(f"кола не завантажились: {e}") from e
    if laps is None or len(laps) == 0:
        return None, weather, "unavailable", "сесія без кіл"
    agg = practice_aggregates(laps, fsession.results)
    return agg, weather, "ok", f"{int((agg['long_run_laps'] > 0).sum())} пілотів з довгими серіями"


# ---------------------------------------------------------------------------
# Фічі вікенду з агрегатів сесій
# ---------------------------------------------------------------------------

WEEKEND_FEATURES = [
    "pr_long_run_gap_pct", "pr_long_run_pos", "pr_pace_vs_grid",
    "pr_best_lap_gap_pct", "pr_deg_rel",
    "pr_long_run_laps", "pr_total_laps",
    "wx_fp_rain", "wx_q_rain", "wx_track_temp",
]


def weekend_features(practice: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """practice: рядки [raceId, driverId, session, PRACTICE_COLUMNS...],
    weather: рядки [raceId, session, WEATHER_COLUMNS...] ->
    рядок на (raceId, driverId) з WEEKEND_FEATURES, крім pr_pace_vs_grid
    (потребує quali_position - див. add_pace_vs_grid).

    Однакова для всієї історії (dataset.py) і для одного вікенду (live.py).
      pr_long_run_gap_pct - темп довгих серій відносно медіани сесії, %
                            (від'ємне = швидше), зважено по колах серій
                            між практиками вікенду;
      pr_long_run_pos     - місце за цим темпом серед пілотів з даними;
      pr_best_lap_gap_pct - найкраще коло проти найшвидшого в сесії, % (мін.
                            по сесіях);
      pr_deg_rel          - деградація відносно медіани сесії, с/коло;
      pr_*_laps           - скільки проїхав (проблеми на практиці = мало кіл);
      wx_*                - дощ у практиках/кваліфікації, температура траси.
    """
    key = ["raceId", "session"]
    p = practice.copy()
    p["lr_gap"] = (p["long_run_median_s"] / p.groupby(key)["long_run_median_s"].transform("median") - 1) * 100
    p["best_gap"] = (p["best_lap_s"] / p.groupby(key)["best_lap_s"].transform("min") - 1) * 100
    p["deg_rel"] = p["long_run_deg_s"] - p.groupby(key)["long_run_deg_s"].transform("median")

    w_gap = p["long_run_laps"].where(p["lr_gap"].notna(), 0)
    w_deg = p["long_run_laps"].where(p["deg_rel"].notna(), 0)
    p = p.assign(_w_gap=w_gap, _gap=p["lr_gap"].fillna(0) * w_gap,
                 _w_deg=w_deg, _deg=p["deg_rel"].fillna(0) * w_deg)
    g = p.groupby(["raceId", "driverId"])
    out = pd.DataFrame({
        "pr_long_run_gap_pct": g["_gap"].sum() / g["_w_gap"].sum().replace(0, np.nan),
        "pr_deg_rel": g["_deg"].sum() / g["_w_deg"].sum().replace(0, np.nan),
        "pr_best_lap_gap_pct": g["best_gap"].min(),
        "pr_long_run_laps": g["long_run_laps"].sum(),
        "pr_total_laps": g["valid_laps"].sum(),
    }).reset_index()
    out["pr_long_run_pos"] = out.groupby("raceId")["pr_long_run_gap_pct"].rank(method="min")

    wx = weather.copy()
    is_q = wx["session"] == "Q"
    fp = wx[~is_q].groupby("raceId").agg(wx_fp_rain=("rain_fraction", "max"),
                                         _fp_temp=("track_temp", "mean"))
    q = wx[is_q].groupby("raceId").agg(wx_q_rain=("rain_fraction", "max"),
                                       _q_temp=("track_temp", "mean"))
    wx = fp.join(q, how="outer")
    wx["wx_track_temp"] = wx["_q_temp"].fillna(wx["_fp_temp"])
    wx = wx[["wx_fp_rain", "wx_q_rain", "wx_track_temp"]].reset_index()

    return out.merge(wx, on="raceId", how="left")


def add_pace_vs_grid(df: pd.DataFrame) -> None:
    """pr_pace_vs_grid = позиція в кваліфікації - місце за темпом довгих
    серій (> 0: гоночний темп кращий за стартову позицію). Змінює df."""
    df["pr_pace_vs_grid"] = df["quali_position"] - df["pr_long_run_pos"]

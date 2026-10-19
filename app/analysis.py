"""
Аналіз прогнозу однієї гонки (v2.0 two-stage ансамбль).

Розраховується ОДИН РАЗ на етап - при першому запиті результат
пишеться в analysis_cache/{year}_{round}.json і на всі наступні запити
віддається звідти (без refresh=true перерахунок не робиться, навіть
якщо з моменту першого запиту з'явились нові дані - однакова поведінка
для одного й того самого etapу під час усього його вікенду).

Наповнення залежить від прогресу вікенду:
  - гонка вже відбулась (є в df_model АБО дата гонки в минулому) - беремо
    всі доступні дані ВКЛЮЧНО з кваліфікацією цього етапу, САМУ гонку як
    вхід моделі не використовуємо (щоб не було leakage: прогноз має бути
    таким, яким він був би ДО старту). Якщо результат вже відомий - він
    додається окремо, лише для порівняння "прогноз vs факт", не як фіча.
  - гонка ще не відбулась (поточний/майбутній етап) - беремо, що є:
    кваліфікацію, якщо вона вже проїхана, інакше лише форму
    пілотів/команд за попередніми етапами.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import fastf1_client
from app.ml import config, live
from app.ml.dataset import get_dataset
from app.ml.predict import predict_two_stage
from app.ml.train import get_model_bundle
from app.i18n import canonical, get_lang, translate_cached_uk, tr

MODEL_VERSION = "v2.3"  # змінюється разом з моделлю - інвалідує analysis_cache/

# Скорочені людські підписи фіч для feature_importance у відповіді
# (підмножина FEAT_LABELS з візуалізаційного аддону скрипта).
FEATURE_LABELS = {
    "quali_position": "Позиція в кваліфікації",
    "best_quali_sec": "Час кола в кваліфікації",
    "delta_to_pole_sec": "Відставання від поула (с)",
    "delta_to_pole_pct": "Відставання від поула (%)",
    "grid": "Стартова позиція",
    "grid_penalty": "Грід-пенальті",
    "driver_quali_pace_3": "Темп пілота в кваліфікації (3 етапи)",
    "driver_quali_pace_5": "Темп пілота в кваліфікації (5 етапів)",
    "con_quali_pace_3": "Темп команди в кваліфікації (3 етапи)",
    "con_quali_pace_5": "Темп команди в кваліфікації (5 етапів)",
    "driver_avg_pos_5": "Середній фініш пілота (5 етапів)",
    "driver_avg_pos_10": "Середній фініш пілота (10 етапів)",
    "driver_avg_quali_5": "Середня кваліфікація пілота (5 етапів)",
    "driver_pts_5": "Очки пілота (5 етапів)",
    "driver_pts_10": "Очки пілота (10 етапів)",
    "driver_consistency_5": "Стабільність пілота",
    "driver_dnf_rate_5": "Частка сходів пілота (5 етапів)",
    "driver_gain_5": "Середній приріст позицій пілота",
    "driver_overtakes_5": "Обгони пілота (5 етапів)",
    "driver_cum_avg_pos": "Середня позиція за кар'єру",
    "driver_career_races": "Етапів у кар'єрі",
    "is_rookie": "Новачок",
    "con_avg_pos_5": "Форма команди (5 етапів)",
    "con_avg_pos_10": "Форма команди (10 етапів)",
    "con_pts_5": "Очки команди (5 етапів)",
    "con_dnf_rate_5": "Частка сходів команди",
    "track_avg_pos": "Середній фініш пілота на трасі",
    "track_avg_quali": "Середня кваліфікація пілота на трасі",
    "track_affinity": "Афінність до траси",
    "circuit_avg_dnf": "Частка сходів на трасі",
    "circuit_avg_finish": "Середній фініш на трасі",
    "teammate_quali_delta_5": "Проти напарника (кваліфікація)",
    "teammate_finish_delta_5": "Проти напарника (фініш)",
    "new_team_period": "Новий період у команді",
    "early_pace_gap_5": "Темп на ранніх колах (відрив, 5 етапів)",
    "early_pace_rank_5": "Темп на ранніх колах (місце, 5 етапів)",
    "early_pace_std_5": "Стабільність темпу на ранніх колах",
    "wet_skill": "Майстерність у дощ",
    "comeback_score": "Здатність відігруватись",
    "pr_long_run_gap_pct": "Темп довгих серій на практиці",
    "pr_long_run_pos": "Місце за темпом довгих серій",
    "pr_pace_vs_grid": "Гоночний темп проти стартової позиції",
    "pr_best_lap_gap_pct": "Найкраще коло практик",
    "pr_deg_rel": "Деградація шин на практиці",
    "pr_long_run_laps": "Кіл у довгих серіях",
    "pr_total_laps": "Кіл на практиках",
    "wx_fp_rain": "Дощ на практиках",
    "wx_q_rain": "Дощ у кваліфікації",
    "wx_track_temp": "Температура траси",
    "champ_points": "Очки в заліку",
    "champ_position": "Позиція в заліку",
    "champ_wins": "Перемоги в заліку",
    "round": "Раунд сезону",
    "driver_age": "Вік пілота",
    "driverId_cat": "Пілот (ID)",
    "constructorId_cat": "Команда (ID)",
    "circuitId_cat": "Траса (ID)",
    # Решта фіч моделі (FEATURE_ORDER) і фічі практик старіших версій моделі,
    # що ще лежать у кеші прогнозів.
    "q1_q2_delta": "Прогрес Q1 → Q2 (с)",
    "q2_q3_delta": "Прогрес Q2 → Q3 (с)",
    "progressed_to_q3": "Пройшов у Q3",
    "driver_avg_quali_10": "Середня кваліфікація пілота (10 етапів)",
    "driver_quali_std_10": "Стабільність кваліфікацій пілота",
    "driver_gain_10": "Приріст позицій пілота (10 етапів)",
    "driver_lap1_gain_5": "Позиції на першому колі (5 етапів)",
    "driver_dnf_rate_10": "Частка сходів пілота (10 етапів)",
    "con_avg_quali_5": "Кваліфікація команди (5 етапів)",
    "track_races": "Гонок пілота на трасі",
    "fp_median_ms": "Медіанне коло практик",
    "fp_pace_std": "Розкид темпу на практиках",
    "fp_rank": "Місце за темпом практик",
    "fp_gap_to_leader_pct": "Відставання від лідера практик (%)",
}

# Англійські підписи тих самих фіч. Підпис у відповіді підставляється мовою
# запиту в момент видачі (_localize), а не при побудові - кеш прогнозів
# лишається одномовним (українським) і спільним для обох мов.
FEATURE_LABELS_EN = {
    "quali_position": "Qualifying position",
    "best_quali_sec": "Qualifying lap time",
    "delta_to_pole_sec": "Gap to pole (s)",
    "delta_to_pole_pct": "Gap to pole (%)",
    "grid": "Grid position",
    "grid_penalty": "Grid penalty",
    "driver_quali_pace_3": "Driver qualifying pace (3 rounds)",
    "driver_quali_pace_5": "Driver qualifying pace (5 rounds)",
    "con_quali_pace_3": "Team qualifying pace (3 rounds)",
    "con_quali_pace_5": "Team qualifying pace (5 rounds)",
    "driver_avg_pos_5": "Driver average finish (5 rounds)",
    "driver_avg_pos_10": "Driver average finish (10 rounds)",
    "driver_avg_quali_5": "Driver average qualifying (5 rounds)",
    "driver_pts_5": "Driver points (5 rounds)",
    "driver_pts_10": "Driver points (10 rounds)",
    "driver_consistency_5": "Driver consistency",
    "driver_dnf_rate_5": "Driver DNF rate (5 rounds)",
    "driver_gain_5": "Driver average places gained",
    "driver_overtakes_5": "Driver overtakes (5 rounds)",
    "driver_cum_avg_pos": "Career average position",
    "driver_career_races": "Career races",
    "is_rookie": "Rookie",
    "con_avg_pos_5": "Team form (5 rounds)",
    "con_avg_pos_10": "Team form (10 rounds)",
    "con_pts_5": "Team points (5 rounds)",
    "con_dnf_rate_5": "Team DNF rate",
    "track_avg_pos": "Driver average finish at the circuit",
    "track_avg_quali": "Driver average qualifying at the circuit",
    "track_affinity": "Circuit affinity",
    "circuit_avg_dnf": "Circuit DNF rate",
    "circuit_avg_finish": "Average finish at the circuit",
    "teammate_quali_delta_5": "Versus teammate (qualifying)",
    "teammate_finish_delta_5": "Versus teammate (finish)",
    "new_team_period": "New period with the team",
    "early_pace_gap_5": "Early-lap pace (gap, 5 rounds)",
    "early_pace_rank_5": "Early-lap pace (rank, 5 rounds)",
    "early_pace_std_5": "Early-lap pace consistency",
    "wet_skill": "Wet-weather skill",
    "comeback_score": "Comeback ability",
    "pr_long_run_gap_pct": "Practice long-run pace",
    "pr_long_run_pos": "Long-run pace rank",
    "pr_pace_vs_grid": "Race pace versus grid position",
    "pr_best_lap_gap_pct": "Best practice lap",
    "pr_deg_rel": "Practice tyre degradation",
    "pr_long_run_laps": "Long-run laps",
    "pr_total_laps": "Practice laps",
    "wx_fp_rain": "Rain in practice",
    "wx_q_rain": "Rain in qualifying",
    "wx_track_temp": "Track temperature",
    "champ_points": "Championship points",
    "champ_position": "Championship position",
    "champ_wins": "Championship wins",
    "round": "Season round",
    "driver_age": "Driver age",
    "driverId_cat": "Driver (ID)",
    "constructorId_cat": "Team (ID)",
    "circuitId_cat": "Circuit (ID)",
    "q1_q2_delta": "Q1 → Q2 improvement (s)",
    "q2_q3_delta": "Q2 → Q3 improvement (s)",
    "progressed_to_q3": "Reached Q3",
    "driver_avg_quali_10": "Driver average qualifying (10 rounds)",
    "driver_quali_std_10": "Driver qualifying consistency",
    "driver_gain_10": "Driver places gained (10 rounds)",
    "driver_lap1_gain_5": "First-lap places gained (5 rounds)",
    "driver_dnf_rate_10": "Driver DNF rate (10 rounds)",
    "con_avg_quali_5": "Team qualifying (5 rounds)",
    "track_races": "Driver races at the circuit",
    "fp_median_ms": "Median practice lap",
    "fp_pace_std": "Practice pace spread",
    "fp_rank": "Practice pace rank",
    "fp_gap_to_leader_pct": "Gap to practice leader (%)",
}


def _localize(result: dict) -> dict:
    """Прогноз мовою запиту: підписи фіч - за кодом фічі (для обох мов: у
    старішому кеші частина фіч лишилась без підпису), примітки - за
    шаблонами (у т.ч. в кеші, побудованому раніше). Оригінал не змінюється."""
    labels = FEATURE_LABELS_EN if get_lang() == "en" else FEATURE_LABELS
    out = dict(result)
    fi = result.get("feature_importance") or {}
    out["feature_importance"] = {
        model: [{**item, "label": labels.get(item.get("feature"), item.get("label"))} for item in items]
        for model, items in fi.items()
    }
    if get_lang() != "uk" and isinstance(result.get("data"), dict) and result["data"].get("notes"):
        out["data"] = {**result["data"], "notes": [translate_cached_uk(n) for n in result["data"]["notes"]]}
    return out


def _cache_path(year: int, round_number: int) -> str:
    return os.path.join(config.ANALYSIS_CACHE_DIR, f"{year}_{round_number}.json")


def _native(value):
    """numpy/pandas скаляр -> звичайний Python-тип, придатний для json.dump."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if np.isnan(v) else round(v, 4)
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    return value


def _feature_importance(model, top_n: int) -> list[dict]:
    fi = pd.DataFrame({
        "feature": model.feature_name_,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)
    return [
        {
            "feature": r.feature,
            "label": FEATURE_LABELS.get(r.feature, r.feature),
            "importance": _native(r.importance),
        }
        for r in fi.itertuples()
    ]


def _determine_stage(race_date: pd.Timestamp, quali_date) -> str:
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if pd.notna(race_date) and today >= pd.Timestamp(race_date).normalize():
        return "completed"
    if pd.notna(quali_date) and today >= pd.Timestamp(quali_date).normalize():
        return "post_quali"
    return "not_started"


def _fetch_actual_results(year: int, round_number: int) -> dict[str, dict]:
    """code (Abbreviation) -> {position, status} з живих даних FastF1.
    Тільки для показу "прогноз vs факт" - НІКОЛИ не йде у фічі моделі."""
    try:
        results = fastf1_client.get_race_results(year, round_number)
    except Exception:
        return {}
    out = {}
    for r in results:
        code = r.get("Abbreviation")
        if not code:
            continue
        out[code.upper()] = {
            "position": r.get("Position"),
            "status": r.get("Status"),
        }
    return out


def _compute(year: int, round_number: int) -> dict:
    dataset = get_dataset()
    bundle = get_model_bundle(dataset)

    race_row = dataset.races[(dataset.races["year"] == year) & (dataset.races["round"] == round_number)]
    if race_row.empty:
        raise ValueError(tr("round_not_in_calendar", round=round_number, year=year))
    race_row = race_row.iloc[0]
    race_id = int(race_row["raceId"])
    race_date = race_row["date"]
    quali_date = race_row.get("quali_date")
    quali_date = pd.to_datetime(quali_date, errors="coerce") if quali_date is not None else pd.NaT

    stage = _determine_stage(race_date, quali_date)

    static_rows = dataset.df_model[dataset.df_model["raceId"] == race_id]
    notes: list[str] = []

    if not static_rows.empty and len(static_rows) >= 10:
        rows = static_rows
        source = "static_dataset"
        quali_available = True
    else:
        rows, quali_available, live_notes = live.build_feature_rows(dataset, year, round_number)
        source = "live_fastf1_quali" if quali_available else "live_no_quali"
        notes.extend(live_notes)
    display = rows[["driverId", "grid", "quali_position"]].copy()

    pred = predict_two_stage(dataset, bundle, rows, alpha=bundle.best_alpha)
    pred = pred.merge(display, on="driverId", how="left")
    pred = pred.sort_values("pred_rank")

    # Факт для порівняння - тільки коли гонка вже відбулась.
    actual_by_code: dict[str, dict] = {}
    if stage == "completed" and source != "static_dataset":
        actual_by_code = _fetch_actual_results(year, round_number)
        if actual_by_code:
            notes.append(canonical("note_actual_added"))

    code_by_driver = dict(zip(dataset.drivers["driverId"], dataset.drivers["code"]))

    predictions = []
    for r in pred.itertuples():
        actual_position = None
        if source == "static_dataset" and pd.notna(r.positionOrder):
            actual_position = _native(r.positionOrder)
        elif actual_by_code:
            code = str(code_by_driver.get(r.driverId, "")).upper()
            info = actual_by_code.get(code)
            if info and info.get("position") is not None:
                actual_position = _native(info["position"])

        hit = None
        if actual_position is not None:
            hit = abs(actual_position - r.pred_rank) <= 2

        predictions.append({
            "driver_id": _native(r.driverId),
            "driver": r.driver_name,
            "team": r.constructor_name,
            "grid": _native(r.grid),
            "quali_position": _native(r.quali_position),
            "predicted_position": _native(r.pred_rank),
            "expected_position": _native(r.expected_pos),
            "p_dnf": _native(r.p_dnf),
            "actual_position": actual_position,
            "hit": hit,
        })

    result = {
        "race": {
            "year": year,
            "round": int(round_number),
            "race_id": race_id,
            "name": str(race_row["name"]),
            "date": str(race_row["date"].date()) if pd.notna(race_row["date"]) else None,
            "circuit_id": _native(race_row["circuitId"]),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "stage": stage,
            "source": source,
            "quali_available": quali_available,
            "notes": notes,
        },
        "model": {
            "version": MODEL_VERSION,
            "best_alpha": _native(bundle.best_alpha),
            "use_dnf": bool(bundle.use_dnf),
            "metrics": {k: _native(v) for k, v in bundle.metrics.items()},
        },
        "predictions": predictions,
        "feature_importance": {
            "ranker": _feature_importance(bundle.ranker, 15),
            "dnf_classifier": _feature_importance(bundle.dnf_clf, 10),
        },
    }
    return result


def analysis_status(year: int, round_number: int) -> dict:
    """Чи порахований прогноз етапу для поточної MODEL_VERSION - лише
    читає analysis_cache/, нічого не тренує і не рахує (миттєво). Для
    «Огляду»: є прогноз - показати його проти факту, нема - кнопку
    «Порахувати» (GET /api/analysis/...).

    hits - скільки з трьох пілотів, яких модель поставила на подіум,
    реально фінішували в топ-3 (None, поки результату гонки немає)."""
    path = _cache_path(year, round_number)
    cached = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
    if not cached or cached.get("model", {}).get("version") != MODEL_VERSION:
        return {"ready": False, "generated_at": None, "top3": [], "hits": None}

    preds = sorted(cached["predictions"], key=lambda p: p["predicted_position"])
    top3 = [{k: p.get(k) for k in ("driver", "team", "predicted_position", "actual_position", "p_dnf")}
            for p in preds[:3]]
    has_actual = any(p.get("actual_position") is not None for p in preds)
    hits = (sum(1 for p in top3 if p["actual_position"] is not None and p["actual_position"] <= 3)
            if has_actual else None)
    return {"ready": True, "generated_at": cached.get("generated_at"), "top3": top3, "hits": hits}


def analyze_race(year: int, round_number: int, refresh: bool = False) -> dict:
    """Аналіз прогнозу гонки (year, round_number). Кешується на диск -
    перший виклик рахує й зберігає, наступні віддають той самий
    результат, поки не передати refresh=True або не зміниться
    MODEL_VERSION (прогноз іншої моделі - вже не "той самий" результат)."""
    path = _cache_path(year, round_number)
    if not refresh and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("model", {}).get("version") == MODEL_VERSION:
            return _localize(cached)

    result = _compute(year, round_number)
    os.makedirs(config.ANALYSIS_CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return _localize(result)

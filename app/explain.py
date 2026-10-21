"""
Пояснення прогнозу для кожного пілота: які фічі і наскільки посунули його
прогнозовану позицію відносно середнього пілота в полі цього етапу.

Внески рахують самі дерева (SHAP, вбудований у LightGBM/XGBoost:
pred_contrib / pred_contribs) - це точний розклад прогнозу моделі, а не
окрема "пояснювальна" модель, яка могла б пояснювати щось інше.

Позиція у прогнозі - суміш двох моделей (див. _predict_two_stage_internal):
    finisher_pos = alpha * pseudo_pos(ranker) + (1 - alpha) * xgb_pos
XGBoost прогнозує позицію напряму, тож його внески вже в позиціях.
LambdaRank видає бал (більший = краще), а позицію дає ранг цього балу -
переводимо бал у позиції нахилом лінійної залежності "ранг ~ бал" у цьому
полі. Тому сума внесків пілота приблизно (не до сотих) дорівнює тому,
наскільки його прогноз кращий за середній у полі.

impact > 0 - фіча тягне пілота вгору (до P1), < 0 - вниз.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

# Скільки фіч показувати поіменно - решта йде одним рядком "інші фактори".
TOP_FACTORS = 8

# Групи фіч для зведення "за рахунок чого": кваліфікація, форма пілота тощо.
GROUP_ORDER = ["quali", "driver", "team", "track", "practice", "weather", "championship"]
_GROUP_EXACT = {
    "grid": "quali",
    "grid_penalty": "quali",
    "progressed_to_q3": "quali",
    "best_quali_sec": "quali",
    "is_rookie": "driver",
    "wet_skill": "driver",
    "comeback_score": "driver",
    "driverId_cat": "driver",
    "new_team_period": "team",
    "constructorId_cat": "team",
    "circuitId_cat": "track",
    "round": "championship",
}
_GROUP_PREFIX = [
    ("quali_", "quali"),
    ("delta_to_pole", "quali"),
    ("q1_", "quali"),
    ("q2_", "quali"),
    ("driver_quali_pace", "driver"),
    ("driver_", "driver"),
    ("teammate_", "driver"),
    ("early_pace", "driver"),
    ("con_", "team"),
    ("track_", "track"),
    ("circuit_", "track"),
    ("pr_", "practice"),
    ("fp_", "practice"),
    ("wx_", "weather"),
    ("champ_", "championship"),
]


def feature_group(feature: str) -> str:
    if feature in _GROUP_EXACT:
        return _GROUP_EXACT[feature]
    for prefix, group in _GROUP_PREFIX:
        if feature.startswith(prefix):
            return group
    return "driver"


def _value(v):
    """Значення фічі для відповіді: число (округлене) або None, якщо даних
    немає (NaN - напр. кваліфікації ще не було)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else round(f, 4)


def position_impacts(dataset, bundle, rows: pd.DataFrame, alpha: float) -> np.ndarray:
    """Матриця (пілоти x фічі dataset.features) внесків у позиції відносно
    середнього пілота поля. Рядки - у тому ж порядку, що й rows."""
    features = dataset.features
    Xn = rows[features]
    ranker_contrib = bundle.ranker.predict(Xn, pred_contrib=True)[:, :-1]

    Xx = rows[dataset.num_features].copy()
    for c in dataset.categorical:
        Xx[c] = rows[c].cat.codes
    xgb_contrib = bundle.xgb_reg.get_booster().predict(xgb.DMatrix(Xx), pred_contribs=True)[:, :-1]
    xgb_cols = list(Xx.columns)
    xgb_contrib = xgb_contrib[:, [xgb_cols.index(f) for f in features]]

    # Бал ранкера -> позиції: нахил "ранг ~ бал" у цьому полі.
    score = ranker_contrib.sum(axis=1) + 0.0
    if len(rows) > 2 and np.ptp(score) > 0:
        ranks = pd.Series(score).rank(ascending=False, method="first").to_numpy()
        slope = -np.polyfit(score, ranks, 1)[0]
    else:
        slope = 0.0

    impact = alpha * slope * ranker_contrib - (1 - alpha) * xgb_contrib
    # Відносно середнього в полі: базове значення моделі однакове для всіх
    # і нічого не пояснює, а "краще/гірше за суперників" - пояснює.
    return impact - impact.mean(axis=0, keepdims=True)


def explain_rows(dataset, bundle, rows: pd.DataFrame, alpha: float) -> dict:
    """{driverId: explanation} для кожного рядка rows."""
    impacts = position_impacts(dataset, bundle, rows, alpha)
    features = dataset.features
    groups = [feature_group(f) for f in features]

    out = {}
    for i, (_, row) in enumerate(rows.iterrows()):
        imp = impacts[i]
        order = np.argsort(-np.abs(imp))
        top = order[:TOP_FACTORS]
        factors = [
            {
                "feature": features[j],
                "group": groups[j],
                # ID-фічі (пілот/команда/траса) - категорія, а не число.
                "value": None if features[j].endswith("_cat") else _value(row[features[j]]),
                "impact": round(float(imp[j]), 2),
            }
            for j in top
        ]
        by_group = {}
        for j, g in enumerate(groups):
            by_group[g] = by_group.get(g, 0.0) + float(imp[j])
        out[int(row["driverId"])] = {
            "total": round(float(imp.sum()), 2),
            "factors": factors,
            "other": round(float(imp[order[TOP_FACTORS:]].sum()), 2),
            "groups": [{"group": g, "impact": round(by_group[g], 2)} for g in GROUP_ORDER if g in by_group],
        }
    return out

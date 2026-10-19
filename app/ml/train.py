"""
Тренування трьох моделей v2.0 (DNF-класифікатор, LambdaRank ranker,
XGBoost regressor) + підбір ваги ансамблю alpha. Перенесено з CELL 18-24
f1_pipeline_v2.py, з відмінностями (MODEL_VERSION 2):
  - DNF-класифікатор без class_weight="balanced" і з early stopping по
    logloss - p_dnf має бути каліброваною ймовірністю, а не скором;
  - чи домішувати p_dnf у expected_pos (use_dnf) підбирається на val разом
    з alpha. Без leakage DNF погано передбачуваний (AUC ~0.55), і
    домішування "(1-p)*pos + p*20" лише зсуває всіх до хвоста;
  - NaN не імпутуються - LightGBM/XGBoost обробляють їх нативно.

Тренування ~15-20с - тому результат (ModelBundle) кешується на диск
(joblib) і в пам'яті процесу, прив'язаний до fingerprint датасету, на
якому його навчено (див. get_model_bundle())."""
from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import (brier_score_loss, mean_absolute_error, mean_squared_error,
                             ndcg_score, roc_auc_score)

from app.ml import config
from app.ml.dataset import Dataset

warnings.filterwarnings("ignore")

CACHE_FILE = os.path.join(config.MODELS_DIR, "bundle.joblib")
ALPHA_GRID = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]

# Версія логіки тренування - кеш бандла інвалідується при її зміні так
# само, як при зміні датасету.
MODEL_VERSION = 2


@dataclass
class ModelBundle:
    dnf_clf: lgb.LGBMClassifier
    ranker: lgb.LGBMRanker
    xgb_reg: xgb.XGBRegressor
    best_alpha: float
    use_dnf: bool = False
    metrics: dict = field(default_factory=dict)
    dataset_fingerprint: str = ""
    model_version: int = 0


def _top_n_metrics(pred_df) -> dict:
    actual = pred_df["positionOrder"]
    pred = pred_df["pred_rank"]
    return {
        "top1": float(((actual == 1) & (pred == 1)).sum() / max((actual == 1).sum(), 1)),
        "podium_set": float(pred_df.groupby("raceId").apply(
            lambda x: len(set(x[x["positionOrder"] <= 3]["driverId"]) &
                          set(x[x["pred_rank"] <= 3]["driverId"])) / 3
        ).mean()),
        "top5_set": float(pred_df.groupby("raceId").apply(
            lambda x: len(set(x[x["positionOrder"] <= 5]["driverId"]) &
                          set(x[x["pred_rank"] <= 5]["driverId"])) / 5
        ).mean()),
        "top10_set": float(pred_df.groupby("raceId").apply(
            lambda x: len(set(x[x["positionOrder"] <= 10]["driverId"]) &
                          set(x[x["pred_rank"] <= 10]["driverId"])) / 10
        ).mean()),
    }


def _per_race_spearman(pred_df) -> tuple[float, float]:
    rho_list = []
    for _, gr in pred_df.groupby("raceId"):
        if len(gr) < 5:
            continue
        rho, _ = spearmanr(gr["positionOrder"], gr["pred_rank"])
        if not np.isnan(rho):
            rho_list.append(rho)
    if not rho_list:
        return float("nan"), float("nan")
    return float(np.mean(rho_list)), float(np.median(rho_list))


def _per_race_ndcg(pred_df, k=10) -> float:
    ndcg_list = []
    for _, gr in pred_df.groupby("raceId"):
        if len(gr) < 10:
            continue
        rel_true = (21 - gr["positionOrder"]).clip(lower=0, upper=20).values
        rel_pred = -gr["pred_rank"].values
        try:
            ndcg_list.append(ndcg_score([rel_true], [rel_pred], k=k))
        except Exception:
            pass
    return float(np.mean(ndcg_list)) if ndcg_list else float("nan")


def _predict_two_stage_internal(dataset: Dataset, dnf_clf, ranker, xgb_reg, df_in,
                                alpha: float, use_dnf: bool):
    """Ядро app.ml.predict.predict_two_stage() без залежності від готового
    ModelBundle - потрібно тут для підбору alpha/use_dnf й оцінки метрик
    під час самого тренування."""
    features, num_features, categorical = dataset.features, dataset.num_features, dataset.categorical
    Xn = df_in[features]
    Xx = df_in[num_features].copy()
    for c in categorical:
        Xx[c] = df_in[c].cat.codes

    p_dnf = dnf_clf.predict_proba(Xn)[:, 1]
    rank_score = ranker.predict(Xn)
    df_tmp = df_in[["raceId"]].copy()
    df_tmp["rs"] = rank_score
    df_tmp["pseudo_pos"] = df_tmp.groupby("raceId")["rs"].rank(ascending=False, method="first")
    xgb_pos = xgb_reg.predict(Xx)

    finisher_pos = alpha * df_tmp["pseudo_pos"].values + (1 - alpha) * xgb_pos
    expected_pos = (1 - p_dnf) * finisher_pos + p_dnf * 20.0 if use_dnf else finisher_pos

    out = df_in[["raceId", "driverId", "driver_name", "constructor_name",
                  "name", "year", "positionOrder", "dnf"]].copy()
    out["p_dnf"] = p_dnf
    out["finisher_pos"] = finisher_pos
    out["expected_pos"] = expected_pos
    out["pred_rank"] = out.groupby("raceId")["expected_pos"].rank(method="first").astype(int)
    return out


def fit_bundle(dataset: Dataset, train, val) -> ModelBundle:
    """Тренує 3 моделі на train, early stopping і підбір alpha/use_dnf - на
    val. Test тут не використовується взагалі (див. evaluate_bundle), тому
    ту саму функцію використовує walk-forward оцінка (app/ml/evaluate.py)."""
    features, num_features, categorical = dataset.features, dataset.num_features, dataset.categorical

    # ── Stage 1: DNF classifier ──
    X_tr, y_tr_dnf = train[features], train["dnf"]
    X_va, y_va_dnf = val[features], val["dnf"]

    dnf_clf = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbose=-1,
    )
    dnf_clf.fit(
        X_tr, y_tr_dnf,
        eval_set=[(X_va, y_va_dnf)],
        eval_metric="binary_logloss",
        categorical_feature=categorical,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    auc_val = float(roc_auc_score(y_va_dnf, dnf_clf.predict_proba(X_va)[:, 1]))

    # ── Stage 2: Position ranker (LambdaRank), тільки фінішери ──
    train_fin = train[train["dnf"] == 0].copy()
    val_fin = val[val["dnf"] == 0].copy()

    train_fin["rel"] = (21 - train_fin["positionOrder"]).clip(lower=0, upper=config.MAX_REL).astype(int)
    val_fin["rel"] = (21 - val_fin["positionOrder"]).clip(lower=0, upper=config.MAX_REL).astype(int)
    train_fin = train_fin.sort_values(["raceId"])
    val_fin = val_fin.sort_values(["raceId"])

    X_tr_r, y_tr_r = train_fin[features], train_fin["rel"]
    X_va_r, y_va_r = val_fin[features], val_fin["rel"]
    group_tr = train_fin.groupby("raceId").size().values
    group_va = val_fin.groupby("raceId").size().values

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=1500, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbose=-1,
        label_gain=[i for i in range(config.MAX_REL + 1)],
    )
    ranker.fit(
        X_tr_r, y_tr_r, group=group_tr,
        eval_set=[(X_va_r, y_va_r)], eval_group=[group_va], eval_at=[3, 5, 10],
        categorical_feature=categorical,
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )

    # ── Stage 3: XGBoost regressor (для ансамблю) ──
    X_tr_x = train_fin[num_features].copy()
    for c in categorical:
        X_tr_x[c] = train_fin[c].cat.codes
    X_va_x = val_fin[num_features].copy()
    for c in categorical:
        X_va_x[c] = val_fin[c].cat.codes
    y_tr_pos = train_fin["positionOrder"]
    y_va_pos = val_fin["positionOrder"]

    xgb_reg = xgb.XGBRegressor(
        n_estimators=1500, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbosity=0, eval_metric="mae",
        early_stopping_rounds=80,
    )
    xgb_reg.fit(X_tr_x, y_tr_pos, eval_set=[(X_va_x, y_va_pos)], verbose=False)

    # ── Підбір alpha і use_dnf на val ──
    best_alpha, best_use_dnf, best_mae_val = 0.5, False, float("inf")
    for use_dnf in (False, True):
        for a in ALPHA_GRID:
            p = _predict_two_stage_internal(dataset, dnf_clf, ranker, xgb_reg, val, a, use_dnf)
            m = mean_absolute_error(p["positionOrder"], p["expected_pos"])
            if m < best_mae_val:
                best_mae_val, best_alpha, best_use_dnf = m, a, use_dnf

    return ModelBundle(
        dnf_clf=dnf_clf, ranker=ranker, xgb_reg=xgb_reg,
        best_alpha=best_alpha, use_dnf=best_use_dnf,
        metrics={"dnf_auc_val": auc_val, "best_alpha": best_alpha, "use_dnf": best_use_dnf,
                 "best_alpha_val_mae": float(best_mae_val)},
        dataset_fingerprint=dataset.fingerprint, model_version=MODEL_VERSION,
    )


def evaluate_bundle(dataset: Dataset, bundle: ModelBundle, test) -> dict:
    """Метрики моделі й baseline "фініш = кваліфікація" на test."""
    pred_test_final = _predict_two_stage_internal(dataset, bundle.dnf_clf, bundle.ranker,
                                                  bundle.xgb_reg, test, bundle.best_alpha,
                                                  bundle.use_dnf)
    y_te_dnf, p_dnf_test = test["dnf"], pred_test_final["p_dnf"]
    mae_final = float(mean_absolute_error(pred_test_final["positionOrder"], pred_test_final["expected_pos"]))
    rmse_final = float(np.sqrt(mean_squared_error(pred_test_final["positionOrder"], pred_test_final["expected_pos"])))
    tm_final = _top_n_metrics(pred_test_final)
    rho_mean, rho_med = _per_race_spearman(pred_test_final)
    ndcg10 = _per_race_ndcg(pred_test_final, k=10)
    ndcg5 = _per_race_ndcg(pred_test_final, k=5)

    # Baseline "фініш = позиція в кваліфікації" - модель має бути помітно
    # кращою за нього, інакше вона нічого не додає до стартової решітки.
    baseline = test[["raceId", "driverId", "positionOrder", "quali_position"]].copy()
    baseline["pred_rank"] = baseline.groupby("raceId")["quali_position"].rank(method="first")
    baseline_rho, _ = _per_race_spearman(baseline)
    baseline_tm = _top_n_metrics(baseline)

    return {
        "mae_test": mae_final,
        "rmse_test": rmse_final,
        "top1": tm_final["top1"],
        "podium_set": tm_final["podium_set"],
        "top5_set": tm_final["top5_set"],
        "top10_set": tm_final["top10_set"],
        "spearman_mean": rho_mean,
        "spearman_median": rho_med,
        "ndcg5": ndcg5,
        "ndcg10": ndcg10,
        "dnf_auc_test": float(roc_auc_score(y_te_dnf, p_dnf_test)),
        "dnf_brier_test": float(brier_score_loss(y_te_dnf, p_dnf_test)),
        "dnf_rate_test": float(y_te_dnf.mean()),
        "p_dnf_mean_test": float(p_dnf_test.mean()),
        "baseline_quali_mae_test": float(mean_absolute_error(test["positionOrder"], test["quali_position"])),
        "baseline_quali_spearman_mean": baseline_rho,
        "baseline_quali_top1": baseline_tm["top1"],
        "baseline_quali_podium_set": baseline_tm["podium_set"],
    }


def train_bundle(dataset: Dataset) -> ModelBundle:
    bundle = fit_bundle(dataset, dataset.train, dataset.val)
    bundle.metrics.update(evaluate_bundle(dataset, bundle, dataset.test))
    bundle.metrics.update({
        "train_size": int(len(dataset.train)),
        "val_size": int(len(dataset.val)),
        "test_size": int(len(dataset.test)),
    })
    return bundle


_lock = threading.Lock()
_cached: ModelBundle | None = None


def get_model_bundle(dataset: Dataset, force_retrain: bool = False) -> ModelBundle:
    """Лінива побудова моделей: тренуємо один раз за життя процесу, з
    диск-кешем між рестартами, доки датасет (fingerprint) не зміниться."""
    global _cached
    with _lock:
        def _fresh(b: ModelBundle) -> bool:
            return (b.dataset_fingerprint == dataset.fingerprint
                    and getattr(b, "model_version", 0) == MODEL_VERSION)

        if _cached is not None and not force_retrain and _fresh(_cached):
            return _cached

        if not force_retrain and os.path.exists(CACHE_FILE):
            try:
                loaded: ModelBundle = joblib.load(CACHE_FILE)
                if _fresh(loaded):
                    _cached = loaded
                    return _cached
            except Exception:
                pass

        bundle = train_bundle(dataset)
        try:
            joblib.dump(bundle, CACHE_FILE)
        except Exception:
            pass
        _cached = bundle
        return _cached

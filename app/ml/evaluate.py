"""
Надійна офлайн-оцінка моделі: walk-forward по сезонах + bootstrap
довірчі інтервали, завжди в парі з baseline "фініш = кваліфікація".

Один фіксований test (2024+) - це ~60 гонок, і різниця в 0.01-0.02 між
варіантами тоне в шумі. Тут для кожного сезону Y модель вчиться на
роках <= Y-2, early stopping/alpha - на Y-1, оцінка - на Y; метрики
рахуються по кожній гонці окремо, а bootstrap по гонках дає інтервал
для різниці "модель - baseline" (парний: ті самі гонки в обох).

Запуск: python scripts/evaluate.py (див. там параметри).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.ml import config
from app.ml.dataset import Dataset
from app.ml.train import _predict_two_stage_internal, fit_bundle

# Метрики на гонку. Для mae менше = краще, для решти більше = краще.
METRICS = ["mae", "spearman", "podium", "top1"]
LOWER_IS_BETTER = {"mae"}


def per_race_metrics(df: pd.DataFrame, pred_pos: str) -> pd.DataFrame:
    """df: рядки (raceId, driverId, positionOrder, <pred_pos>) ->
    один рядок метрик на гонку."""
    rows = []
    for race_id, g in df.groupby("raceId"):
        rank = g[pred_pos].rank(method="first")
        actual = g["positionOrder"]
        rho = spearmanr(actual, rank)[0] if len(g) >= 5 else np.nan
        rows.append({
            "raceId": race_id,
            "year": int(g["year"].iloc[0]),
            "mae": float((actual - g[pred_pos]).abs().mean()),
            "spearman": rho,
            "podium": len(set(g.loc[actual <= 3, "driverId"]) & set(g.loc[rank <= 3, "driverId"])) / 3,
            "top1": float(((actual == 1) & (rank == 1)).any()),
        })
    return pd.DataFrame(rows)


@dataclass
class WalkForwardResult:
    races: pd.DataFrame  # на гонку: <metric>_model, <metric>_base, year, alpha, use_dnf

    def summary(self, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
        """Середнє по всіх гонках + 95% bootstrap CI для моделі, baseline
        і їхньої різниці (model - base)."""
        return paired_summary(self.races, "model", "base", ("model", "baseline"), n_boot, seed)

    def by_year(self) -> pd.DataFrame:
        cols = [f"{m}_{w}" for m in ("mae", "spearman") for w in ("model", "base")]
        g = self.races.groupby("year")
        return g[cols].mean().join(g.size().rename("races")).join(g[["alpha", "use_dnf"]].first())


def paired_summary(races: pd.DataFrame, a: str, b: str, names: tuple[str, str],
                   n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Парний bootstrap по гонках: колонки <metric>_<a> і <metric>_<b> ->
    середні, різниця a - b, її 95% CI і вердикт для a."""
    rng = np.random.default_rng(seed)
    n = len(races)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = []
    for m in METRICS:
        va = races[f"{m}_{a}"].to_numpy(dtype=float)
        vb = races[f"{m}_{b}"].to_numpy(dtype=float)
        diff = va - vb
        boot = np.nanmean(diff[idx], axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        better = (hi < 0) if m in LOWER_IS_BETTER else (lo > 0)
        worse = (lo > 0) if m in LOWER_IS_BETTER else (hi < 0)
        out.append({
            "metric": m,
            names[0]: np.nanmean(va),
            names[1]: np.nanmean(vb),
            "diff": np.nanmean(diff),
            "diff_ci_low": lo,
            "diff_ci_high": hi,
            "verdict": "краще" if better else ("гірше" if worse else "нерозрізнимо"),
        })
    return pd.DataFrame(out)


def without_features(dataset: Dataset, drop_prefixes: tuple[str, ...]) -> Dataset:
    """Копія датасету без фіч з такими префіксами - для A/B (ablation)."""
    keep = [f for f in dataset.features if not f.startswith(drop_prefixes)]
    return dataclasses.replace(dataset, features=keep,
                               num_features=[f for f in keep if f not in dataset.categorical])


def compare(with_: WalkForwardResult, without: WalkForwardResult, n_boot: int = 2000) -> pd.DataFrame:
    """Модель A проти моделі B на тих самих гонках (парно)."""
    cols = ["raceId"] + [f"{m}_model" for m in METRICS]
    races = with_.races[cols].merge(without.races[cols], on="raceId", suffixes=("_a", "_b"))
    races = races.rename(columns={f"{m}_model_{s}": f"{m}_{s}" for m in METRICS for s in ("a", "b")})
    return paired_summary(races, "a", "b", ("з фічами", "без фіч"), n_boot)


def walk_forward(dataset: Dataset, test_years: list[int] | None = None,
                 log=print) -> WalkForwardResult:
    dm = dataset.df_model
    if test_years is None:
        # перші два сезони після MIN_YEAR потрібні як мінімальні train+val
        test_years = list(range(config.MIN_YEAR + 6, int(dm["year"].max()) + 1))

    parts = []
    for year in test_years:
        train = dm[dm["year"] <= year - 2]
        val = dm[dm["year"] == year - 1]
        test = dm[dm["year"] == year]
        if train.empty or val.empty or test.empty:
            continue
        bundle = fit_bundle(dataset, train, val)
        pred = _predict_two_stage_internal(dataset, bundle.dnf_clf, bundle.ranker, bundle.xgb_reg,
                                           test, bundle.best_alpha, bundle.use_dnf)
        model = per_race_metrics(pred, "expected_pos")
        base = per_race_metrics(test.assign(quali_pred=test["quali_position"]), "quali_pred")
        merged = model.merge(base, on=["raceId", "year"], suffixes=("_model", "_base"))
        merged["alpha"], merged["use_dnf"] = bundle.best_alpha, bundle.use_dnf
        parts.append(merged)
        log(f"  {year}: {len(merged)} гонок, MAE {merged['mae_model'].mean():.3f} "
            f"(baseline {merged['mae_base'].mean():.3f}), alpha={bundle.best_alpha}, use_dnf={bundle.use_dnf}")

    return WalkForwardResult(races=pd.concat(parts, ignore_index=True))

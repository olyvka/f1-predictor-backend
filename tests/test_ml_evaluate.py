"""Юніт-тести метрик walk-forward оцінки (app/ml/evaluate.py) на
синтетичних даних - без датасету й тренування."""
import numpy as np
import pandas as pd

from app.ml.evaluate import WalkForwardResult, per_race_metrics


def _race(race_id, pred):
    return pd.DataFrame({
        "raceId": race_id, "year": 2024, "driverId": range(1, 7),
        "positionOrder": range(1, 7), "pred": pred,
    })


def test_per_race_metrics_perfect_and_reversed():
    df = pd.concat([_race(1, [1, 2, 3, 4, 5, 6]), _race(2, [6, 5, 4, 3, 2, 1])])
    m = per_race_metrics(df, "pred").set_index("raceId")

    assert m.loc[1, "mae"] == 0 and m.loc[1, "spearman"] == 1
    assert m.loc[1, "podium"] == 1 and m.loc[1, "top1"] == 1
    assert m.loc[2, "spearman"] == -1
    assert m.loc[2, "podium"] == 0 and m.loc[2, "top1"] == 0


def test_summary_verdicts():
    n = 200
    rng = np.random.default_rng(1)
    races = pd.DataFrame({
        "raceId": range(n), "year": 2024,
        # модель чітко краща за MAE, однакова з baseline за рештою
        "mae_model": rng.normal(3.0, 0.2, n), "mae_base": rng.normal(4.0, 0.2, n),
        **{f"{m}_{w}": np.full(n, 0.5) for m in ("spearman", "podium", "top1") for w in ("model", "base")},
    })
    s = WalkForwardResult(races).summary(n_boot=300).set_index("metric")

    assert s.loc["mae", "verdict"] == "краще"
    assert s.loc["spearman", "verdict"] == "нерозрізнимо"

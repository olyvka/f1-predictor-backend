"""predict_two_stage() - перенесення CELL 21 з f1_pipeline_v2.py.
Працює однаково і для рядків з df_model (історична гонка), і для
"живих" рядків, зібраних app/ml/live.py для поточної/майбутньої гонки -
формат колонок той самий."""
from __future__ import annotations

import pandas as pd

from app.ml.dataset import Dataset
from app.ml.train import ModelBundle, _predict_two_stage_internal


def predict_two_stage(dataset: Dataset, bundle: ModelBundle, df_in: pd.DataFrame,
                       alpha: float | None = None) -> pd.DataFrame:
    """
    Двостадійне передбачення:
      1. P(DNF) - від класифікатора (калібрована ймовірність, віддається
         окремою колонкою p_dnf).
      2. finisher_pos - ансамбль ranker + xgb: alpha * ranker + (1-alpha) * xgb.
      3. expected_pos = finisher_pos, або (1 - P(DNF)) * finisher_pos + P(DNF) * 20,
         якщо на val це дало менший MAE (bundle.use_dnf).

    alpha - вага ranker (best_alpha з бандла, якщо не задано явно).
    """
    if alpha is None:
        alpha = bundle.best_alpha
    return _predict_two_stage_internal(dataset, bundle.dnf_clf, bundle.ranker, bundle.xgb_reg,
                                       df_in, alpha, bundle.use_dnf)

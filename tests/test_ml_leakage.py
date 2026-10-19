"""
Тест на data leakage у feature engineering (app/ml/dataset.py).

Ідея: фічі гонки X мають бути відомі ДО її старту. Тому якщо зіпсувати
результат гонки X (results.csv) і її кола (lap_times.csv), а потім
перебудувати датасет - жодна фіча в рядках гонки X змінитись не має.
Якщо змінилась - вона "бачить" результат гонки, яку прогнозує (саме так
було з fp_* і wet, що рахувались з кіл самої гонки).

Кваліфікацію, grid і standings не чіпаємо: вони законно відомі до старту
(standings у датасеті і так зсунуті на етап назад).
"""
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from app.ml import config, dataset as dataset_mod

pytestmark = pytest.mark.integration

TARGET_YEAR, TARGET_ROUND = 2024, 5


def _perturb_race(src_dir: str, dst_dir: str, race_id: int) -> None:
    for name in os.listdir(src_dir):
        if name.endswith(".csv") and name not in ("results.csv", "lap_times.csv"):
            shutil.copy(os.path.join(src_dir, name), os.path.join(dst_dir, name))

    results = pd.read_csv(os.path.join(src_dir, "results.csv"), dtype=str, keep_default_na=False)
    m = results["raceId"] == str(race_id)
    # Фінішний порядок і очки навиворіт, половина пілотів "зійшла".
    for col in ("positionOrder", "points", "statusId"):
        results.loc[m, col] = results.loc[m, col].values[::-1]
    results.loc[m & (results.index % 2 == 0), "statusId"] = "5"  # Engine
    results.to_csv(os.path.join(dst_dir, "results.csv"), index=False)

    laps = pd.read_csv(os.path.join(src_dir, "lap_times.csv"))
    lm = laps["raceId"] == race_id
    rng = np.random.default_rng(0)
    laps.loc[lm, "milliseconds"] = (laps.loc[lm, "milliseconds"] *
                                    rng.uniform(0.9, 1.3, lm.sum())).round().astype(int)
    laps.loc[lm, "position"] = rng.permutation(laps.loc[lm, "position"].values)
    # Прибираємо кола 2-6 у частини пілотів - імітація сходу на старті.
    early_drop = lm & laps["lap"].between(2, 6) & (laps["driverId"] % 3 == 0)
    laps = laps[~early_drop]
    laps.to_csv(os.path.join(dst_dir, "lap_times.csv"), index=False)


def test_features_do_not_depend_on_target_race_outcome(tmp_path, monkeypatch):
    original = dataset_mod._build_uncached()
    races = original.races
    race_id = int(races[(races["year"] == TARGET_YEAR) & (races["round"] == TARGET_ROUND)]["raceId"].iloc[0])

    _perturb_race(config.DATA_DIR, str(tmp_path), race_id)
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    perturbed = dataset_mod._build_uncached()

    cols = ["driverId"] + original.features

    def rows(ds):
        r = ds.raw[ds.raw["raceId"] == race_id][cols].copy()
        for c in ds.categorical:
            r[c] = r[c].astype(object)
        return r.sort_values("driverId").reset_index(drop=True)

    before, after = rows(original), rows(perturbed)
    assert len(before) >= 15

    leaked = [c for c in original.features
              if not before[c].equals(after[c])
              and not np.allclose(before[c].astype(float), after[c].astype(float), equal_nan=True)]
    assert not leaked, f"Фічі залежать від результату самої гонки (leakage): {leaked}"

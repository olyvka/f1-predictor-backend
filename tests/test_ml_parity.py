"""
Тест відповідності тренування і прогнозу: фічі, які app/ml/live.py
рахує "на льоту" для гонки, мають збігатися з тими, на яких модель
вчилась (dataset.raw), якщо дати live тій самій історичній гонці ту саму
кваліфікацію. Розбіжність = модель у проді бачить не те, що в тренуванні.

Єдиний навмисний виняток - grid/grid_penalty: наперед реальні
грід-пенальті невідомі, live бере grid = позиція в кваліфікації.
"""
import numpy as np
import pandas as pd
import pytest

from app.ml import live
from app.ml.dataset import get_dataset

pytestmark = pytest.mark.integration

EXPECTED_DIFFERENT = {"grid", "grid_penalty"}
RACES = [(2019, 10), (2022, 3), (2024, 5), (2025, 8)]


@pytest.fixture(scope="module")
def dataset():
    return get_dataset()


def _quali_as_fastf1(dataset, race_id: int) -> list[dict]:
    """Кваліфікація з CSV у форматі fastf1_client.get_session_results(..., "Q")."""
    q = dataset.qualifying[dataset.qualifying["raceId"] == race_id]
    teams = dataset.raw[dataset.raw["raceId"] == race_id][["driverId", "constructor_name"]]
    q = (q.merge(teams, on="driverId")
          .merge(dataset.drivers[["driverId", "code", "forename", "surname"]], on="driverId"))
    none_if_nan = lambda v: None if pd.isna(v) else float(v)
    return [{
        "Abbreviation": r.code, "FullName": f"{r.forename} {r.surname}",
        "TeamName": r.constructor_name, "Position": float(r.position),
        "Q1": none_if_nan(r.q1_sec), "Q2": none_if_nan(r.q2_sec), "Q3": none_if_nan(r.q3_sec),
    } for r in q.itertuples()]


@pytest.mark.parametrize("year,round_number", RACES)
def test_live_features_match_training_features(dataset, monkeypatch, year, round_number):
    races = dataset.races
    race_id = int(races[(races["year"] == year) & (races["round"] == round_number)]["raceId"].iloc[0])
    monkeypatch.setattr(live, "_fetch_live_quali",
                        lambda *a: _quali_as_fastf1(dataset, race_id))
    # Практики - лише з data/fastf1_*.csv, як і в тренуванні; без мережі
    # (якщо етапу там нема - і тренування, і live мають NaN).
    monkeypatch.setattr(live, "_fetch_live_weekend", lambda *a: (pd.DataFrame(), pd.DataFrame()))

    out, quali_available, _ = live.build_feature_rows(dataset, year, round_number)
    assert quali_available

    live_rows = out.set_index("driverId")
    train_rows = dataset.raw[dataset.raw["raceId"] == race_id].set_index("driverId")
    common = live_rows.index.intersection(train_rows.index)
    assert len(common) == len(train_rows), "не всіх пілотів вдалось зіставити"

    mismatched = {}
    for f in dataset.num_features:
        if f in EXPECTED_DIFFERENT:
            continue
        a = live_rows.loc[common, f].astype(float).values
        b = train_rows.loc[common, f].astype(float).values
        bad = ~np.isclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-6)
        if bad.any():
            i = int(np.argmax(bad))
            mismatched[f] = f"{bad.sum()} пілотів, напр. live={a[i]} train={b[i]}"
    for c in dataset.categorical:
        if not (live_rows.loc[common, c].astype(object) == train_rows.loc[common, c].astype(object)).all():
            mismatched[c] = "категорії не збігаються"

    assert not mismatched, f"live і тренування рахують фічі по-різному: {mismatched}"

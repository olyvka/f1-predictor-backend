"""
Тести app/predictor.py. predict_race() імпортує get_race_results напряму
(`from app.fastf1_client import get_race_results`), тому мокати треба саме
app.predictor.get_race_results, а не app.fastf1_client.get_race_results.
"""
import pytest

from app import predictor


def test_predict_race_maps_results_to_predictions(monkeypatch):
    fake_results = [
        {"Position": 1.0, "FullName": "Max Verstappen", "TeamName": "Red Bull Racing"},
        {"Position": 2.0, "FullName": "Lewis Hamilton", "TeamName": "Mercedes"},
    ]
    monkeypatch.setattr(predictor, "get_race_results", lambda year, round_number: fake_results)

    predictions = predictor.predict_race(2024, 1)

    assert predictions == [
        {
            "predicted_position": 1.0,
            "driver": "Max Verstappen",
            "team": "Red Bull Racing",
            "confidence": None,
        },
        {
            "predicted_position": 2.0,
            "driver": "Lewis Hamilton",
            "team": "Mercedes",
            "confidence": None,
        },
    ]


def test_predict_race_empty_results(monkeypatch):
    monkeypatch.setattr(predictor, "get_race_results", lambda year, round_number: [])

    assert predictor.predict_race(2024, 1) == []


def test_predict_race_propagates_errors(monkeypatch):
    def boom(year, round_number):
        raise RuntimeError("no data")

    monkeypatch.setattr(predictor, "get_race_results", boom)

    with pytest.raises(RuntimeError, match="no data"):
        predictor.predict_race(2024, 1)

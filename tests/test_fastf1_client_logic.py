"""
Тест бізнес-логіки app/fastf1_client.get_driver_standings_snapshot(), яка не
залежить від мережі напряму, а лише від get_race_results (мокається).
"""
from app import fastf1_client


def test_driver_standings_snapshot_sorts_by_position_none_last(monkeypatch):
    fake_results = [
        {"Abbreviation": "SAI", "Position": None},  # DNF - без позиції
        {"Abbreviation": "HAM", "Position": 2.0},
        {"Abbreviation": "VER", "Position": 1.0},
    ]
    monkeypatch.setattr(
        fastf1_client, "get_race_results", lambda year, round_number: fake_results
    )

    snapshot = fastf1_client.get_driver_standings_snapshot(2024, 1)

    assert [r["Abbreviation"] for r in snapshot] == ["VER", "HAM", "SAI"]

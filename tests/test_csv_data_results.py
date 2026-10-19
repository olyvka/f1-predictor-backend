"""
Тести CSV-джерела результатів сесій (R/Q/S) і піт-стопів у app/csv_data.py -
на реальних рядках датасету (2023 Bahrain GP = round 1, чемпіон VER; 2023
Azerbaijan GP = round 4, спринт), без моків.
"""
import json

import pytest

from app import csv_data


def test_race_results_2023_round1_winner_is_verstappen():
    results = csv_data.get_race_results(2023, 1)
    assert len(results) > 0
    assert results[0]["position"] == 1
    assert results[0]["code"] == "VER"
    assert results[0]["status"] == "Finished"
    positions = [r["position"] for r in results if r["position"] is not None]
    assert positions == sorted(positions)


def test_race_results_time_fields_are_seconds_not_strings():
    results = csv_data.get_race_results(2023, 1)
    winner = results[0]
    assert isinstance(winner["total_time"], float)
    assert isinstance(winner["fastest_lap_time"], float)
    assert winner["total_time"] > 0


def test_race_results_dnf_has_null_total_time_but_status_set():
    results = csv_data.get_race_results(2023, 1)
    dnf = next(r for r in results if r["status"] != "Finished" and "Lap" not in r["status"])
    assert dnf["total_time"] is None
    assert dnf["position"] is None
    assert dnf["position_text"] not in (None, "")


def test_race_results_unknown_round_raises():
    with pytest.raises(ValueError, match="2023"):
        csv_data.get_race_results(2023, 999)


def test_qualifying_results_2023_round1_pole_is_verstappen():
    results = csv_data.get_qualifying_results(2023, 1)
    assert results[0]["position"] == 1
    assert results[0]["code"] == "VER"
    assert isinstance(results[0]["q3"], float)


def test_sprint_results_2023_round4_exists():
    results = csv_data.get_sprint_results(2023, 4)
    assert len(results) > 0
    assert results[0]["position"] == 1


def test_sprint_results_missing_for_non_sprint_round_raises():
    with pytest.raises(ValueError, match="спринт"):
        csv_data.get_sprint_results(2023, 1)  # Bahrain 2023 не мав спринту


@pytest.mark.parametrize("session", ["R", "race", "Race", "Q", "qualifying", "S", "sprint"])
def test_get_session_results_accepts_aliases_case_insensitive(session):
    # не має кидати - лише перевіряємо, що аліас розпізнається (round 4 має і R, і Q, і S)
    results = csv_data.get_session_results(2023, 4, session)
    assert isinstance(results, list)
    assert len(results) > 0


def test_get_session_results_rejects_practice_sessions():
    with pytest.raises(ValueError, match="FP1"):
        csv_data.get_session_results(2023, 1, "FP1")


def test_get_session_drivers_projects_subset_of_fields():
    drivers = csv_data.get_session_drivers(2023, 1, "R")
    assert drivers[0] == {
        "driver_number": 33,
        "code": "VER",
        "full_name": "Max Verstappen",
        "team": "Red Bull",
        "position": 1,
        "points": 25.0,
    }


def test_pit_stops_2023_round1():
    stops = csv_data.get_pit_stops(2023, 1)
    assert len(stops) > 0
    first = stops[0]
    assert first["lap"] >= 1
    assert first["stop"] >= 1
    assert isinstance(first["duration"], float)
    # відсортовано по колу (і по стопу в межах кола)
    laps = [s["lap"] for s in stops]
    assert laps == sorted(laps)


def test_pit_stops_unknown_round_raises():
    with pytest.raises(ValueError, match="2023"):
        csv_data.get_pit_stops(2023, 999)


def test_all_results_json_serializable():
    json.dumps(csv_data.get_race_results(2023, 1), allow_nan=False)
    json.dumps(csv_data.get_qualifying_results(2023, 1), allow_nan=False)
    json.dumps(csv_data.get_sprint_results(2023, 4), allow_nan=False)
    json.dumps(csv_data.get_pit_stops(2023, 1), allow_nan=False)

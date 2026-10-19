"""
Тести app/csv_data.py - шар над статичним датасетом у data/ (сезонні
зведення: календар, залік пілотів/конструкторів). На відміну від
fastf1_client, тут нічого не мокається: датасет статичний, лежить у
репозиторії, і саме факт коректного читання/агрегації реальних CSV -
предмет цих тестів.
"""
import pytest

from app import csv_data


def test_list_seasons_includes_known_years():
    seasons = csv_data.list_seasons()
    assert isinstance(seasons, list)
    assert seasons == sorted(seasons)  # відсортовано за зростанням
    assert 2023 in seasons


def test_get_season_races_returns_full_calendar_sorted_by_round():
    races = csv_data.get_season_races(2023)
    assert len(races) > 0
    rounds = [r["round"] for r in races]
    assert rounds == sorted(rounds)
    assert races[0]["round"] == 1
    assert races[0]["name"]  # непорожня назва Гран-прі


def test_get_season_races_unknown_year_raises():
    with pytest.raises(ValueError, match="2999"):
        csv_data.get_season_races(2999)


def test_get_driver_standings_2023_champion_is_verstappen():
    standings = csv_data.get_driver_standings(2023)
    assert standings[0]["position"] == 1
    assert standings[0]["code"] == "VER"
    assert standings[0]["surname"] == "Verstappen"
    # відсортовано за позицією за зростанням
    positions = [s["position"] for s in standings]
    assert positions == sorted(positions)


def test_get_driver_standings_unknown_year_raises():
    with pytest.raises(ValueError, match="2999"):
        csv_data.get_driver_standings(2999)


def test_get_constructor_standings_2023_champion_is_red_bull():
    standings = csv_data.get_constructor_standings(2023)
    assert standings[0]["position"] == 1
    assert standings[0]["name"] == "Red Bull"
    positions = [s["position"] for s in standings]
    assert positions == sorted(positions)


def test_get_constructor_standings_unknown_year_raises():
    with pytest.raises(ValueError, match="2999"):
        csv_data.get_constructor_standings(2999)


def test_standings_are_json_serializable():
    import json

    json.dumps(csv_data.get_driver_standings(2023), allow_nan=False)
    json.dumps(csv_data.get_constructor_standings(2023), allow_nan=False)
    json.dumps(csv_data.get_season_races(2023), allow_nan=False)

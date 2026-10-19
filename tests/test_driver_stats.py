"""
app/driver_stats.py на реальному CSV-датасеті - перевіряємо на фактах, які
відомі напевно (сезон 2021, кар'єра Гамільтона й Сенни).
"""
import pytest

from app import driver_stats


def test_season_progress_2021_final_table():
    d = driver_stats.get_season_progress(2021)

    assert d["final"] is True
    assert d["completed_rounds"] == d["total_rounds"] == 22
    ver, ham = d["drivers"][0], d["drivers"][1]
    assert (ver["code"], ver["points"], ver["wins"]) == ("VER", 395.5, 10)
    assert (ham["code"], ham["points"], ham["wins"]) == ("HAM", 387.5, 8)
    # Сезон завершено - шанси на титул лише в чемпіона.
    assert ver["can_win_title"] and not ham["can_win_title"]
    # Накопичені очки по етапах закінчуються підсумковими.
    assert len(ver["points_by_round"]) == 22 and ver["points_by_round"][-1] == 395.5
    # Переможці етапів: Бахрейн 2021 - Гамільтон, Абу-Дабі - Ферстаппен.
    assert d["rounds"][0]["winner"]["code"] == "HAM"
    assert d["rounds"][-1]["winner"]["code"] == "VER"
    assert d["constructors"][0]["team"] == "Mercedes"
    assert d["constructors"][0]["points"] == 613.5
    # Внесок пілотів: гонки + спринти Гамільтона й Боттаса.
    split = {x["code"]: x["points"] for x in d["constructors"][0]["drivers"]}
    assert split == {"HAM": 387.5, "BOT": 226.0}


def test_season_progress_unfinished_season_keeps_title_open():
    d = driver_stats.get_season_progress(2026)
    assert d["final"] is False
    assert d["max_remaining_points"] > 0
    assert d["drivers"][0]["can_win_title"] is True


def test_driver_season_teammate_and_rating():
    d = driver_stats.get_driver_season(2021, "max_verstappen")

    assert d["championship"] == {"position": 1, "points": 395.5, "after_round": 22}
    assert d["stats"]["starts"] == 22 and d["stats"]["wins"] == 10
    assert len(d["races"]) == 22
    (perez,) = d["teammates"]
    assert perez["code"] == "PER"
    assert perez["race"] == {"won": 16, "lost": 0}  # лише гонки, де фінішували обидва
    rating = d["rating"]
    assert 0 <= rating["score"] <= 100
    assert 1 <= rating["rank"] <= rating["of"]
    assert {c["key"] for c in rating["components"]} == {k for k, _, _ in driver_stats.RATING_COMPONENTS}


def test_driver_career_hamilton():
    d = driver_stats.get_driver_career("hamilton")

    assert d["debut"]["year"] == 2007
    assert d["totals"]["titles"] == 7
    assert d["totals"]["wins"] >= 105
    assert d["seasons"][0]["teams"] == ["McLaren"]


def test_fastest_laps_hidden_for_careers_before_2004():
    # До 2004 найшвидші кола в датасеті не заповнені - нуль був би неправдою.
    assert driver_stats.get_driver_career("senna")["totals"]["fastest_laps"] is None


def test_driver_event_context():
    d = driver_stats.get_driver_event(2024, 8, "leclerc")  # Монако 2024 - перемога Леклера

    assert d["championship"]["after"]["points"] - d["championship"]["before"]["points"] == 25
    assert d["nationality"] == "Monegasque"
    assert d["teammate"]["code"] == "SAI"
    assert d["teammate"]["race"] == {"mine": "1", "mate": "3"}
    assert all(h["year"] < 2024 for h in d["circuit_history"])


def test_unknown_driver():
    with pytest.raises(ValueError):
        driver_stats.get_driver_season(2024, "nobody")

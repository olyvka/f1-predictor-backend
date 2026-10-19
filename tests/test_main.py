"""
Юніт-тести для всіх ендпоінтів app/main.py.

Викликів до fastf1/мережі тут немає: app.fastf1_client і app.predictor
підміняються моками через monkeypatch, тому тести швидкі, детерміновані
і не залежать від інтернету чи FastF1-кешу. Перевіряють:
- що кожен ендпоінт віддає дані з fastf1_client/predictor як є (200);
- що будь-який виняток з fastf1_client/predictor перетворюється на 404
  з очікуваним префіксом повідомлення;
- валідацію вхідних параметрів (список пілотів у /api/laps, /api/compare).
"""
import pytest

from app import csv_data, fastf1_client, predictor
from app.main import _parse_drivers


# ---------------------------------------------------------------------------
# _parse_drivers - чиста функція
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("VER,HAM", ["VER", "HAM"]),
        ("ver,ham", ["VER", "HAM"]),
        (" ver , ham ", ["VER", "HAM"]),
        ("VER,,HAM", ["VER", "HAM"]),  # порожні елементи відкидаються
        ("ver", ["VER"]),
        ("", []),
        (",, ,", []),
    ],
)
def test_parse_drivers(raw, expected):
    assert _parse_drivers(raw) == expected


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "f1-predictor-api"}


# ---------------------------------------------------------------------------
# GET /api/seasons, /api/standings/drivers/{year}, /api/standings/constructors/{year}
# (джерело даних - локальний CSV-датасет через app.csv_data, не FastF1)
# ---------------------------------------------------------------------------

def test_list_seasons_ok(client, monkeypatch):
    monkeypatch.setattr(csv_data, "list_seasons", lambda: [2023, 2024, 2025])

    resp = client.get("/api/seasons")

    assert resp.status_code == 200
    assert resp.json() == [2023, 2024, 2025]


def test_driver_standings_ok(client, monkeypatch):
    fake = [{"position": 1, "code": "VER", "points": 575}]
    captured = {}

    def fake_get_driver_standings(year):
        captured["year"] = year
        return fake

    monkeypatch.setattr(csv_data, "get_driver_standings", fake_get_driver_standings)

    resp = client.get("/api/standings/drivers/2023")

    assert resp.status_code == 200
    assert resp.json() == fake
    assert captured["year"] == 2023


def test_driver_standings_unknown_season_is_404(client, monkeypatch):
    def boom(year):
        raise ValueError(f"Сезон {year} відсутній у датасеті")

    monkeypatch.setattr(csv_data, "get_driver_standings", boom)

    resp = client.get("/api/standings/drivers/2999")

    assert resp.status_code == 404
    assert "2999" in resp.json()["detail"]


def test_constructor_standings_ok(client, monkeypatch):
    fake = [{"position": 1, "name": "Red Bull", "points": 860}]
    monkeypatch.setattr(csv_data, "get_constructor_standings", lambda year: fake)

    resp = client.get("/api/standings/constructors/2023")

    assert resp.status_code == 200
    assert resp.json() == fake


def test_constructor_standings_unknown_season_is_404(client, monkeypatch):
    def boom(year):
        raise ValueError(f"Сезон {year} відсутній у датасеті")

    monkeypatch.setattr(csv_data, "get_constructor_standings", boom)

    resp = client.get("/api/standings/constructors/2999")

    assert resp.status_code == 404
    assert "2999" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/csv/results/{year}/{round}/{session}, /api/csv/drivers/..., /api/csv/pitstops/...
# (джерело - csv_data, не FastF1)
# ---------------------------------------------------------------------------

def test_session_results_csv_ok(client, monkeypatch):
    fake = [{"position": 1, "code": "VER"}]
    captured = {}

    def fake_get_session_results(year, round_number, session):
        captured["args"] = (year, round_number, session)
        return fake

    monkeypatch.setattr(csv_data, "get_session_results", fake_get_session_results)

    resp = client.get("/api/csv/results/2023/1/R")

    assert resp.status_code == 200
    assert resp.json() == fake
    assert captured["args"] == (2023, 1, "R")


def test_session_results_csv_not_found(client, monkeypatch):
    def boom(year, round_number, session):
        raise ValueError(f"Сесія '{session}' недоступна в CSV-датасеті")

    monkeypatch.setattr(csv_data, "get_session_results", boom)

    resp = client.get("/api/csv/results/2023/1/FP1")

    assert resp.status_code == 404
    assert "недоступна в CSV-датасеті" in resp.json()["detail"]


def test_session_drivers_csv_ok(client, monkeypatch):
    fake = [{"code": "VER", "full_name": "Max Verstappen"}]
    monkeypatch.setattr(csv_data, "get_session_drivers", lambda year, round_number, session: fake)

    resp = client.get("/api/csv/drivers/2023/1/R")

    assert resp.status_code == 200
    assert resp.json() == fake


def test_session_drivers_csv_not_found(client, monkeypatch):
    def boom(year, round_number, session):
        raise ValueError("немає даних")

    monkeypatch.setattr(csv_data, "get_session_drivers", boom)

    resp = client.get("/api/csv/drivers/2023/1/R")

    assert resp.status_code == 404


def test_pit_stops_ok(client, monkeypatch):
    fake = [{"code": "VER", "lap": 14, "stop": 1, "duration": 2.3}]
    captured = {}

    def fake_get_pit_stops(year, round_number):
        captured["args"] = (year, round_number)
        return fake

    monkeypatch.setattr(csv_data, "get_pit_stops", fake_get_pit_stops)

    resp = client.get("/api/csv/pitstops/2023/1")

    assert resp.status_code == 200
    assert resp.json() == fake
    assert captured["args"] == (2023, 1)


def test_pit_stops_not_found(client, monkeypatch):
    def boom(year, round_number):
        raise ValueError(f"Дані по піт-стопах етапу {round_number} сезону {year} відсутні")

    monkeypatch.setattr(csv_data, "get_pit_stops", boom)

    resp = client.get("/api/csv/pitstops/2023/999")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/races/{year}
# ---------------------------------------------------------------------------

def test_list_races_ok(client, monkeypatch):
    fake = [{"RoundNumber": 1, "EventName": "Bahrain Grand Prix"}]
    monkeypatch.setattr(fastf1_client, "get_season_schedule", lambda year: fake)

    resp = client.get("/api/races/2024")

    assert resp.status_code == 200
    assert resp.json() == fake


def test_list_races_not_found(client, monkeypatch):
    def boom(year):
        raise RuntimeError("no such season")

    monkeypatch.setattr(fastf1_client, "get_season_schedule", boom)

    resp = client.get("/api/races/1900")

    assert resp.status_code == 404
    assert resp.json()["detail"].startswith("Season not found:")


# ---------------------------------------------------------------------------
# GET /api/sessions/{year}/{round_number}
# ---------------------------------------------------------------------------

def test_list_sessions_ok(client, monkeypatch):
    fake = [{"number": 5, "name": "Race", "code": "R"}]
    captured = {}

    def fake_get_event_sessions(year, round_number):
        captured["args"] = (year, round_number)
        return fake

    monkeypatch.setattr(fastf1_client, "get_event_sessions", fake_get_event_sessions)

    resp = client.get("/api/sessions/2024/1")

    assert resp.status_code == 200
    assert resp.json() == fake
    assert captured["args"] == (2024, 1)


def test_list_sessions_not_found(client, monkeypatch):
    monkeypatch.setattr(
        fastf1_client,
        "get_event_sessions",
        lambda year, round_number: (_ for _ in ()).throw(RuntimeError("no event")),
    )

    resp = client.get("/api/sessions/2024/999")

    assert resp.status_code == 404
    assert resp.json()["detail"].startswith("Event not found:")


# ---------------------------------------------------------------------------
# GET /api/results/{year}/{round_number}  (скорочення для сесії "R")
# ---------------------------------------------------------------------------

def test_race_results_ok_calls_session_results_with_R(client, monkeypatch):
    fake = [{"Abbreviation": "VER", "Position": 1}]
    captured = {}

    def fake_get_session_results(year, round_number, session):
        captured["args"] = (year, round_number, session)
        return fake

    monkeypatch.setattr(fastf1_client, "get_session_results", fake_get_session_results)

    resp = client.get("/api/results/2024/1")

    assert resp.status_code == 200
    assert resp.json() == fake
    assert captured["args"] == (2024, 1, "R")


def test_race_results_not_found(client, monkeypatch):
    def boom(year, round_number, session):
        raise RuntimeError("no race")

    monkeypatch.setattr(fastf1_client, "get_session_results", boom)

    resp = client.get("/api/results/2024/1")

    assert resp.status_code == 404
    assert resp.json()["detail"].startswith("Race not found:")


# ---------------------------------------------------------------------------
# GET /api/results/{year}/{round_number}/{session}
# ---------------------------------------------------------------------------

def test_session_results_ok(client, monkeypatch):
    fake = [{"Abbreviation": "HAM", "Position": 2}]
    captured = {}

    def fake_get_session_results(year, round_number, session):
        captured["args"] = (year, round_number, session)
        return fake

    monkeypatch.setattr(fastf1_client, "get_session_results", fake_get_session_results)

    resp = client.get("/api/results/2024/1/Q")

    assert resp.status_code == 200
    assert resp.json() == fake
    assert captured["args"] == (2024, 1, "Q")


def test_session_results_not_found(client, monkeypatch):
    def boom(year, round_number, session):
        raise RuntimeError("bad session")

    monkeypatch.setattr(fastf1_client, "get_session_results", boom)

    resp = client.get("/api/results/2024/1/Q")

    assert resp.status_code == 404
    assert resp.json()["detail"].startswith("Session not found:")


# ---------------------------------------------------------------------------
# GET /api/drivers/{year}/{round_number}/{session}
# ---------------------------------------------------------------------------

def test_session_drivers_ok(client, monkeypatch):
    fake = [{"Abbreviation": "VER", "FullName": "Max Verstappen"}]
    monkeypatch.setattr(
        fastf1_client, "get_session_drivers", lambda year, round_number, session: fake
    )

    resp = client.get("/api/drivers/2024/1/R")

    assert resp.status_code == 200
    assert resp.json() == fake


def test_session_drivers_not_found(client, monkeypatch):
    def boom(year, round_number, session):
        raise RuntimeError("no drivers")

    monkeypatch.setattr(fastf1_client, "get_session_drivers", boom)

    resp = client.get("/api/drivers/2024/1/R")

    assert resp.status_code == 404
    assert resp.json()["detail"].startswith("Session not found:")


# ---------------------------------------------------------------------------
# GET /api/driver/{year}/{round_number}/{session}/{driver}
# ---------------------------------------------------------------------------

def test_driver_summary_ok(client, monkeypatch):
    fake = {"driver": {"Abbreviation": "VER"}, "result": None, "stints": [], "laps": []}
    captured = {}

    def fake_get_driver_summary(year, round_number, session, driver):
        captured["args"] = (year, round_number, session, driver)
        return fake

    monkeypatch.setattr(fastf1_client, "get_driver_summary", fake_get_driver_summary)

    resp = client.get("/api/driver/2024/1/R/VER")

    assert resp.status_code == 200
    assert resp.json() == fake
    assert captured["args"] == (2024, 1, "R", "VER")


def test_driver_summary_not_found(client, monkeypatch):
    def boom(year, round_number, session, driver):
        raise ValueError(f"Пілот '{driver}' не знайдений")

    monkeypatch.setattr(fastf1_client, "get_driver_summary", boom)

    resp = client.get("/api/driver/2024/1/R/ZZZ")

    assert resp.status_code == 404
    assert resp.json()["detail"].startswith("Дані пілота недоступні:")


# ---------------------------------------------------------------------------
# GET /api/laps/{year}/{round_number}/{session}
# ---------------------------------------------------------------------------

def test_session_laps_ok_without_drivers_filter(client, monkeypatch):
    fake = [{"Driver": "VER", "LapNumber": 1}]
    captured = {}

    def fake_get_session_laps(year, round_number, session, driver_list):
        captured["args"] = (year, round_number, session, driver_list)
        return fake

    monkeypatch.setattr(fastf1_client, "get_session_laps", fake_get_session_laps)

    resp = client.get("/api/laps/2024/1/R")

    assert resp.status_code == 200
    assert resp.json() == fake
    # без query-параметра drivers список пілотів має бути None (усі пілоти)
    assert captured["args"] == (2024, 1, "R", None)


def test_session_laps_ok_with_drivers_filter(client, monkeypatch):
    captured = {}

    def fake_get_session_laps(year, round_number, session, driver_list):
        captured["args"] = (year, round_number, session, driver_list)
        return []

    monkeypatch.setattr(fastf1_client, "get_session_laps", fake_get_session_laps)

    resp = client.get("/api/laps/2024/1/R", params={"drivers": "ver, ham"})

    assert resp.status_code == 200
    assert captured["args"] == (2024, 1, "R", ["VER", "HAM"])


def test_session_laps_not_found(client, monkeypatch):
    def boom(year, round_number, session, driver_list):
        raise RuntimeError("no laps")

    monkeypatch.setattr(fastf1_client, "get_session_laps", boom)

    resp = client.get("/api/laps/2024/1/R")

    assert resp.status_code == 404
    assert resp.json()["detail"].startswith("Кола недоступні:")


# ---------------------------------------------------------------------------
# GET /api/compare/{year}/{round_number}/{session}
# ---------------------------------------------------------------------------

def test_compare_drivers_ok(client, monkeypatch):
    fake = {"VER": {"laps": [], "fastest_lap": None}, "HAM": {"laps": [], "fastest_lap": None}}
    captured = {}

    def fake_compare(year, round_number, session, driver_list):
        captured["args"] = (year, round_number, session, driver_list)
        return fake

    monkeypatch.setattr(fastf1_client, "compare_drivers_laps", fake_compare)

    resp = client.get("/api/compare/2024/1/R", params={"drivers": "VER,HAM"})

    assert resp.status_code == 200
    assert resp.json() == fake
    assert captured["args"] == (2024, 1, "R", ["VER", "HAM"])


def test_compare_drivers_requires_query_param(client):
    # drivers - обов'язковий Query(...), без нього FastAPI поверне 422
    resp = client.get("/api/compare/2024/1/R")
    assert resp.status_code == 422


def test_compare_drivers_requires_at_least_two(client):
    resp = client.get("/api/compare/2024/1/R", params={"drivers": "VER"})

    assert resp.status_code == 400
    assert "щонайменше 2 пілоти" in resp.json()["detail"]


def test_compare_drivers_not_found(client, monkeypatch):
    def boom(year, round_number, session, driver_list):
        raise RuntimeError("comparison broke")

    monkeypatch.setattr(fastf1_client, "compare_drivers_laps", boom)

    resp = client.get("/api/compare/2024/1/R", params={"drivers": "VER,HAM"})

    assert resp.status_code == 404
    assert resp.json()["detail"].startswith("Порівняння недоступне:")


# ---------------------------------------------------------------------------
# GET /api/weather/{year}/{round_number}/{session}
# ---------------------------------------------------------------------------

def test_session_weather_ok(client, monkeypatch):
    fake = [{"AirTemp": 25.0, "TrackTemp": 35.0, "Rainfall": False}]
    monkeypatch.setattr(
        fastf1_client, "get_session_weather", lambda year, round_number, session: fake
    )

    resp = client.get("/api/weather/2024/1/R")

    assert resp.status_code == 200
    assert resp.json() == fake


def test_session_weather_not_found(client, monkeypatch):
    def boom(year, round_number, session):
        raise RuntimeError("no weather")

    monkeypatch.setattr(fastf1_client, "get_session_weather", boom)

    resp = client.get("/api/weather/2024/1/R")

    assert resp.status_code == 404
    assert resp.json()["detail"].startswith("Погода недоступна:")


# ---------------------------------------------------------------------------
# GET /api/predict/{year}/{round_number}
# ---------------------------------------------------------------------------

def test_predict_race_ok(client, monkeypatch):
    fake = [{"predicted_position": 1, "driver": "Max Verstappen", "team": "Red Bull", "confidence": None}]
    captured = {}

    def fake_predict(year, round_number):
        captured["args"] = (year, round_number)
        return fake

    monkeypatch.setattr(predictor, "predict_race", fake_predict)

    resp = client.get("/api/predict/2024/1")

    assert resp.status_code == 200
    assert resp.json() == fake
    assert captured["args"] == (2024, 1)


def test_predict_race_not_found(client, monkeypatch):
    def boom(year, round_number):
        raise RuntimeError("no prediction")

    monkeypatch.setattr(predictor, "predict_race", boom)

    resp = client.get("/api/predict/2024/1")

    assert resp.status_code == 404
    assert resp.json()["detail"].startswith("Could not predict race:")

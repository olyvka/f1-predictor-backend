"""
Інтеграційні тести: реальний TestClient без моків, реальний FastF1 (Гран-прі
Бахрейну 2024, round=1 - вже є в локальному cache/, тож дані не тягнуться
з мережі повторно). Перевіряють, що весь стек (роутинг -> fastf1_client ->
FastF1 -> серіалізація в JSON) справді працює на реальних даних, а не лише
з моками.

Повільніші за юніт-тести і в CI без прогрітого кешу можуть піти в мережу -
запускай окремо: pytest -m integration
"""
import pytest

pytestmark = pytest.mark.integration

YEAR = 2024
ROUND = 1  # Bahrain Grand Prix - закешовано в cache/2024


def test_csv_race_results_endpoint(client):
    resp = client.get("/api/csv/results/2023/1/R")
    assert resp.status_code == 200
    results = resp.json()
    assert results[0]["code"] == "VER"


def test_csv_qualifying_results_endpoint(client):
    resp = client.get("/api/csv/results/2023/1/Q")
    assert resp.status_code == 200
    assert resp.json()[0]["position"] == 1


def test_csv_results_rejects_practice_session(client):
    resp = client.get("/api/csv/results/2023/1/FP1")
    assert resp.status_code == 404


def test_csv_drivers_endpoint(client):
    resp = client.get("/api/csv/drivers/2023/1/R")
    assert resp.status_code == 200
    assert resp.json()[0]["code"] == "VER"


def test_csv_pitstops_endpoint(client):
    resp = client.get("/api/csv/pitstops/2023/1")
    assert resp.status_code == 200
    assert len(resp.json()) > 0


def test_seasons_endpoint_lists_real_seasons(client):
    resp = client.get("/api/seasons")
    assert resp.status_code == 200
    seasons = resp.json()
    assert 2023 in seasons


def test_driver_standings_endpoint_2023(client):
    resp = client.get("/api/standings/drivers/2023")
    assert resp.status_code == 200
    standings = resp.json()
    assert standings[0]["code"] == "VER"


def test_constructor_standings_endpoint_2023(client):
    resp = client.get("/api/standings/constructors/2023")
    assert resp.status_code == 200
    standings = resp.json()
    assert standings[0]["name"] == "Red Bull"


def test_races_returns_full_calendar(client):
    resp = client.get(f"/api/races/{YEAR}")
    assert resp.status_code == 200
    races = resp.json()
    assert isinstance(races, list)
    assert len(races) >= 20  # повний календар сезону
    assert races[0]["RoundNumber"] == 1


def test_sessions_lists_race_weekend_sessions(client):
    resp = client.get(f"/api/sessions/{YEAR}/{ROUND}")
    assert resp.status_code == 200
    sessions = resp.json()
    codes = {s["code"] for s in sessions}
    assert "R" in codes
    assert "Q" in codes


def test_race_results_shortcut_matches_explicit_R(client):
    short = client.get(f"/api/results/{YEAR}/{ROUND}")
    explicit = client.get(f"/api/results/{YEAR}/{ROUND}/R")

    assert short.status_code == 200
    assert explicit.status_code == 200
    assert short.json() == explicit.json()

    results = short.json()
    assert len(results) > 0
    winner = min(results, key=lambda r: r["Position"] or 999)
    assert winner["Position"] == 1
    assert winner["Abbreviation"]  # непорожній код пілота


def test_qualifying_results(client):
    resp = client.get(f"/api/results/{YEAR}/{ROUND}/Q")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) > 0
    assert "Q1" in results[0]


def test_session_drivers(client):
    resp = client.get(f"/api/drivers/{YEAR}/{ROUND}/R")
    assert resp.status_code == 200
    drivers = resp.json()
    assert len(drivers) > 15  # F1 стартова решітка ~20 пілотів
    assert any(d["Abbreviation"] == "VER" for d in drivers)


def test_driver_summary_for_verstappen(client):
    resp = client.get(f"/api/driver/{YEAR}/{ROUND}/R/VER")
    assert resp.status_code == 200
    data = resp.json()
    assert data["driver"]["Abbreviation"] == "VER"
    assert data["result"] is not None
    assert isinstance(data["laps"], list)
    assert len(data["laps"]) > 0


def test_driver_summary_unknown_driver_is_404(client):
    resp = client.get(f"/api/driver/{YEAR}/{ROUND}/R/ZZZ")
    assert resp.status_code == 404


def test_laps_filtered_by_driver(client):
    resp = client.get(f"/api/laps/{YEAR}/{ROUND}/R", params={"drivers": "VER"})
    assert resp.status_code == 200
    laps = resp.json()
    assert len(laps) > 0
    assert all(lap["Driver"] == "VER" for lap in laps)


def test_compare_two_drivers(client):
    resp = client.get(f"/api/compare/{YEAR}/{ROUND}/R", params={"drivers": "VER,HAM"})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"VER", "HAM"}
    assert len(data["VER"]["laps"]) > 0


def test_weather(client):
    resp = client.get(f"/api/weather/{YEAR}/{ROUND}/R")
    assert resp.status_code == 200
    weather = resp.json()
    assert len(weather) > 0
    assert "AirTemp" in weather[0]


def test_predict_race_uses_actual_results_as_stub(client):
    resp = client.get(f"/api/predict/{YEAR}/{ROUND}")
    assert resp.status_code == 200
    predictions = resp.json()
    assert len(predictions) > 0
    assert predictions[0]["predicted_position"] == 1


def test_unknown_round_is_404(client):
    resp = client.get(f"/api/results/{YEAR}/999")
    assert resp.status_code == 404

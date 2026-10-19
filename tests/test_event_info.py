"""
Тести вкладки «Огляд» (app/event_info.py + ендпоінти) на реальних рядках
датасету, без моків і мережі:
  - 2024 R6 Miami: перемога NOR, поул VER, найшвидше коло PIA, спринт (VER);
  - 2024 R1 Bahrain: перший етап - дельт у чемпіонаті нема;
  - 2024 R16 Monza; 2021 R9 Austria (Styrian/Austrian - одна траса).
"""
import json

import pytest

from app import analysis, event_info


# ---------------------------------------------------------------------------
# Задача 1: підсумок гонки
# ---------------------------------------------------------------------------

def test_event_summary_miami_2024():
    s = event_info.get_event_summary(2024, 6)
    assert [p["code"] for p in s["podium"]] == ["NOR", "VER", "LEC"]
    assert s["podium"][0]["gap"] is None
    assert s["podium"][1]["gap"] == pytest.approx(7.612)
    assert s["pole"]["code"] == "VER" and isinstance(s["pole"]["time"], float)
    assert s["fastest_lap"]["code"] == "PIA" and s["fastest_lap"]["lap"] == 43
    assert s["laps"] == 57
    assert s["dnf_count"] == len(s["dnfs"]) == 1
    assert s["sprint"]["podium"][0]["code"] == "VER"


def test_event_summary_biggest_gainer_miami_2024():
    g = event_info.get_event_summary(2024, 6)["biggest_gainer"]
    assert (g["code"], g["grid"], g["position"], g["gained"]) == ("ALO", 15, 9, 6)
    assert g["pit_lane_start"] is False


def test_event_summary_pit_lane_start_counts_as_last_place():
    # Китай-2019: Албон стартував з піт-лейну (grid = 0) і фінішував 10-м з 20.
    g = event_info.get_event_summary(2019, 3)["biggest_gainer"]
    assert (g["code"], g["grid"], g["pit_lane_start"], g["position"], g["gained"]) == ("ALB", 0, True, 10, 10)


def test_event_summary_future_race_is_none():
    assert event_info.get_event_summary(2026, 23) is None


def test_event_summary_unknown_round_raises():
    with pytest.raises(ValueError):
        event_info.get_event_summary(2024, 99)


@pytest.mark.parametrize("status, dnf", [
    ("Finished", False), ("+1 Lap", False), ("+3 Laps", False),
    ("Engine", True), ("Collision", True), ("Disqualified", True), (None, False),
])
def test_is_dnf(status, dnf):
    assert event_info.is_dnf(status) is dnf


# ---------------------------------------------------------------------------
# Задача 2: чемпіонат після етапу
# ---------------------------------------------------------------------------

def test_event_standings_deltas_and_gap():
    s = event_info.get_event_standings(2024, 6)
    assert s["as_of_round"] == 6
    leader = s["drivers"][0]
    assert leader["code"] == "VER" and leader["gap_to_leader"] == 0
    assert all(d["delta_points"] is not None for d in s["drivers"])
    second = s["drivers"][1]
    assert second["gap_to_leader"] == leader["points"] - second["points"]
    assert s["constructors"][0]["team"] == "Red Bull"


def test_event_standings_first_round_has_no_deltas():
    s = event_info.get_event_standings(2024, 1)
    assert all(d["delta_position"] is None and d["delta_points"] is None for d in s["drivers"])


def test_event_standings_future_round_uses_last_completed():
    s = event_info.get_event_standings(2026, 23)
    assert s["requested_round"] == 23
    assert s["as_of_round"] < 23
    assert s["drivers"]


# ---------------------------------------------------------------------------
# Задачі 3-4: історія й факти траси
# ---------------------------------------------------------------------------

def test_circuit_history_monza_before_2024():
    h = event_info.get_circuit_history(2024, 16)
    assert h["circuit"] == "Autodromo Nazionale di Monza"
    assert len(h["winners"]) == 10
    assert h["winners"][0]["year"] == 2023  # найсвіжіша перша, поточний рік не входить
    assert all(w["year"] < 2024 for w in h["winners"])
    assert {m["code"] for m in h["most_wins"][:2]} == {"HAM", "MSC"}
    assert h["pole_conversion"]["last_10"]["races"] == 10
    assert h["first_gp_year"] == 1950


def test_circuit_history_groups_by_circuit_not_name():
    names = {w["race_name"] for w in event_info.get_circuit_history(2021, 9)["winners"]}
    assert {"Styrian Grand Prix", "Austrian Grand Prix"} <= names


def test_circuit_facts_monza():
    f = event_info.get_circuit_facts(2024, 16)
    assert f["length_km"] == 5.793
    assert f["typical_laps"] == 53
    assert 1 <= f["avg_pit_stops_per_driver"] <= 3
    assert f["fastest_pit_stop"]["duration"] > 20  # drive-through не рахується
    assert f["pit_data_races"] == 10


def test_circuit_length_respects_layout_changes():
    assert event_info.circuit_length_km("yas_marina", 2020) == 5.554
    assert event_info.circuit_length_km("yas_marina", 2021) == 5.281
    assert event_info.circuit_length_km("unknown_track", 2024) is None


# ---------------------------------------------------------------------------
# Задача 5: статус прогнозу (лише читає кеш)
# ---------------------------------------------------------------------------

def _write_cache(tmp_path, monkeypatch, version, preds):
    monkeypatch.setattr(analysis, "_cache_path",
                        lambda y, r: str(tmp_path / f"{y}_{r}.json"))
    (tmp_path / "2024_6.json").write_text(json.dumps({
        "generated_at": "2026-10-01T10:00:00+00:00",
        "model": {"version": version},
        "predictions": preds,
    }))


def test_analysis_status_not_ready_without_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "_cache_path", lambda y, r: str(tmp_path / "none.json"))
    assert analysis.analysis_status(2024, 6) == {"ready": False, "generated_at": None, "top3": [], "hits": None}


def test_analysis_status_ignores_cache_of_other_model(tmp_path, monkeypatch):
    _write_cache(tmp_path, monkeypatch, "v0-old", [])
    assert analysis.analysis_status(2024, 6)["ready"] is False


def test_analysis_status_top3_and_hits(tmp_path, monkeypatch):
    preds = [
        {"driver": "A", "team": "T", "predicted_position": 2, "actual_position": 1, "p_dnf": 0.1},
        {"driver": "B", "team": "T", "predicted_position": 1, "actual_position": 5, "p_dnf": 0.1},
        {"driver": "C", "team": "T", "predicted_position": 3, "actual_position": 3, "p_dnf": 0.1},
        {"driver": "D", "team": "T", "predicted_position": 4, "actual_position": 2, "p_dnf": 0.1},
    ]
    _write_cache(tmp_path, monkeypatch, analysis.MODEL_VERSION, preds)
    s = analysis.analysis_status(2024, 6)
    assert s["ready"] is True
    assert [p["driver"] for p in s["top3"]] == ["B", "A", "C"]
    assert s["hits"] == 2


def test_analysis_status_hits_none_before_race(tmp_path, monkeypatch):
    preds = [{"driver": "A", "team": "T", "predicted_position": 1, "actual_position": None, "p_dnf": 0.1}]
    _write_cache(tmp_path, monkeypatch, analysis.MODEL_VERSION, preds)
    assert analysis.analysis_status(2024, 6)["hits"] is None


# ---------------------------------------------------------------------------
# Задача 6 + ендпоінти
# ---------------------------------------------------------------------------

def test_driver_media_missing_season_is_empty():
    assert event_info.season_media(1990) == {}
    assert event_info.driver_media(1990, "SEN") == {"headshot_url": None, "team_color": None}


@pytest.mark.parametrize("url", [
    "/api/event-summary/2024/6", "/api/event-standings/2024/6",
    "/api/circuit-history/2024/16", "/api/circuit-facts/2024/16",
    "/api/analysis/2024/6/status", "/api/driver-media/2024",
])
def test_endpoints_ok(client, url):
    res = client.get(url)
    assert res.status_code == 200
    assert res.json() is not None


def test_event_summary_future_race_endpoint_returns_null(client):
    res = client.get("/api/event-summary/2026/23")
    assert res.status_code == 200 and res.json() is None


@pytest.mark.parametrize("url", [
    "/api/event-summary/2024/99", "/api/event-standings/2024/99",
    "/api/circuit-history/2024/99", "/api/circuit-facts/2024/99",
])
def test_endpoints_unknown_round_404(client, url):
    assert client.get(url).status_code == 404

"""
Запасне джерело кіл/погоди (app/fallback.py) і перемикання на нього в
app/main.py, коли FastF1 не може завантажити сесію. Мережа не потрібна:
OpenF1 замокано, CSV-шлях працює на локальному датасеті.
"""
import pytest

from app import fallback, fastf1_client, main


@pytest.fixture(autouse=True)
def _reset_memo():
    main._NO_LIVE_TIMING.clear()
    yield
    main._NO_LIVE_TIMING.clear()


def _not_loaded(*args, **kwargs):
    raise RuntimeError("The data you are trying to access has not been loaded yet. See `Session.load`")


# ---------------------------------------------------------------------------
# Перемикання в main.py
# ---------------------------------------------------------------------------

def test_uses_fallback_when_fastf1_has_no_data(client, monkeypatch):
    calls = {"fastf1": 0}

    def fastf1_laps(*args, **kwargs):
        calls["fastf1"] += 1
        _not_loaded()

    monkeypatch.setattr(fastf1_client, "get_session_laps", fastf1_laps)
    monkeypatch.setattr(fallback, "get_session_laps", lambda *a, **k: [{"LapNumber": 1.0}])

    first = client.get("/api/laps/2026/17/R")
    second = client.get("/api/laps/2026/17/R")

    assert first.status_code == second.status_code == 200
    assert first.json() == [{"LapNumber": 1.0}]
    # Другий запит іде одразу в запасне джерело, без марної спроби FastF1.
    assert calls["fastf1"] == 1


def test_other_fastf1_errors_do_not_use_fallback(client, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("щось інше зламалось")

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("fallback не мав викликатись")

    monkeypatch.setattr(fastf1_client, "get_session_weather", broken)
    monkeypatch.setattr(fallback, "get_session_weather", must_not_be_called)

    resp = client.get("/api/weather/2024/1/R")

    assert resp.status_code == 404
    assert "щось інше зламалось" in resp.json()["detail"]


def test_explains_when_no_source_has_data(client, monkeypatch):
    monkeypatch.setattr(fastf1_client, "get_driver_summary", _not_loaded)

    def no_data(*args, **kwargs):
        raise fallback.FallbackUnavailable("OpenF1 має дані лише з 2023 року")

    monkeypatch.setattr(fallback, "get_driver_summary", no_data)

    resp = client.get("/api/driver/2019/1/FP1/HAM")

    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail.startswith("Дані пілота недоступні")
    assert "FastF1 більше не завантажує нові сесії" in detail
    assert "2023" in detail


# ---------------------------------------------------------------------------
# Стан траси з повідомлень дирекції гонки (OpenF1 race_control)
# ---------------------------------------------------------------------------

def test_track_status_by_lap():
    messages = [
        {"date": "1", "lap_number": 7, "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED"},
        {"date": "2", "lap_number": 9, "category": "SafetyCar", "message": "SAFETY CAR IN THIS LAP"},
        {"date": "3", "lap_number": 20, "category": "SafetyCar", "message": "VIRTUAL SAFETY CAR DEPLOYED"},
        {"date": "4", "lap_number": 21, "category": "SafetyCar", "message": "VIRTUAL SAFETY CAR ENDING"},
        {"date": "5", "lap_number": 30, "category": "Flag", "flag": "RED", "scope": "Track"},
        # З 2026 дирекція пише VSC скорочено.
        {"date": "7", "lap_number": 39, "category": "SafetyCar", "message": "VSC DEPLOYED"},
        {"date": "8", "lap_number": 40, "category": "SafetyCar", "message": "VSC ENDING"},
        {"date": "6", "lap_number": 31, "category": "Flag", "flag": "YELLOW", "scope": "Sector"},
    ]

    status = fallback._track_status_by_lap(messages)

    assert status == {7: "4", 8: "4", 9: "4", 20: "6", 21: "6", 30: "5", 39: "6", 40: "6"}


# ---------------------------------------------------------------------------
# CSV-шлях (гонки до 2023) - на реальному датасеті
# ---------------------------------------------------------------------------

def test_csv_laps_bahrain_2022_leclerc():
    laps = [l for l in fallback._csv_laps(2022, 1, "R") if l["Driver"] == "LEC"]

    assert len(laps) == 57
    assert [l["LapNumber"] for l in laps] == [float(n) for n in range(1, 58)]
    assert laps[-1]["Position"] == 1.0
    # Леклер тричі заїжджав у бокси - чотири стінти.
    assert sorted({l["Stint"] for l in laps}) == [1.0, 2.0, 3.0, 4.0]
    # Кола з піт-лейном і перше коло не рахуються як показові для темпу.
    assert laps[0]["IsAccurate"] is False
    assert all(not l["IsAccurate"] for l in laps if l["PitInTime"] or l["PitOutTime"])
    # Шин і секторів у датасеті немає.
    assert all(l["Compound"] is None and l["Sector1Time"] is None for l in laps)


def test_csv_has_only_races():
    with pytest.raises(fallback.FallbackUnavailable):
        fallback._csv_laps(2022, 1, "Q")

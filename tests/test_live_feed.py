"""
Тести live-режиму (app/live_feed.py) і токена OpenF1 (app/openf1_client.py).

Ні мережі, ні справжнього MQTT: брокер підмінено FakeFeed, у який тест сам
"публікує" записи, REST-частина (prepare_live) і календар - монкіпатчами.
"""
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import data_update, live_feed, openf1_client
from app.live_feed import LiveHub, LiveState, find_live_session
from app.main import app

T0 = datetime(2026, 12, 6, 13, 0, tzinfo=timezone.utc)


def at(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


SCHEDULE = [
    {"RoundNumber": 23, "Session5": "Race", "Session5DateUtc": "2026-11-29T16:00:00"},
    {"RoundNumber": 24, "Session1": "Practice 1", "Session1DateUtc": "2026-12-04T09:30:00",
     "Session5": "Race", "Session5DateUtc": "2026-12-06T13:00:00"},
]

SESSION = {"session_key": 9999, "session_name": "Race", "session_type": "Race",
           "date_start": at(0), "date_end": at(7200), "circuit_key": 70, "year": 2026}


# ---------------------------------------------------------------------------
# LiveState
# ---------------------------------------------------------------------------

def test_lap_is_forwarded_only_when_completed():
    s = LiveState()
    started = {"driver_number": 4, "lap_number": 12, "date_start": at(10), "lap_duration": None, "_key": "a", "_id": 1}
    assert s.ingest("laps", started) == []
    done = dict(started, lap_duration=81.2, _id=2)
    (event,) = s.ingest("laps", done)
    assert event["topic"] == "laps" and event["data"]["lap_duration"] == 81.2
    assert "_key" not in event["data"] and "_id" not in event["data"]
    assert s.ingest("laps", done) == []  # повтор


def test_unfinished_lap_is_flushed_when_next_lap_starts():
    s = LiveState()
    s.ingest("laps", {"driver_number": 4, "lap_number": 1, "date_start": at(0), "lap_duration": None})
    out = s.ingest("laps", {"driver_number": 4, "lap_number": 2, "date_start": at(95), "lap_duration": None})
    assert [e["data"]["lap_number"] for e in out] == [1]
    # коло іншого пілота не чіпається
    assert s.ingest("laps", {"driver_number": 81, "lap_number": 3, "date_start": at(96), "lap_duration": None}) == []


def test_state_topics_keep_latest_and_catchup_is_ordered():
    s = LiveState()
    s.ingest("position", {"date": at(5), "driver_number": 4, "position": 2})
    s.ingest("position", {"date": at(9), "driver_number": 4, "position": 1})
    s.ingest("race_control", {"date": at(7), "message": "GREEN LIGHT"})
    s.ingest("weather", {"date": at(1), "air_temperature": 24})
    assert s.ingest("race_control", {"date": at(7), "message": "GREEN LIGHT"}) == []
    s.ingest("location", {"date": at(11), "driver_number": 4, "x": 1, "y": 2})  # не в catch-up
    catchup = s.catchup()
    assert [(e["topic"], e["data"].get("position")) for e in catchup] == [
        ("weather", None), ("race_control", None), ("position", 1),
    ]
    assert s.clock == T0 + timedelta(seconds=11)


# ---------------------------------------------------------------------------
# Пошук live-сесії
# ---------------------------------------------------------------------------

def test_find_live_session_window_upcoming_and_cancelled(monkeypatch):
    fp1 = dict(SESSION, session_key=1, date_start=at(-86400), date_end=at(-82800))
    cancelled = dict(SESSION, session_key=2, date_start=at(-600), date_end=at(3000), is_cancelled=True)
    race = dict(SESSION, session_key=3)
    monkeypatch.setattr(openf1_client, "_season_sessions", lambda year: [fp1, cancelled, race] if year == 2026 else [])
    assert find_live_session(T0 - timedelta(minutes=29)) == ("live", race)       # вікно відкрите за 30 хв до старту
    assert find_live_session(T0 - timedelta(minutes=31)) == ("upcoming", race)
    assert find_live_session(T0 + timedelta(hours=2, minutes=29)) == ("live", race)
    assert find_live_session(T0 + timedelta(hours=3)) == ("none", None)


# ---------------------------------------------------------------------------
# Токен OpenF1
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setenv("OPENF1_USERNAME", "me@example.com")
    monkeypatch.setenv("OPENF1_PASSWORD", "secret")
    monkeypatch.setattr(openf1_client, "_token", None)
    monkeypatch.setattr(openf1_client, "_token_expires", 0.0)
    monkeypatch.setattr(openf1_client, "_token_failed_until", 0.0)
    monkeypatch.setattr(openf1_client, "_respect_limits", lambda: None)


def test_token_is_cached_and_refreshed_before_expiry(creds, monkeypatch):
    issued = []

    def post(url, data, timeout):
        issued.append(data["username"])
        return FakeResponse(200, {"access_token": f"tok{len(issued)}", "expires_in": "3600"})

    monkeypatch.setattr(openf1_client.requests, "post", post)
    assert openf1_client.get_token() == "tok1"
    assert openf1_client.get_token() == "tok1"
    monkeypatch.setattr(openf1_client, "_token_expires", time.time() + 100)  # < 5 хв до кінця
    assert openf1_client.get_token() == "tok2"
    assert issued == ["me@example.com", "me@example.com"]


def test_no_credentials_means_no_token(monkeypatch):
    monkeypatch.delenv("OPENF1_USERNAME", raising=False)
    monkeypatch.delenv("OPENF1_PASSWORD", raising=False)
    assert openf1_client.configured() is False
    assert openf1_client.get_token() is None


def test_get_sends_bearer_and_retries_once_after_401(creds, monkeypatch):
    tokens = iter(["old", "new"])
    monkeypatch.setattr(openf1_client.requests, "post",
                        lambda url, data, timeout: FakeResponse(200, {"access_token": next(tokens), "expires_in": 3600}))
    seen = []

    def get(url, headers, timeout):
        seen.append(headers.get("Authorization"))
        return FakeResponse(401) if len(seen) == 1 else FakeResponse(200, [{"x": 1}])

    monkeypatch.setattr(openf1_client.requests, "get", get)
    assert openf1_client._get("location", {"session_key": 1}, disk_cache=False) == [{"x": 1}]
    assert seen == ["Bearer old", "Bearer new"]


# ---------------------------------------------------------------------------
# Хаб і WS /ws/live
# ---------------------------------------------------------------------------

class FakeFeed:
    instances = []

    def __init__(self, on_record):
        self.on_record = on_record
        self.started = self.stopped = False
        FakeFeed.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def publish(self, topic, record):
        # як paho - з чужого потоку
        t = threading.Thread(target=self.on_record, args=(topic, dict(record, session_key=SESSION["session_key"])))
        t.start()
        t.join()


INTRO = {"type": "session", "mode": "live", "session": {"session_key": 9999}, "drivers": [], "track": None, "grid": None, "replay": None}


@pytest.fixture
def live(monkeypatch):
    FakeFeed.instances.clear()
    now = [T0 + timedelta(minutes=10)]
    hub = LiveHub(feed_factory=FakeFeed, now=lambda: now[0], batch_seconds=0.02, idle_stop_seconds=0.05)
    monkeypatch.setattr(live_feed, "hub", hub)
    monkeypatch.setattr(openf1_client, "configured", lambda: True)
    monkeypatch.setattr(live_feed, "find_live_session", lambda _now: ("live", SESSION))
    initial = [("laps", {"driver_number": 4, "lap_number": 1, "date_start": at(60), "lap_duration": 90.0}),
               ("position", {"date": at(100), "driver_number": 4, "position": 1})]
    monkeypatch.setattr(live_feed, "prepare_live", lambda session: (INTRO, initial))
    monkeypatch.setattr(data_update, "start_background_updates", lambda: None)
    return hub, now


def receive_until(ws, predicate, limit=200):
    for _ in range(limit):
        message = ws.receive_json()
        if predicate(message):
            return message
    raise AssertionError("очікуване повідомлення не прийшло")


def test_ws_live_streams_to_several_viewers_over_one_feed(live):
    hub, now = live
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as a:
            assert a.receive_json() == INTRO
            catchup = a.receive_json()
            assert catchup["catchup"] is True
            assert [e["topic"] for e in catchup["events"]] == ["laps", "position"]
            (feed,) = FakeFeed.instances
            assert feed.started

            feed.publish("location", {"date": at(120), "driver_number": 4, "x": 5, "y": 6, "_id": 7})
            batch = receive_until(a, lambda m: m["type"] == "batch")
            assert batch["events"] == [{"topic": "location", "data": {"date": at(120), "driver_number": 4, "x": 5, "y": 6}}]
            assert batch["clock"] == at(120)

            with client.websocket_connect("/ws/live") as b:
                b.receive_json()
                assert [e["topic"] for e in b.receive_json()["events"]] == ["laps", "position"]
                assert len(FakeFeed.instances) == 1  # другий глядач - те саме підключення
                feed.publish("race_control", {"date": at(130), "message": "SAFETY CAR DEPLOYED", "category": "SafetyCar"})
                for ws in (a, b):
                    msg = receive_until(ws, lambda m: m["type"] == "batch")
                    assert msg["events"][0]["data"]["message"] == "SAFETY CAR DEPLOYED"

            now[0] = T0 + timedelta(hours=2, minutes=31)  # live-вікно закрилось
            assert receive_until(a, lambda m: m["type"] == "end")["type"] == "end"
    assert feed.stopped


def test_ws_live_waiting_and_not_configured(monkeypatch):
    monkeypatch.setattr(live_feed.fastf1_client, "get_season_schedule", lambda year: SCHEDULE)
    monkeypatch.setattr(openf1_client, "configured", lambda: False)
    client = TestClient(app)
    with client.websocket_connect("/ws/live?lang=en") as ws:
        message = ws.receive_json()
    assert message["type"] == "error" and "OPENF1_USERNAME" in message["detail"]

    monkeypatch.setattr(openf1_client, "configured", lambda: True)
    monkeypatch.setattr(live_feed, "find_live_session", lambda _now: ("upcoming", SESSION))
    with client.websocket_connect("/ws/live") as ws:
        message = ws.receive_json()
    assert message == {"type": "waiting", "status": "upcoming", "opens_at": at(-1800), "calendar": {"year": 2026, "round": 24, "session": "R", "event": None},
                       "session": {k: SESSION.get(k) for k in live_feed.live_replay._INTRO_SESSION_FIELDS}}


def test_live_status_endpoint(monkeypatch):
    monkeypatch.setattr(live_feed.fastf1_client, "get_season_schedule", lambda year: SCHEDULE)
    previous_race = dict(SESSION, session_key=8888, date_start="2026-11-29T16:00:00+00:00", date_end="2026-11-29T18:00:00+00:00")
    monkeypatch.setattr(openf1_client, "_season_sessions", lambda year: [previous_race, SESSION] if year == 2026 else [])
    monkeypatch.setattr(openf1_client, "configured", lambda: False)  # таймер видно й без підписки
    body = live_feed.live_status(now=T0 - timedelta(days=1))
    assert TestClient(app).get("/api/live/status").status_code == 200
    assert body["configured"] is False and body["status"] == "upcoming"
    assert body["opens_at"] == at(-1800) and body["session"]["session_key"] == 9999
    assert body["calendar"] == {"year": 2026, "round": 24, "session": "R", "event": None}
    assert body["latest_replay"] == {"year": 2026, "round": 23, "session": "R", "event": None}


def test_failed_token_falls_back_to_anonymous_requests(creds, monkeypatch):
    posts = []

    def post(url, data, timeout):
        posts.append(1)
        return FakeResponse(401)  # неправильний пароль

    monkeypatch.setattr(openf1_client.requests, "post", post)
    seen = []
    monkeypatch.setattr(openf1_client.requests, "get",
                        lambda url, headers, timeout: seen.append(headers) or FakeResponse(200, [{"ok": 1}]))
    assert openf1_client._get("sessions", {"year": 2026}, disk_cache=False) == [{"ok": 1}]
    assert seen == [{}]
    openf1_client._get("sessions", {"year": 2025}, disk_cache=False)
    assert len(posts) == 1  # невдачу запам'ятали - /token не смикаємо щоразу


def test_team_radio_is_deduplicated_and_in_catchup():
    s = LiveState()
    clip = {"date": at(30), "driver_number": 44, "recording_url": "https://x/HAM_44.mp3", "_key": "k"}
    assert [e["topic"] for e in s.ingest("team_radio", clip)] == ["team_radio"]
    assert s.ingest("team_radio", dict(clip, _id=9)) == []
    assert [e["data"]["recording_url"] for e in s.catchup() if e["topic"] == "team_radio"] == ["https://x/HAM_44.mp3"]

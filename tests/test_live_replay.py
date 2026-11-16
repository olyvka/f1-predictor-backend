"""
Тести replay сесії (app/live_replay.py) і WebSocket /ws/replay.

Мережі немає: OpenF1 підмінений фейковими записами, а годинник або
фейковий (для паузи/швидкості), або справжній із мікро-тіком і великою
швидкістю (щоб replay на кілька секунд сесії програвався за частку секунди).
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import live_replay, openf1_client
from app.live_replay import ReplayPlayer, apply_command, build_timeline, replay_end
from app.main import app

T0 = datetime(2026, 11, 8, 17, 0, tzinfo=timezone.utc)


def at(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


# ---------------------------------------------------------------------------
# build_timeline / replay_end
# ---------------------------------------------------------------------------

def test_timeline_orders_dated_topics_and_drops_session_keys():
    small = {
        "race_control": [{"date": at(5), "message": "GREEN", "session_key": 1, "meeting_key": 2}],
        "position": [{"date": at(1), "driver_number": 4, "position": 1, "session_key": 1}],
    }
    timeline = build_timeline(small, T0)
    assert [e[2] for e in timeline] == ["position", "race_control"]
    assert timeline[1][3] == {"date": at(5), "message": "GREEN"}


def test_timeline_lap_arrives_when_completed():
    laps = [
        # є тривалість - коло приходить на date_start + lap_duration
        {"driver_number": 4, "lap_number": 2, "date_start": at(100), "lap_duration": 70.5},
        # немає тривалості - на старті наступного кола
        {"driver_number": 4, "lap_number": 1, "date_start": at(10), "lap_duration": None},
        # немає ні тривалості, ні наступного кола - на власному старті
        {"driver_number": 4, "lap_number": 3, "date_start": at(170.5), "lap_duration": None},
    ]
    times = {e[3]["lap_number"]: e[0] for e in build_timeline({"laps": laps}, T0)}
    assert times == {1: T0 + timedelta(seconds=100), 2: T0 + timedelta(seconds=170.5), 3: T0 + timedelta(seconds=170.5)}


def test_timeline_stint_arrives_at_its_first_lap_or_session_start():
    small = {
        "laps": [{"driver_number": 4, "lap_number": 15, "date_start": at(900), "lap_duration": 70}],
        "stints": [
            {"driver_number": 4, "stint_number": 2, "lap_start": 15},
            {"driver_number": 81, "stint_number": 1, "lap_start": 1},  # кола 1 немає в даних
        ],
    }
    stints = {e[3]["driver_number"]: e[0] for e in build_timeline(small, T0) if e[2] == "stints"}
    assert stints == {4: T0 + timedelta(seconds=900), 81: T0}


def test_replay_end_extends_for_overrun_but_is_capped():
    session = {"date_start": at(0), "date_end": at(7200)}
    late = [(T0 + timedelta(hours=2, minutes=40), 0, "race_control", {})]
    assert replay_end(session, late) == T0 + timedelta(hours=2, minutes=40)
    very_late = [(T0 + timedelta(hours=9), 0, "race_control", {})]
    assert replay_end(session, very_late) == T0 + timedelta(hours=2) + live_replay.MAX_OVERRUN
    assert replay_end(session, []) == T0 + timedelta(hours=2)


# ---------------------------------------------------------------------------
# ReplayPlayer
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def no_stream(topic, since, until):
    return []


def test_catch_up_keeps_only_latest_state_per_driver():
    small = {
        "intervals": [{"date": at(s), "driver_number": d, "interval": s} for s in (1, 2, 3) for d in (4, 81)],
        "weather": [{"date": at(1), "air_temperature": 20}, {"date": at(2), "air_temperature": 21}],
        "race_control": [{"date": at(1), "message": "A"}, {"date": at(2), "message": "B"}],
        "position": [{"date": at(50), "driver_number": 4, "position": 1}],  # уже після старту
    }
    player = ReplayPlayer(build_timeline(small, T0), no_stream, T0 + timedelta(seconds=10), T0 + timedelta(seconds=60))
    events = player.catch_up()
    assert sorted((e[2], e[3].get("driver_number"), e[3].get("interval")) for e in events if e[2] == "intervals") == [
        ("intervals", 4, 3), ("intervals", 81, 3),
    ]
    assert [e[3]["air_temperature"] for e in events if e[2] == "weather"] == [21]
    assert [e[3]["message"] for e in events if e[2] == "race_control"] == ["A", "B"]
    assert all(e[2] != "position" for e in events)


def test_clock_speed_and_pause_do_not_jump():
    clock = FakeClock()
    player = ReplayPlayer([], no_stream, T0, T0 + timedelta(hours=2), speed=2, clock=clock)
    clock.t += 10
    assert player.now() == T0 + timedelta(seconds=20)

    player.pause()
    clock.t += 100
    assert player.now() == T0 + timedelta(seconds=20)

    player.resume()
    player.set_speed(5)
    clock.t += 10
    assert player.now() == T0 + timedelta(seconds=70)

    player.set_speed(1000)  # обмежується MAX_SPEED
    assert player.speed == live_replay.MAX_SPEED
    clock.t += 10_000
    assert player.now() == player.end  # не далі кінця сесії


def test_run_delivers_every_event_once_in_order():
    small = {"race_control": [{"date": at(s), "message": str(s)} for s in (0, 12, 31, 59)]}
    calls = []

    def stream(topic, since, until):
        calls.append((topic, since, until))
        # Межі запиту включні: запис рівно на межі вікна OpenF1 віддасть
        # обом сусіднім вікнам - гравець має надіслати його один раз.
        seconds = range(int((since - T0).total_seconds()), int((until - T0).total_seconds()) + 1, 5)
        return [{"date": at(s), "driver_number": 4, "x": s, "session_key": 1} for s in seconds]

    end = T0 + timedelta(seconds=60)
    player = ReplayPlayer(build_timeline(small, T0), stream, T0, end, speed=20, tick=0.001, window=timedelta(seconds=30))
    sent = []

    async def send(message):
        sent.append(message)

    # 60 с сесії на ×20 - це ~3 с. Прискорюємо: справжній годинник, але
    # кожен виклик "проходить" 1 с.
    fake = FakeClock()

    def ticking():
        fake.t += 1
        return fake.t

    player._clock = ticking
    asyncio.run(asyncio.wait_for(player.run(send), timeout=10))

    assert sent[0]["catchup"] is True
    assert [e["data"]["message"] for e in sent[0]["events"]] == ["0"]
    assert sent[-1]["type"] == "end"
    events = [e for m in sent[1:-1] for e in m["events"]]
    xs = [e["data"]["x"] for e in events if e["topic"] == "location"]
    assert xs == list(range(0, 61, 5))  # кожен запис рівно раз, разом з кінцем сесії
    assert [e["data"]["message"] for e in events if e["topic"] == "race_control"] == ["12", "31", "59"]
    dates = [e["data"]["date"] for e in events]
    assert dates == sorted(dates)
    assert all("session_key" not in e["data"] for e in events)
    assert {(c[1], c[2]) for c in calls} == {
        (T0, T0 + timedelta(seconds=30)), (T0 + timedelta(seconds=30), end),
    }


@pytest.mark.parametrize(
    "message, ok",
    [
        ({"action": "pause"}, True),
        ({"action": "resume"}, True),
        ({"action": "speed", "value": 5}, True),
        ({"action": "speed", "value": "fast"}, False),
        ({"action": "speed"}, False),
        ({"action": "seek"}, False),
        (None, False),
        ([1, 2], False),
    ],
)
def test_apply_command(message, ok):
    player = ReplayPlayer([], no_stream, T0, T0 + timedelta(hours=1))
    assert (apply_command(player, message) is None) is ok


# ---------------------------------------------------------------------------
# WebSocket /ws/replay
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_session(monkeypatch):
    session = {"session_key": 9, "session_name": "Race", "date_start": at(0), "date_end": at(4)}
    seen = {}

    def find_session(year, round_number, name):
        seen["args"] = (year, round_number, name)
        return session

    def prepare(info, offset, speed):
        seen["prepare"] = (info, offset, speed)
        small = {"race_control": [{"date": at(2), "message": "GREEN"}]}
        player = ReplayPlayer(build_timeline(small, T0), no_stream, T0, T0 + timedelta(seconds=4), speed, tick=0.01)
        return {"type": "session", "mode": "replay", "session": info}, player

    monkeypatch.setattr(openf1_client, "find_session", find_session)
    monkeypatch.setattr(live_replay, "prepare_replay", prepare)
    return seen


def test_ws_replay_streams_until_end(fake_session):
    client = TestClient(app)
    with client.websocket_connect("/ws/replay/2026/20/R?speed=20&offset=-30") as ws:
        assert ws.receive_json()["type"] == "session"
        assert ws.receive_json()["catchup"] is True
        messages = []
        while not messages or messages[-1]["type"] != "end":
            messages.append(ws.receive_json())
    assert fake_session["args"] == (2026, 20, "R")
    assert fake_session["prepare"][1:] == (-30.0, 20.0)
    assert [e["data"]["message"] for m in messages[:-1] for e in m["events"]] == ["GREEN"]


def test_ws_replay_answers_commands(fake_session):
    client = TestClient(app)
    with client.websocket_connect("/ws/replay/2026/20/R") as ws:
        ws.receive_json()
        ws.receive_json()
        ws.send_json({"action": "pause"})
        state = next(m for m in iter(ws.receive_json, None) if m["type"] != "batch")
        assert state["type"] == "state" and state["paused"] is True

        ws.send_text("not json")
        error = next(m for m in iter(ws.receive_json, None) if m["type"] != "batch")
        assert error["type"] == "error"


def test_ws_replay_reports_unknown_session(monkeypatch):
    def missing(*_):
        raise openf1_client.OpenF1Unavailable("no such session")

    monkeypatch.setattr(openf1_client, "find_session", missing)
    client = TestClient(app)
    with client.websocket_connect("/ws/replay/2019/1/R?lang=en") as ws:
        message = ws.receive_json()
    assert message["type"] == "error"
    assert message["detail"] == "Could not start the session replay: no such session"


def test_starting_grid_takes_first_position_of_race_only():
    positions = [
        {"date": at(600), "driver_number": 4, "position": 2},
        {"date": at(-3000), "driver_number": 4, "position": 1},
        {"date": at(-3000), "driver_number": 81, "position": 2},
        {"date": at(700), "driver_number": 81, "position": 1},
    ]
    assert live_replay.starting_grid({"session_type": "Race"}, positions) == {4: 1, 81: 2}
    assert live_replay.starting_grid({"session_type": "Qualifying"}, positions) is None
    assert live_replay.starting_grid({"session_type": "Race"}, []) is None


def test_windows_are_aligned_to_session_start_and_cached_across_players(monkeypatch):
    calls = []

    def session_window(session, topic, since, until, disk_cache=True):
        calls.append((topic, since, until))
        return [{"date": since.isoformat(), "driver_number": 4, "x": 1}]

    monkeypatch.setattr(openf1_client, "session_window", session_window)
    monkeypatch.setattr(live_replay, "_WINDOW_CACHE", live_replay.OrderedDict())
    session = {"session_key": 77}
    fetch = live_replay._fetch_window(session)
    # Старт посеред вікна: тягнеться все вирівняне вікно, записи до старту відкидаються.
    player = ReplayPlayer([], fetch, T0 + timedelta(seconds=90), T0 + timedelta(hours=1), align_base=T0)
    records, until = player._load_window(T0 + timedelta(seconds=90))
    assert until == T0 + timedelta(seconds=120)
    assert records == []  # єдиний запис - на початку вікна (60 с), раніше за старт
    assert {(c[1], c[2]) for c in calls} == {(T0 + timedelta(seconds=60), T0 + timedelta(seconds=120))}
    # Другий глядач того ж вікна не робить нових запитів.
    other = ReplayPlayer([], fetch, T0 + timedelta(seconds=70), T0 + timedelta(hours=1), align_base=T0)
    records, _ = other._load_window(T0 + timedelta(seconds=70))
    assert len(calls) == 2  # по одному запиту на location і car_data


def test_openf1_client_waits_instead_of_exceeding_per_minute_limit(monkeypatch):
    now = [1000.0]
    slept = []
    monkeypatch.setattr(openf1_client.time, "monotonic", lambda: now[0])

    def sleep(s):
        slept.append(round(s, 2))
        now[0] += s

    monkeypatch.setattr(openf1_client.time, "sleep", sleep)
    monkeypatch.setattr(openf1_client, "_PER_MINUTE", 3)
    monkeypatch.setattr(openf1_client, "_recent", openf1_client.collections.deque([970.0, 980.0, 990.0]))
    monkeypatch.setattr(openf1_client, "_last_request", 990.0)
    openf1_client._respect_limits()
    # Найстаріший запит (970) випадає з хвилини о 1030: чекаємо ~30 с, а не отримуємо 429.
    assert slept == [30.05]
    assert list(openf1_client._recent) == [980.0, 990.0]

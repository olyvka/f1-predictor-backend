"""
Регресійний тест на потокобезпеку: FastF1 не розрахований на одночасний
доступ з кількох потоків (SQLite HTTP-кеш, pickle-кеш), а Starlette виконує
кожен синхронний `def`-ендпоінт у власному потоці зі свого threadpool. Без
_FASTF1_LOCK у app/fastf1_client.py кілька одночасних запитів можуть валити
процес по SIGSEGV усередині libsqlite3 (реальний крах, який стався на проді).

Тест підміняє fastf1.get_session() функцією, що навмисно "тримає" виконання
(time.sleep) і фіксує проміжок часу свого виконання, і б'є в _load_session()
з кількох потоків одночасно. Якщо лок працює - жоден із зафіксованих
проміжків не перетинається (звернення до FastF1 виконуються строго по черзі).
"""
import time
from concurrent.futures import ThreadPoolExecutor

from app import fastf1_client


def _intervals_overlap(a, b):
    (a_start, a_end), (b_start, b_end) = a, b
    return a_start < b_end and b_start < a_end


def test_load_session_serializes_concurrent_calls(monkeypatch):
    intervals = []
    intervals_lock = __import__("threading").Lock()

    class _FakeSession:
        def load(self, **kwargs):
            pass

    def fake_get_session(year, round_number, session):
        start = time.monotonic()
        time.sleep(0.05)  # імітує повільний мережевий/кеш-виклик FastF1
        end = time.monotonic()
        with intervals_lock:
            intervals.append((start, end))
        return _FakeSession()

    monkeypatch.setattr(fastf1_client.fastf1, "get_session", fake_get_session)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(fastf1_client._load_session, 2024, 1, "R") for _ in range(8)
        ]
        for f in futures:
            f.result()

    assert len(intervals) == 8
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            assert not _intervals_overlap(intervals[i], intervals[j]), (
                "два виклики fastf1.get_session() виконувались одночасно - "
                "_FASTF1_LOCK не серіалізує доступ, ризик SIGSEGV як на проді"
            )

"""
Завантаження історії практик і погоди з FastF1 у data/fastf1_*.csv
(агрегати - app/ml/fastf1_features.py). Потрібно один раз для історії +
після кожного вікенду; далі модель вчиться з CSV без мережі.

    python scripts/fetch_fastf1.py                     # 2018..поточний рік, усі проходи
    python scripts/fetch_fastf1.py --years 2023-2024 --passes 1

FastF1 обмежує 500 запитів/год (~8 на сесію з колами), тому сесії йдуть
проходами за цінністю, щоб перший прохід уже був придатний для навчання:
  1 - головна практика вікенду: FP2 (довгі серії), на спринт-вікендах FP1;
  2 - погода кваліфікації (без кіл - дешево);
  3 - решта практик.
Скрипт відновлюваний: кожна сесія пишеться в data/fastf1_fetch_log.csv,
ok/unavailable при повторному запуску пропускаються, error - повторюються
(до MAX_ATTEMPTS разів). Ліміт 500 запитів/год FastF1 при перевищенні
лише пише в лог, а сесія лишається без кіл - тому між сесіями є пауза
(SESSION_PAUSE, ~60 сесій/год), а після кількох невдач поспіль -
наростаюча перерва. Уся історія 2018+ - ~10 год, прохід 1 - ~3 год.
"""
import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastf1  # noqa: E402
import pandas as pd  # noqa: E402
from fastf1.req import RateLimitExceededError  # noqa: E402

from app import fastf1_client  # noqa: E402,F401 (кеш FastF1 + timeout-патч requests)
from app.ml import config  # noqa: E402
from app.ml.fastf1_features import load_session_aggregates  # noqa: E402

PRACTICE_CSV = os.path.join(config.DATA_DIR, "fastf1_practice.csv")
WEATHER_CSV = os.path.join(config.DATA_DIR, "fastf1_weather.csv")
LOG_CSV = os.path.join(config.DATA_DIR, "fastf1_fetch_log.csv")

PRACTICE_NAMES = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3"}
RATE_LIMIT_SLEEP = 10 * 60
# FastF1 сам обмежує 500 запитів/год (і, досягнувши ліміту, мовчки лишає
# сесію без кіл), а сесія з колами - ~8 запитів: 60 с між сесіями = ~480/год.
SESSION_PAUSE = 60
MAX_ATTEMPTS = 3           # стільки error поспіль для сесії - і вона пропускається
BACKOFF_AFTER_ERRORS = 3   # стільки невдач поспіль - перерва BACKOFF_BASE * 2^n
BACKOFF_BASE = 5 * 60


def _append(path: str, rows: pd.DataFrame) -> None:
    rows.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _done() -> set:
    if not os.path.exists(LOG_CSV):
        return set()
    log = pd.read_csv(LOG_CSV)
    key = ["year", "round", "session"]
    finished = log[log["status"].isin(["ok", "unavailable"])]
    errors = log[log["status"] == "error"].groupby(key).size()
    return set(zip(*[finished[k] for k in key])) | set(errors[errors >= MAX_ATTEMPTS].index)


def _tasks(years: list[int], passes: set[int]) -> list[tuple]:
    """(pass, year, round, session_code, with_laps) для вікендів, що вже минули."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    tasks = []
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for ev in schedule.itertuples():
            if pd.isna(ev.EventDate) or ev.EventDate >= now:
                continue
            names = [getattr(ev, f"Session{i}") for i in range(1, 6)]
            practices = [PRACTICE_NAMES[n] for n in names if n in PRACTICE_NAMES]
            if not practices:
                continue
            main = "FP2" if ev.EventFormat == "conventional" and "FP2" in practices else practices[0]
            rnd = int(ev.RoundNumber)
            tasks.append((1, year, rnd, main, True))
            if "Qualifying" in names:
                tasks.append((2, year, rnd, "Q", False))
            tasks += [(3, year, rnd, p, True) for p in practices if p != main]
    return sorted(t for t in tasks if t[0] in passes)


def _fetch(year: int, rnd: int, code: str, with_laps: bool) -> tuple[str, str]:
    agg, weather, status, msg = load_session_aggregates(year, rnd, code, with_laps)
    base = {"year": year, "round": rnd, "session": code}
    _append(WEATHER_CSV, pd.DataFrame([{**base, **weather}]))
    if agg is not None:
        _append(PRACTICE_CSV, agg.assign(**base)[list(base) + list(agg.columns)])
    return status, msg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", default=f"2018-{datetime.now().year}")
    parser.add_argument("--passes", default="1,2,3")
    args = parser.parse_args()
    start, _, end = args.years.partition("-")
    years = list(range(int(start), int(end or start) + 1))
    passes = {int(p) for p in args.passes.split(",")}

    fastf1.set_log_level("ERROR")
    done = _done()
    tasks = [t for t in _tasks(years, passes) if (t[1], t[2], t[3]) not in done]
    print(f"До завантаження: {len(tasks)} сесій", flush=True)

    consecutive_errors, backoffs = 0, 0
    for i, (p, year, rnd, code, with_laps) in enumerate(tasks, 1):
        if consecutive_errors >= BACKOFF_AFTER_ERRORS:
            pause = BACKOFF_BASE * 2 ** min(backoffs, 3)
            print(f"  {consecutive_errors} невдач поспіль - перерва {pause // 60} хв", flush=True)
            time.sleep(pause)
            consecutive_errors, backoffs = 0, backoffs + 1
        while True:
            t0 = time.monotonic()
            try:
                status, msg = _fetch(year, rnd, code, with_laps)
                break
            except RateLimitExceededError:
                print(f"  ліміт FastF1 - чекаю {RATE_LIMIT_SLEEP // 60} хв", flush=True)
                time.sleep(RATE_LIMIT_SLEEP)
            except Exception as e:
                status, msg = "error", f"{type(e).__name__}: {e}"[:300]
                break
        _append(LOG_CSV, pd.DataFrame([{
            "year": year, "round": rnd, "session": code, "status": status, "message": msg,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }]))
        print(f"[{i}/{len(tasks)}] прохід {p} {year} R{rnd:>2} {code}: {status} "
              f"({time.monotonic() - t0:.1f}с) {msg}", flush=True)
        if status == "error":
            consecutive_errors += 1
        else:
            consecutive_errors, backoffs = 0, 0
        time.sleep(SESSION_PAUSE)


if __name__ == "__main__":
    main()

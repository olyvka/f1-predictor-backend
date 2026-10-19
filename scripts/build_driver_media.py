"""
Довідник фото й кольорів пілотів data/driver_media.json з OpenF1
(api.openf1.org: /v1/drivers?session_key=... має headshot_url і
team_colour). OpenF1 покриває сезони з 2023 - для раніших фото не буде.

    python scripts/build_driver_media.py            # 2023..поточний рік
    python scripts/build_driver_media.py 2025 2026  # діапазон

Формат: {"source": "OpenF1", "generated_at": ...,
         "seasons": {"2024": {"VER": {"headshot_url": ..., "team_color": "#3671C6",
                                      "team": "Red Bull Racing"}}}}
На сезон береться найсвіжіше значення з гонок сезону (переходи між
командами посеред сезону - за останньою гонкою пілота). Бекенд читає
файл при старті (app/event_info.py) - після оновлення треба рестарт.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "driver_media.json")
API = "https://api.openf1.org/v1"
PAUSE = 2.1  # с між запитами - OpenF1 без ключа пускає ~30 запитів/хв


def get(path: str, **params):
    for attempt in range(5):
        res = requests.get(f"{API}/{path}", params=params, timeout=(10, 30))
        if res.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        if res.status_code == 404:  # OpenF1 так відповідає, коли для сесії даних нема
            time.sleep(PAUSE)
            return []
        res.raise_for_status()
        time.sleep(PAUSE)
        return res.json()
    raise RuntimeError(f"OpenF1: забагато запитів ({path})")


def season(year: int) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    sessions = sorted((s for s in get("sessions", year=year, session_name="Race") if s["date_start"] < now),
                      key=lambda s: s["date_start"])
    drivers = {}
    for s in sessions:
        for d in get("drivers", session_key=s["session_key"]):
            code = d.get("name_acronym")
            if not code:
                continue
            entry = drivers.setdefault(code, {"headshot_url": None, "team_color": None, "team": None})
            if d.get("headshot_url"):
                entry["headshot_url"] = d["headshot_url"]
            if d.get("team_colour"):
                entry["team_color"] = f"#{d['team_colour'].upper()}"
            if d.get("team_name"):
                entry["team"] = d["team_name"]
    print(f"{year}: {len(sessions)} гонок, {len(drivers)} пілотів", flush=True)
    return drivers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("years", nargs="*", type=int, help="рік або діапазон: 2024 / 2023 2026")
    args = parser.parse_args()
    years = args.years or [2023, datetime.now().year]
    years = list(range(years[0], years[-1] + 1))

    data = {"seasons": {}}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            data = json.load(f)
    for y in years:
        data["seasons"][str(y)] = season(y)
        data["source"] = "OpenF1 (api.openf1.org)"
        data["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(OUT, "w", encoding="utf-8") as f:  # після кожного сезону - не губимо прогрес
            json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"Записано {OUT}")


if __name__ == "__main__":
    main()

"""
Прогрів локального кешу FastF1 (cache/fastf1_http_cache.sqlite) для
цілого сезону наперед - щоб перший реальний запит через API вже брав
дані з диска, а не чекав мережу.

Легкий обсяг: розклад сезону + результати й погода кожної сесії
кожного етапу (без кіл/лапів - вони суттєво важчі й довше тягнуться).

Запуск з кореня проєкту (venv має бути активований):
    python scripts/warm_cache.py            # сезон 2025
    python scripts/warm_cache.py 2023 2025  # діапазон років
"""
import sys
import time
from pathlib import Path

# Дозволяє запускати скрипт напряму (python scripts/warm_cache.py),
# не встановлюючи проєкт як пакет.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import fastf1_client  # noqa: E402 (застосовує й timeout-патч на requests)


def warm_season(year: int) -> None:
    print(f"\n=== Сезон {year} ===")
    schedule = fastf1_client.get_season_schedule(year)
    print(f"Етапів у календарі: {len(schedule)}")

    ok, failed = 0, 0
    t_start = time.monotonic()

    for event in schedule:
        round_number = event["RoundNumber"]
        event_name = event["EventName"]
        try:
            sessions = fastf1_client.get_event_sessions(year, round_number)
        except Exception as e:
            print(f"  [{round_number:>2}] {event_name}: не вдалось отримати сесії - {e}")
            failed += 1
            continue

        for session in sessions:
            code = session["code"]
            label = f"  [{round_number:>2}] {event_name} / {code}"
            t0 = time.monotonic()
            try:
                fastf1_client.get_session_results(year, round_number, code)
                fastf1_client.get_session_weather(year, round_number, code)
            except Exception as e:
                print(f"{label}: ПРОПУЩЕНО ({e})")
                failed += 1
                continue
            print(f"{label}: OK ({time.monotonic() - t0:.1f}с)")
            ok += 1

    elapsed = time.monotonic() - t_start
    print(
        f"\nСезон {year} готовий: {ok} сесій закешовано, "
        f"{failed} пропущено, {elapsed / 60:.1f} хв."
    )


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or [2025]
    if len(years) == 2:
        years = list(range(years[0], years[1] + 1))

    for y in years:
        warm_season(y)

    cache_dir = Path(fastf1_client.CACHE_DIR)
    size_mb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"\nРозмір кеша ({cache_dir}): {size_mb:.1f} МБ")

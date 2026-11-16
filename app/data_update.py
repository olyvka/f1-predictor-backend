"""
Автоматичне оновлення data/*.csv з Kaggle (датасет jtrotman/formula-1-race-data -
та сама Ergast-схема, оновлюється після кожного етапу).

Сервер перевіряє нову версію при старті й далі кожні DATA_UPDATE_HOURS
годин (фоновий потік, див. start_background_updates() і lifespan у
app/main.py). Публічний датасет качається анонімно - ключ Kaggle не потрібен.

Що відбувається при новій версії:
  1. Перевірка: є всі потрібні файли, колонки ті самі, жоден файл не
     "схуд" (обрізаний реліз не має затерти дані). Не пройшла - нічого не
     змінюємо, лише пишемо помилку в data/kaggle_meta.json.
  2. Змінені CSV атомарно (os.replace) підміняють data/*.csv. Файли, яких
     немає в Kaggle (fastf1_*.csv, driver_media.json), не чіпаються.
  3. Модулі, що читають CSV при імпорті (DATA_MODULES), перезавантажуються -
     нові дані видно одразу, без рестарту.
  4. Для гонок, у яких щойно з'явився результат, видаляється кеш прогнозу
     (analysis_cache/) - наступний запит порахує його з фактом для порівняння.
  5. У фоні перебудовується ML-датасет і перенавчається модель (fingerprint
     CSV змінився - див. app/ml/dataset.get_dataset).

Вручну: python scripts/update_data.py (або POST /api/data/update).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from app.csv_data import DATA_DIR

log = logging.getLogger(__name__)

KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "jtrotman/formula-1-race-data")
# Як часто перевіряти нову версію, год. 0 - вимкнути фонову перевірку.
UPDATE_HOURS = float(os.getenv("DATA_UPDATE_HOURS", "6"))
# Ручний запуск (POST /api/data/update) - не частіше ніж раз на стільки секунд.
MIN_MANUAL_INTERVAL = 10 * 60

META_FILE = os.path.join(DATA_DIR, "kaggle_meta.json")

# Без цих файлів бекенд не працює - реліз без них відкидається.
REQUIRED = [
    "races.csv", "results.csv", "drivers.csv", "constructors.csv", "qualifying.csv",
    "circuits.csv", "driver_standings.csv", "constructor_standings.csv", "status.csv",
    "seasons.csv", "sprint_results.csv", "pit_stops.csv", "lap_times.csv",
]
# Новий файл може бути трохи меншим (Ergast іноді чистить дублікати), але
# не помітно: інакше це обрізаний/зламаний реліз.
MIN_ROWS_RATIO = 0.99

# Модулі, що читають CSV у змінні модуля при імпорті, - у порядку
# залежностей. Перезавантаження одного без наступних лишило б у них старі
# таблиці. Новий такий модуль треба додати сюди (стежить
# tests/test_data_update.py::test_all_data_modules_are_reloaded).
DATA_MODULES = ("app.csv_data", "app.event_info", "app.driver_stats")

_lock = threading.Lock()
_last_manual_run = 0.0


class UpdateRejected(Exception):
    """Нова версія не пройшла перевірку - data/ лишається як була."""


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _header(path: str) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_meta() -> dict:
    if not os.path.exists(META_FILE):
        return {}
    with open(META_FILE, encoding="utf-8") as f:
        return json.load(f)


def _write_meta(**fields) -> dict:
    meta = {**read_meta(), **fields, "dataset": KAGGLE_DATASET}
    tmp = META_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    os.replace(tmp, META_FILE)
    return meta


def _download() -> tuple[str, str]:
    """Остання версія датасету -> (локальна папка, номер версії).
    kagglehub кешує версії в ~/.cache/kagglehub і качає лише нову."""
    import kagglehub  # тут: не потрібен для решти бекенду й тестів

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    return path, os.path.basename(os.path.normpath(path))


def plan_changes(src_dir: str, dst_dir: str | None = None) -> list[str]:
    """Які CSV з src_dir треба скопіювати в dst_dir (за замовчуванням
    data/). Кидає UpdateRejected, якщо реліз неповний, змінилась схема або
    файл помітно зменшився."""
    dst_dir = dst_dir or DATA_DIR
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(src_dir, f))]
    if missing:
        raise UpdateRejected(f"у релізі бракує файлів: {missing}")

    changed = []
    for name in sorted(os.listdir(src_dir)):
        src, dst = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if not name.endswith(".csv"):
            continue
        if not os.path.exists(dst):
            if name in REQUIRED:
                changed.append(name)
            continue  # невідомий новий файл - бекенду не потрібен
        if _md5(src) == _md5(dst):
            continue
        if _header(src) != _header(dst):
            raise UpdateRejected(f"{name}: змінились колонки {_header(dst)} -> {_header(src)}")
        old_rows, new_rows = _rows(dst), _rows(src)
        if new_rows < old_rows * MIN_ROWS_RATIO:
            raise UpdateRejected(f"{name}: рядків стало {new_rows} замість {old_rows}")
        changed.append(name)
    return changed


def _apply(src_dir: str, files: list[str]) -> None:
    for name in files:
        tmp = os.path.join(DATA_DIR, f".{name}.tmp")
        shutil.copyfile(os.path.join(src_dir, name), tmp)
        os.replace(tmp, os.path.join(DATA_DIR, name))


def reload_data_modules() -> None:
    """Перечитати CSV у пам'ять: модулі з DATA_MODULES перезавантажуються
    на місці, тож app.main та інші, хто тримає посилання на модуль, одразу
    бачать нові таблиці."""
    for name in DATA_MODULES:
        if name in sys.modules:
            importlib.reload(sys.modules[name])


def _races_with_results() -> set[int]:
    path = os.path.join(DATA_DIR, "results.csv")
    return set(pd.read_csv(path, usecols=["raceId"])["raceId"].unique().tolist())


def _invalidate_analysis(race_ids: set[int]) -> list[str]:
    """Прогноз, порахований до гонки, не має факту для порівняння -
    видаляємо його кеш, наступний запит перерахує з результатом."""
    if not race_ids:
        return []
    from app import analysis

    races = pd.read_csv(os.path.join(DATA_DIR, "races.csv"), usecols=["raceId", "year", "round"])
    removed = []
    for r in races[races["raceId"].isin(race_ids)].itertuples():
        path = analysis._cache_path(int(r.year), int(r.round))
        if os.path.exists(path):
            os.remove(path)
            removed.append(f"{r.year}_{r.round}")
    return removed


def _warm_models() -> None:
    try:
        from app.ml.dataset import get_dataset
        from app.ml.train import get_model_bundle

        t0 = time.monotonic()
        get_model_bundle(get_dataset())
        log.info("Модель перенавчена на нових даних за %.0f с", time.monotonic() - t0)
    except Exception:
        log.exception("Не вдалось перенавчити модель після оновлення даних")


def _cleanup_old_versions(current_path: str) -> None:
    """kagglehub лишає кожну завантажену версію (~35 МБ) - на сервері, що
    працює місяцями, це росте без меж. Лишаємо тільки поточну."""
    versions_dir = os.path.dirname(os.path.normpath(current_path))
    if os.path.basename(versions_dir) != "versions":
        return
    for name in os.listdir(versions_dir):
        path = os.path.join(versions_dir, name)
        if path != os.path.normpath(current_path) and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def update(warm_models: bool = True) -> dict:
    """Одна перевірка/оновлення. Повертає {status, version, changed, ...}:
    status - "updated" | "up_to_date" | "rejected" | "error"."""
    with _lock:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            src, version = _download()
            changed = plan_changes(src)
        except UpdateRejected as e:
            log.warning("Оновлення з Kaggle відхилено: %s", e)
            return {**_write_meta(checked_at=now, last_error=str(e)), "status": "rejected", "changed": []}
        except Exception as e:
            log.warning("Не вдалось перевірити Kaggle: %s", e)
            return {**_write_meta(checked_at=now, last_error=f"{type(e).__name__}: {e}"),
                    "status": "error", "changed": []}

        if not changed:
            meta = _write_meta(checked_at=now, version=version, last_error=None)
            return {**meta, "status": "up_to_date", "changed": []}

        before = _races_with_results()
        _apply(src, changed)
        reload_data_modules()
        new_races = _races_with_results() - before
        invalidated = _invalidate_analysis(new_races)
        _cleanup_old_versions(src)

        meta = _write_meta(checked_at=now, updated_at=now, version=version, last_error=None,
                           changed=changed, new_races=len(new_races))
        log.info("Дані оновлено з Kaggle (версія %s): %s; нових гонок з результатом: %d",
                 version, ", ".join(changed), len(new_races))

    if warm_models:
        threading.Thread(target=_warm_models, name="ml-warmup", daemon=True).start()
    return {**meta, "status": "updated", "changed": changed, "invalidated_analysis": invalidated}


def update_manually() -> dict:
    """Для POST /api/data/update: не частіше ніж раз на MIN_MANUAL_INTERVAL."""
    global _last_manual_run
    wait = MIN_MANUAL_INTERVAL - (time.monotonic() - _last_manual_run)
    if _last_manual_run and wait > 0:
        return {**read_meta(), "status": "throttled", "retry_after": int(wait)}
    _last_manual_run = time.monotonic()
    return update()


def _loop() -> None:
    while True:
        update()
        time.sleep(UPDATE_HOURS * 3600)


def start_background_updates() -> threading.Thread | None:
    """Перевірка при старті й далі кожні UPDATE_HOURS. Під pytest і з
    DATA_UPDATE_HOURS=0 не запускається."""
    if UPDATE_HOURS <= 0 or "pytest" in sys.modules:
        return None
    t = threading.Thread(target=_loop, name="kaggle-data-update", daemon=True)
    t.start()
    return t

"""
Тести автооновлення даних з Kaggle (app/data_update.py) - без мережі:
"реліз" Kaggle і папка data/ - тимчасові каталоги з маленькими CSV.
"""
import ast
import json
import os
from pathlib import Path

import pytest

from app import data_update

APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _write(dir_: Path, name: str, text: str) -> None:
    (dir_ / name).write_text(text, encoding="utf-8")


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """(release, data): повний "реліз" з REQUIRED-файлами і його копія в data/."""
    release, data = tmp_path / "release", tmp_path / "data"
    release.mkdir()
    data.mkdir()
    for name in data_update.REQUIRED:
        body = "id,value\n" + "".join(f"{i},{i}\n" for i in range(200))
        _write(release, name, body)
        _write(data, name, body)
    _write(release, "races.csv", "raceId,year,round\n1,2026,1\n2,2026,2\n")
    _write(data, "races.csv", "raceId,year,round\n1,2026,1\n2,2026,2\n")
    _write(release, "results.csv", "raceId,driverId\n1,1\n")
    _write(data, "results.csv", "raceId,driverId\n1,1\n")
    monkeypatch.setattr(data_update, "DATA_DIR", str(data))
    monkeypatch.setattr(data_update, "META_FILE", str(data / "kaggle_meta.json"))
    return release, data


# ---------------------------------------------------------------------------
# plan_changes - перевірка релізу
# ---------------------------------------------------------------------------

def test_identical_release_has_no_changes(dirs):
    release, data = dirs
    assert data_update.plan_changes(str(release), str(data)) == []


def test_appended_rows_are_a_change(dirs):
    release, data = dirs
    with open(release / "drivers.csv", "a") as f:
        f.write("999,999\n")
    assert data_update.plan_changes(str(release), str(data)) == ["drivers.csv"]


def test_missing_required_file_is_rejected(dirs):
    release, data = dirs
    os.remove(release / "results.csv")
    with pytest.raises(data_update.UpdateRejected, match="results.csv"):
        data_update.plan_changes(str(release), str(data))


def test_changed_columns_are_rejected(dirs):
    release, data = dirs
    _write(release, "drivers.csv", "id,other\n1,1\n" * 1)
    with pytest.raises(data_update.UpdateRejected, match="колонки"):
        data_update.plan_changes(str(release), str(data))


def test_truncated_file_is_rejected(dirs):
    release, data = dirs
    _write(release, "drivers.csv", "id,value\n1,1\n")
    with pytest.raises(data_update.UpdateRejected, match="рядків"):
        data_update.plan_changes(str(release), str(data))


def test_files_not_in_release_and_unknown_files_are_ignored(dirs):
    release, data = dirs
    _write(data, "fastf1_practice.csv", "a\n1\n")      # лише в нас - не чіпаємо
    _write(release, "brand_new_table.csv", "a\n1\n")   # лише в релізі - не потрібен
    assert data_update.plan_changes(str(release), str(data)) == []


# ---------------------------------------------------------------------------
# update - повний цикл
# ---------------------------------------------------------------------------

@pytest.fixture
def no_side_effects(monkeypatch):
    calls = {"reload": 0, "invalidated": None}
    monkeypatch.setattr(data_update, "reload_data_modules",
                        lambda: calls.__setitem__("reload", calls["reload"] + 1))
    monkeypatch.setattr(data_update, "_invalidate_analysis",
                        lambda ids: calls.__setitem__("invalidated", ids) or [])
    monkeypatch.setattr(data_update, "_cleanup_old_versions", lambda path: None)
    return calls


def test_update_applies_new_release(dirs, monkeypatch, no_side_effects):
    release, data = dirs
    _write(release, "results.csv", "raceId,driverId\n1,1\n2,1\n")  # гонка 2 відбулась
    monkeypatch.setattr(data_update, "_download", lambda: (str(release), "136"))

    result = data_update.update(warm_models=False)

    assert result["status"] == "updated"
    assert result["changed"] == ["results.csv"]
    assert (data / "results.csv").read_text() == "raceId,driverId\n1,1\n2,1\n"
    assert no_side_effects["reload"] == 1
    assert no_side_effects["invalidated"] == {2}
    meta = json.loads((data / "kaggle_meta.json").read_text())
    assert meta["version"] == "136" and meta["new_races"] == 1 and meta["last_error"] is None


def test_update_up_to_date_changes_nothing(dirs, monkeypatch, no_side_effects):
    release, _ = dirs
    monkeypatch.setattr(data_update, "_download", lambda: (str(release), "135"))
    result = data_update.update(warm_models=False)
    assert result["status"] == "up_to_date"
    assert no_side_effects["reload"] == 0


def test_rejected_release_keeps_data(dirs, monkeypatch, no_side_effects):
    release, data = dirs
    before = (data / "drivers.csv").read_text()
    _write(release, "drivers.csv", "id,value\n1,1\n")
    monkeypatch.setattr(data_update, "_download", lambda: (str(release), "137"))

    result = data_update.update(warm_models=False)

    assert result["status"] == "rejected"
    assert (data / "drivers.csv").read_text() == before
    assert "рядків" in json.loads((data / "kaggle_meta.json").read_text())["last_error"]


def test_network_error_is_reported_not_raised(dirs, monkeypatch, no_side_effects):
    def boom():
        raise ConnectionError("kaggle.com недоступний")
    monkeypatch.setattr(data_update, "_download", boom)
    result = data_update.update(warm_models=False)
    assert result["status"] == "error" and "ConnectionError" in result["last_error"]


def test_manual_update_is_throttled(monkeypatch):
    monkeypatch.setattr(data_update, "update", lambda: {"status": "up_to_date"})
    monkeypatch.setattr(data_update, "_last_manual_run", 0.0)
    assert data_update.update_manually()["status"] == "up_to_date"
    second = data_update.update_manually()
    assert second["status"] == "throttled" and second["retry_after"] > 0


def test_background_updates_disabled_under_pytest():
    assert data_update.start_background_updates() is None


# ---------------------------------------------------------------------------
# Перезавантаження модулів з даними
# ---------------------------------------------------------------------------

def _module_level_data_users() -> set[str]:
    """Модулі app/, які на рівні модуля (поза функціями) читають щось із
    csv_data/event_info/driver_stats - їм потрібне перезавантаження."""
    sources = {"csv_data", "event_info", "driver_stats"}
    users = set()
    for path in APP_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set(sources)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] in sources:
                # КОНСТАНТИ (DATA_DIR тощо) - не таблиці, перезавантаження не потребують
                imported |= {a.asname or a.name for a in node.names if not (a.asname or a.name).isupper()}
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                if names & imported:
                    users.add(f"app.{path.stem}")
    return users


def test_all_data_modules_are_reloaded():
    missing = _module_level_data_users() - set(data_update.DATA_MODULES)
    assert not missing, f"додай у data_update.DATA_MODULES: {missing}"


def test_reload_rebinds_derived_tables():
    from app import csv_data, event_info

    old = csv_data._races
    data_update.reload_data_modules()
    assert csv_data._races is not old
    assert event_info._races is csv_data._races


def test_data_status_endpoint(client):
    res = client.get("/api/data/status")
    assert res.status_code == 200
    assert "update_every_hours" in res.json()

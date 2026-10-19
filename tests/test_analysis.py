"""
Тести app/analysis.py + app/ml/*. Датасет і моделі важкі (~20-25с на
холодному кеші) - тому будуються ОДИН РАЗ на весь файл (session-scoped
фікстура) і повторно використовуються в усіх тестах нижче.

Позначені @pytest.mark.integration за тим самим принципом, що й
tests/test_integration.py: реальні дані (тут - локальний data/*.csv,
без мережі), повільніші за звичайні юніт-тести.
"""
import json

import pytest

from app import analysis
from app.ml.dataset import get_dataset
from app.ml.live import _resolve_driver_id
from app.ml.train import get_model_bundle

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dataset():
    return get_dataset()


@pytest.fixture(scope="module")
def bundle(dataset):
    return get_model_bundle(dataset)


@pytest.fixture(autouse=True)
def _clean_analysis_cache(tmp_path, monkeypatch):
    """Кожен тест пише свій кеш аналізу в окремий tmp-каталог, щоб не
    засмічувати analysis_cache/ проєкту і не залежати від результатів
    попередніх запусків."""
    monkeypatch.setattr(analysis.config, "ANALYSIS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(analysis, "_cache_path",
                         lambda year, round_number: str(tmp_path / f"{year}_{round_number}.json"))


def test_resolve_driver_id_disambiguates_shared_code(dataset):
    """'VER' - код одразу двох пілотів в історії Ergast-датасету
    (Jean-Éric Vergne, driverId 818, і Max Verstappen, driverId 830).
    Резолвер має брати того, хто реально їздив пізніше, а не першого
    за driverId - інакше live-фічі рахуються для геть іншої людини."""
    resolved = _resolve_driver_id(dataset, code="VER", full_name=None)
    assert resolved == 830


def test_resolve_driver_id_prefers_exact_full_name(dataset):
    resolved = _resolve_driver_id(dataset, code=None, full_name="Max Verstappen")
    assert resolved == 830


def test_analyze_completed_race_from_static_dataset(dataset, bundle):
    """Гонка, що вже повністю є в df_model (кваліфікація+результат) -
    статичний шлях, без звернень до FastF1."""
    result = analysis.analyze_race(2023, 1, refresh=True)

    assert result["data"]["stage"] == "completed"
    assert result["data"]["source"] == "static_dataset"
    assert result["race"]["name"] == "Bahrain Grand Prix"
    assert len(result["predictions"]) >= 15

    # predicted_position - суцільний ранг 1..N без пропусків/дублів
    ranks = sorted(p["predicted_position"] for p in result["predictions"])
    assert ranks == list(range(1, len(ranks) + 1))

    # переможець гонки був відомий (raceId уже в датасеті) - actual_position
    # має бути виставлений хоч комусь, і використаний лише для порівняння
    assert any(p["actual_position"] == 1 for p in result["predictions"])

    for p in result["predictions"]:
        assert 0.0 <= p["p_dnf"] <= 1.0


def test_analyze_race_unknown_round_raises(dataset, bundle):
    with pytest.raises(ValueError):
        analysis.analyze_race(2023, 999, refresh=True)


def test_analyze_future_race_without_quali_uses_last_known_lineup(dataset, bundle, monkeypatch):
    """Етап, де кваліфікації ще нема ні в CSV, ні через FastF1 (мережа
    недоступна/сесія не існує) - має впасти на fallback-склад пілотів і
    не кидати виняток. p_dnf не повинен вироджуватись у майже-100% для
    всіх (регрес на баг з fp_* фічами, що імітують "зійшов на 1 колі")."""
    monkeypatch.setattr(analysis.fastf1_client, "get_session_results",
                         lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no data")))
    monkeypatch.setattr(analysis.fastf1_client, "get_race_results",
                         lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no data")))
    monkeypatch.setattr(analysis.fastf1_client, "_load_session",
                         lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no data")))

    # Раунд без кваліфікації в CSV - будь-який з 2026, що ще не проїхано.
    result = analysis.analyze_race(2026, 22, refresh=True)

    assert result["data"]["source"] == "live_no_quali"
    assert result["data"]["quali_available"] is False
    assert len(result["predictions"]) >= 15
    assert all(p["actual_position"] is None for p in result["predictions"])

    p_dnf_values = [p["p_dnf"] for p in result["predictions"]]
    assert max(p_dnf_values) < 0.7, (
        "p_dnf для живого прогнозу не має вироджуватись у майже-100% "
        "(регрес на fp_* leak - див. коментар у app/ml/live.py)"
    )


def test_analysis_is_cached_between_calls(dataset, bundle):
    first = analysis.analyze_race(2023, 1, refresh=True)
    second = analysis.analyze_race(2023, 1)  # без refresh - з кешу
    assert first["generated_at"] == second["generated_at"]


def test_analysis_cache_file_is_valid_json(dataset, bundle, tmp_path):
    analysis.analyze_race(2023, 1, refresh=True)
    cached_files = list(tmp_path.glob("*.json"))
    assert len(cached_files) == 1
    with open(cached_files[0], encoding="utf-8") as f:
        data = json.load(f)
    assert data["race"]["year"] == 2023


def test_analysis_cache_from_older_model_is_recomputed(dataset, bundle, tmp_path):
    first = analysis.analyze_race(2023, 1, refresh=True)
    path = tmp_path / "2023_1.json"
    stale = json.loads(path.read_text(encoding="utf-8"))
    stale["model"]["version"] = "v0-stale"
    path.write_text(json.dumps(stale), encoding="utf-8")

    again = analysis.analyze_race(2023, 1)  # без refresh, але кеш від іншої моделі
    assert again["model"]["version"] == analysis.MODEL_VERSION
    assert again["generated_at"] != first["generated_at"]

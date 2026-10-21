"""
Пояснення прогнозу (app/explain.py): внески фіч у позицію кожного пілота.
"""
import statistics

import pytest

from app import analysis, explain, i18n


@pytest.fixture(scope="module")
def race():
    # Етап з датасету - рахується локально, без мережі.
    return analysis._compute(2024, 10)


def test_every_feature_has_a_group():
    from app.ml.dataset import FEATURE_ORDER

    assert {explain.feature_group(f) for f in FEATURE_ORDER} <= set(explain.GROUP_ORDER)
    assert explain.feature_group("quali_position") == "quali"
    assert explain.feature_group("driver_quali_pace_5") == "driver"
    assert explain.feature_group("con_avg_pos_5") == "team"
    assert explain.feature_group("pr_long_run_gap_pct") == "practice"


def test_each_prediction_has_explanation(race):
    for p in race["predictions"]:
        e = p["explanation"]
        assert len(e["factors"]) == explain.TOP_FACTORS
        impacts = [abs(f["impact"]) for f in e["factors"]]
        assert impacts == sorted(impacts, reverse=True)
        # Групи разом - це всі фічі: збігаються з сумою топу та "інших".
        assert sum(g["impact"] for g in e["groups"]) == pytest.approx(e["total"], abs=0.1)
        assert sum(f["impact"] for f in e["factors"]) + e["other"] == pytest.approx(e["total"], abs=0.1)


def test_total_matches_gap_to_field_average(race):
    # Сума внесків ~ наскільки прогноз пілота кращий за середній у полі
    # (наближено: бал ранкера переводиться в позиції лінійно).
    preds = race["predictions"]
    mean = statistics.mean(p["expected_position"] for p in preds)
    errors = [abs((mean - p["expected_position"]) - p["explanation"]["total"]) for p in preds]
    assert statistics.mean(errors) < 1.0
    # Хто попереду в прогнозі - в того й більший сумарний внесок.
    first, last = preds[0], preds[-1]
    assert first["explanation"]["total"] > 0 > last["explanation"]["total"]


def test_pole_sitter_is_explained_by_qualifying(race):
    pole = next(p for p in race["predictions"] if p["quali_position"] == 1)
    top = pole["explanation"]["factors"][0]

    assert top["group"] == "quali"
    assert top["impact"] > 0


def test_factors_get_labels_in_request_language(race):
    token = i18n._lang.set("en")
    try:
        en = analysis._localize(race)
    finally:
        i18n._lang.reset(token)

    factor = next(f for p in en["predictions"] for f in p["explanation"]["factors"] if f["feature"] == "quali_position")
    assert factor["label"] == "Qualifying position"
    # Збережений результат (кеш) не змінюється.
    assert "label" not in race["predictions"][0]["explanation"]["factors"][0]


def test_status_includes_model_metrics(tmp_path, monkeypatch, race):
    import json

    monkeypatch.setattr(analysis.config, "ANALYSIS_CACHE_DIR", str(tmp_path))
    (tmp_path / "2024_10.json").write_text(json.dumps(race), encoding="utf-8")

    status = analysis.analysis_status(2024, 10)

    assert status["ready"] and status["hits"] is not None
    assert set(status["metrics"]) == set(analysis.STATUS_METRICS)
    assert status["metrics"]["mae_test"] == race["model"]["metrics"]["mae_test"]

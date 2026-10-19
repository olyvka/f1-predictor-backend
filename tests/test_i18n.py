"""
Мова відповідей API (app/i18n.py): визначення мови запиту, переклад
повідомлень про помилки, прогнозу моделі й оцінки сезону.
"""
import pytest

from app import analysis, i18n


# ---------------------------------------------------------------------------
# Визначення мови
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "header, expected",
    [
        ("en-GB,en;q=0.9,uk;q=0.8", "en"),
        ("uk-UA,uk;q=0.9,en;q=0.8", "uk"),
        ("de-DE,de;q=0.9,en;q=0.5", "en"),  # перша підтримувана за вагою
        ("en;q=0.3,uk;q=0.7", "uk"),  # вага важливіша за порядок
        ("de-DE", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_accept_language(header, expected):
    assert i18n.parse_accept_language(header) == expected


def test_default_language_is_ukrainian(client):
    resp = client.get("/api/event-summary/2024/99")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Етап 99 сезону 2024 відсутній у датасеті"
    assert resp.headers["content-language"] == "uk"
    assert "Accept-Language" in resp.headers["vary"]


def test_accept_language_english(client):
    # Синхронний ендпоінт виконується в потоці threadpool - мова з
    # middleware має дійти й туди.
    resp = client.get("/api/event-summary/2024/99", headers={"Accept-Language": "en-US,en;q=0.9"})

    assert resp.json()["detail"] == "Round 99 of the 2024 season is not in the dataset"
    assert resp.headers["content-language"] == "en"


def test_query_param_overrides_header(client):
    resp = client.get("/api/driver-season/2024/nobody?lang=uk", headers={"Accept-Language": "en"})

    assert resp.json()["detail"] == "Пілот 'nobody' відсутній у датасеті"


def test_compare_needs_two_in_english(client):
    resp = client.get("/api/compare/2024/1/R", params={"drivers": "VER", "lang": "en"})

    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("At least 2 drivers")


def test_rating_components_follow_language(client):
    en = client.get("/api/driver-season/2021/max_verstappen", headers={"Accept-Language": "en"}).json()
    uk = client.get("/api/driver-season/2021/max_verstappen").json()

    labels_en = {c["key"]: c["label"] for c in en["rating"]["components"]}
    labels_uk = {c["key"]: c["label"] for c in uk["rating"]["components"]}
    assert labels_en["consistency"] == "Consistency"
    assert labels_uk["consistency"] == "Стабільність"
    # Цифри від мови не залежать.
    assert en["rating"]["score"] == uk["rating"]["score"]


# ---------------------------------------------------------------------------
# Каталог повідомлень
# ---------------------------------------------------------------------------

def test_every_message_has_both_languages_with_same_placeholders():
    import string

    for code, entry in i18n.MESSAGES.items():
        assert set(entry) == {"uk", "en"}, code
        fields = [{f for _, f, _, _ in string.Formatter().parse(entry[lng]) if f} for lng in ("uk", "en")]
        assert fields[0] == fields[1], code


# ---------------------------------------------------------------------------
# Прогноз моделі: кеш українською, видача - мовою запиту
# ---------------------------------------------------------------------------

def _with_lang(lang, fn):
    token = i18n._lang.set(lang)
    try:
        return fn()
    finally:
        i18n._lang.reset(token)


def test_cached_notes_are_translated_by_template():
    stored = i18n.canonical("note_driver_unmatched", name="Kimi Antonelli")

    translated = _with_lang("en", lambda: i18n.translate_cached_uk(stored))

    assert translated == "Driver 'Kimi Antonelli' could not be matched to the local dataset - skipped"
    assert _with_lang("uk", lambda: i18n.translate_cached_uk(stored)) == stored
    assert _with_lang("en", lambda: i18n.translate_cached_uk("довільний текст")) == "довільний текст"


def test_analysis_localized_without_touching_cache():
    cached = {
        "data": {"notes": [i18n.canonical("note_quali_missing")]},
        "feature_importance": {"ranker": [{"feature": "grid", "label": "Стартова позиція", "importance": 80}]},
    }

    en = _with_lang("en", lambda: analysis._localize(cached))

    assert en["feature_importance"]["ranker"][0]["label"] == "Grid position"
    assert en["data"]["notes"][0].startswith("Qualifying for this round is not available yet")
    # Збережений об'єкт (той, що в кеші) лишився українським.
    assert cached["feature_importance"]["ranker"][0]["label"] == "Стартова позиція"
    uk = _with_lang("uk", lambda: analysis._localize(cached))
    assert uk["data"]["notes"] == cached["data"]["notes"]


def test_old_cache_features_get_labels_in_ukrainian_too():
    cached = {"feature_importance": {"ranker": [{"feature": "q1_q2_delta", "label": "q1_q2_delta", "importance": 5}]}}

    uk = _with_lang("uk", lambda: analysis._localize(cached))

    assert uk["feature_importance"]["ranker"][0]["label"] == "Прогрес Q1 → Q2 (с)"


def test_every_feature_has_labels_in_both_languages():
    from app.ml.dataset import FEATURE_ORDER

    assert set(analysis.FEATURE_LABELS) == set(analysis.FEATURE_LABELS_EN)
    # Кожна фіча моделі має людський підпис - інакше на графіку важливості
    # фіч видно сирий код на кшталт "q1_q2_delta".
    assert set(FEATURE_ORDER) <= set(analysis.FEATURE_LABELS)

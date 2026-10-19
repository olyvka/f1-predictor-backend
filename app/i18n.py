"""
Мова відповідей API: українська (за замовчуванням) або англійська.

Мову запиту визначає LanguageMiddleware: параметр ?lang=uk|en, інакше
заголовок Accept-Language (фронт шле мову інтерфейсу), інакше українська -
так поводився бекенд і досі. Мова живе в contextvar на час запиту, тож
функція tr() у будь-якому модулі формує текст потрібною мовою без
протягування параметра через усі виклики.

Повідомлення - каталог MESSAGES {код: {uk, en}} з шаблонами str.format.
Тексти, що вже лежать у кеші прогнозів (analysis_cache/*.json) українською,
перекладає translate_cached_uk(): він розпізнає їх за українськими
шаблонами тих самих кодів - перебудовувати кеш (а з ним і модель) не треба.
"""
from __future__ import annotations

import re
from contextvars import ContextVar

LANGUAGES = ("uk", "en")
DEFAULT_LANGUAGE = "uk"

_lang: ContextVar[str] = ContextVar("lang", default=DEFAULT_LANGUAGE)


def get_lang() -> str:
    return _lang.get()


def parse_accept_language(header: str | None) -> str | None:
    """Перша підтримувана мова з Accept-Language з урахуванням ваг q:
    "en-GB,en;q=0.9,uk;q=0.8" -> "en"."""
    if not header:
        return None
    ranked = []
    for i, part in enumerate(header.split(",")):
        tag, _, params = part.strip().partition(";")
        q = 1.0
        if params.strip().startswith("q="):
            try:
                q = float(params.strip()[2:])
            except ValueError:
                q = 0.0
        ranked.append((-q, i, tag.strip().lower().split("-")[0]))
    for _, _, primary in sorted(ranked):
        if primary in LANGUAGES:
            return primary
    return None


class LanguageMiddleware:
    """Чистий ASGI-middleware: виставляє мову запиту й додає до відповіді
    Content-Language та Vary: Accept-Language (відповідь залежить від мови -
    проміжні кеші мають це знати)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        query = scope.get("query_string", b"").decode("latin-1")
        from_query = next(
            (p.split("=", 1)[1] for p in query.split("&") if p.startswith("lang=")), None
        )
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        lang = (
            from_query if from_query in LANGUAGES else parse_accept_language(headers.get("accept-language"))
        ) or DEFAULT_LANGUAGE

        async def send_with_language(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"] = list(message["headers"]) + [
                    (b"content-language", lang.encode()),
                    (b"vary", b"Accept-Language"),
                ]
            await send(message)

        token = _lang.set(lang)
        try:
            await self.app(scope, receive, send_with_language)
        finally:
            _lang.reset(token)


MESSAGES: dict[str, dict[str, str]] = {
    # --- відсутні дані в CSV-датасеті ---
    "round_not_found": {
        "uk": "Етап {round} сезону {year} відсутній у датасеті",
        "en": "Round {round} of the {year} season is not in the dataset",
    },
    "round_not_in_calendar": {
        "uk": "Етап {round} сезону {year} відсутній у календарі датасету",
        "en": "Round {round} of the {year} season is not in the dataset calendar",
    },
    "season_not_found": {
        "uk": "Сезон {year} відсутній у датасеті",
        "en": "The {year} season is not in the dataset",
    },
    "race_results_missing": {
        "uk": "Результати гонки {round} сезону {year} відсутні в датасеті",
        "en": "Race results for round {round} of {year} are not in the dataset",
    },
    "quali_results_missing": {
        "uk": "Результати кваліфікації етапу {round} сезону {year} відсутні в датасеті",
        "en": "Qualifying results for round {round} of {year} are not in the dataset",
    },
    "sprint_results_missing": {
        "uk": "Результати спринту етапу {round} сезону {year} відсутні в датасеті "
              "(спринтів не було до 2021 і не на кожному етапі)",
        "en": "Sprint results for round {round} of {year} are not in the dataset "
              "(there were no sprints before 2021 and not every round has one)",
    },
    "csv_session_unsupported": {
        "uk": "Сесія '{session}' недоступна в CSV-датасеті - тільки R/Race, Q/Qualifying, "
              "S/Sprint (практики FP1-3 результатів у Ergast не мають; скористайся "
              "FastF1-ендпоінтами /api/results, /api/laps)",
        "en": "Session '{session}' is not available in the CSV dataset - only R/Race, Q/Qualifying, "
              "S/Sprint (Ergast has no results for practice sessions FP1-3; use the FastF1 "
              "endpoints /api/results, /api/laps)",
    },
    "pitstops_missing": {
        "uk": "Дані по піт-стопах етапу {round} сезону {year} відсутні в датасеті",
        "en": "Pit stop data for round {round} of {year} is not in the dataset",
    },
    # --- пілоти ---
    "driver_not_found": {
        "uk": "Пілот '{driver}' відсутній у датасеті",
        "en": "Driver '{driver}' is not in the dataset",
    },
    "driver_not_in_season": {
        "uk": "Пілот '{driver}' не їхав у сезоні {year}",
        "en": "Driver '{driver}' did not race in the {year} season",
    },
    "driver_no_races": {
        "uk": "У датасеті немає гонок пілота '{driver}'",
        "en": "The dataset has no races for driver '{driver}'",
    },
    "driver_not_in_session": {
        "uk": "Пілот '{driver}' не знайдений у цій сесії",
        "en": "Driver '{driver}' was not found in this session",
    },
    # --- запасні джерела кіл і погоди (OpenF1 / CSV) ---
    "openf1_rate_limited": {
        "uk": "OpenF1 не відповідає (ліміт запитів): {path}",
        "en": "OpenF1 is not responding (rate limit): {path}",
    },
    "openf1_since": {
        "uk": "OpenF1 має дані лише з {year} року",
        "en": "OpenF1 only has data from {year} onwards",
    },
    "openf1_session_not_in_calendar": {
        "uk": "Сесії {session} немає в календарі етапу {round}",
        "en": "Session {session} is not in the schedule of round {round}",
    },
    "openf1_session_not_found": {
        "uk": "OpenF1 не знайшов сесію {session} етапу {round} {year}",
        "en": "OpenF1 could not find session {session} of round {round} {year}",
    },
    "openf1_no_laps": {
        "uk": "OpenF1 не має кіл для цієї сесії",
        "en": "OpenF1 has no laps for this session",
    },
    "openf1_no_weather": {
        "uk": "OpenF1 не має погоди для цієї сесії",
        "en": "OpenF1 has no weather for this session",
    },
    "csv_laps_races_only": {
        "uk": "У датасеті є кола лише гонок, без практик, кваліфікацій і спринтів",
        "en": "The dataset only has race laps - no practice, qualifying or sprint laps",
    },
    "csv_no_laps": {
        "uk": "У датасеті немає кіл цієї гонки",
        "en": "The dataset has no laps for this race",
    },
    "weather_not_in_dataset": {
        "uk": "{error}; у датасеті погоди немає",
        "en": "{error}; the dataset has no weather",
    },
    # --- обгортки помилок ендпоінтів (main.py) ---
    "driver_data_unavailable": {"uk": "Дані пілота недоступні", "en": "Driver data is unavailable"},
    "laps_unavailable": {"uk": "Кола недоступні", "en": "Laps are unavailable"},
    "compare_unavailable": {"uk": "Порівняння недоступне", "en": "Comparison is unavailable"},
    "weather_unavailable": {"uk": "Погода недоступна", "en": "Weather is unavailable"},
    "unavailable_with_reason": {"uk": "{what}: {error}", "en": "{what}: {error}"},
    "fallback_failed": {
        "uk": "{what}. FastF1 більше не завантажує нові сесії, а запасні джерела не мають цих даних: {error}",
        "en": "{what}. FastF1 can no longer load new sessions, and the fallback sources do not have this data: {error}",
    },
    "compare_needs_two": {
        "uk": "Потрібно щонайменше 2 пілоти для порівняння, напр. ?drivers=VER,HAM",
        "en": "At least 2 drivers are needed for a comparison, e.g. ?drivers=VER,HAM",
    },
    "analysis_failed": {
        "uk": "Не вдалось побудувати аналіз прогнозу: {error}",
        "en": "Could not build the prediction analysis: {error}",
    },
    # --- примітки до прогнозу (лежать і в кеші analysis_cache - див. translate_cached_uk) ---
    "note_actual_added": {
        "uk": "Фактичний результат гонки додано лише для порівняння - модель його не бачила",
        "en": "The actual race result is added for comparison only - the model did not see it",
    },
    "note_practice_missing": {
        "uk": "Практики цього етапу недоступні - фічі вікенду (темп довгих серій, погода) відсутні",
        "en": "Practice sessions for this round are unavailable - weekend features (long-run pace, weather) are missing",
    },
    "note_driver_unmatched": {
        "uk": "Пілота '{name}' не вдалось зіставити з локальним датасетом - пропущено",
        "en": "Driver '{name}' could not be matched to the local dataset - skipped",
    },
    "note_quali_unmatched": {
        "uk": "Кваліфікація завантажилась, але жодного пілота не вдалось зіставити - використано попередній відомий склад",
        "en": "Qualifying loaded, but no driver could be matched - the last known line-up was used",
    },
    "note_quali_missing": {
        "uk": "Кваліфікація цього етапу ще недоступна - склад пілотів і фічі кваліфікації "
              "взято/проігноровано за останнім відомим етапом; прогноз ґрунтується лише на формі команд/пілотів",
        "en": "Qualifying for this round is not available yet - the line-up and qualifying features "
              "were taken from / ignored based on the last known round; the prediction relies only on team and driver form",
    },
    # --- складові оцінки сезону (driver_stats.RATING_COMPONENTS) ---
    "rating_quali_vs_teammate": {"uk": "Кваліфікація проти напарника", "en": "Qualifying vs teammate"},
    "rating_quali_vs_teammate_hint": {"uk": "частка виграних дуелей", "en": "share of duels won"},
    "rating_race_vs_teammate": {"uk": "Гонка проти напарника", "en": "Race vs teammate"},
    "rating_race_vs_teammate_hint": {
        "uk": "частка виграних дуелей, коли обидва фінішували",
        "en": "share of duels won when both finished",
    },
    "rating_positions_gained": {"uk": "Позиції в гонці", "en": "Places gained"},
    "rating_positions_gained_hint": {
        "uk": "середньо виграних позицій від старту до фінішу",
        "en": "average places gained from start to finish",
    },
    "rating_consistency": {"uk": "Стабільність", "en": "Consistency"},
    "rating_consistency_hint": {
        "uk": "розкид фінішних позицій (менше - краще)",
        "en": "spread of finishing positions (lower is better)",
    },
    "rating_incident_free": {"uk": "Без аварій", "en": "No accidents"},
    "rating_incident_free_hint": {
        "uk": "частка гонок без сходу через аварію чи зіткнення (вину датасет не розрізняє)",
        "en": "share of races without retiring from an accident or collision (fault not recorded)",
    },
    "rating_results": {"uk": "Результати", "en": "Results"},
    "rating_results_hint": {"uk": "середньо очок за гонку", "en": "average points per race"},
    "note_no_history": {
        "uk": "Немає історичних даних до цієї гонки - неможливо побудувати аналіз",
        "en": "There is no historical data before this race - the analysis cannot be built",
    },
}


def tr(code: str, **params) -> str:
    """Текст повідомлення мовою поточного запиту."""
    entry = MESSAGES[code]
    return entry.get(get_lang(), entry[DEFAULT_LANGUAGE]).format(**params)


def canonical(code: str, **params) -> str:
    """Український текст незалежно від мови запиту - для того, що
    зберігається (кеш прогнозів), щоб кеш не залежав від того, хто першим
    його побудував. Показувати користувачу - через translate_cached_uk()."""
    return MESSAGES[code][DEFAULT_LANGUAGE].format(**params)


def _uk_pattern(template: str) -> re.Pattern:
    parts = re.split(r"\{(\w+)\}", template)
    regex = "".join(
        re.escape(p) if i % 2 == 0 else f"(?P<{p}>.+?)" for i, p in enumerate(parts)
    )
    return re.compile(f"^{regex}$", re.S)


_NOTE_PATTERNS = [
    (code, _uk_pattern(entry["uk"])) for code, entry in MESSAGES.items() if code.startswith("note_")
]


def translate_cached_uk(text: str) -> str:
    """Збережений український текст примітки -> мова запиту. Невідомий
    текст лишається як є."""
    if get_lang() == DEFAULT_LANGUAGE:
        return text
    for code, pattern in _NOTE_PATTERNS:
        m = pattern.match(text)
        if m:
            return tr(code, **m.groupdict())
    return text

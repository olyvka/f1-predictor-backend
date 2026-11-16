import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

# .env до імпорту модулів app: вони читають змінні середовища під час імпорту
# (ліміти й облікові дані OpenF1, FRONTEND_ORIGIN тощо).
load_dotenv()

from fastapi import FastAPI, HTTPException, Query, WebSocket  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app import (analysis, csv_data, data_update, driver_stats, event_info, fallback, fastf1_client,  # noqa: E402
                 live_feed, live_replay, openf1_client, predictor)
from app.i18n import LanguageMiddleware, tr  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Свіжі data/*.csv з Kaggle: перевірка при старті й далі кожні
    # DATA_UPDATE_HOURS годин у фоновому потоці (app/data_update.py).
    data_update.start_background_updates()
    yield


app = FastAPI(
    title="F1 Predictor API",
    description="Бекенд для дипломної роботи: аналіз та передбачення результатів Формули-1",
    version="0.1.0",
    lifespan=lifespan,
)

# У проді краще задавати через змінну середовища FRONTEND_ORIGIN,
# щоб не хардкодити адресу фронта в коді.
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    os.getenv("FRONTEND_ORIGIN", ""),
]
# Мова відповідей (?lang= / Accept-Language) - див. app/i18n.py. Додано
# першим, тож CORS-middleware (додане пізніше) обгортає його ззовні.
app.add_middleware(LanguageMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in origins if o],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_drivers(drivers: str) -> list[str]:
    return [d.strip().upper() for d in drivers.split(",") if d.strip()]


# Сесії, для яких FastF1 уже не зміг дати кола/погоду (живий таймінг F1
# закритий з жовтня 2026 - працює лише вже закешоване). Наступні запити по
# них ідуть одразу в запасне джерело, без марної спроби FastF1 через мережу.
_NO_LIVE_TIMING: set[tuple[int, int, str]] = set()


def _not_loaded(e: Exception) -> bool:
    # FastF1 кидає DataNotLoadedError, коли load() не зміг дістати дані.
    return "has not been loaded yet" in str(e)


def _fastf1_or_fallback(year: int, round_number: int, session: str, primary, backup, not_found: str):
    """FastF1 з кешу, а якщо там цієї сесії немає - app/fallback.py
    (OpenF1 для 2023+, CSV для гонок раніших сезонів)."""
    key = (year, round_number, session.upper())
    if key not in _NO_LIVE_TIMING:
        try:
            return primary()
        except Exception as e:
            if not _not_loaded(e):
                raise HTTPException(status_code=404, detail=tr("unavailable_with_reason", what=tr(not_found), error=e))
            _NO_LIVE_TIMING.add(key)
    try:
        return backup()
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=tr("fallback_failed", what=tr(not_found), error=e),
        )


@app.get("/")
def health_check():
    return {"status": "ok", "service": "f1-predictor-api"}


@app.get("/api/seasons")
def list_seasons():
    """Усі сезони, наявні в локальному датасеті (data/seasons.csv) - для
    випадаючого списку сезонів на фронті. Без мережі й без FastF1."""
    return csv_data.list_seasons()


@app.get("/api/standings/drivers/{year}")
def driver_standings(year: int):
    """Підсумковий залік пілотів сезону з локального датасету
    (data/driver_standings.csv + data/drivers.csv): позиція, очки, перемоги.
    Дані історичні й статичні - не тягнемо їх через FastF1."""
    try:
        return csv_data.get_driver_standings(year)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/standings/constructors/{year}")
def constructor_standings(year: int):
    """Підсумковий залік конструкторів сезону з локального датасету
    (data/constructor_standings.csv + data/constructors.csv)."""
    try:
        return csv_data.get_constructor_standings(year)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/csv/results/{year}/{round_number}/{session}")
def session_results_csv(year: int, round_number: int, session: str):
    """Результати сесії (R/Race, Q/Qualifying, S/Sprint) з локального
    датасету (data/results.csv, data/qualifying.csv, data/sprint_results.csv)
    - без мережі й без FastF1. НЕ те саме, що /api/results: тут менший
    набір полів (немає TeamColor/HeadshotUrl/BroadcastName - їх у CSV
    немає), зате швидко й офлайн. Практики (FP1-3) сюди не входять -
    Ergast не веде результатів практик, лише FastF1 (/api/results, /api/laps)."""
    try:
        return csv_data.get_session_results(year, round_number, session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/csv/drivers/{year}/{round_number}/{session}")
def session_drivers_csv(year: int, round_number: int, session: str):
    """Короткий список пілотів сесії (R/Q/S) з локального датасету -
    CSV-аналог /api/drivers."""
    try:
        return csv_data.get_session_drivers(year, round_number, session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/csv/pitstops/{year}/{round_number}")
def pit_stops(year: int, round_number: int):
    """Піт-стопи гонки з локального датасету (data/pit_stops.csv): коло,
    номер стопу, час доби, тривалість у секундах. Функціонал, якого немає
    у FastF1-частині API."""
    try:
        return csv_data.get_pit_stops(year, round_number)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/data/status")
def data_status():
    """Стан локального датасету: версія Kaggle, коли перевіряли й
    оновлювали, які файли змінились, остання помилка."""
    return {**data_update.read_meta(), "update_every_hours": data_update.UPDATE_HOURS}


@app.post("/api/data/update")
def data_update_now():
    """Перевірити Kaggle й оновити дані зараз (не чекаючи планової
    перевірки) - напр. одразу після гонки. Не частіше ніж раз на 10 хв."""
    return data_update.update_manually()


@app.get("/api/event-summary/{year}/{round_number}")
def event_summary(year: int, round_number: int):
    """Підсумок етапу з CSV (миттєво, без FastF1): подіум, поул, найшвидше
    коло, сходи, найбільший приріст позицій, кількість кіл і те саме для
    спринту. null - якщо гонка ще не відбулась."""
    try:
        return event_info.get_event_summary(year, round_number)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/event-standings/{year}/{round_number}")
def event_standings(year: int, round_number: int):
    """Чемпіонат пілотів і конструкторів після етапу з дельтами до
    попереднього етапу й відставанням від лідера. Для етапу, що ще не
    відбувся, - після останнього завершеного (as_of_round)."""
    try:
        return event_info.get_event_standings(year, round_number)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/circuit-history/{year}/{round_number}")
def circuit_history(year: int, round_number: int):
    """Історія траси етапу (за circuitId, не за назвою гран-прі) станом до
    етапу: переможці останніх 10 гонок, топ-3 за перемогами, конверсія
    поулу, рік першого гран-прі й кількість гонок."""
    try:
        return event_info.get_circuit_history(year, round_number)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/circuit-facts/{year}/{round_number}")
def circuit_facts(year: int, round_number: int):
    """Факти траси: довжина кола, звичайна кількість кіл, піт-стопи
    (середня кількість на пілота, найшвидший), координати. Втрата часу на
    піт-стоп - у статичних JSON трас фронтенду (pit_loss)."""
    try:
        return event_info.get_circuit_facts(year, round_number)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/driver-media/{year}")
def driver_media(year: int):
    """Фото й кольори команд пілотів сезону: {код: {headshot_url,
    team_color, team}}. Є лише для сезонів 2023+ (OpenF1), для решти - {}."""
    return event_info.season_media(year)


@app.get("/api/races/{year}")
def list_races(year: int):
    """Календар сезону: етапи з датами всіх сесій вікенду (практики,
    кваліфікація, спринт, гонка), форматом вікенду й офіційною назвою."""
    try:
        return fastf1_client.get_season_schedule(year)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Season not found: {e}")


@app.get("/api/sessions/{year}/{round_number}")
def list_sessions(year: int, round_number: int):
    """Сесії конкретного етапу (FP1/FP2/FP3/Q/S/SQ/R) з датами та коротким
    кодом, який можна одразу підставляти в /api/results, /api/laps тощо."""
    try:
        return fastf1_client.get_event_sessions(year, round_number)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Event not found: {e}")


@app.get("/api/results/{year}/{round_number}")
def race_results(year: int, round_number: int):
    """Фактичні результати гонки (скорочення для /api/results/{year}/{round}/R)."""
    try:
        return event_info.fill_session_media(fastf1_client.get_session_results(year, round_number, "R"), year)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Race not found: {e}")


@app.get("/api/results/{year}/{round_number}/{session}")
def session_results(year: int, round_number: int, session: str):
    """Повні результати довільної сесії: R, Q, S, SQ, FP1, FP2, FP3
    (або повні назви: Race, Qualifying, Sprint, Sprint Qualifying, Practice 1...).
    Включає біо пілота, команду, стартову решітку, Q1/Q2/Q3, час/статус, очки."""
    try:
        return event_info.fill_session_media(
            fastf1_client.get_session_results(year, round_number, session), year)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")


@app.get("/api/drivers/{year}/{round_number}/{session}")
def session_drivers(year: int, round_number: int, session: str):
    """Короткий список пілотів сесії - для вибору перед /api/driver або /api/compare."""
    try:
        return event_info.fill_session_media(
            fastf1_client.get_session_drivers(year, round_number, session), year)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")


@app.get("/api/season-progress/{year}")
def season_progress(year: int):
    """Чемпіонат сезону для вкладок "Пілоти"/"Команди": очки й позиції після
    кожного етапу, перемоги/подіуми/поули/сходи, форма за 5 гонок, внесок
    пілотів в очки команди і чи можна ще математично виграти титул."""
    try:
        return driver_stats.get_season_progress(year)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/driver-season/{year}/{driver_ref}")
def driver_season(year: int, driver_ref: str):
    """Профіль пілота за сезон: статистика, результати по етапах, порівняння
    з напарником і оцінка сезону 0-100 зі складовими. driver_ref - як у
    drivers.csv і полі DriverId FastF1 ("max_verstappen")."""
    try:
        return driver_stats.get_driver_season(year, driver_ref)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/driver-career/{driver_ref}")
def driver_career(driver_ref: str):
    """Кар'єра пілота: підсумки, титули, команди, таблиця по сезонах."""
    try:
        return driver_stats.get_driver_career(driver_ref)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/driver-event/{year}/{round_number}/{driver_ref}")
def driver_event(year: int, round_number: int, driver_ref: str):
    """Контекст пілота на етапі: місце в чемпіонаті до/після, дуель з
    напарником, його результати на цій трасі в попередні роки."""
    try:
        return driver_stats.get_driver_event(year, round_number, driver_ref)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/driver/{year}/{round_number}/{session}/{driver}")
def driver_summary(year: int, round_number: int, session: str, driver: str):
    """Детальна інформація по одному пілоту в сесії: біо, підсумковий
    результат, стінти (шини/кількість кіл), найшвидше коло, усі кола."""
    return _fastf1_or_fallback(
        year, round_number, session,
        lambda: fastf1_client.get_driver_summary(year, round_number, session, driver),
        lambda: fallback.get_driver_summary(year, round_number, session, driver),
        "driver_data_unavailable",
    )


@app.get("/api/laps/{year}/{round_number}/{session}")
def session_laps(
    year: int,
    round_number: int,
    session: str,
    drivers: Optional[str] = Query(
        None, description="Коди пілотів через кому, напр. VER,HAM (якщо не задано - всі)"
    ),
):
    """Усі кола сесії: час кола, сектори, швидкості, шини, стінт, позиція
    на колі. Опційно фільтрується списком пілотів."""
    driver_list = _parse_drivers(drivers) if drivers else None
    return _fastf1_or_fallback(
        year, round_number, session,
        lambda: fastf1_client.get_session_laps(year, round_number, session, driver_list),
        lambda: fallback.get_session_laps(year, round_number, session, driver_list),
        "laps_unavailable",
    )


@app.get("/api/compare/{year}/{round_number}/{session}")
def compare_drivers(
    year: int,
    round_number: int,
    session: str,
    drivers: str = Query(..., description="Коди пілотів через кому, напр. VER,HAM,LEC"),
):
    """Порівняння кількох пілотів по колах однієї сесії: час кола, сектори,
    шини, позиція та найшвидше коло для кожного пілота."""
    driver_list = _parse_drivers(drivers)
    if len(driver_list) < 2:
        raise HTTPException(
            status_code=400,
            detail=tr("compare_needs_two"),
        )
    return _fastf1_or_fallback(
        year, round_number, session,
        lambda: fastf1_client.compare_drivers_laps(year, round_number, session, driver_list),
        lambda: fallback.compare_drivers_laps(year, round_number, session, driver_list),
        "compare_unavailable",
    )


@app.get("/api/weather/{year}/{round_number}/{session}")
def session_weather(year: int, round_number: int, session: str):
    """Погодні дані сесії протягом заїзду: температура повітря/траси,
    вологість, тиск, вітер, дощ."""
    return _fastf1_or_fallback(
        year, round_number, session,
        lambda: fastf1_client.get_session_weather(year, round_number, session),
        lambda: fallback.get_session_weather(year, round_number, session),
        "weather_unavailable",
    )


@app.get("/api/predict/{year}/{round_number}")
def predict_race(year: int, round_number: int):
    """Передбачення результатів гонки (поки що заглушка, підключиш свою модель)."""
    try:
        return predictor.predict_race(year, round_number)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not predict race: {e}")


@app.get("/api/analysis/{year}/{round_number}/status")
def race_analysis_status(year: int, round_number: int):
    """Чи порахований прогноз етапу для поточної версії моделі - лише
    перевіряє кеш, нічого не тренує (миттєво). {ready, generated_at, top3,
    hits}: hits - скільки з прогнозованого подіуму реально фінішували в топ-3."""
    return analysis.analysis_status(year, round_number)


@app.get("/api/analysis/{year}/{round_number}")
def race_analysis(year: int, round_number: int, refresh: bool = False):
    """Аналіз прогнозу однієї гонки: двоетапний ансамбль v2.0
    (DNF-класифікатор + LambdaRank ranker + XGBoost regressor),
    натренований на data/*.csv. Наповнення залежить від прогресу етапу -
    для гонки, що вже відбулась, використано всі дані включно з її
    кваліфікацією (саму гонку модель не бачить, щоб не було leakage); для
    поточного/майбутнього етапу - лише те, що вже доступно (форма
    пілотів/команд, і кваліфікація, якщо вона вже проїхана).

    Рахується один раз на етап і кешується в analysis_cache/ - повторні
    запити віддають той самий результат; передай ?refresh=true, щоб
    примусово перерахувати (наприклад, після появи нових даних)."""
    try:
        return analysis.analyze_race(year, round_number, refresh=refresh)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=tr("analysis_failed", error=e))


@app.websocket("/ws/replay/{year}/{round_number}/{session}")
async def session_replay(
    websocket: WebSocket, year: int, round_number: int, session: str,
    speed: float = 1.0, offset: float = 0.0,
):
    """Replay минулої сесії (OpenF1, 2023+) "як наживо": позиції на карті,
    телеметрія, кола, інтервали, race control - тим самим потоком, що й
    майбутній live-режим. speed - прискорення (0.25-20), offset - з якої
    секунди від запланованого старту сесії почати. Під час програвання
    клієнт може слати {"action": "pause" | "resume"} і
    {"action": "speed", "value": 5}. Протокол - README, "Replay сесії"."""
    await websocket.accept()
    try:
        # Обидва кроки блокуючі (мережа, ліміт OpenF1, календар FastF1) -
        # у потоці, щоб не зупиняти event loop для інших клієнтів.
        info = await asyncio.to_thread(openf1_client.find_session, year, round_number, session)
        intro, player = await asyncio.to_thread(live_replay.prepare_replay, info, offset, speed)
    except Exception as e:
        await websocket.send_json({"type": "error", "detail": tr("replay_unavailable", error=e)})
        await websocket.close(code=1011)
        return
    await websocket.send_json(intro)
    await live_replay.stream(websocket, player)


@app.get("/api/live/status")
def live_status():
    """Чи можна зараз дивитись гонку наживо: {configured (є облікові дані
    підписки OpenF1), status: live | upcoming | none | not_configured, session,
    opens_at (коли відкриється live-вікно, за 30 хв до старту), viewers}."""
    return live_feed.live_status()


@app.websocket("/ws/live")
async def live_session(websocket: WebSocket):
    """Поточна сесія наживо (платна підписка OpenF1) - той самий протокол, що й
    /ws/replay, але mode "live" і без керування. Поза live-вікном приходить
    {"type": "waiting", session, opens_at} і з'єднання закривається. Протокол -
    README, "Live-сесія через WebSocket"."""
    await live_feed.serve(websocket)

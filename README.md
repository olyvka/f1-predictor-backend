# F1 Predictor — Backend

FastAPI + [FastF1](https://github.com/theOehrly/Fast-F1) для аналізу та передбачення результатів гонок Формули-1.

## Запуск локально

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API буде доступне на `http://localhost:8000`, документація Swagger — на `http://localhost:8000/docs`.

⚠️ **macOS: LightGBM (`/api/analysis`) потребує `libomp`**, якого немає в системі за
замовчуванням — без нього `import lightgbm` падає з `OSError: Library not loaded: @rpath/libomp.dylib`:

```bash
brew install libomp
```

На Linux (у т.ч. `Dockerfile`) відповідний пакет — `libgomp1`, він уже прописаний у Dockerfile.

## Джерела даних

Два незалежні джерела, кожне для своєї задачі:

- **`data/*.csv`** (`app/csv_data.py`) - статичний історичний датасет (Ergast-схема): сезони,
  календар, підсумковий залік пілотів/конструкторів. Читається один раз при старті, без мережі.
- **FastF1** (`app/fastf1_client.py`) - деталі конкретної сесії, яких у CSV немає: кола, сектори,
  темп, погода, телеметрія, порівняння пілотів. Тягнеться з мережі (з диск-кешем у `cache/`).
- **ML-пайплайн** (`app/ml/`, `app/analysis.py`) - двоетапний ансамбль (DNF-класифікатор +
  LambdaRank ranker + XGBoost regressor), натренований на `data/*.csv` при першому зверненні до
  `/api/analysis`; для етапу, якого ще нема в `data/*.csv` (поточний/майбутній), добудовує фічі
  живими даними з FastF1. Деталі й обмеження - розділ нижче.

## Ендпоінти

| Метод | Шлях | Джерело | Опис |
|---|---|---|---|
| GET | `/api/seasons` | CSV | Усі сезони в датасеті |
| GET | `/api/standings/drivers/{year}` | CSV | Підсумковий залік пілотів сезону |
| GET | `/api/standings/constructors/{year}` | CSV | Підсумковий залік конструкторів сезону |
| GET | `/api/csv/results/{year}/{round}/{session}` | CSV | Результати R/Q/S (легша форма, без мережі) |
| GET | `/api/csv/drivers/{year}/{round}/{session}` | CSV | Короткий список пілотів сесії R/Q/S |
| GET | `/api/csv/pitstops/{year}/{round}` | CSV | Піт-стопи гонки |
| GET | `/api/event-summary/{year}/{round}` | CSV | Підсумок етапу: подіум, поул, найшвидше коло, сходи, найбільший приріст, спринт (`null` до гонки) |
| GET | `/api/event-standings/{year}/{round}` | CSV | Чемпіонат після етапу з дельтами й відставанням від лідера |
| GET | `/api/circuit-history/{year}/{round}` | CSV | Історія траси: переможці, рекордсмени, конверсія поулу |
| GET | `/api/circuit-facts/{year}/{round}` | CSV | Довжина кола, звичайна кількість кіл, піт-стопи, координати |
| GET | `/api/driver-media/{year}` | JSON | Фото й кольори команд пілотів (2023+, `data/driver_media.json` з OpenF1) |
| GET | `/api/analysis/{year}/{round}/status` | кеш | Чи порахований прогноз (миттєво, без тренування) |
| GET | `/api/races/{year}` | FastF1 | Календар сезону з датами всіх сесій вікенду |
| GET | `/api/sessions/{year}/{round}` | FastF1 | Сесії етапу (FP1-3/Q/S/SQ/R) |
| GET | `/api/results/{year}/{round}[/{session}]` | FastF1 | Результати сесії |
| GET | `/api/drivers/{year}/{round}/{session}` | FastF1 | Пілоти сесії |
| GET | `/api/driver/{year}/{round}/{session}/{driver}` | FastF1 | Деталі пілота: кола, стінти, найшвидше коло |
| GET | `/api/laps/{year}/{round}/{session}` | FastF1 | Усі кола сесії |
| GET | `/api/compare/{year}/{round}/{session}` | FastF1 | Порівняння пілотів по колах |
| GET | `/api/weather/{year}/{round}/{session}` | FastF1 | Погода сесії |
| GET | `/api/predict/{year}/{round}` | FastF1 | Передбачення (зараз заглушка) |
| GET | `/api/analysis/{year}/{round}?refresh=true` | ML | Аналіз прогнозу гонки (реальна модель v2.0, кешується на диск) |

`/api/csv/results` і `/api/csv/drivers` - **не заміна** `/api/results`/`/api/drivers`:
менший набір полів (немає `TeamColor`/`HeadshotUrl`/`BroadcastName` - їх у CSV нема),
підтримують лише R/Q/S (практики FP1-3 у Ergast-датасеті результатів не мають).
`FRONTEND_REQUIREMENTS.md` описує контракт саме FastF1-ендпоінтів - на нього й
орієнтуйся, якщо вже підключив фронт до `/api/results`.

## Мова відповідей

API відповідає українською (за замовчуванням) або англійською. Мова береться
з параметра `?lang=uk|en`, інакше із заголовка `Accept-Language` (фронт шле
мову інтерфейсу). Відповідь містить `Content-Language` і `Vary: Accept-Language`.

Перекладаються повідомлення про помилки, підписи фіч і примітки прогнозу
(`/api/analysis`) та назви складових оцінки сезону. Тексти — у
`app/i18n.py` (`MESSAGES`), підписи фіч — `FEATURE_LABELS` / `FEATURE_LABELS_EN`
у `app/analysis.py`. Кеш прогнозів (`analysis_cache/`) лишається українським
і перекладається в момент відповіді, тож його не треба перебудовувати.

```bash
curl -H "Accept-Language: en" http://localhost:8000/api/event-summary/2024/99
# {"detail": "Round 99 of the 2024 season is not in the dataset"}
```

## Тести

```bash
pip install -r requirements-dev.txt

pytest -m "not integration"   # юніт-тести: моки, без мережі, ~секунда
pytest -m integration         # інтеграційні: реальний FastF1 (2024 Bahrain GP, є в cache/)
pytest                        # усі разом
```

Юніт-тести (`tests/test_main.py`, `tests/test_predictor.py`,
`tests/test_fastf1_client_*.py`) мокають `app.fastf1_client`/`app.predictor` і
перевіряють роутинг, парсинг параметрів (`drivers=VER,HAM`), відповідність
кожної помилки → 404 з очікуваним повідомленням.

Інтеграційні тести (`tests/test_integration.py`) ганяють реальні запити через
`TestClient` без моків проти закешованого Гран-прі Бахрейну 2024 (round=1) і
перевіряють усі ендпоінти на справжніх даних.

`tests/test_analysis.py` (теж `integration` - будує реальний датасет/моделі з
`data/*.csv`, без мережі) перевіряє `/api/analysis` на всіх трьох станах прогресу
етапу (уже в датасеті / live з кваліфікацією / live без кваліфікації) і зіставлення
пілотів за кодом (регрес-тест на колізію `VER` = Jean-Éric Vergne ще й Max Verstappen).

## Підключення власної логіки передбачення

`app/predictor.py` (`/api/predict`) усе ще заглушка - лишили як є. Робоча модель живе окремо,
в `/api/analysis` (розділ 15 `FRONTEND_REQUIREMENTS.md`):

- `app/ml/config.py` - шляхи й константи спліту (train ≤2021 / val 2022-2023 / test ≥2024).
- `app/ml/dataset.py` - завантаження `data/*.csv` + feature engineering (leakage-safe
  rolling/expanding фічі: форма пілота/команди, темп кваліфікації, трек-афінність тощо).
- `app/ml/train.py` - тренування DNF-класифікатора (LightGBM), position ranker (LightGBM
  LambdaRank) і XGBoost-регресора + підбір ваги ансамблю.
- `app/ml/predict.py` - `predict_two_stage()`: P(DNF) окремо, очікувана позиція серед
  фінішерів - ансамбль ranker+XGB, фінальний `expected_pos = (1-P(DNF))·finisher_pos + P(DNF)·20`.
- `app/ml/live.py` - те саме, але для етапу, якого ще нема в `data/*.csv` (не оновлюється в
  реальному часі): фічі форми рахує "на льоту" з історії, кваліфікацію й склад пілотів тягне
  з FastF1, коли вона вже доступна.
- `app/analysis.py` - оркестрація: визначає прогрес етапу (`not_started`/`post_quali`/
  `completed`), вибирає статичний чи live-шлях, кешує готовий аналіз на диск
  (`analysis_cache/{year}_{round}.json`) - рахується один раз, далі віддається з кешу.

Датасет і моделі важкі (~20с на повному `data/*.csv`) - будуються один раз при першому
зверненні до `/api/analysis` і кешуються на диск (`models/dataset.joblib`, `models/bundle.joblib`),
прив'язані до fingerprint CSV-файлів (зміниш `data/*.csv` - перебудується сам). На Render
free-плані диск ефемерний (як і `cache/` для FastF1, див. нижче) - кеш просто збереться
заново на наступному холодному старті, це не помилка.

Обидва старих джерела (`app/csv_data.py`, `app/fastf1_client.py`) лишаються доступні, якщо
захочеш підключити інші фічі до `app/predictor.py` окремо від `app/ml/`.

## Деплой на Render.com (безкоштовно)

1. Заливаєш цей репозиторій на GitHub.
2. На [render.com](https://render.com) → New → Web Service → підключаєш репозиторій.
3. Render сам підхопить `Dockerfile`. Обери план **Free**.
4. У Environment Variables додай `FRONTEND_ORIGIN` = адреса твого фронта (додаси після деплою фронта).
5. Деплой займе кілька хвилин. Отримаєш URL типу `https://f1-predictor-backend.onrender.com`.

⚠️ На безкоштовному плані Render "засинає" після 15 хв бездіяльності — перший запит після паузи буде повільним (~30-50 сек).

## Важливо про FastF1 кеш

FastF1 обов'язково кешує завантажені дані на диск (`cache/` локально, `/tmp/fastf1-cache` на Render,
бо на free-плані диск не персистентний — кеш обнулиться при кожному рестарті контейнера).

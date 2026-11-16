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

- **`data/*.csv`** (`app/csv_data.py`) - історичний датасет (Ergast-схема): сезони,
  календар, результати, заліки. Читається в пам'ять, запити до нього - без мережі.
  **Оновлюється автоматично з Kaggle** - див. «Автооновлення даних» нижче.
- **FastF1** (`app/fastf1_client.py`) - деталі конкретної сесії, яких у CSV немає: кола, сектори,
  темп, погода, телеметрія, порівняння пілотів. Тягнеться з мережі (з диск-кешем у `cache/`).
- **ML-пайплайн** (`app/ml/`, `app/analysis.py`) - двоетапний ансамбль (DNF-класифікатор +
  LambdaRank ranker + XGBoost regressor), натренований на `data/*.csv` при першому зверненні до
  `/api/analysis`; для етапу, якого ще нема в `data/*.csv` (поточний/майбутній), добудовує фічі
  живими даними з FastF1. Деталі й обмеження - розділ нижче.

### Автооновлення даних

`data/*.csv` береться з Kaggle-датасету
[`jtrotman/formula-1-race-data`](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data)
(та сама схема, оновлюється після кожного етапу; ключ Kaggle не потрібен). Сервер
(`app/data_update.py`) перевіряє нову версію **при старті й далі кожні 6 годин**:

1. перевіряє реліз (усі файли на місці, колонки ті самі, жоден файл не «схуд») - інакше нічого не міняє;
2. атомарно підміняє змінені CSV (`fastf1_*.csv`, `driver_media.json` не чіпає);
3. перечитує дані в пам'ять - нові гонки видно одразу, без рестарту;
4. видаляє кеш прогнозу для гонок, у яких щойно з'явився результат (щоб показати прогноз проти факту);
5. у фоні перенавчає модель.

| | |
|---|---|
| `GET /api/data/status` | версія, коли перевіряли/оновлювали, які файли змінились, остання помилка |
| `POST /api/data/update` | перевірити зараз, не чекаючи (не частіше ніж раз на 10 хв) |
| `python scripts/update_data.py` | те саме з командного рядка |
| `DATA_UPDATE_HOURS` | інтервал перевірки в годинах; `0` - вимкнути |

На Render free-інстанс засинає і після пробудження стартує з даними, запеченими в Docker-образ:
якщо вони відстали від Kaggle, кожен холодний старт качатиме ~9 МБ і перенавчатиме модель (~хвилина).
Щоб цього уникнути, час від часу комітьте оновлені `data/*.csv` і передеплоюйте.

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
| WS | `/ws/replay/{year}/{round}/{session}?speed=&offset=` | OpenF1 | Replay минулої сесії "як наживо" - див. нижче |
| WS | `/ws/live` | OpenF1 (підписка) | Поточна сесія наживо - див. нижче |
| GET | `/api/live/status` | OpenF1 | Чи є зараз live і коли найближча сесія |

`/api/csv/results` і `/api/csv/drivers` - **не заміна** `/api/results`/`/api/drivers`:
менший набір полів (немає `TeamColor`/`HeadshotUrl`/`BroadcastName` - їх у CSV нема),
підтримують лише R/Q/S (практики FP1-3 у Ergast-датасеті результатів не мають).
`FRONTEND_REQUIREMENTS.md` описує контракт саме FastF1-ендпоінтів - на нього й
орієнтуйся, якщо вже підключив фронт до `/api/results`.

## Replay сесії через WebSocket

`WS /ws/replay/{year}/{round}/{session}?speed=1&offset=0&lang=uk` програє минулу сесію
(OpenF1, з 2023) **як наживо**: позиції машин на карті, телеметрію, кола, інтервали,
race control. Історичні дані OpenF1 безкоштовні, ключ не потрібен. Модуль -
`app/live_replay.py`.

Навіщо: перевірити live-режим фронта без справжньої гонки. Live-фід OpenF1
(MQTT/WebSocket, платний) шле ті самі записи, що й REST, тому майбутнє live-джерело
віддаватиме **той самий протокол**, і фронт не доведеться переписувати.

- `speed` - прискорення від 0.25 до 20. Без кешу мережа OpenF1 витримує приблизно ×10;
  швидше replay буферизується (годинник чекає на дані, а не губить їх).
- `offset` - з якої секунди від **запланованого** старту сесії почати. Наприклад,
  `offset=600` починає з 10-ї хвилини гонки. Може бути від'ємним (до години).

```js
const ws = new WebSocket("ws://localhost:8000/ws/replay/2026/20/R?speed=5&offset=600");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.send(JSON.stringify({ action: "pause" }));            // і "resume"
ws.send(JSON.stringify({ action: "speed", value: 10 }));
```

Повідомлення сервера (JSON, поле `type`):

| `type` | Коли | Вміст |
|---|---|---|
| `session` | першим (через ~3-8 с: сервер тягне дані сесії) | `session` (назва, траса, дати, `gmt_offset` траси, наприклад `"-03:00:00"`), `drivers` (номер, абревіатура, команда, `team_colour`, фото), `track.points` (контур траси `[[x, y], ...]` з найшвидшого кола, або `null`), `grid` (`{номер_пілота: стартова_позиція}` для гонки/спринту, інакше `null`), `replay` (`start`, `end`, `speed`) |
| `batch` + `catchup: true` | другим | стан на момент `start`: усі кола, стінти й race control до цієї миті, а також останні `position` / `intervals` на пілота й погода |
| `batch` | ~4 рази на секунду | `clock` (час replay, UTC) і `events: [{topic, data}]`, впорядковані за `data.date` |
| `state` | у відповідь на команду | `paused`, `speed`, `clock` |
| `error` | невідома команда або збій | `detail` (мовою `?lang=`) |
| `end` | кінець сесії | `clock`; після цього сервер закриває з'єднання |

`topic` - назва ендпоінта OpenF1, `data` - його запис без `session_key`/`meeting_key`:

- `location` (x, y, z, ~3.7 Гц на пілота);
- `car_data` (speed, throttle, brake, n_gear, rpm, drs, ~3.7 Гц);
- `position`, `intervals` (gap_to_leader, interval; лише гонки);
- `laps` (приходить, коли коло завершене);
- `stints` (приходить на старті стінту);
- `pit`, `race_control`, `weather`.

Поля збігаються з [документацією OpenF1](https://openf1.org/docs/).

Карта: `location.x/y` і `track.points` у тій самій системі координат, тож досить одного
масштабу. Координати лише вздовж траси (без положення поперек), тож між точками
(~0.27 с) варто інтерполювати.

## Live-сесія через WebSocket

`WS /ws/live?lang=uk` - поточна сесія **наживо** тим самим протоколом, що й replay
(`mode: "live"`, `replay: null`, без команд pause/speed). Потрібна платна підписка
OpenF1 (Sponsor, €9.90/міс). Модуль - `app/live_feed.py`.

**Налаштування:** скопіюйте `.env.example` у `.env` і впишіть `OPENF1_USERNAME` /
`OPENF1_PASSWORD` (логін і пароль, які OpenF1 видає після оплати) та
`OPENF1_PER_MINUTE=58`. Без облікових даних live вимкнено, replay працює як раніше.

Як це влаштовано:

- **Токен.** Бекенд міняє логін і пароль на OAuth2-токен (`POST https://api.openf1.org/token`),
  оновлює його за 5 хв до кінця години й після 401. Облікові дані ніколи не йдуть на фронт.
  Якщо отримати токен не вдалося, REST-запити йдуть анонімно: історичні дані й replay
  працюють, live - ні.
- **Одне MQTT-підключення на сервер** (`mqtt.openf1.org:8883`, TLS, пароль = токен) на всіх
  глядачів. Підключається з першим глядачем і закривається через хвилину після останнього.
- **Стан для нових глядачів:** на старті хаб тягне сесію через REST (з токеном), далі веде
  стан сам. Глядач отримує `session` → `batch` з `catchup: true` → батчі ~4 рази на секунду.
- **Кола** live-фід шле на старті кола й потім оновлює. Хаб пересилає коло, коли воно
  завершене (як replay), тож фронт не відрізняє live від replay.
- **Вікно.** OpenF1 віддає live від 30 хв до старту до 30 хв після кінця сесії. Поза ним
  `/ws/live` відповідає `{"type": "waiting", status, session, opens_at, calendar}` і
  закриває з'єднання. Наприкінці вікна глядачі отримують `end`.

| | |
|---|---|
| `GET /api/live/status` | `{configured, status: live \| upcoming \| none \| not_configured, session, opens_at, calendar: {year, round, session}, viewers}` |
| `WS /ws/live` | `session` / `batch` / `end` / `waiting` / `error` |

Ліцензія OpenF1: дані для особистого некомерційного використання (CC BY-NC-SA 4.0).
Для комерційного застосунку потрібна окрема домовленість з OpenF1.

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

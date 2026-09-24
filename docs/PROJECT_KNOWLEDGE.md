# PROJECT_KNOWLEDGE — база знаний проекта

Это главный human/AI knowledge document репозитория `mostransport-ml`.
Цель — чтобы новый участник команды или AI-агент (Codex, Claude/Sonnet)
мог понять проект без чтения истории чатов и без домыслов о том, чего
здесь ещё нет.

Документ не дублирует код и не копирует source files. Там, где нужны
точные технические детали, он ссылается на другие документы:

- [`../README.md`](../README.md) — короткий вход, с чего начать;
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — строгие технические границы,
  диаграммы, инварианты;
- [`ML_SERVING_CONTRACT.md`](ML_SERVING_CONTRACT.md) — точный HTTP-контракт
  serving-слоя;
- [`DEVELOPMENT.md`](DEVELOPMENT.md) — команды и workflow разработки;
- [`HACKATHON_RUNBOOK.md`](HACKATHON_RUNBOOK.md) — что делать после
  публикации официального ТЗ;
- [`../AGENTS.md`](../AGENTS.md) — операционные правила для AI-агентов.

Каждое существенное архитектурное утверждение ниже помечено одной из меток:

- **ТЕКУЩЕЕ СОСТОЯНИЕ** — существует и подтверждено текущим кодом/
  checkpoint'ом. Для ML-компонентов этого репозитория это дополнительно
  означает: покрыто нашей test suite. Для внешнего backend'а (§10.16)
  это означает: подтверждено чтением его исходного кода READ-ONLY — он не
  покрывается нашей test suite, и мы не выдаём его за проверенный нашими
  тестами;
- **ЗАПЛАНИРОВАНО** — согласованное направление, кода ещё нет;
- **TBD** — неизвестно, ждём официальных материалов хакатона;
- **ПОДТВЕРЖДЕНО ОРГАНИЗАТОРАМИ** — факт, заявленный организаторами
  хакатона, а не наше предположение.

Если что-то не попадает ни в одну из категорий явно — считайте это TBD,
а не фактом.

---

## 10.1. Назначение проекта

`mostransport-ml` — офлайн- и serving-часть ML-решения для хакатона
Мостранспорта: раннее прогнозирование задержек наземного городского
транспорта. Сейчас реализация ориентирована на автобусы, но доменные
имена (`vehicle`, `route`, `trip`, `stop`) намеренно generic, чтобы
архитектура масштабировалась на другой наземный транспорт без
переименований.

Бизнес-смысл: перевести диспетчера из реактивного режима ("узнал о
задержке — отреагировал") в проактивный ("предсказал задержку заранее —
успел скорректировать движение"). Примеры упреждающих действий диспетчера
— изменение количества ТС на линии, изменение времени стоянки на
остановках. **Это product/backend-логика, не ML output.** Python-сервис
прогнозирует численную задержку (`predicted_delay`); превращение прогноза
в решение диспетчера — вне зоны ответственности этого репозитория.

Хакатон длится **48 часов** — это определяет уровень сложности всей
инфраструктуры: она должна окупаться за 48 часов, а не быть
production-платформой.

Официальная метрика — **MAE фактической задержки** (ПОДТВЕРЖДЕНО
ОРГАНИЗАТОРАМИ). Вся offline-инфраструктура (`evaluation/`) построена
вокруг того, чтобы эту метрику можно было честно и воспроизводимо
посчитать, как только появится реальный target.

## 10.2. Подтверждённые факты организаторов

Список ниже — только то, что реально заявлено организаторами, без домыслов:

- хакатон длится 48 часов;
- текущая предметная реализация — автобусы; архитектура должна оставаться
  масштабируемой на другой наземный транспорт;
- обучающие данные будут выданы в виде custom CSV;
- полная схема CSV станет известна на старте хакатона;
- возможны дополнительные labels сверх базовых полей;
- организаторы предоставят Docker-образ online-эмулятора телеметрии;
- полная схема полей эмулятора будет опубликована на старте;
- будет опубликован официальный mapping: CSV fields ↔ emulator fields;
- официальная primary-метрика — **MAE фактической задержки**;
- внешние источники данных использовать можно;
- организаторские датасеты нельзя публиковать или использовать за
  пределами разрешённых условий хакатона;
- примеры упреждающих действий диспетчера: изменение количества ТС на
  линии, изменение времени стоянки на остановках (это backend/product
  слой, не ML).

## 10.3. Явный TBD

Ниже — то, что организаторы **ещё не сообщили**. Ни один из этих пунктов
не должен превращаться в тихое допущение внутри кода или документации:

- точная схема полей CSV (имена колонок, типы, единицы измерения);
- точная схема полей эмулятора;
- официальный mapping CSV ↔ emulator fields;
- точная семантика `target`/`delay` (что именно считается задержкой);
- единицы измерения delay (секунды? минуты?);
- официальная семантика prediction horizon: обязательность поля, какое
  конкретное значение использовать, и совпадает ли оно с provisional
  полем `horizon_minutes` из текущего serving-контракта. Само поле
  `horizon_minutes` в текущем provisional API уже означает «число минут»
  (это факт нашего API, а не организаторов) — TBD именно официальная
  семантика организаторов, см. §10.13 и
  [ML_SERVING_CONTRACT.md](ML_SERVING_CONTRACT.md) §4;
- детали подсчёта MAE (какие строки участвуют в scoring, есть ли
  фильтрация, агрегация по маршрутам и т.д.);
- Feature Contract — как именно offline feature-логика будет вызываться
  из online-пути (см. §10.15);
- Artifact Contract — точная схема ML Artifact Bundle сверх сегодняшнего
  `ArtifactMetadata` (см. §10.14);
- финальная схема `context` в serving-запросе — сегодня это намеренно
  непрозрачный JSON-объект (см. §10.13).

## 10.4. Владение (ownership) в команде

Четыре участника, у каждого своя зона. Важно различать **ответственность**
(кто в итоге отвечает за область) и **что реализовано сейчас** — они не
всегда совпадают по времени.

| Участник | Ответственность (planned + current) |
|----------|--------------------------------------|
| **fz** | offline-данные, data inspection, `DatasetManifest`, canonicalization boundary, target construction (после публикации ТЗ), temporal validation, MAE, baseline, experiment evaluation, leakage correctness |
| **Valeria** | Shared Feature Builder, preprocessing, обучение модели, model runtime, Artifact Bundle, artifact-backed predictor, Python ML serving |
| **Andrey** | backend, приём телеметрии, operational `VehicleState`, scheduler, `PredictionService`, ML HTTP client, risk/alerts, REST/WebSocket, frontend |
| **Lisa** | БД, reference data, BI / аналитическая работа с данными |

Что из этого уже реализовано **ТЕКУЩЕЕ СОСТОЯНИЕ**, а что только
**ЗАПЛАНИРОВАНО** — см. §10.7 (offline/serving модули) и §10.16 (backend
Андрея). Ни один planned-компонент не должен описываться как
существующий.

## 10.5. Границы репозитория

**Внутри этого репозитория:**

- offline-инструменты fz (инспекция CSV, manifest, canonicalization
  mechanism, метрики, temporal split, baseline, experiment log);
- provisional serving-фундамент Valeria (FastAPI shell, Predictor
  boundary, `MockPredictor`, минимальные artifact-метаданные);
- документация проекта.

**Намеренно вне этого репозитория:**

- backend Андрея — отдельный репозиторий `/home/fz/projects/backend`
  (NestJS/TypeScript), доступен нам только READ-ONLY для сверки границы;
- реальная модель, реальный Feature Builder, реальный Artifact Bundle —
  появятся здесь же, но только после публикации официального ТЗ;
- организаторский emulator, PostgreSQL, Redis, Kafka, scheduler,
  фронтенд — не наша ответственность и не должны появляться в этом
  репозитории.

## 10.6. Актуальная структура репозитория

Показаны только содержательные файлы и директории.

```
mostransport-ml/
├── README.md                       короткий вход
├── AGENTS.md                       правила для AI coding agents
├── CLAUDE.md                       короткий pointer на AGENTS.md/PROJECT_KNOWLEDGE
├── pyproject.toml                  зависимости, extras: dev, serving
├── uv.lock
│
├── docs/
│   ├── PROJECT_KNOWLEDGE.md        этот документ
│   ├── ARCHITECTURE.md             технические границы и инварианты
│   ├── ML_SERVING_CONTRACT.md      точный HTTP-контракт serving-слоя
│   ├── DEVELOPMENT.md              команды и workflow разработки
│   └── HACKATHON_RUNBOOK.md        план действий после публикации ТЗ
│
├── data/                           organizer-данные; НЕ в git (см. §10.17)
│   ├── raw/  interim/  processed/
│   └── README.md
│
├── notebooks/{fz,valeria}/         рабочие ноутбуки участников
├── experiments/README.md           куда пишется experiment log (JSONL)
│
├── src/mostransport_ml/
│   ├── data/                       inspection, manifest, canonicalization (fz)
│   ├── target/                     TargetSpec — метаданные таргета (fz)
│   ├── evaluation/                 MAE, temporal split, baseline (fz)
│   ├── experiments/                append-only JSONL experiment log (fz)
│   ├── features/                   ЗАПЛАНИРОВАНО, пусто (Valeria)
│   ├── models/                     ЗАПЛАНИРОВАНО, пусто (Valeria)
│   ├── artifacts/                  минимальный ArtifactMetadata (Valeria)
│   └── serving/                    provisional FastAPI-shell + MockPredictor (Valeria)
│
├── scripts/
│   ├── inspect_csv.py              CLI: первичный осмотр CSV
│   └── smoke_offline.py            synthetic end-to-end smoke offline-пайплайна
│
└── tests/                          61 offline-тест + 84 serving-теста (см. §10.11/10.12)
```

## 10.7. Реализованные модули

Ниже — по каждому production-пакету: назначение, что реально есть,
основные публичные сущности, зависимости, кто использует, чего намеренно
нет. Все версии — **ТЕКУЩЕЕ СОСТОЯНИЕ**, если не сказано иное.

### `data/` (fz)

Назначение: понять и подготовить сырой CSV, не выдумывая схему.

- `inspection.py` — `inspect_csv()`/`inspect_dataframe()` строят
  `InspectionReport` (колонки, dtypes, missing, дубликаты строк, опциональный
  time-range) без знания о конкретных transport-полях.
- `manifest.py` — `build_manifest()` строит `DatasetManifest`: hash,
  размер, схема, row count, time-range **отдельно по каждому файлу**
  (`SourceFile`), не предполагая единую схему для всех CSV в датасете.
  Сырые строки датасета никогда не сохраняются.
- `canonical.py` — `CanonicalMapping` + `apply_canonical_mapping()`:
  явный, caller-заданный механизм переименования колонок и проверки
  обязательных полей. Никакого захардкоженного списка полей. Подробнее о
  смысле этого модуля — см. §10.10, это важное уточнение.

Зависимости: только `pandas` + stdlib. Не импортирует ничего из
`serving/`/`artifacts/`. Используется: `scripts/inspect_csv.py`, будущим
target-builder'ом fz после публикации ТЗ.

### `target/` (fz)

- `spec.py` — `TargetSpec`: стабильная метаданная-обёртка для описания
  таргета (`name`, `unit`, `description`, `version`,
  `horizon_minutes: float | None`). Ничего не хардкодит.

**Чего намеренно нет:** построения самого таргета (target construction) —
это невозможно сделать честно до публикации официальной семантики
delay/horizon. Появится в первые часы хакатона (см.
[HACKATHON_RUNBOOK.md](HACKATHON_RUNBOOK.md)).

### `evaluation/` (fz)

- `metrics.py` — `mae()`: тонкая, но строгая обёртка над
  `sklearn.metrics.mean_absolute_error` с проверкой shape и finite-values.
- `temporal.py` — `split_by_time_boundaries()`: строго хронологический
  train/validation/test split по явно заданным границам. Никогда не
  использует random split — при раннем прогнозировании задержек порядок
  времени важнее почти всего остального в оценке качества.
- `baseline.py` — `MedianBaselineRegressor`: константный baseline
  (медиана target на train). Даёт честный baseline MAE в первые минуты
  после появления реального таргета — планку, которую обязана превзойти
  любая последующая модель.

Зависимости: `pandas`/`numpy`/`sklearn` + stdlib. Никаких внутренних
зависимостей на `data/`/`serving/`.

**Чего намеренно нет:** purge/embargo-окон, walk-forward CV — они
появятся только если станут нужны после понимания реального
horizon/leakage-профиля данных.

### `experiments/` (fz)

- `log.py` — `ExperimentLogger`/`ExperimentRecord`: append-only JSONL лог
  экспериментов. Обязательны только `model_name` и `validation_mae`;
  `validation_mae`, если задан, обязан быть конечным числом (NaN/±Infinity
  отклоняются с понятной ошибкой ещё до записи на диск) — это защищает
  лог от превращения в невалидный JSON.

### `features/` (Valeria) — ЗАПЛАНИРОВАНО

Пакет существует, но пуст. Здесь появится Shared Feature Builder — общая
feature-логика для offline-обучения и online-serving (см. §10.15).

### `models/` (Valeria) — ЗАПЛАНИРОВАНО

Пакет существует, но пуст. Здесь появится pipeline
предобработки/обучения модели.

### `artifacts/` (Valeria)

- `metadata.py` — `ArtifactMetadata`: минимальная, стабильная метаданная
  обученной модели (`model_version`, `created_at`, `model_type`,
  `target_name` обязательны; `target_unit`/`target_version`/
  `feature_schema_version`/`validation_mae` опциональны).
  `validation_mae`, если задан, обязан быть конечным и `>= 0`.

**Важно:** это **не** загрузчик реального Artifact Bundle. Модельного
файла (`.cbm`, joblib и т.п.) здесь нет и не может быть, пока не выбран
реальный формат модели. См. §10.14.

### `serving/` (Valeria)

Provisional FastAPI-фундамент, построенный заранее, чтобы не блокировать
интеграцию с backend Андрея на ожидании реальной модели. Подробно — §10.12
и [ML_SERVING_CONTRACT.md](ML_SERVING_CONTRACT.md).

- `schemas.py` — Pydantic DTO: `PredictionBatchRequest`/`Response`,
  `VehicleRequest`, `VehiclePrediction`, `PredictionStatus`,
  `HealthResponse`, `ReadyResponse`, `ErrorDetail`,
  `ValidationErrorResponse`, `ErrorResponse`.
- `service.py` — `Predictor` (`typing.Protocol`), `InferenceService`,
  `RawPrediction`, `PredictorNotReadyError`, `PredictorContractError`.
- `mock.py` — `MockPredictor`: детерминированный, всегда готов,
  `predicted_delay = 0.0` для каждого vehicle.
- `app.py` — `create_app(predictor)`: фабрика FastAPI-приложения,
  никогда не имеет предиктора по умолчанию.
- `mock_app.py` — единственное место, где `MockPredictor` подключается
  явно (`app = create_app(MockPredictor())`).

**Чего намеренно нет:** реальной feature-логики, реальной модели,
реального Artifact-loader'а, Python `VehicleState`, scheduler'а,
emulator-клиента — ничего из этого в serving-слое нет и не должно
появиться до Feature/Artifact Contract sync points (§10.18).

## 10.8. Граф зависимостей

Ниже — граф **по фактическим import'ам в коде**, а не по желаемой
архитектуре.

**Offline-пакеты (fz) — independent siloes.** `data/`, `target/`,
`evaluation/`, `experiments/` не импортируют друг друга и не имеют общих
внутренних зависимостей: каждый — самостоятельный, маленький модуль,
который сегодня используется только внешним кодом (`scripts/`, будущий
target-builder). Ни один из них не импортирует `pydantic` или `fastapi`.

**Serving-пакет (Valeria):**

```
schemas.py   (Pydantic DTO; внутренних зависимостей нет)
    ↑
service.py   (Predictor Protocol, InferenceService; зависит от schemas)
    ↑    ↑
mock.py  app.py   (оба зависят от schemas + service, друг от друга — нет)
    ↑
mock_app.py   (зависит от app.py + mock.py)
```

**Запрещённые зависимости** (проверено по коду, это текущий факт, а не
пожелание):

- offline-код (`data/`, `target/`, `evaluation/`, `experiments/`) НЕ
  должен зависеть от `fastapi`/`pydantic`/`uvicorn` — сегодня это так;
- `serving/` НЕ должен владеть операционным `VehicleState` или
  scheduler'ом — сегодня в `serving/` такого кода нет;
- `serving/` сегодня НЕ импортирует ничего из `data/`/`target/`/
  `evaluation/`/`experiments/` — общей feature-логики ещё нет, это честное
  текущее состояние, а не архитектурная ошибка (см. §10.15).

## 10.9. Offline data flow

Важно не путать два разных факта: «инструмент реализован и протестирован»
и «инструмент уже применён к реальным organizer-данным». Ниже они
разделены явно.

**ТЕКУЩИЙ ИНСТРУМЕНТАРИЙ** (весь код ниже реализован и покрыт тестами
прямо сейчас — не ждёт ничего, кроме реальных входных данных, чтобы быть
запущенным):

- `inspect_csv()` / `inspect_dataframe()` — профиль CSV/dataframe;
- `build_manifest()` — `DatasetManifest` для воспроизводимости;
- `apply_canonical_mapping()` — сам *механизм* явного rename +
  required-fields (готов и протестирован; конкретная конфигурация
  mapping для организаторской схемы — ещё нет, см. §10.10);
- `TargetSpec` — метаданная-обёртка для описания таргета (готова;
  реального таргета внутри неё ещё нет, см. §10.7 `target/`);
- `split_by_time_boundaries()` — утилита хронологического split'а,
  полностью реализована и протестирована;
- `mae()` — утилита расчёта MAE, полностью реализована и протестирована;
- `MedianBaselineRegressor` — median baseline, полностью реализован и
  протестирован.

Все семь пунктов выше покрыты offline-тестами уже сегодня — их
реализация не является ЗАПЛАНИРОВАННОЙ.

**ЗАПЛАНИРОВАННЫЙ data flow после публикации ТЗ** (что предстоит
*применить*, а не заново изобрести — плюс то, чего действительно ещё
нет):

```
organizer CSV
  ↓
inspect_csv() / build_manifest()        — уже готовы: просто запустить на реальном файле
  ↓
реальная конфигурация CanonicalMapping  — НЕ РЕАЛИЗОВАНО: rename/required_fields
                                           для организаторской схемы пока не заданы
  ↓
построение реального target             — НЕ РЕАЛИЗОВАНО: ждёт официальной семантики delay
  ↓
split_by_time_boundaries()               — уже готова: прогнать на реальных данных с target
  ↓
MedianBaselineRegressor + mae()          — уже готовы: прогнать и получить honest baseline MAE
  ↓
(далее) Feature Builder + обучение модели (Valeria) — не реализовано
```

## 10.10. Семантика `CanonicalMapping` — важное уточнение

Здесь легко ошибиться, поэтому формулируем максимально точно.

`CanonicalMapping`/`apply_canonical_mapping()` в `data/canonical.py` — это
**offline-механизм работы с `pandas.DataFrame`**. Он используется fz при
построении offline-пайплайна:

```
offline CSV
    ↓
CanonicalMapping (явный rename + required_fields)
    ↓
canonical domain-представление (тот же DataFrame, с переименованными колонками)
```

**Это НЕ обязательный runtime-конвертер** для online-пути и не
Python/Node мост между backend'ом и ML-сервисом. `data/canonical.py` не
импортируется и не будет импортироваться из `serving/`.

Online-путь (организаторский эмулятор → backend Андрея →
`PredictionBatchRequest` → Python ML Service) должен прийти к
**совместимой** доменной семантике (тем же именам сущностей —
`vehicle`, `route`, `trip`, `stop`) через будущий **Feature Contract**
(§10.15, §10.18), но не обязан физически вызывать `data/canonical.py`.
Это две разные точки одной и той же доменной модели, а не один общий
код-путь.

## 10.11. Стратегия оценки (evaluation)

- официальная метрика — **MAE фактической задержки** (ПОДТВЕРЖДЕНО
  ОРГАНИЗАТОРАМИ);
- split — строго хронологический (`split_by_time_boundaries`), random
  split не используется по умолчанию нигде в этом репозитории;
- baseline — медиана target на train (`MedianBaselineRegressor`);
- leakage: границы train/validation/test обязаны идти по времени;
  purge/embargo-логика — TBD, будет добавлена только если понадобится
  после появления реального target/horizon;
- 61 offline-тест покрывает `data/`, `target/`, `evaluation/`,
  `experiments/` (границы, ошибки, round-trip сериализации, детерминизм
  split'а).

## 10.12. Serving-архитектура

```
FastAPI (app.py: create_app(predictor))
    ↓
InferenceService (service.py)
    ↓
Predictor (typing.Protocol)
    ↓
MockPredictor (mock.py) — ТЕКУЩЕЕ СОСТОЯНИЕ
    ↓ (после Artifact Contract)
artifact-backed predictor — ЗАПЛАНИРОВАНО
```

Ключевые свойства, все — **ТЕКУЩЕЕ СОСТОЯНИЕ**:

- **explicit mock**: `create_app()` никогда не подставляет
  `MockPredictor` по умолчанию — его явно выбирает только
  `mock_app.py`. Так production-запуск не может тихо оказаться на mock'е.
- **stateless serving**: между запросами не хранится operational-состояние
  — ни `VehicleState`, ни история конкретного vehicle, ни состояние
  scheduler'а. Вся request-specific информация приходит в каждом запросе
  заново. Это **не** запрет на persistent-объекты внутри predictor'а:
  загруженный model-артефакт, неизменяемые lookup-данные, объекты
  препроцессинга или read-only исторические агрегаты — это ожидаемая
  часть будущего artifact-backed predictor'а (см. §10.14), а не нарушение
  инварианта. Запрещено конкретно то, что делает backend: mutable история,
  производная от отдельных запросов и накапливаемая между ними.
- **readiness**: `GET /ready` отражает реальную готовность predictor'а, а
  не просто факт, что процесс жив (для этого есть отдельный `/health`).
  Ошибка внутри проверки готовности превращается в безопасный `503`, а не
  в необработанное падение.
- **request snapshot**: перед вызовом predictor'а `InferenceService`
  снимает неизменяемый снимок (`_RequestSnapshot`) полей, влияющих на
  контракт (`prediction_time`, `horizon_minutes`, упорядоченные
  `vehicle_id`). После вызова текущее состояние запроса сверяется с этим
  снимком — так predictor не может незаметно подменить данные запроса
  (например, очистить список vehicles) и обойти проверку идентичности
  1:1. Ответ всегда строится из снимка, а не из потенциально изменённого
  запроса.
- **runtime Predictor contract**: `typing.Protocol` документирует
  контракт, но не проверяет его в рантайме — это делает
  `InferenceService`. `is_ready()` обязан вернуть ровно `bool` (не
  truthy-строку вроде `"false"`), а `model_version()` готового predictor'а
  обязан быть непустой строкой. Нарушение — это `PredictorContractError`,
  который превращается в безопасный `503` (`/ready`) или `500`
  (`/predict`), но никогда не в необработанное падение.
- **safe error handling / strict response semantics**: см. §10.13 и
  [ML_SERVING_CONTRACT.md](ML_SERVING_CONTRACT.md) §6 — 422 санитизируется
  (не эхо́ит `input`/`ctx`/неизвестные имена полей), server-side логи не
  содержат сообщений исключений и трасировок, ответ никогда не содержит
  `NaN`/`Infinity`.

## 10.13. Контракт прогнозирования — краткое summary

Полный контракт — в [ML_SERVING_CONTRACT.md](ML_SERVING_CONTRACT.md).
Здесь только сводка.

Запрос (`PredictionBatchRequest`):

```
prediction_time     — datetime
horizon_minutes?    — None либо конечное число > 0, в минутах (текущий provisional API;
                       официальная семантика horizon у организаторов — TBD, см. §10.3)
vehicles[]
    vehicle_id       — строка
    context          — непрозрачный JSON-объект
```

Ответ (`PredictionBatchResponse`):

```
model_version
predictions[]
    vehicle_id
    predicted_delay  — конечное число, обязателен при status == "ok"
    status           — "ok" | "insufficient_data" | "error"
```

Почему `context` непрозрачен: официальная схема CSV/эмулятора ещё TBD, и
никакое поле внутри `context` сегодня не читается ни `MockPredictor`, ни
serving-слоем — это чистый HTTP-мост для интеграции до появления
официального mapping. Как только он появится, `context` будет заменён
или ужесточён в типизированную domain-схему (Canonical Schema sync point,
§10.18).

## 10.14. Artifact-фундамент

Сегодня существует только `ArtifactMetadata` (см. §10.7) — стабильная,
строго-JSON-сериализуемая метаданная обученной модели. Это **не**
Artifact Bundle и не загрузчик модели.

Планируемый (**ЗАПЛАНИРОВАНО**) полноценный Artifact Bundle должен
содержать:

- саму модель;
- версию/схему набора признаков (feature schema/version);
- метаданные препроцессинга;
- исторические агрегаты, если понадобятся;
- метаданные таргета/модели.

## 10.15. Согласованность offline ↔ online (training-serving consistency)

Главный будущий инвариант:

    ОДНА И ТА ЖЕ FEATURE-ЛОГИКА
    OFFLINE ↔ ONLINE

Идея: feature-логика должна жить в одном месте (`features/`) и
импортироваться и офлайн-обучением, и online-сервингом — никогда не
дублироваться (например, копия в ноутбуке и другая копия внутри FastAPI).
Раздельные копии почти гарантированно разойдутся (training-serving skew) и
сделают offline-качество модели недостижимым в проде.

**Прямо сейчас реального `FeatureBuilder` не существует.** `features/` —
пустой пакет-заглушка (см. §10.7). Он появится только после Feature
Contract sync point (§10.18).

## 10.16. Внешняя граница: backend Андрея

Backend — отдельный репозиторий (`/home/fz/projects/backend`, NestJS/
TypeScript), доступен нам только READ-ONLY.

**ТЕКУЩЕЕ СОСТОЯНИЕ backend'а** (проверено по коду на момент написания):

```
POST /telemetry/events
    ↓
TelemetryService
    ↓
VehicleStateService
```

- `VehicleStateService.updateState()` сегодня **не** хранит состояние —
  формирует и возвращает объект состояния по одному входящему событию, без
  персистентного `Map<vehicleId, VehicleState>`.
- `PredictionSchedulerService` существует, но это ранний scaffold (в
  исходнике backend'а он прямо помечен как заглушка): проверяет только
  `Boolean(event.vehicle_id)`, не содержит ни расписания, ни вызова
  ML-сервиса.
- `PredictionService` и ML HTTP client, которые должны вызывать наш
  `POST /api/v1/predict/batch`, **ещё не существуют** в backend'е.
- Официальной интеграции с organizer emulator в backend'е тоже пока нет —
  TBD.

**ЗАПЛАНИРОВАННАЯ граница** (backend → Python):

```
backend PredictionService
    ↓
ML HTTP Client
    ↓
POST /api/v1/predict/batch  (этот репозиторий)
```

Разделение ответственности:

| Владеет backend (Андрей) | Владеет Python (Valeria/fz) |
|---------------------------|-------------------------------|
| операционное состояние (`VehicleState`) | feature-трансформации |
| недавняя телеметрия конкретного vehicle | инференс модели |
| scheduling | ML-артефакты |
| reference context, risk/alerts | — |

Python-репозиторий никогда не должен дублировать левую колонку: ни своего
`VehicleState`, ни своего scheduler'а, ни emulator-клиента. Это про
*operational* состояние конкретных vehicle/запросов — не про сам
predictor: загруженный внутри него model-артефакт или статичные
lookup-данные владением backend'а не являются и этому правилу не
противоречат (см. §10.12).

## 10.17. Безопасность и организаторские данные

- организаторские датасеты не коммитятся: `data/raw/`, `data/interim/`,
  `data/processed/` в `.gitignore`;
- перед отправкой organizer-данных в любой внешний инструмент/сервис/AI —
  сначала проверить правила хакатона;
- serving-слой не пишет payload запроса в логи (особенно `context`,
  который после старта будет содержать организаторские данные) — в логах
  только метаданные (размер батча, версия модели, тип ошибки,
  успех/неудача);
- ответ `422` санитизирован — не содержит `input`, `ctx`, эхо
  произвольных имён полей запроса;
- runtime-ошибки (`500`/`503`) не содержат сообщения исключения или
  traceback — ни в ответе клиенту, ни в логах.

Подробнее — [ML_SERVING_CONTRACT.md](ML_SERVING_CONTRACT.md) §6 и
[ARCHITECTURE.md](ARCHITECTURE.md) §13.

## 10.18. Точки расширения после публикации ТЗ

Порядок, в котором команда должна синхронизироваться:

1. **ML Task Spec** — семантика target/delay/horizon/метрики.
2. **Canonical Schema** — общая доменная семантика offline ↔ online.
3. **Feature Contract** — интерфейс `features/`, который вызывают и
   обучение, и serving.
4. **Artifact Contract** — точная схема ML Artifact Bundle, которую
   грузит serving-слой.

Только после этого имеет смысл строить: target builder, `FeatureBuilder`,
модель, реальный Artifact Bundle, artifact-backed predictor, ужесточённую
схему `context` в запросе. Подробный план по фазам —
[HACKATHON_RUNBOOK.md](HACKATHON_RUNBOOK.md).

## 10.19. Non-goals (до официального ТЗ)

Намеренно не строится до публикации официального ТЗ:

- Kafka, Redis, MLflow, DVC, Airflow;
- feature store, сложный model registry;
- Python-реализация `VehicleState`;
- scheduler в Python;
- база данных в этом репозитории;
- risk rules / классификация LOW-MEDIUM-HIGH;
- любая угаданная transport-схема (придуманные названия полей CSV или
  эмулятора).

## 10.20. Правила для AI-агентов

Полная версия — [`../AGENTS.md`](../AGENTS.md). Кратко:

Перед любым изменением агент обязан:

1. прочитать этот документ (`PROJECT_KNOWLEDGE.md`);
2. прочитать [`ARCHITECTURE.md`](ARCHITECTURE.md);
3. если задача касается serving — прочитать
   [`ML_SERVING_CONTRACT.md`](ML_SERVING_CONTRACT.md);
4. проверить тесты (`uv run pytest -q`);
5. явно определить для своей задачи, что из затрагиваемого —
   ТЕКУЩЕЕ СОСТОЯНИЕ, что ЗАПЛАНИРОВАНО, а что TBD.

Агенту запрещено:

- угадывать organizer-схему (поля CSV/эмулятора, единицы delay, horizon);
- превращать TBD в факт;
- дублировать feature-логику вместо использования общего `features/`
  (после того как он появится);
- создавать в Python операционный `VehicleState` или scheduler;
- менять ownership без явного решения команды;
- расширять scope "про запас", не запросив об этом;
- менять backend Андрея, если это явно не поставленная задача.

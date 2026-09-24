# Архитектура

Источник истины по техническим границам системы: что подтверждено, что
только запланировано, кто чем владеет. Читать перед добавлением новой
подсистемы или абстракции.

Это документ 48-часового хакатона. Он намеренно тонкий. Там, где что-то
пока нельзя знать, стоит явная пометка **TBD**, а не догадка.

Роль этого документа — строгие технические границы и потоки данных. Более
широкое, "человеческое" описание проекта (зачем он нужен, кто чем владеет
в терминах процесса, как читать репозиторий) — в
[`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md).

---

## 1. Назначение

Раннее прогнозирование задержек/изменений наземного городского транспорта
(хакатон Мостранспорта). Бизнес-цель — перевести диспетчеров из
реактивного режима в проактивный: например, скорректировать количество ТС
на маршруте или время стоянки на остановке до того, как задержка
накопится.

Этот репозиторий содержит **offline ML-часть** (fz): превращение
организаторского CSV в валидированный target, leakage-safe схему оценки
и (когда появится реальная модель) обученный артефакт. С текущего
checkpoint'а он также содержит **provisional serving-фундамент**
(Valeria): FastAPI-сервис с детерминированным mock-предиктором, собранный
заранее, чтобы не блокировать интеграцию с backend'ом на ожидании
реальной модели. Контракт serving-слоя — в
[`ML_SERVING_CONTRACT.md`](ML_SERVING_CONTRACT.md).

## 2. Подтверждённые факты организаторов

На момент написания, до публикации полного ТЗ:

- обучающие данные будут выданы в виде custom CSV;
- полная схема по полям пока не опубликована — появится на старте
  хакатона;
- отдельный Docker-образ будет эмулировать online-поток телеметрии;
- будет предоставлен официальный mapping CSV ↔ emulator fields;
- primary scoring-метрика — **MAE фактической задержки**;
- точная семантика target/delay/horizon пока не определена;
- внешние источники данных разрешены для обогащения;
- организаторские датасеты нельзя публиковать или использовать за
  пределами разрешённых условий хакатона.

## 3. Целевая end-to-end архитектура (CURRENT + PLANNED)

Диаграмма ниже показывает **согласованное направление системы целиком** —
она сознательно смешивает уже реализованные шаги с ещё не реализованными,
чтобы показать, как они должны состыковаться в итоге. Она не описывает
«систему, как она работает сегодня»: большая часть online-цепочки
(`PredictionService`, `ML Client`, `Risk Rules / Alerts`) и часть
offline-цепочки (`target construction`, `shared feature logic`, `model
training`, `ML Artifact Bundle`) — ЗАПЛАНИРОВАНЫ, кода для них ещё нет.
Что реально существует прямо сейчас — по каждой стороне отдельно и точно
— описано ниже в §5 (backend Андрея) и §6–§7 (offline/serving fz и
Valeria); там прямо разделены CURRENT и PLANNED части.

```
OFFLINE                                   ONLINE

organizer CSV                             Organizer Emulator (Docker)
    |                                          |
offline ingest                            Node TelemetryService
    |                                          |
data understanding / quality              Node VehicleStateService
    |                                          |
alignment / canonical representation      PredictionScheduler
    |                                          |
target construction                       PredictionService
    |                                          |
temporal evaluation                       ML Client
    |                                          |  HTTP/JSON
shared feature logic  ------------------> Python ML Service
    |                                          |
model training                            predicted delay
    |                                          |
evaluation (primary metric: MAE)          Backend Risk Rules / Alerts
    |                                          |
ML Artifact Bundle  ---------------------> (loaded by Python ML Service)
                                                |
                                           WebSocket / REST
                                                |
                                           React Dispatcher Dashboard
```

**Ключевой инвариант:** backend владеет операционным состоянием; Python
владеет ML-трансформациями и инференсом. Будущий Python ML Service не
хранит operational-историю по конкретным vehicle (её приносит каждый
запрос от backend'а) — но при этом обычным образом держит в памяти
загруженный model-артефакт и другие статичные объекты инференса между
запросами, как и любой обычный inference-сервис. «Не хранит историю
по vehicle» и «не имеет вообще никакого состояния в процессе» — разные
утверждения; верно первое, не второе.

## 4. Границы ownership

| Владелец | Ответственность |
|--------|----------------|
| **fz** (этот репозиторий) | data → target → evaluation correctness |
| **Valeria** | features → model → serving |
| **Andrey** | backend / операционное состояние / scheduler / интеграция / realtime / frontend |
| **Lisa** | БД / reference data / BI / аналитика |

## 5. Текущий checkpoint backend'а Андрея

Read-only ссылка (backend лежит в соседнем репозитории, NestJS/TypeScript).

**Текущая реализация** (сверено с исходным кодом backend'а):

- `POST /telemetry/events` → `TelemetryController`
- `TelemetryService`, вызывающий `VehicleStateService`
- `VehicleStateService` — ранний scaffold: сейчас в основном
  формирует/возвращает объект состояния по одному входящему событию — он
  **ещё не** реализует персистентный `Map<vehicleId, VehicleState>`
- `PredictionScheduler` существует как scaffold (в исходнике backend'а
  прямо помечен комментарием как заглушка); проверяет только
  `Boolean(event.vehicle_id)`, без реальной логики планирования

**Запланированная архитектура** (ещё не реализована):

- memory-first `VehicleState`, концептуально `Map<vehicleId, VehicleState>`
- окно недавней телеметрии по каждому vehicle
- необязательность постоянного хранения `VehicleState` в PostgreSQL

Наблюдаемая текущая форма события (`TelemetryEventDto`): `vehicle_id`,
`event_time`, `lon`, `speed`, `direction`, опционально `route_id`,
`trip_id`, `stop_id`. Это **не** трактуется здесь как подтверждённая
каноническая схема — это текущий checkpoint backend'а, показан только
чтобы держать доменные имена (`vehicle`/`route`/`trip`/`stop`)
согласованными по команде. Скорее всего изменится, как только появится
официальная схема CSV/эмулятора.

Backend владеет, и этот репозиторий не должен дублировать (ни текущее, ни
запланированное):

- runtime-приём телеметрии
- операционный `VehicleState`
- окно недавней телеметрии
- `PredictionScheduler` / `PredictionService`
- ML Client
- risk rules, alerts
- WebSocket, REST, интеграцию с frontend
- PostgreSQL/Prisma (routes, stops, trips, schedule, prediction_history, alerts)

Этот репозиторий не реализует и не будет реализовывать ничего из
перечисленного выше.

## 6. Offline-ответственность (fz)

Что реализовано здесь, в `src/mostransport_ml/`:

- `data/inspection.py` — generic, schema-agnostic профилирование
  CSV/dataframe.
- `data/manifest.py` — метаданные воспроизводимости для версии датасета
  (hash, схема, row count), никогда не сырые строки.
- `data/canonical.py` — явный механизм rename + required-field, без
  захардкоженного списка полей (см. §9).
- `target/spec.py` — метаданная-обёртка для описания таргета, не
  построение самого таргета (см. §10).
- `evaluation/metrics.py` — MAE, строго валидированная.
- `evaluation/temporal.py` — хронологический train/validation/test split.
- `evaluation/baseline.py` — константный baseline по медиане train.
- `experiments/log.py` — append-only JSONL лог экспериментов.

Граница ответственности: **data → target → evaluation correctness**. Этот
репозиторий гарантирует, что как только появятся реальный CSV и
определение таргета, их можно будет понять, канонизировать, разбить по
времени и оценить корректно и воспроизводимо.

## 7. Будущая интеграция Valeria

Зарезервированные пакеты: `features/`, `models/`, `artifacts/`,
`serving/`. Они существуют, чтобы Valeria могла добавить реальное
содержимое без реорганизации репозитория.

**Текущая реализация** (provisional, pre-task-release фундамент):

- `serving/` — FastAPI-shell: `GET /health`, `GET /ready`,
  `POST /api/v1/predict/batch`. Никакой model/feature-логики на уровне
  эндпоинтов — см. границу `Predictor` в `serving/service.py`. Эта
  граница проверяется в рантайме, а не является просто типовой подсказкой
  `typing.Protocol`: predictor не должен мутировать переданный ему
  request, `is_ready()` обязан вернуть ровно `bool`, а `model_version()`
  готового predictor'а обязан быть непустой строкой — нарушения
  превращаются в безопасный `503`/`500`, а не в неконтролируемое падение
  или тихо неверный ответ.
- `serving/mock.py` — `MockPredictor`: детерминированный, всегда готов,
  фиксированный `model_version="mock-v0"`, `predicted_delay=0.0` для
  каждого vehicle. Используется для интеграции с backend/frontend до
  появления реальной модели и рассчитан оставаться полезным и после (например,
  для локальной разработки), а не быть одноразовым. Подключается только
  явно (`serving/mock_app.py`) — `serving/app.py::create_app` никогда не
  выбирает его по умолчанию.
- `artifacts/metadata.py` — `ArtifactMetadata`: минимальная, стабильная
  форма метаданных (`model_version`, `created_at`, `model_type`,
  `target_name`, опционально `target_unit`/`target_version`/
  `feature_schema_version`/`validation_mae`). Не загрузчик артефакта —
  реальный формат модельного файла пока неизвестен.
- `features/` и `models/` остаются пустыми — feature- и training-логики
  ещё нет.

Полный контракт запроса/ответа — [`ML_SERVING_CONTRACT.md`](ML_SERVING_CONTRACT.md).
**Поле `context` у каждого vehicle в запросе намеренно непрозрачно
(нетипизированный JSON-объект) и ДОЛЖНО быть заменено/ужесточено, как
только появится официальный mapping CSV ↔ emulator** — ни `MockPredictor`,
ни serving-слой не читают из него transport-specific ключи.

**Запланировано после публикации полного ТЗ** (ещё не реализовано):

- общий Feature Builder (`features/`), используемый и offline-обучением,
  и online-сервингом
- pipeline предобработки/обучения модели (`models/`)
- реальная реализация Artifact Bundle (`artifacts/`) — настоящий
  загрузчик модели поверх сегодняшнего `ArtifactMetadata`
- artifact-backed `Predictor`, заменяющий `MockPredictor`, подключаемый в
  тот же `serving/app.py::create_app` без изменения HTTP-слоя

Всё перечисленное, текущее и запланированное, — зона Valeria.

## 8. Offline data flow

См. диаграмму в §3. Конкретно, offline (шаги 1–4, 6–7 — уже реализованный
и протестированный инструментарий, готовый к запуску на реальных данных;
шаги 5 и 8 — ещё не реализованы, появятся после публикации ТЗ; подробное
разделение — [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md) §10.9):

1. Организаторский CSV попадает в `data/raw/` (никогда не коммитится).
2. `scripts/inspect_csv.py` профилирует его — схема, missing, дубликаты.
3. Строится `DatasetManifest` для воспроизводимости.
4. `CanonicalMapping` (как только реальные поля станут известны)
   переименовывает/валидирует в общее доменное представление.
5. Построение таргета (появится после публикации ТЗ) даёт реальный label.
6. `split_by_time_boundaries` строит train/validation/test хронологически.
7. `MedianBaselineRegressor` + `mae()` сразу дают честный baseline.
8. Далее: feature/model pipeline Valeria обучается на том же split'е и
   каноническом представлении, логируя запуски через `experiments/log.py`.

## 9. Стратегия канонической схемы

Ни один канонический список полей нигде в этом репозитории не
захардкожен. Официальная схема CSV и схема эмулятора неизвестны до старта
хакатона, организатор предоставит mapping CSV ↔ emulator.

Вместо этого `data/canonical.py` даёт только *механизм*:

```python
CanonicalMapping(
    source_name=...,
    rename={...},  # только явно заданное, ничего не выводится автоматически
    required_fields=(...),  # только явно заданное
)
```

`apply_canonical_mapping()` переименовывает ровно перечисленные колонки и
проверяет ровно перечисленные обязательные поля. Никакого feature
engineering здесь нет. Как только реальная схема CSV станет известна, это
точка, где offline-поля организаторского CSV приводятся к общему
каноническому доменному представлению.

**Важное уточнение семантики — читать внимательно:** `CanonicalMapping`
— это **только** offline-механизм работы с `pandas.DataFrame`, применяемый
к организаторскому CSV. Он не обязателен и не должен физически
вызываться для online/emulator-пути:

```
organizer CSV  →  CanonicalMapping  →  совместимая каноническая семантика  (offline, этот репозиторий)
online/backend →  будущий Feature Contract  →  совместимая доменная семантика  (НЕ через data/canonical.py)
```

Backend/online-путь должен прийти к **совместимой** (не обязательно
буквально тем же кодом достигнутой) доменной семантике через будущий
Feature Contract — но никогда не обязан и не должен физически вызывать
`data/canonical.py`. Это два разных механизма для одной и той же целевой
доменной модели, а не общий код-путь. Подробнее —
[`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md) §10.10.

## 10. Стратегия target/evaluation

Организатор подтвердил метрику (MAE), но не семантику delay/horizon.
`target/spec.py` даёт стабильную форму метаданных `TargetSpec` (name,
unit, description, version, опциональный `horizon_minutes`), ничего не
хардкодя — ни предполагаемого horizon, ни единицы измерения, ни названия.

Реальное построение таргета (превращение сырых полей в колонку-label)
намеренно не реализовано и будет построено fz сразу после публикации ТЗ.

## 11. MAE baseline

`evaluation/baseline.MedianBaselineRegressor` предсказывает медиану
training-таргета, полностью игнорируя признаки. В сочетании с
`evaluation/metrics.mae` это даёт baseline validation MAE в первые минуты
после появления реального таргета — планку, которую обязана превзойти
любая последующая модель.

## 12. Согласованность training/serving

Критическое будущее правило: **одна и та же feature-логика должна
использоваться offline и online.** Никогда не должно появиться второй,
параллельной реализации feature-логики в ноутбуке или продублированной
внутри FastAPI-сервиса. Как только `features/` получит реальное
содержимое, и offline-обучение, и online-serving будут импортировать его
из одного и того же места.

## 13. Конфиденциальность данных / политика репозитория

- Организаторские данные никогда не коммитятся и не публикуются.
  `data/raw/`, `data/interim/`, `data/processed/` в `.gitignore`
  (отслеживаются только README/`.gitkeep`).
- Перед отправкой организаторских данных в любой внешний сервис,
  инструмент или AI-ассистента — сначала свериться с правилами хакатона.
- Код и метаданные (схема, hash, row count) отделены от confidential
  сырых данных — см. `DatasetManifest`, который никогда не хранит
  реальные строки.
- Входящие serving-запросы — особенно `context` каждого vehicle, который
  после старта хакатона будет нести организаторские данные — не должны
  сбрасываться в логи. В логах допустима только метаданные (размер
  батча, версия модели, тип ошибки, успех/неудача); сообщения исключений
  и трассировки намеренно исключены, поскольку исключение predictor'а
  само может содержать данные, производные от запроса. См.
  [`ML_SERVING_CONTRACT.md`](ML_SERVING_CONTRACT.md) §6.

## 14. Подтверждённые инварианты

- Официальная primary-метрика = MAE.
- Точная семантика delay: TBD.
- Точная семантика horizon: TBD.
- Схема CSV: TBD.
- Схема эмулятора: TBD.
- Официальный mapping CSV ↔ emulator предоставят организаторы.
- Основной online-источник телеметрии = организаторский Docker-эмулятор.
- Node-backend владеет `VehicleState`, недавней телеметрией и scheduler'ом.
- Python не должен дублировать операционный `VehicleState`.
- Offline и online обязаны сойтись на совместимом каноническом доменном
  представлении.
- Один и тот же Feature Builder должен в итоге использоваться offline и
  online.
- Организаторские датасеты нельзя коммитить или публиковать.
- Автобусы — текущая цель реализации; доменные имена остаются
  transport-generic (`vehicle`/`route`/`trip`/`stop`).
- Внешнее обогащение данных разрешено, но не входит в pre-hackathon
  critical path.
- Поле `context` в provisional serving-контракте намеренно непрозрачно и
  будет заменено/ужесточено, как только появится официальный mapping
  CSV ↔ emulator.
- Python ML Service (`serving/`) не хранит operational-состояние между
  запросами — ни `VehicleState`, ни историю по vehicle, ни состояние
  scheduler'а; см. §3. Это не запрещает predictor'у держать в памяти
  загруженный model-артефакт, статичные lookup-данные или объекты
  препроцессинга между запросами — это обычная часть inference-сервиса,
  а не нарушение инварианта.

### Точки синхронизации интеграции

Точки, где работа fz, Valeria и Andrey обязана сойтись, в порядке
актуальности:

1. **ML Task Spec** — семантика target/horizon/metric, как только
   опубликована.
2. **Canonical Schema** — общее доменное представление offline и online.
3. **Feature Contract** — общий интерфейс `features/`, который вызывают
   и обучение, и serving.
4. **Artifact Contract** — форма ML Artifact Bundle, которую загружает
   serving-слой.

## 15. TBD после публикации полного ТЗ

- Реальные названия и типы колонок CSV.
- Реальное определение delay/target и единица измерения.
- Реальный prediction horizon.
- Схема событий эмулятора и её mapping на схему CSV.
- Нужны ли purge/embargo-окна или walk-forward CV для temporal evaluation
  (зависит от horizon/leakage-характеристик реальных данных).
- Набор признаков (Valeria, после появления таргета).
- Выбор модели (Valeria).
- Конкретная схема Artifact Bundle (Valeria).

## 16. Явные non-goals (для этого репозитория)

Намеренно не реализовано здесь:

- Serving-реализация с реальной моделью: реальный feature engineering,
  реальный инференс, загрузка артефакта/модельного файла. (Provisional
  FastAPI-shell + `MockPredictor` уже существуют — см. §7 и
  [`ML_SERVING_CONTRACT.md`](ML_SERVING_CONTRACT.md) — но модели за ними
  нет.)
- CatBoost/XGBoost/LightGBM или любой training pipeline.
- Feature engineering или придуманный набор признаков.
- Emulator-клиент или схема эмулятора.
- Второе хранилище `VehicleState` или scheduler на Python.
- PostgreSQL/Prisma, Redis, Kafka.
- MLflow, DVC, Airflow/Prefect/Ray, Optuna/Hydra.
- CI/CD, дашборды, Docker/Compose для этого репозитория.
- Придуманный target, придуманные поля телеметрии, придуманная
  каноническая схема.

## 17. Чек-лист первого часа хакатона

Как только опубликуют организаторский CSV и task spec:

1. `uv run python scripts/inspect_csv.py --path <csv>` — первый взгляд на
   схему, missing, дубликаты.
2. Построить `DatasetManifest` для полученного файла (`data/manifest.py`).
3. Определить реальный `CanonicalMapping` для организаторской CSV-схемы
   (offline). Отдельно, не через `CanonicalMapping` — согласовать с
   online-путём backend'а совместимую доменную семантику для эмулятора
   через будущий Feature Contract (см. §9).
4. Прочитать полный task spec; заполнить реальный `TargetSpec` и
   реализовать построение таргета.
5. Выбрать границы `train_end`/`validation_end`, запустить
   `split_by_time_boundaries`.
6. Обучить `MedianBaselineRegressor`, посчитать baseline MAE через
   `mae()`, залогировать через `experiments/log.py`.
7. Передать канонические данные + target + границы split'а Valeria для
   feature/model-работы.
8. Держать этот документ обновлённым по мере разрешения TBD.

Подробный пошаговый план по фазам после публикации ТЗ —
[`HACKATHON_RUNBOOK.md`](HACKATHON_RUNBOOK.md).

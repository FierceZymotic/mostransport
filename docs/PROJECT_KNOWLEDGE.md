# PROJECT_KNOWLEDGE — база знаний проекта

Это главный human/AI knowledge document репозитория `mostransport-ml`.
Цель — чтобы новый участник команды или AI-агент мог понять проект без
чтения истории чатов и без домыслов.

Документ не дублирует код. Точные детали — в других документах:

- [`../README.md`](../README.md) — короткий вход;
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — технические границы, диаграммы, инварианты;
- [`BACKEND_ML_INTEGRATION.md`](BACKEND_ML_INTEGRATION.md) — Backend → ML Contract v1
  (единственное каноническое описание контракта + handoff для Backend);
- [`ML_SERVING_CONTRACT.md`](ML_SERVING_CONTRACT.md) — где контракт реализован в ML;
- [`DEVELOPMENT.md`](DEVELOPMENT.md) — команды и workflow;
- [`HACKATHON_RUNBOOK.md`](HACKATHON_RUNBOOK.md) — цикл modeling → artifact → submission/serving;
- [`../AGENTS.md`](../AGENTS.md) — правила для AI-агентов.

Метки утверждений:

- **ТЕКУЩЕЕ СОСТОЯНИЕ** — существует в коде этого репозитория и покрыто
  тестами;
- **ИСТОРИЧЕСКИЙ СНИМОК** — было верно на указанном milestone; не текущий
  статус;
- **ПОДТВЕРЖДЕНО ОРГАНИЗАТОРАМИ** — заявлено в официальных материалах;
- **ДОКАЗАНО ДАННЫМИ** — доказано Dataset Evidence v1
  (`notebooks/fz/01_official_dataset_evidence.ipynb`, тег `dataset-evidence-v1`;
  тег содержит 92-ячеечную версию notebook, а не исторический 94-ячеечный
  артефакт SHA-256 `9eb2c78b…` — см.
  `notebooks/fz/01_official_dataset_evidence.PROVENANCE.md`);
- **ОТКРЫТО** — research/modeling-вопрос, ещё не решён.

---

## 10.1. Назначение проекта

ML-часть решения хакатона Мостранспорта: раннее прогнозирование задержки
прибытия наземного транспорта, чтобы диспетчер действовал проактивно.
Превращение прогноза в решение диспетчера (изменение числа ТС, стоянок) —
backend/product-слой, не ML output.

Python-сервис прогнозирует численную задержку `delay_seconds`. Метрика —
**MAE** (ПОДТВЕРЖДЕНО ОРГАНИЗАТОРАМИ). Хакатон длится 48 часов.

## 10.2. Подтверждённые факты

- Точка прогноза — `(tr_id, T)`; target — `target_delay_s =
  time_fact_begin − time_begin`, секунды, `+` опоздание, `−` опережение,
  для целевого действия расписания в горизонте `(T+10м, T+15м]`.
  ДАННЫМИ ДОКАЗАНО на train/test/validate: официальный target всегда
  находится среди действий с минимальным `time_begin` в этом окне.
  При одинаковом минимальном времени точный organizer tie-break не
  восстановлен — см. `BACKEND_ML_INTEGRATION.md` §5.1.
- Для прогноза в `T` разрешена только telemetry с `event_time <= T`
  (ПОДТВЕРЖДЕНО ОРГАНИЗАТОРАМИ).
- Официальные naive timestamps — UTC wall-clock без метки пояса.
- Исторический `traffic.csv speed` = `G6CellNav00 speedAvg` (официальный
  mapping); runtime mapping `G6CellNav00`: `timestamp → event_time`,
  `longitude → lon`, `latitude → lat`, `locationValid → location_valid`,
  `speedAvg → speed`. В runtime `speed` всегда число (подтверждено Backend) —
  это wire-инвариант Contract v1. В историческом CSV ~6% строк не имеют всех
  навигационных полей сразу, включая `speed` (ДОКАЗАНО ДАННЫМИ).
- Ответ организаторов на вопросы (устно): строки исторического датасета и
  данные эмулятора — не буквально один и тот же уровень данных. Датасет
  получен внутренним процессингом организаторов из telemetry того типа,
  что даёт эмулятор; перейти от данных эмулятора к представлению, похожему
  на датасет, можно, но точное преобразование не раскрыто. (Что процессинг
  именно прореживает 1 Гц до ~12–15 с — не доказано.)
- `labels_test` — официальный local model-selection split (с ограничением
  temporal nearness); `validate` — скрытый leaderboard.
- `test/traffic.csv` байт-в-байт равен `validate/traffic.csv`, а факт
  `test/schedule.csv` структурно раскрывает hidden target validate —
  использование этого факта для validate запрещено (ДОКАЗАНО ДАННЫМИ).
- Train содержит 13 общих с test/validate schedule `tr_id` (группа A) и
  26 train-only/synthetic-candidate `tr_id` (группа B, ~74% строк);
  что группа B — именно «synthetic» ТС, официально не подтверждено.
- Организаторские датасеты нельзя публиковать или использовать за
  пределами правил хакатона.

## 10.3. Открытые вопросы (не инфраструктура)

- **ОТКРЫТО**: пригодность группы B для обучения (M1 показал большой
  выигрыш от неё, происхождение выигрыша не проверено).
- **ОТКРЫТО**: DIRECT vs RESIDUAL, режим обучения, финальная модель.
- **ОТКРЫТО**: сдвиг распределения count-признаков (`rows_*`,
  `valid_gps_count_*`) из-за частоты telemetry (CSV ~12–15 с между пакетами,
  эмулятор ~1 Гц; преобразование организаторов не раскрыто, см. §10.2).
  Общий канонический Feature Builder (code-path parity) — да; parity
  источников и распределений — нет. `tabular-v1` не меняется, telemetry не
  прореживается и молча не нормализуется.
- **ОТКРЫТО (modeling debt)**: official `cur_dev_s` ≠ гарантированно runtime
  `current_deviation_seconds`. Runtime-значение по контракту — point-in-time-safe
  отклонение последнего подтверждённо пройденного события (иначе `0`, см.
  [BACKEND_ML_INTEGRATION.md](BACKEND_ML_INTEGRATION.md) §5.1). Forensic-анализ
  показал, что official `cur_dev_s` в real-time один в один не
  воспроизводится: примерно в 44–45% исследованных real/test случаев
  соответствующее фактическое событие находится после `T`. `cur_dev_s` —
  признак `tabular-v1` и основа RESIDUAL-формулировки, поэтому этот skew
  нужно учесть до финальной production-модели. Это вопрос модели, не API.
- **ОТКРЫТО**: соответствие исторических строк без навигационных полей
  (включая `speed`) runtime-пакетам, где `speed` всегда число, — часть того же
  исследования распределений.
- **ОТКРЫТО**: probability/reason — модели нет, `reason = null`.

## 10.4. Владение (ownership) в команде

| Участник | Ответственность |
|----------|-----------------|
| **fz** — canonical ML track | данные, target, evaluation, Feature Builder, контракты, валидация, Artifact Bundle, inference, serving, promotion gates |
| **Valeria** | изолированное feature/model research; кандидаты передаются как Artifact Bundle v1, совместимый со схемой признаков |
| **Andrey** | backend: NDTP, `VehicleState`, schedule/route matching, текущее отклонение, eligibility/cadence, alerts, REST/WebSocket, frontend |
| **Lisa** | БД, reference data, BI / аналитика |

## 10.5. Границы репозитория

**Сейчас (ТЕКУЩЕЕ СОСТОЯНИЕ):** репозиторий содержит ML-часть —
безопасные official loaders, Feature Builder `tabular-v1`, канонический
ML-контекст и адаптеры, M1 baseline и эксперименты, ArtifactManifest/Artifact
Bundle, artifact-backed inference, serving Contract v1, validate submission,
документация — и становится начальной основой общего командного
репозитория.

**Запланировано (ближайшая интеграция):** Backend и Frontend (Андрей)
добавляются в этот же репозиторий на верхнем уровне, рядом с текущей
структурой; ML не переносится и не переименовывается. Пока их здесь нет.

**Вне репозитория:** организаторские данные и эмулятор.

## 10.6. Актуальная структура репозитория

Текущая структура (Backend/Frontend будут добавлены позже как соседние
каталоги верхнего уровня):

```
<repo>/
├── README.md  AGENTS.md  CLAUDE.md  pyproject.toml  uv.lock
├── Dockerfile  .dockerignore       минимальный образ ML-сервиса (artifact монтируется)
├── docs/                           этот документ, ARCHITECTURE, BACKEND_ML_INTEGRATION,
│                                   ML_SERVING_CONTRACT, DEVELOPMENT, HACKATHON_RUNBOOK
├── data/                           локальная область для данных; НЕ в git
├── notebooks/fz/                   Dataset Evidence v1 (+ PROVENANCE.md)
├── experiments/                    README; JSONL-логи в .gitignore
├── src/mostransport_ml/
│   ├── data/                       official loaders + generic inspection/manifest/canonical
│   ├── target/                     TargetSpec, TARGET_SPEC, DIRECT/RESIDUAL
│   ├── evaluation/                 MAE, temporal split, median baseline
│   ├── experiments/                append-only JSONL experiment log
│   ├── features/                   schema, builder (tabular-v1), spatial, context, adapters
│   ├── models/                     M1 CatBoost config, train regimes, 6-run grid
│   ├── artifacts/                  ArtifactManifest v1, Artifact Bundle v1, legacy ArtifactMetadata
│   ├── inference/                  ArtifactPredictor, export_bundle, model families, submission
│   └── serving/                    FastAPI Contract v1, artifact_app, mock_app
├── scripts/                        run_offline_baseline.py (M1), make_submission.py,
│                                   build_integration_artifact.py
│                                     (INTEGRATION TEST ONLY · NOT FOR SUBMISSION · NOT A QUALITY MODEL),
│                                   smoke_contract_v1.py, smoke_offline.py, inspect_csv.py
└── tests/                          unit/contract/E2E тесты (+ fixtures/contract_v1_request.json)
```

## 10.7. Реализованные модули (ТЕКУЩЕЕ СОСТОЯНИЕ)

- `data/official.py` — train/test loaders (target отделён; schedule по
  allowlist, `time_fact_begin` не читается); validate inference loader
  (только `validate/points.csv`, `validate/traffic.csv`,
  `validate/schedule_plan.csv`; target-колонки в points — ошибка);
  шаблон `sample_submission.csv`; плановый fingerprint schedule.
- `data/safe_deviation.py` — offline/training-only point-in-time-safe текущее
  отклонение из factual schedule (`time_fact_begin <= T`, research tie-break);
  наружу отдаёт только значения для `offline_context(..., current_deviation_seconds=...)`.
- `data/inspection.py`, `data/manifest.py`, `data/canonical.py` — generic
  инструменты вне production-пути признаков.
- `target/` — `TargetSpec`, `TARGET_SPEC`, `training_target` /
  `final_prediction` (DIRECT: ŷ = f(X); RESIDUAL: ŷ = cur_dev_s + f(X)).
- `evaluation/`, `experiments/` — MAE, temporal split, baseline, JSONL лог.
- `features/` — `schema.py` (37 признаков `tabular-v1`; frozen `runtime-safe-v1` —
  29 признаков без raw counts, в artifact/serving пока не активна), `builder.py`
  (point-in-time), `spatial.py`, `context.py` (`CanonicalBatch`,
  `build_features_from_context`, единственная проекция 37 → 29
  `project_runtime_safe_features`), `adapters.py` (`offline_context` с keyword-only
  override `current_deviation_seconds` по `sample_id`, `runtime_context`).
- `models/` — замороженные M1 CatBoost config и train regimes
  (`scripts/run_offline_baseline.py`).
- `artifacts/` — `manifest.py` (ArtifactManifest v1: канонический JSON,
  SHA-256, строгий разбор, совместимость), `bundle.py` (Artifact Bundle
  v1), `metadata.py` (LEGACY `ArtifactMetadata`, для нового кода не
  использовать). Только stdlib.
- `inference/` — `ArtifactPredictor`, `export_bundle`, model family
  `catboost` (= обученная скалярная регрессия задержки: экспорт только
  `CatBoostRegressor`; при загрузке проверяются сохранённый objective и
  форма выхода — классификатор/ранкер/многомерный выход отклоняются),
  validate submission.
- `serving/` — `POST /api/v1/predict` (Contract v1), `/health`, `/ready`;
  `artifact_app.py` (production composition root), `mock_app.py`.

## 10.8. Граф зависимостей

```
data   features   target   artifacts(stdlib)   evaluation   experiments
                     ▲
                  models
inference → artifacts, features, target, data
serving   → inference, features
```

Без циклов; `artifacts` не зависит ни от чего внутреннего; `features` не
зависит от serving/inference; serving не содержит формул признаков.

## 10.9. Offline data flow

```
official dataset (MOSTRANSPORT_DATASET, вне репозитория)
  → data/official.py (allowlist, target отдельно)
  → features/adapters.offline_context → CanonicalBatch
  → build_features_from_context (tabular-v1)
  → обучение кандидата → ArtifactManifest → inference.export_bundle
  → ArtifactPredictor: оценка на labels_test / scripts/make_submission.py (validate)
```

## 10.10. Каноническое представление

Внутреннее представление одной логической точки прогноза —
`CanonicalBatch` (`features/context.py`): точки (`point_id`, `tr_id`, `T`,
target_time, координаты цели, `current_deviation_s`, `manual_fill`,
идентификаторы для трассировки) + колоночная история telemetry. Оба
адаптера только нормализуют и сходятся в один builder.
`data/canonical.py::CanonicalMapping` — generic offline-механизм rename для
прочих CSV, в production-пути не используется.

## 10.11. Стратегия оценки

- Метрика — MAE (ПОДТВЕРЖДЕНО ОРГАНИЗАТОРАМИ).
- Model selection — `labels_test`; validate — только submission.
- Baselines на `labels_test`: zero ≈ 103.34 с, train median ≈ 100.87 с,
  `cur_dev_s` ≈ 93.36 с.
- M1: 6 заранее заданных CatBoost-экспериментов (3 режима × DIRECT/
  RESIDUAL); результаты — лог `experiments/m1_runs.jsonl` (локально).

## 10.12. Serving-архитектура (ТЕКУЩЕЕ СОСТОЯНИЕ)

```
FastAPI (serving/app.py: create_app(predictor, clock))
  → Pydantic Contract v1 (serving/schemas.py)
  → InferenceService: runtime_context → CanonicalBatch → Predictor.predict
  → ArtifactPredictor (inference/) | MockPredictor (только явно, mock_app.py)
```

- `create_app` не выбирает predictor сам; production — `artifact_app.py`
  (`MOSTRANSPORT_ARTIFACT_DIR`), при проблеме с bundle — `503`.
- Stateless по отношению к истории vehicle; artifact держится в памяти.
- Runtime-проверка контракта predictor'а: `is_ready` — ровно `bool`,
  версии — непустые строки, выход — одно конечное число на точку.
- `422` санитизирован, `500`/`503` — фиксированные сообщения, в логах нет
  тел запросов.

## 10.13. Контракт прогнозирования — краткое summary

Полный контракт — [BACKEND_ML_INTEGRATION.md](BACKEND_ML_INTEGRATION.md).
Запрос: `request_id` (opaque, UUID рекомендуется), `prediction_time`,
`vehicle_context{unit_id, tr_id, route_id}`, `schedule_context{target_action_id,
target_time_begin, target_lat, target_lon, current_deviation_seconds,
manual_fill}`, `telemetry[]{event_time, lat, lon, location_valid, speed}`;
время — aware ISO-8601 (`Z`); `speed` — всегда конечное число. Ответ:
`request_id`, `status="success"`, `prediction{delay_seconds, target_time =
target_time_begin + delay (согласовано), reason=null}`, `generated_at` (UTC),
`model_version`, `feature_schema_version`. История: `(T-15m, T]` + последний
пакет + последний strict-valid GPS пакет `<= T`; producer не шлёт
`event_time > T`, ML отбрасывает такие пакеты defensively. Contract v1
согласован и заморожен. Как Backend выбирает целевое событие (включая
ничью на минимальном `time_begin`), считает `current_deviation_seconds` и
передаёт `manual_fill` — BACKEND_ML_INTEGRATION.md §5.1.

## 10.14. Artifact-фундамент (ТЕКУЩЕЕ СОСТОЯНИЕ)

- **ArtifactManifest v1** — модельно-независимые метаданные: версии,
  `model_family`, `feature_schema_version`, `target_formulation`,
  `train_regime`, params, provenance, evaluation, `created_at` (UTC);
  канонический JSON, SHA-256 fingerprint, строгий разбор (дубликаты
  ключей, NaN/Infinity, пути отклоняются).
- **Artifact Bundle v1** — каталог `manifest.json` + бинарник модели
  (`model.cbm`) + `bundle.json` (`bundle_schema_version`, имена и SHA-256
  обоих файлов). Загрузка проверяет состав каталога, имена, хеши,
  manifest, совместимость схемы признаков, family и формулировку — до
  загрузки модели; модель грузится из проверенных байтов (без pickle).
- `ArtifactPredictor` дополнительно сверяет тип задачи модели (скалярная
  регрессия, см. §10.7) и что модель обучена ровно на `FEATURE_NAMES`;
  иначе artifact не готов (`503`), прогноз не выдаётся.
- `scripts/build_integration_artifact.py` — детерминированный
  `integration-fixture-v1` bundle на синтетике:
  **INTEGRATION TEST ONLY · NOT FOR SUBMISSION · NOT A QUALITY MODEL**.

## 10.15. Согласованность offline ↔ online

ТЕКУЩЕЕ СОСТОЯНИЕ: offline и runtime используют один канонический Feature
Builder. Тесты и проверка на реальных данных показывают, что эквивалентный
канонический входной контекст через оба пути даёт побитово одинаковые
признаки, а один bundle — одинаковый прогноз (в т.ч. для эквивалентных
timestamps в разных поясах). Это code-path parity, а не полная parity
источников: runtime `current_deviation_seconds` и official `cur_dev_s`
семантически не идентичны, распределения telemetry (частота) тоже
различаются — ОТКРЫТО, см. §10.3.

## 10.16. Граница Backend ↔ ML

Backend (NestJS/TypeScript, Андрей) будет добавлен в этот репозиторий на
верхнем уровне; сейчас его кода здесь нет. Его обязанности по отношению к
ML — [BACKEND_ML_INTEGRATION.md](BACKEND_ML_INTEGRATION.md) (§1, §5.1, §7).

ИСТОРИЧЕСКИЙ СНИМОК (M2, 2026-09-25, отдельная рабочая копия backend до
объединения репозиториев): приём телеметрии `POST /telemetry/events`,
`event_time` через `toISOString()` (UTC `Z`), клиента Contract v1 не было; по
коммуникации в команде — NDTP-эмулятор подключён, TCP receiver и парсинг
`G6CellNav00` работают, `VehicleState` есть. Не текущий статус.

| Владеет backend (Андрей) | Владеет Python (canonical ML track) |
|---|---|
| NDTP, `VehicleState`, история, schedule matching | point-in-time валидация, канонизация |
| текущее отклонение, `manual_fill`, цель, eligibility, cadence | model-specific признаки `tabular-v1` |
| alerts, REST/WebSocket, dashboard | Artifact Bundle, инференс |

ML не дублирует левую колонку и не меняет Backend/Frontend/DB без задачи
на соответствующую зону.

## 10.17. Безопасность и организаторские данные

- Организаторские данные не коммитятся (`data/*` в `.gitignore`) и не
  пишутся в репозиторий; submission не пишется внутрь каталога датасета.
- Перед отправкой organizer-данных во внешний инструмент — сверяться с
  правилами хакатона.
- Serving не логирует тела запросов; `422` без `input`/`ctx`; `500`/`503`
  без текста исключений и traceback.
- Hidden validate target не вычисляется; факт test schedule не
  используется для validate.

## 10.18. Точки расширения

- Новый кандидат `tabular-v1`: обучить → `ArtifactManifest` →
  `export_bundle` → `ArtifactPredictor` / `make_submission.py` / serving —
  без изменения инфраструктуры ([HACKATHON_RUNBOOK.md](HACKATHON_RUNBOOK.md)).
- Новые признаки: новая версия схемы рядом с `tabular-v1`; bundle указывает
  свою схему, совместимость проверяется при загрузке.
- Новый model family: запись в `inference/model_families.MODEL_FAMILIES`.

## 10.19. Non-goals

- Kafka, Redis, MLflow, DVC, Airflow, feature store, model registry;
- Python `VehicleState`, scheduler, schedule/map matching;
- база данных в этом репозитории, risk rules, dashboard;
- автоматическое прореживание/нормализация telemetry;
- угаданные поля или семантика сверх официальных материалов и Dataset
  Evidence v1.

## 10.20. Правила для AI-агентов

Полная версия — [`../AGENTS.md`](../AGENTS.md). Кратко: прочитать этот
документ, [`ARCHITECTURE.md`](ARCHITECTURE.md) и (для serving)
[`BACKEND_ML_INTEGRATION.md`](BACKEND_ML_INTEGRATION.md); прогнать тесты.
Запрещено: менять замороженные M1-семантики и `tabular-v1`, использовать
hidden validate target или факт test schedule для validate, дублировать
feature-логику, возвращать provisional v0 serving-контракт, создавать
Python `VehicleState`/scheduler, молча выбирать модель, менять
Backend/Frontend/DB без задачи на соответствующую зону.

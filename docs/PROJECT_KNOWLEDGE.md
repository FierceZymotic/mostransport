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
- **ЗАКРЫТО** — решение принято и воспроизведено в коде репозитория;
- **ОТКРЫТО** — research/modeling-вопрос, ещё не решён;
- **ИНТЕГРАЦИЯ** — открыто на стороне Backend ↔ ML, не ML-модель.

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
  на датасет, можно, но точное преобразование не раскрыто. Исторический CSV
  имеет шаг ~12–15 с между пакетами; что runtime-поток имеет ту же частоту —
  не гарантировано и не доказано.
- `labels_test` — официальный local evaluation split организаторов (с
  ограничением temporal nearness); `validate` — скрытый leaderboard.
  ИСТОРИЧЕСКИ `labels_test` использовался для оценки M1; текущее
  production-решение его не использует (§10.11).
- `test/traffic.csv` байт-в-байт равен `validate/traffic.csv`, а факт
  `test/schedule.csv` структурно раскрывает hidden target validate —
  использование этого факта для validate запрещено (ДОКАЗАНО ДАННЫМИ).
- Train содержит 13 общих с test/validate schedule `tr_id` (группа A) и
  26 train-only/synthetic-candidate `tr_id` (группа B, ~74% строк);
  что группа B — именно «synthetic» ТС, официально не подтверждено (по
  данным это клоны группы A — §10.3).
- Организаторские датасеты нельзя публиковать или использовать за
  пределами правил хакатона.

## 10.3. Статус решений и открытые вопросы

Research закрыт; production-решение заморожено и воспроизведено (P1–P3).

- **ЗАКРЫТО — группа B** (research, train-only; notebooks вне репозитория):
  не используется для обучения. Каждый `tr_id` группы B — сдвинутая во
  времени копия одного `tr_id` группы A (по две на каждый) с зашумлёнными
  метками; на невиденных real `tr_id` без их собственных клонов B не улучшает
  MAE — выигрыш M1 от B объяснялся доступом к копиям тех же ТС того же дня.
- **ЗАКРЫТО — модель**: `HistGradientBoostingRegressor` DIRECT на
  `runtime-safe-v1`, обучение только на группе A (1141 точка, 13 `tr_id`),
  фиксированная конфигурация H0 (`models/hgb_v1.py`) — консервативный
  кандидат, а не доказанный глобальный оптимум.
- **ЗАКРЫТО — P3**: production training path (`scripts/train_final_hgb.py`)
  воспроизводит research GroupKFold OOF MAE точно (77.34108978455868, допуск
  1e-9) и строит финальный artifact `hgb-h0-runtime-safe-v1-group-a-v1`
  (локально, в `artifacts/`, не коммитится).
- **ЗАКРЫТО на стороне ML — raw packet counts**: исторический CSV имеет шаг
  ~12–15 с, а частота runtime-потока не гарантирована и может отличаться,
  поэтому восемь count-признаков (`rows_*`, `valid_gps_count_*`) исключены из
  `runtime-safe-v1` (P1). `tabular-v1` не меняется, telemetry не прореживается
  и молча не нормализуется.
- **ЗАКРЫТО на стороне ML — current deviation**: обучение использует
  point-in-time-safe offline-отклонение (`data/safe_deviation.py`: последний
  подтверждённый факт `<= T`, research tie-break), а не official `cur_dev_s`.
  ИСТОРИЧЕСКИ: forensic-анализ показал, что official `cur_dev_s` в real-time
  не воспроизводится (в ~44–45% real/test случаев соответствующее
  фактическое событие — после `T`).
- **ЗАКРЫТО — вехи**: P1 (`ml-runtime-safe-v1`), P2 (`ml-hgb-artifact-v1`),
  P3 (final training path, этот раздел).
- **ИНТЕГРАЦИЯ**: совпадение runtime `current_deviation_seconds` Backend'а с
  той же point-in-time семантикой ([BACKEND_ML_INTEGRATION.md](BACKEND_ML_INTEGRATION.md)
  §5.1); выбор целевого события при ничьей `time_begin`; реальная частота
  runtime-telemetry. Это обязанности/риски Backend ↔ ML, не ML-модели.
- **ОТКРЫТО**: соответствие исторических строк без навигационных полей
  (включая `speed`) runtime-пакетам, где `speed` всегда число.
- **ОТКРЫТО**: probability/reason — модели нет, `reason = null`.

## 10.4. Владение (ownership) в команде

| Участник | Ответственность |
|----------|-----------------|
| **fz** — canonical ML track | данные, target, evaluation, Feature Builder, контракты, валидация, Artifact Bundle, inference, serving, promotion gates |
| **Valeria** | изолированное feature/model research; кандидаты передаются как Artifact Bundle v1, совместимый со схемой признаков |
| **Andrey** | backend: NDTP, `VehicleState`, schedule/route matching, текущее отклонение, eligibility/cadence, alerts, REST/WebSocket, frontend |
| **Lisa** | БД, reference data, BI / аналитика |

## 10.5. Границы репозитория

**Сейчас (ТЕКУЩЕЕ СОСТОЯНИЕ):** общий командный репозиторий. ML-часть —
безопасные official loaders, Feature Builder `tabular-v1` и проекция
`runtime-safe-v1`, канонический ML-контекст и адаптеры, M1 baseline и
эксперименты, ArtifactManifest/Artifact Bundle, artifact-backed inference
(CatBoost и HGB), serving Contract v1, validate submission, документация.
Рядом на верхнем уровне — Backend (`backend/`), Frontend (`frontend/`), БД
(`database/`), `docker/` и `docker-compose.yml`: вне зоны ML, для ML-задач
только чтение.

**Вне репозитория:** организаторские данные и эмулятор.

## 10.6. Актуальная структура репозитория

Текущая структура (ML-часть подробно; соседние зоны — одной строкой):

```
<repo>/
├── backend/  frontend/  database/  docker/  docker-compose.yml   вне зоны ML (только чтение)
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
│   ├── features/                   schema (tabular-v1, runtime-safe-v1), builder, spatial,
│   │                               context (+ проекция 37 → 29), adapters
│   ├── models/                     M1 CatBoost config, HGB H0 config, train regimes
│   ├── artifacts/                  ArtifactManifest v1, Artifact Bundle v1, legacy ArtifactMetadata
│   ├── inference/                  ArtifactPredictor, export_bundle, model families, submission
│   ├── training/                   final_hgb: official Group A → OOF gate → final HGB artifact (P3)
│   └── serving/                    FastAPI Contract v1, artifact_app, mock_app
├── artifacts/                      локальные artifacts/отчёты (в .gitignore)
├── scripts/                        train_final_hgb.py (P3), run_offline_baseline.py (M1), make_submission.py,
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
  наружу отдаёт только значения для `offline_context(..., current_deviation_seconds=...)`;
  `load_train_schedule_facts` — единственный loader фактов (только
  `train/schedule.csv`, только факт-колонки, порядок CSV).
- `data/inspection.py`, `data/manifest.py`, `data/canonical.py` — generic
  инструменты вне production-пути признаков.
- `target/` — `TargetSpec`, `TARGET_SPEC`, `training_target` /
  `final_prediction` (DIRECT: ŷ = f(X); RESIDUAL: ŷ = cur_dev_s + f(X)).
- `evaluation/`, `experiments/` — MAE, temporal split, baseline, JSONL лог.
- `features/` — `schema.py` (37 признаков `tabular-v1`; frozen `runtime-safe-v1` —
  29 признаков без raw counts; lookup `feature_names_for_schema` /
  `SUPPORTED_FEATURE_SCHEMA_VERSIONS`), `builder.py` (point-in-time), `spatial.py`,
  `context.py` (`CanonicalBatch`, `build_features_from_context`, единственная
  проекция 37 → 29 `project_runtime_safe_features`, выбор признаков схемы
  `features_for_schema`), `adapters.py` (`offline_context` с keyword-only
  override `current_deviation_seconds` по `sample_id`, `runtime_context`).
- `models/` — замороженные M1 CatBoost config и train regimes
  (`scripts/run_offline_baseline.py`); `hgb_v1.py` — HGB H0 (`HGB_H0_PARAMS`,
  `fit_h0`: строго `runtime-safe-v1`, NaN допустимы, ±inf нет).
- `artifacts/` — `manifest.py` (ArtifactManifest v1: канонический JSON,
  SHA-256, строгий разбор, совместимость), `bundle.py` (Artifact Bundle
  v1), `metadata.py` (LEGACY `ArtifactMetadata`, для нового кода не
  использовать). Только stdlib.
- `inference/` — `ArtifactPredictor` (схема признаков — из проверенного
  manifest: `tabular-v1` как есть, `runtime-safe-v1` через P1-проекцию),
  `export_bundle`, validate submission и model families: `catboost` (legacy;
  обученная скалярная регрессия задержки: экспорт только `CatBoostRegressor`;
  при загрузке проверяются сохранённый objective и форма выхода) и
  `hist_gradient_boosting` (ровно `HistGradientBoostingRegressor`, `skops`
  `model.skops` без pickle; доверен единственный тип `TreePredictor`, деревья
  структурно проверяются до любого `predict`).
- `training/final_hgb.py` — финальный training path (P3): Group A из
  planned-only real universe, target по `sample_id`, safe deviation, канонический
  builder → `runtime-safe-v1`, GroupKFold(5) OOF gate против research reference,
  финальный H0 fit, manifest с честным provenance (`git_head`, `git_dirty`,
  `source_tree_sha256`), export и load-back parity; CLI — `scripts/train_final_hgb.py`.
- `serving/` — `POST /api/v1/predict` (Contract v1), `/health`, `/ready`;
  `artifact_app.py` (production composition root), `mock_app.py`.

## 10.8. Граф зависимостей

```
data   features   target   artifacts(stdlib)   evaluation   experiments
                     ▲
                  models
inference → artifacts, features, target, data
training  → data, features, models, evaluation, artifacts, inference
serving   → inference, features
```

Без циклов; `artifacts` не зависит ни от чего внутреннего; `features` не
зависит от serving/inference; serving не содержит формул признаков.

## 10.9. Offline data flow

Текущий production-путь (P3, `scripts/train_final_hgb.py`):

```
official dataset (MOSTRANSPORT_DATASET, вне репозитория)
  → Group A: train points ∩ planned-only real tr_id universe (load_shared_real_vehicle_ids)
  → target по sample_id (labels_train)
  → load_train_schedule_facts → safe_current_deviation_seconds (P1)
  → offline_context(..., current_deviation_seconds=safe) → CanonicalBatch
  → build_features_from_context (tabular-v1, 37) → project_runtime_safe_features (29)
  → GroupKFold(5, tr_id) OOF HGB H0 → gate против research reference
  → fit H0 на всей Group A → ArtifactManifest v1 → export_bundle → load-back parity
  → scripts/make_submission.py (validate) / serving
```

ИСТОРИЧЕСКИ (M1): `tabular-v1` + CatBoost, оценка на `labels_test`.

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
- ТЕКУЩЕЕ ПРАВИЛО: семейство, конфигурация и признаки заморожены;
  production-приёмка — воспроизведение GroupKFold(5) OOF MAE на группе A
  (`training/final_hgb.py`). `labels_test` и validate для нового выбора
  модели не используются; validate — только submission.
- ИСТОРИЧЕСКИ (M1): model selection на `labels_test`; baselines на
  `labels_test`: zero ≈ 103.34 с, train median ≈ 100.87 с, `cur_dev_s` ≈ 93.36 с;
  6 заранее заданных CatBoost-экспериментов (3 режима × DIRECT/RESIDUAL),
  лог `experiments/m1_runs.jsonl` (локально).

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
  (`model.cbm` для `catboost`, `model.skops` для `hist_gradient_boosting`) +
  `bundle.json` (`bundle_schema_version`, имена и SHA-256 обоих файлов).
  Загрузка проверяет состав каталога, имена, хеши, manifest, совместимость
  схемы признаков (одна закреплённая или любая поддерживаемая кодом), family
  и формулировку — до загрузки модели; модель грузится из проверенных байтов
  (без pickle). Manifest и bundle остаются v1: family и схема — поля manifest.
- `ArtifactPredictor` дополнительно сверяет тип задачи модели (скалярная
  регрессия, см. §10.7) и что модель обучена ровно на упорядоченных
  признаках схемы из manifest; иначе artifact не готов (`503`), прогноз не
  выдаётся. `/ready` и ответ сообщают `feature_schema_version` artifact'а.
- `scripts/build_integration_artifact.py` — `integration-fixture-v1`
  (CatBoost/`tabular-v1`, байтово детерминированный) и по
  `--model-family hist_gradient_boosting` — `integration-fixture-hgb-v1`
  (HGB H0/`runtime-safe-v1`) на синтетике:
  **INTEGRATION TEST ONLY · NOT FOR SUBMISSION · NOT A QUALITY MODEL**.
- Финальный artifact `hgb-h0-runtime-safe-v1-group-a-v1` строится одной
  командой `scripts/train_final_hgb.py` из официальных данных (локально,
  `artifacts/`, в git не попадает); evaluation в manifest — OOF
  (`group_a_groupkfold5_oof`), не in-sample и не `labels_test`.

## 10.15. Согласованность offline ↔ online

ТЕКУЩЕЕ СОСТОЯНИЕ: offline и runtime используют один канонический Feature
Builder. Тесты и проверка на реальных данных показывают, что эквивалентный
канонический входной контекст через оба пути даёт побитово одинаковые
признаки, а один bundle — одинаковый прогноз (в т.ч. для эквивалентных
timestamps в разных поясах). Это code-path parity, а не полная parity
источников: совпадение runtime `current_deviation_seconds` Backend'а с
offline safe-семантикой и частота runtime-telemetry — ИНТЕГРАЦИЯ, см. §10.3.

## 10.16. Граница Backend ↔ ML

Backend (NestJS/TypeScript, Андрей) находится в этом репозитории (`backend/`,
рядом `frontend/`, `database/`, compose); для ML-задач — только чтение. Его
обязанности по отношению к ML — [BACKEND_ML_INTEGRATION.md](BACKEND_ML_INTEGRATION.md)
(§1, §5.1, §7).

ИСТОРИЧЕСКИЙ СНИМОК (M2, 2026-09-25, отдельная рабочая копия backend до
объединения репозиториев): приём телеметрии `POST /telemetry/events`,
`event_time` через `toISOString()` (UTC `Z`), клиента Contract v1 не было; по
коммуникации в команде — NDTP-эмулятор подключён, TCP receiver и парсинг
`G6CellNav00` работают, `VehicleState` есть. Не текущий статус.

| Владеет backend (Андрей) | Владеет Python (canonical ML track) |
|---|---|
| NDTP, `VehicleState`, история, schedule matching | point-in-time валидация, канонизация |
| текущее отклонение, `manual_fill`, цель, eligibility, cadence | model-specific признаки (`tabular-v1` / `runtime-safe-v1`) |
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

- Новый кандидат поддерживаемой схемы (`tabular-v1` или `runtime-safe-v1`):
  обучить → `ArtifactManifest` →
  `export_bundle` → `ArtifactPredictor` / `make_submission.py` / serving —
  без изменения инфраструктуры ([HACKATHON_RUNBOOK.md](HACKATHON_RUNBOOK.md)).
- Новые признаки: новая версия схемы рядом с `tabular-v1` (запись в
  `features/schema.SUPPORTED_FEATURE_SCHEMAS` и преобразование из канонических
  признаков в `features_for_schema`); bundle указывает свою схему,
  совместимость проверяется при загрузке.
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

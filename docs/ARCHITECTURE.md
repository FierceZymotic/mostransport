# Архитектура

Источник истины по техническим границам системы: что реализовано, какие
инварианты действуют, кто чем владеет. Читать перед добавлением новой
подсистемы или абстракции.

Роль этого документа — строгие технические границы и потоки данных. Более
широкое описание проекта — в [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md);
контракт Backend → ML — в [`BACKEND_ML_INTEGRATION.md`](BACKEND_ML_INTEGRATION.md)
(карта его реализации — [`ML_SERVING_CONTRACT.md`](ML_SERVING_CONTRACT.md)).

---

## 1. Назначение

Раннее прогнозирование задержки прибытия наземного транспорта (хакатон
Мостранспорта): перевести диспетчеров из реактивного режима в проактивный.

Это общий командный репозиторий. ML-часть: безопасное чтение официальных
данных, point-in-time признаки, обучение/оценка, Artifact Bundle,
artifact-backed инференс для runtime (Backend → ML Contract v1) и для
validate submission (Python-пакет, `tests/`, `scripts/`, `pyproject.toml`,
`Dockerfile`). Рядом на верхнем уровне — Backend (`backend/`), Frontend
(`frontend/`), БД (`database/`) и compose; они вне зоны ML.

## 2. Подтверждённые факты (официальный датасет + Dataset Evidence v1)

Dataset Evidence v1 — тег `dataset-evidence-v1` (92-ячеечная версия notebook;
исторический 94-ячеечный артефакт `9eb2c78b…` в теге не содержится — см.
`notebooks/fz/01_official_dataset_evidence.PROVENANCE.md`).

- Точка прогноза — `(tr_id, T)`; target — `target_delay_s =
  time_fact_begin − time_begin` (секунды, `+` опоздание) на первой
  остановке с плановым прибытием в `(T+10м, T+15м]`; метрика — MAE.
- Для прогноза в `T` разрешена только telemetry с `event_time <= T`.
- Официальные naive timestamps — UTC wall-clock без метки пояса.
- `traffic.csv speed` = `G6CellNav00 speedAvg`; в runtime `speed` всегда
  число (wire-инвариант Contract v1).
- Датасет получен внутренним процессингом организаторов из telemetry типа
  эмулятора; точное преобразование не раскрыто (ответ организаторов).
- `labels_test` — официальный local model-selection split; `validate` —
  скрытый leaderboard (target недоступен).
- Факт `test/schedule.csv` структурно раскрывает hidden target validate —
  его использование для validate запрещено.
- Организаторские датасеты нельзя коммитить или публиковать.

## 3. End-to-end архитектура (реализовано)

```
OFFLINE / RESEARCH                          RUNTIME
official dataset                            Backend (NDTP, VehicleState, schedule matching)
  │ data/official.py (allowlist loaders)       │ Contract v1 JSON (aware ISO-8601 UTC)
  ▼                                            ▼
features/adapters.offline_context       serving: Pydantic → features/adapters.runtime_context
  └───────────────┬────────────────────────────┘
                  ▼
        CanonicalBatch (features/context.py, naive UTC)
                  ▼
        build_features_from_context → frozen tabular-v1 builder (features/builder.py)
                  ▼  канонические 37 признаков
        features_for_schema(verified manifest.feature_schema_version)
           ├─ tabular-v1      → 37 как есть
           └─ runtime-safe-v1 → project_runtime_safe_features → 29 (без raw counts)
                  ▼
        inference.ArtifactPredictor ← Artifact Bundle v1 (artifacts/bundle.py + manifest.py)
           model family из manifest: catboost (model.cbm) | hist_gradient_boosting (model.skops)
                  ▼
        delay_seconds (direct | residual via target/formulation.py)
          │                                   │
  inference/submission.py                serving: Contract v1 response
  (validate → sample_id;prediction)       (POST /api/v1/predict)
```

**Ключевой инвариант:** одна feature-реализация и одна prediction-логика для
обоих путей. Backend владеет операционным состоянием и доменными фактами;
Python владеет model-specific признаками и инференсом. Python ML Service не
хранит историю по vehicle между запросами (её приносит каждый запрос), но
держит в памяти загруженный artifact — это обычная часть inference-сервиса.

Зависимости пакетов (однонаправленные, без циклов):

```
data, features, target, artifacts   (artifacts — только stdlib)
        ▲
inference  (→ artifacts, features, target, data)
        ▲
serving    (→ inference, features)
```

`models/` (M1 CatBoost config, HGB H0 config, train regimes) зависит только от
`target/` и `features/schema.py`.

Текущее намеренное production-сопоставление: `tabular-v1` — legacy CatBoost
(M1, integration fixture); `runtime-safe-v1` — production-кандидат HGB H0
(DIRECT). Инфраструктура не связывает family и схему жёстко: схему объявляет
manifest, а модель обязана доказать совпадение своих упорядоченных признаков.

## 4. Границы ownership

| Владелец | Ответственность |
|--------|----------------|
| **fz** — canonical ML track | данные, target, evaluation, Feature Builder, контракты, Artifact Bundle, inference, serving, promotion gates |
| **Valeria** | изолированное feature/model research; кандидаты передаются как совместимый Artifact Bundle |
| **Andrey** | backend / NDTP / операционное состояние / schedule matching / cadence / alerts / realtime / frontend |
| **Lisa** | БД / reference data / BI / аналитика |

## 5. Граница Backend ↔ ML

Backend (NestJS/TypeScript) находится в этом же репозитории как соседний
компонент верхнего уровня (`backend/`); для ML-задач — только чтение. Граница
между компонентами — только HTTP Contract v1
([`BACKEND_ML_INTEGRATION.md`](BACKEND_ML_INTEGRATION.md)); общий репозиторий
не означает общего кода: ML не импортирует Backend, Backend не реализует
признаки модели.

Backend владеет, и ML не дублирует: приём/парсинг NDTP,
операционный `VehicleState` и историю, schedule/route matching, текущее
отклонение от расписания, `manual_fill`, координаты цели, prediction
eligibility и cadence, risk/alerts, WebSocket/REST, frontend, PostgreSQL.

Backend присылает по Contract v1 доменные факты + сырую историю telemetry
(`(T-15m, T]` + последний пакет + последний strict-valid GPS пакет `<= T`).
Он выбирает целевое плановое событие, считает point-in-time-safe
`current_deviation_seconds` и передаёт `manual_fill` из выбранной строки
расписания (правила — `BACKEND_ML_INTEGRATION.md` §5.1). Он не вычисляет
model-specific признаки и не присылает `event_time > T` (producer-правило);
ML всё равно отбрасывает такие пакеты (defense-in-depth).

## 6. Offline-модули

- `data/official.py` — безопасные loaders: train/test (target отделён,
  schedule по allowlist без `time_fact_begin`), validate inference inputs
  (только `validate/points.csv`, `validate/traffic.csv`,
  `validate/schedule_plan.csv`), шаблон `sample_submission.csv`,
  плановый fingerprint schedule.
- `data/inspection.py`, `data/manifest.py`, `data/canonical.py` — generic
  инструменты (не часть production-пути признаков).
- `target/spec.py`, `target/formulation.py` — `TARGET_SPEC`,
  DIRECT/RESIDUAL (`final_prediction` — единая реконструкция).
- `evaluation/` — `mae()`, temporal split, median baseline.
- `experiments/log.py` — append-only JSONL лог.
- `models/catboost_v1.py`, `models/regimes.py` — замороженные M1 config и
  train regimes; `scripts/run_offline_baseline.py` — 6 экспериментов M1.
- `models/hgb_v1.py` — фиксированная H0-конфигурация HGB и строгая
  training-граница `fit_h0` (ровно `runtime-safe-v1`, NaN допустимы, ±inf нет).
- `data/safe_deviation.py` — offline point-in-time-safe текущее отклонение
  (факт `<= T`) для `offline_context(..., current_deviation_seconds=...)`.

## 7. Признаки, artifact, inference, serving

- `features/builder.py` — замороженный `tabular-v1` (37 признаков,
  `features/schema.py`): `event_time <= T`, окна `(T-w, T]`, strict GPS,
  fail-fast на `time_fact_begin`/`target_delay_s`/`target_class`.
- `features/context.py` — `CanonicalPoint`/`CanonicalTelemetry`/
  `CanonicalBatch`, единственный вход `build_features_from_context`,
  единственная проекция `tabular-v1` → `runtime-safe-v1`
  (`project_runtime_safe_features`) и выбор признаков схемы модели
  (`features_for_schema`).
- `features/adapters.py` — нормализация: `offline_context` (official кадры)
  и `runtime_context` (Contract v1: aware ISO → naive UTC, naive → ошибка,
  случайно пришедшие пакеты `> T` отбрасываются defensively, история не
  обрезается). Канонический контекст допускает отсутствующую скорость
  исторических данных; wire-инвариант `speed` — в `serving/schemas.py`.
- `artifacts/manifest.py` — ArtifactManifest v1 (канонический JSON, SHA-256,
  совместимость по `feature_schema_version`: одна закреплённая схема или любая
  из поддерживаемых вызывающим); `artifacts/bundle.py` —
  Artifact Bundle v1 (`manifest.json` + модель + `bundle.json` с хешами;
  проверка до загрузки модели); `artifacts/metadata.py` — LEGACY.
- `inference/` — `ArtifactPredictor` (bundle → builder → признаки схемы из
  manifest → модель → direct/residual; `cur_dev_s` для residual — из
  канонических признаков), `export_bundle` (обе схемы), validate submission,
  model families: `catboost` (legacy; проверяются сохранённый objective и
  форма выхода самой модели, не класс обёртки) и `hist_gradient_boosting`
  (ровно `HistGradientBoostingRegressor`; `skops` без pickle, доверен только
  `TreePredictor`, деревья структурно проверяются до любого `predict`).
- `serving/` — Contract v1 (`POST /api/v1/predict`), `/health`, `/ready`;
  `artifact_app.py` (production), `mock_app.py` (явный mock). Provisional
  v0 (`/api/v1/predict/batch`, непрозрачный `context`) удалён.

## 8. Offline data flow

1. Официальный датасет лежит вне репозитория (`MOSTRANSPORT_DATASET`).
2. `load_official_split` / `load_validate_inputs` читают только разрешённые
   колонки и файлы.
3. `offline_context` → `CanonicalBatch` → `build_features_from_context`.
4. Обучение кандидата (research) → `ArtifactManifest` → `export_bundle`.
5. Оценка на `labels_test` через `ArtifactPredictor`; submission —
   `scripts/make_submission.py`.

## 9. Каноническое представление

Production-путь не использует `data/canonical.py`: официальные поля
известны и читаются allowlist-loader'ами, а общее внутреннее
представление — `CanonicalBatch`, в которое сходятся offline и runtime.
`CanonicalMapping` остаётся generic offline-инструментом для прочих CSV.

## 10. Стратегия target/evaluation

Target берётся из официальной разметки (`labels_*`), не строится заново;
DIRECT и RESIDUAL — только преобразования обучающей цели и обратно.
Model selection — на `labels_test`; validate — только для submission.

## 11. MAE baseline

Воспроизведённые на `labels_test` baselines (Dataset Evidence v1 / M1):
zero ≈ 103.34 с, train median ≈ 100.87 с, `cur_dev_s` ≈ 93.36 с.

## 12. Согласованность training/serving

Offline и runtime используют один канонический Feature Builder: при
эквивалентном каноническом входном контексте путь построения признаков
общий и детерминированный (проверено тестами и на реальных данных —
побитово одинаковые признаки и прогноз из одного bundle). Это code-path
parity, а не полная parity источников:

- runtime `current_deviation_seconds` (point-in-time-safe) не идентичен
  official `cur_dev_s` датасета;
- CSV имеет шаг telemetry ~12–15 с, эмулятор может присылать ~1 Гц —
  count-признаки (`rows_*`, `valid_gps_count_*`) могут быть смещены.

Для production-кандидата (`runtime-safe-v1`) count-признаки исключены, а
offline-отклонение считается point-in-time-safe (`data/safe_deviation.py`);
`tabular-v1` не меняется и молча не нормализуется. Совпадение семантики
отклонения с тем, что присылает Backend, остаётся интеграционным риском.

## 13. Конфиденциальность данных / политика репозитория

- Организаторские данные никогда не коммитятся и не публикуются.
  `data/raw/`, `data/interim/`, `data/processed/` в `.gitignore`.
- Перед отправкой организаторских данных во внешний сервис — сверяться с
  правилами хакатона.
- Serving не логирует тела запросов (telemetry, контекст): только тип
  события/исключения, число пакетов, версия модели. `422` санитизирован,
  `500`/`503` — фиксированные сообщения.
- Submission не пишется внутрь каталога официального датасета.

## 14. Инварианты

- Метрика — MAE; target, horizon и leakage-правила — см. §2.
- `tabular-v1` (37) и `runtime-safe-v1` (29) заморожены; новые признаки — новой
  версией схемы.
- Один Feature Builder для offline и runtime; `runtime-safe-v1` — только
  проекция его выхода. Схему признаков модели объявляет проверенный manifest.
- Runtime-время — только aware ISO-8601; внутри — naive UTC.
- Модель загружается только из проверенного Artifact Bundle v1, после
  проверки хешей, manifest'а, совместимости схемы, family и формулировки;
  затем проверяется тип задачи модели (скалярная регрессия) — иначе `503`.
- Wire: `speed` — всегда конечное число; producer не шлёт `event_time > T`.
- Hidden validate target никогда не вычисляется; факт test schedule не
  используется для validate.
- Python не хранит `VehicleState` и не делает matching/scheduling.

## 15. Открытые вопросы (modeling/research, не инфраструктура)

Решено research (train-only): группа B не используется (клоны группы A),
production-кандидат — HGB H0 DIRECT на `runtime-safe-v1`, обучение на группе A;
финальный official artifact — этап P3. Открыто:

- соответствие исторических строк без навигационных полей runtime-пакетам;
- совпадение runtime `current_deviation_seconds` Backend'а с offline
  safe-семантикой (см. §12).

## 16. Явные non-goals (для ML-части)

- Backend-логика в ML-коде: NDTP, `VehicleState`, schedule/map matching,
  scheduler, alerts, WebSocket, dashboard (это зона Backend/Frontend).
- Вероятности задержки и reason — пока нет модели, `reason = null`.
- PyTorch/ONNX model families (добавляются новой записью `MODEL_FAMILIES`).
- MLflow, DVC, Airflow, Optuna/HPO-фреймворки.
- Оркестрация (Compose/Kubernetes) в ML-части: у ML только минимальный
  `Dockerfile` сервиса (artifact монтируется read-only); контейнеры
  Backend/Frontend — зона Backend.

## 17. Где продолжать

Рабочий цикл modeling → artifact → submission/serving —
[`HACKATHON_RUNBOOK.md`](HACKATHON_RUNBOOK.md).

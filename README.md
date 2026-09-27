# mostransport-ml

ML-часть хакатона Мостранспорта: раннее прогнозирование задержки прибытия
наземного транспорта. Точка прогноза — `(tr_id, T)`; target —
`target_delay_s = time_fact_begin − time_begin` (секунды, `+` опоздание,
`−` опережение) на целевом действии расписания в горизонте `(T+10м, T+15м]`.
Официальная метрика — MAE.

Это общий командный репозиторий: рядом с ML-частью (`src/mostransport_ml`,
`scripts/`, `tests/`) лежат Backend (`backend/`), Frontend (`frontend/`), БД
(`database/`) и `docker-compose.yml` — они вне зоны владения ML и для ML-задач
только читаются. Граница: Backend присылает доменные факты + сырую историю
telemetry по Backend → ML Contract v1, ML считает model-specific признаки и
прогноз. Подробнее — [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Статус ML:** research-решение заморожено — `HistGradientBoostingRegressor`
H0, DIRECT, схема признаков `runtime-safe-v1` (29 признаков, без raw packet
counts), обучение только на группе A (консервативный фиксированный кандидат, не
доказанный оптимум). P1 и P2 закрыты тегами, P3 даёт один воспроизводимый путь:
`scripts/train_final_hgb.py` строит финальный artifact
`hgb-h0-runtime-safe-v1-group-a-v1` из официальных данных, воспроизводя research
GroupKFold OOF MAE 77.34108978455868 с допуском 1e-9. Artifact
создаётся локально (`artifacts/`, в `.gitignore`) и не коммитится.
`integration-fixture-v1` и `integration-fixture-hgb-v1` — синтетические fixtures
только для проверки интеграции, не модели качества.

## С чего начать чтение

1. **README** (этот файл) — быстрый вход и команды.
2. [`docs/PROJECT_KNOWLEDGE.md`](docs/PROJECT_KNOWLEDGE.md) — главная база
   знаний: факты, ownership, модули, инварианты.
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — технические границы,
   диаграммы, инварианты.
4. [`docs/BACKEND_ML_INTEGRATION.md`](docs/BACKEND_ML_INTEGRATION.md) — Backend →
   ML Contract v1 (`POST /api/v1/predict`): канонический контракт и его
   семантика, quickstart, handoff для Backend. Карта реализации —
   [`docs/ML_SERVING_CONTRACT.md`](docs/ML_SERVING_CONTRACT.md).
5. [`docs/HACKATHON_RUNBOOK.md`](docs/HACKATHON_RUNBOOK.md) — рабочий цикл
   modeling → artifact → submission/serving.

Для AI coding agents — сначала [`AGENTS.md`](AGENTS.md).

## Что реализовано (замороженные checkpoint'ы)

- **Dataset Evidence v1** (`dataset-evidence-v1`) — доказательный аудит
  официального датасета (`notebooks/fz/`). Тег содержит 92-ячеечную версию
  notebook, не исторический 94-ячеечный артефакт `9eb2c78b…` — см.
  `notebooks/fz/01_official_dataset_evidence.PROVENANCE.md`.
- **M1** (`m1-tabular-baseline-v1`) — безопасные official loaders,
  point-in-time Feature Builder `tabular-v1` (37 признаков, `event_time <= T`,
  окна `(T-w, T]`, strict GPS), CatBoost baseline, 6 заранее заданных
  экспериментов (`scripts/run_offline_baseline.py`).
- **M2-I1** (`m2-i1-streaming-context-v1`) — канонический ML-контекст
  (`CanonicalBatch`) и offline/runtime адаптеры в один и тот же builder.
- **M2 infrastructure** — aware-UTC timestamps Contract v1,
  `ArtifactManifest` v1, Artifact Bundle v1, `ArtifactPredictor`,
  serving `POST /api/v1/predict`, безопасный validate inference и
  `scripts/make_submission.py`.
- **P1** (`ml-runtime-safe-v1`) — frozen схема `runtime-safe-v1` (строгая
  проекция 37 → 29 канонического builder'а), offline point-in-time-safe
  current deviation и его явная подача в `offline_context`.
- **P2** (`ml-hgb-artifact-v1`) — model family `hist_gradient_boosting`
  (сериализация `skops`, без pickle), конфигурация H0 (`models/hgb_v1.py`),
  `ArtifactPredictor` выбирает схему признаков по проверенному manifest
  (`tabular-v1` или `runtime-safe-v1`).
- **P3** — финальный training path (`training/final_hgb.py`,
  `scripts/train_final_hgb.py`): группа A → safe deviation → канонический
  builder → `runtime-safe-v1` → GroupKFold(5) OOF gate → финальный H0 fit →
  Artifact Bundle v1 → load-back parity; submission и serving проверены на
  финальном artifact'е.

ИСТОРИЧЕСКИ: M1 использовал CatBoost/`tabular-v1` и оценку на `labels_test`;
CatBoost-artifacts по-прежнему загружаются (legacy). Не реализовано:
probability/reason.

## Финальная модель: одна команда

```bash
uv sync --extra dev --extra serving
export MOSTRANSPORT_DATASET=/path/to/official/dataset
uv run python scripts/train_final_hgb.py \
  --artifact-dir artifacts/hgb-h0-runtime-safe-v1-group-a-v1 \
  --report artifacts/reports/p3-hgb-h0-group-a-v1.json
uv run python scripts/make_submission.py \
  --artifact-dir artifacts/hgb-h0-runtime-safe-v1-group-a-v1 \
  --output submission-hgb-h0-runtime-safe-v1.csv
```

Если OOF gate не пройден, artifact не создаётся (exit code 3). `labels_test`,
`test/traffic.csv`, `validate/**` и факт test schedule training не читает.

## Быстрый старт: offline

Из корня клонированного репозитория:

```bash
uv sync --extra dev
export MOSTRANSPORT_DATASET=/path/to/official/dataset
uv run python scripts/smoke_offline.py
uv run python scripts/make_submission.py --artifact-dir <bundle> --output submission.csv
```

Submission использует `cur_dev_s` из `validate/points.csv` (factual validate
schedule не существует); это не доказательство того, что runtime
`current_deviation_seconds` Backend'а имеет ту же семантику.

## Быстрый старт: serving

```bash
uv sync --extra dev --extra serving
# integration-only artifact (INTEGRATION TEST ONLY, NOT FOR SUBMISSION, NOT A QUALITY MODEL):
uv run python scripts/build_integration_artifact.py \
  --output /tmp/mostransport-integration-artifact --replace
MOSTRANSPORT_ARTIFACT_DIR=/tmp/mostransport-integration-artifact \
  uv run uvicorn mostransport_ml.serving.artifact_app:create_app_from_env --factory
uv run python scripts/smoke_contract_v1.py --base-url http://127.0.0.1:8000
```

HGB/`runtime-safe-v1` integration fixture — тот же builder с
`--model-family hist_gradient_boosting`; smoke проверяет согласованность
`model_version`/`feature_schema_version` между `/ready` и ответом, схему можно
закрепить `--expected-feature-schema-version runtime-safe-v1`.

Финальная модель подключается заменой `MOSTRANSPORT_ARTIFACT_DIR` на
`artifacts/hgb-h0-runtime-safe-v1-group-a-v1` (smoke:
`--expected-feature-schema-version runtime-safe-v1`). Затем: `GET /health`, `GET /ready`, `POST /api/v1/predict`,
`/docs`, `/openapi.json`. Минимальный образ — `Dockerfile`; подробности —
[`docs/BACKEND_ML_INTEGRATION.md`](docs/BACKEND_ML_INTEGRATION.md).

## Тесты

```bash
uv sync --extra dev --extra serving   # обязателен для полного набора тестов
uv run pytest -q
uv run ruff check src scripts tests
```

## Куда идти за подробностями

Все команды и нюансы (proxy в WSL, гигиена данных, git) —
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md). Про данные организаторов —
[`data/README.md`](data/README.md).

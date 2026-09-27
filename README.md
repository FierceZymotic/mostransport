# Mostransport live-demo fix

1. Replace `backend/src/prediction/trip-matcher.service.ts` with the file from this package.
2. Add `database/03-demo-vehicles.sql` to the PostgreSQL init scripts in `docker-compose.yml`:

<<<<<<< HEAD
```yaml
- ./database/03-demo-vehicles.sql:/docker-entrypoint-initdb.d/03-demo-vehicles.sql:ro
=======
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

## Quick demo

Нужен только Docker с Docker Compose v2 и образ NDTP-эмулятора организаторов
(`docker load -i ndtp-telemetry-emulator.tar` один раз, либо
`EMULATOR_TAR=/path/ndtp-telemetry-emulator.tar ./run-demo.sh`).

```bash
./run-demo.sh          # Linux / WSL / macOS
.\run-demo.ps1         # Windows PowerShell
```

Скрипт создаёт `.env` из `.env.example`, если его нет (существующий не трогает),
собирает и поднимает стек `docker compose up -d --build`, ждёт готовности
(ML `/ready`, Postgres, Backend `/health`, дашборд), печатает URL дашборда и
открывает его в браузере, если это возможно. Первая сборка образов занимает
несколько минут. Для карты укажите `VITE_YANDEX_MAPS_API_KEY` в `.env`.

Остановка: `docker compose down` (данные БД сохраняются).

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
>>>>>>> d60937b10a334528a44bd65893863c144a851e24
```

Important: Docker init scripts run only when the PostgreSQL data volume is created. The current database already has the mapping, so for the current demo only the TripMatcher code change is required. The SQL file makes a fresh checkout reproducible.

The change keeps the required 10–15 minute prediction horizon. It only skips the 1 km GPS geometry check when `vehicles.current_tr_id` is explicitly assigned. Vehicles without a trusted trip mapping still use the original 1 km spatial matching.

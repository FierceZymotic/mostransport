# mostransport-ml

Offline ML foundation for the Mostransport hackathon (early prediction of
ground urban transport delays). This repo owns **data → target → evaluation
correctness** for the offline side of the project; it is not the runtime
backend and not the model/serving pipeline.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before adding any new
subsystem — it is the source of truth for scope, ownership boundaries, and
what is still unknown (TBD) until the full task spec is released.

## Scope

**Implemented:**

- Generic, schema-agnostic CSV/dataframe inspection (`inspect_csv.py`).
- `DatasetManifest` — reproducibility metadata (hash, schema, row count),
  never raw rows.
- `CanonicalMapping` — an explicit rename/required-field mechanism, with no
  field names hardcoded (the real schema isn't known yet).
- `TargetSpec` — metadata shape for describing a target once one exists.
- Strict chronological `split_by_time_boundaries` (no random splitting).
- `mae()` — validated primary metric.
- `MedianBaselineRegressor` — constant baseline for an honest first MAE.
- `ExperimentLogger` — append-only JSONL run log.
- Reserved (currently empty) `features/`, `models/`, `artifacts/`,
  `serving/` packages for Valeria's later work.

**Intentionally NOT implemented** (see ARCHITECTURE.md §16 for the full
list): FastAPI serving, any training pipeline (CatBoost/XGBoost/etc.),
feature engineering, target construction, emulator client, a second
`VehicleState` store, PostgreSQL/Redis/Kafka, MLflow/DVC, CI/CD, or any
invented schema. Those either belong to Valeria/Andrey, or can't be built
correctly before the real task spec and CSV schema are released.

## Quick start

```bash
cd /home/fz/projects/mostransport-ml
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run python scripts/smoke_offline.py
```

Inspect a real CSV once one exists:

```bash
uv run python scripts/inspect_csv.py --path data/raw/organizer.csv
uv run python scripts/inspect_csv.py --path data/raw/organizer.csv \
    --nrows 5000 --time-column event_time
```

## Data handling

Organizer-provided data is confidential: `data/raw/`, `data/interim/`, and
`data/processed/` are gitignored. Never commit or publish organizer data,
and check the hackathon's data-usage rules before sending any of it to an
external service or tool. See [data/README.md](data/README.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §13.

## Layout

```
src/mostransport_ml/
├── data/          data inspection, manifest, canonicalization (fz)
├── target/        target metadata shape (fz)
├── evaluation/     metrics, temporal split, baseline (fz)
├── experiments/   JSONL run logging (fz)
├── features/      shared feature logic — reserved for Valeria
├── models/        training/preprocessing pipeline — reserved for Valeria
├── artifacts/     ML artifact bundle — reserved for Valeria
└── serving/       FastAPI inference service — reserved for Valeria
```

See docs/ARCHITECTURE.md before adding new subsystems.

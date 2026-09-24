# mostransport-ml

Offline ML foundation for the Mostransport hackathon (early prediction of
ground urban transport delays). fz owns **data → target → evaluation
correctness**; Valeria owns **features → model → serving**, whose
provisional pre-hackathon foundation (FastAPI shell + mock predictor) also
lives here. This repo is not the runtime backend — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §5.

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
- A provisional serving foundation (Valeria's zone) — see the
  [Serving (provisional)](#serving-provisional) section below.
- `features/` and `models/` remain reserved and empty for Valeria's later
  work — no feature or training logic exists yet.

**Intentionally NOT implemented** (see ARCHITECTURE.md §16 for the full
list): a real model or real feature engineering behind the serving shell,
any training pipeline (CatBoost/XGBoost/etc.), target construction, emulator
client, a second `VehicleState` store, PostgreSQL/Redis/Kafka, MLflow/DVC,
CI/CD, or any invented schema. Those either belong to Valeria/Andrey, or
can't be built correctly before the real task spec and CSV schema are
released.

## Quick start

**Offline only** (fz's zone: data/target/evaluation; lightweight, no
FastAPI/pydantic/uvicorn):

```bash
cd /home/fz/projects/mostransport-ml
uv sync --extra dev
uv run ruff check .
uv run python scripts/smoke_offline.py
```

**Full repository / serving development** — needed to run the full test
suite: `tests/test_serving_*.py` import `fastapi`, which only exists under
the `serving` extra, so `uv sync --extra dev` alone is **not** enough for
`pytest -q` (it fails at collection, before any test runs):

```bash
uv sync --extra dev --extra serving
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
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
├── features/      shared feature logic — reserved, empty (Valeria)
├── models/        training/preprocessing pipeline — reserved, empty (Valeria)
├── artifacts/     metadata.py: minimal ArtifactMetadata (Valeria)
└── serving/       provisional FastAPI shell + MockPredictor (Valeria)
```

See docs/ARCHITECTURE.md before adding new subsystems.

## Serving (provisional)

A provisional FastAPI shell exists ahead of any real model, so backend
integration isn't blocked. **This is a pre-hackathon contract — see
[docs/ML_SERVING_CONTRACT.md](docs/ML_SERVING_CONTRACT.md) for the full,
authoritative write-up** (request/response schema, error semantics, curl
examples). Summary:

```bash
# Serving development (on top of the base dev environment):
uv sync --extra dev --extra serving

# Launch the explicit mock dev server (MockPredictor — never the default,
# always opt-in):
uv run uvicorn mostransport_ml.serving.mock_app:app --host 127.0.0.1 --port 8000
```

- Health: `GET http://127.0.0.1:8000/health`
- Readiness: `GET http://127.0.0.1:8000/ready`
- Predict: `POST http://127.0.0.1:8000/api/v1/predict/batch`
- Interactive docs: `http://127.0.0.1:8000/docs`, schema at `/openapi.json`

No real feature engineering, model, or artifact loading exists behind this
yet — see docs/ARCHITECTURE.md §7 and §16.

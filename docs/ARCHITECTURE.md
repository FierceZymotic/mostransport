# Architecture

Source of truth for how the offline ML part of this project fits into the
team's overall system, what is confirmed vs. unknown, and who owns what.
Read this before adding a new subsystem or abstraction.

This document describes a 48-hour hackathon project. It is intentionally
thin. Where something can't be known yet, it is marked TBD rather than
guessed at.

---

## 1. Purpose

Early prediction of delays/changes in ground urban transport (Mostransport
hackathon). The business goal is to move dispatchers from reactive to
proactive management — e.g. adjusting vehicle counts on a route, or stop
dwell time, before a delay compounds.

This repository is the **offline ML side**: turning organizer-provided CSV
data into a validated target, a leakage-safe evaluation setup, and (once
Valeria's part exists) a trained model artifact. It does not implement
runtime/serving behavior.

## 2. Confirmed organizer facts

As of now, before the full task specification is released:

- Training data will be delivered as a custom CSV.
- The full field-by-field schema is not published yet — it arrives at
  hackathon start.
- A separate Docker image will emulate the online telemetry stream.
- An official CSV ↔ emulator field mapping will be provided.
- The primary scoring metric is **MAE of actual delay**.
- Exact target/delay/horizon semantics are not yet defined.
- External data sources are allowed for enrichment.
- Organizer-provided datasets must not be published or used outside the
  hackathon's permitted scope.

## 3. Current team architecture

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

**Key invariant:** the backend owns operational state; Python owns ML
transformations and inference. The future Python ML Service is stateless
with respect to per-vehicle history — recent telemetry is supplied by the
backend on each inference request, not stored in Python.

## 4. Ownership boundaries

| Owner  | Responsibility |
|--------|----------------|
| **fz** (this repo) | data → target → evaluation correctness |
| **Valeria** | features → model → serving |
| **Andrey** | backend / operational state / scheduler / integration / realtime / frontend |
| **Lisa** | DB / reference data / BI / analytics |

## 5. Existing Andrey backend checkpoint

Read-only reference (backend lives in a sibling repo, NestJS/TypeScript).

**Current implementation** (verified against the backend source):

- `POST /telemetry/events` → `TelemetryController`
- `TelemetryService`, which calls into `VehicleStateService`
- `VehicleStateService` is an early scaffold: it currently mostly
  forms/returns a state object per incoming event — it does **not** yet
  implement a persistent `Map<vehicleId, VehicleState>`
- `PredictionScheduler` exists as a scaffold

**Planned architecture** (not yet built):

- memory-first `VehicleState`, conceptually a `Map<vehicleId, VehicleState>`
- a recent-telemetry window per vehicle
- no mandatory PostgreSQL persistence for `VehicleState`

Observed current event shape (`TelemetryEventDto`): `vehicle_id`,
`event_time`, `lon`, `speed`, `direction`, optional `route_id`, `trip_id`,
`stop_id`. This is **not** treated as a confirmed canonical schema here —
it's the backend's current checkpoint, shown only to keep domain naming
(`vehicle`/`route`/`trip`/`stop`) consistent across the team. It will very
likely change once the official CSV/emulator schema lands.

Backend owns, and this repo must never duplicate (current or planned):

- runtime telemetry ingestion
- operational `VehicleState`
- recent telemetry window
- `PredictionScheduler` / `PredictionService`
- ML Client
- risk rules, alerts
- WebSocket, REST, frontend integration
- PostgreSQL/Prisma (routes, stops, trips, schedule, prediction_history, alerts)

This repo does not, and will not, implement any of the above.

## 6. Offline ML responsibility (fz)

What's implemented here, in `src/mostransport_ml/`:

- `data/inspection.py` — generic, schema-agnostic CSV/dataframe profiling.
- `data/manifest.py` — reproducibility metadata for a dataset version
  (hash, schema, row count), never raw rows.
- `data/canonical.py` — an explicit rename + required-field mechanism,
  with no field list hardcoded (see §9).
- `target/spec.py` — metadata shape for describing a target, not a target
  builder (see §10).
- `evaluation/metrics.py` — MAE, strictly validated.
- `evaluation/temporal.py` — chronological train/validation/test split.
- `evaluation/baseline.py` — median-of-train constant baseline.
- `experiments/log.py` — append-only JSONL experiment log.

Responsibility boundary: **data → target → evaluation correctness**. This
repo makes sure that once the real CSV and target definition exist, they can
be understood, canonicalized, split, and scored correctly and reproducibly.

## 7. Future Valeria integration

Reserved, currently-empty packages: `features/`, `models/`, `artifacts/`,
`serving/`. They exist so Valeria can add real content without reorganizing
the repository, not as stub classes to fill in.

Valeria owns:

- the shared Feature Builder (`features/`)
- preprocessing/model pipeline (`models/`)
- ML Artifact Bundle implementation (`artifacts/`)
- Python FastAPI inference service (`serving/`)

Expected (not yet built) shape of an Artifact Bundle: trained model,
feature schema/version, preprocessing metadata, historical aggregates if
needed, target/model metadata, `model_version`.

## 8. Data flow

See the diagram in §3. Concretely, offline:

1. Organizer CSV lands in `data/raw/` (never committed).
2. `scripts/inspect_csv.py` profiles it — schema, missingness, duplicates.
3. A `DatasetManifest` is built for reproducibility.
4. A `CanonicalMapping` (once the real fields are known) renames/validates
   into the shared domain representation.
5. Target construction (to be built after the task spec is released)
   produces the actual label.
6. `split_by_time_boundaries` produces train/validation/test chronologically.
7. `MedianBaselineRegressor` + `mae()` give an honest baseline immediately.
8. Later: Valeria's feature/model pipeline trains against the same split
   and canonical representation, logging runs via `experiments/log.py`.

## 9. Canonical schema strategy

No canonical field list is hardcoded anywhere in this repo. The official CSV
schema and the emulator schema are unknown until hackathon start, and the
organizer will provide a CSV ↔ emulator mapping.

Instead, `data/canonical.py` provides only the *mechanism*:

```python
CanonicalMapping(
    source_name=...,
    rename={...},  # explicit only, nothing inferred
    required_fields=(...),  # explicit only, nothing inferred
)
```

`apply_canonical_mapping()` renames exactly the listed columns and validates
exactly the listed required fields. No feature engineering happens here.
Once the real schema is known, this is the single seam where organizer CSV
fields and emulator fields both get mapped to one shared domain
representation.

## 10. Target/evaluation strategy

The organizer has confirmed the metric (MAE) but not delay/horizon
semantics. `target/spec.py` gives a stable `TargetSpec` metadata shape
(name, unit, description, version, optional `horizon_minutes`) with nothing
hardcoded — no assumed horizon, no assumed unit, no assumed name.

Actual target construction (turning raw fields into a label column) is
intentionally not implemented and will be built by fz immediately after the
task spec is released.

## 11. MAE baseline

`evaluation/baseline.MedianBaselineRegressor` predicts the median of the
training target, ignoring features entirely. Combined with
`evaluation/metrics.mae`, this gives a baseline validation MAE within
minutes of the real target existing — a number every later model must beat.

## 12. Training-serving consistency

Critical future rule: **the same feature logic must be used offline and
online.** There must never be a second, parallel implementation of feature
logic living in a notebook or duplicated inside the FastAPI service. Once
`features/` has real content, both offline training and the online serving
path import it from the same place.

## 13. Data confidentiality / repo policy

- Organizer-provided data must never be committed or published. `data/raw/`,
  `data/interim/`, `data/processed/` are gitignored (only README/`.gitkeep`
  are tracked).
- Before sending any organizer data to an external service, tool, or AI
  assistant, check the hackathon's data-usage rules first.
- Code and metadata (schema, hashes, row counts) are kept separable from
  confidential raw data — see `DatasetManifest`, which never stores actual
  rows.

## 14. Confirmed invariants

- Primary official metric = MAE.
- Exact delay semantics: TBD.
- Exact horizon semantics: TBD.
- CSV schema: TBD.
- Emulator schema: TBD.
- Official CSV ↔ emulator mapping will be provided by organizers.
- Online primary telemetry source = organizer Docker emulator.
- Node backend owns `VehicleState`, recent telemetry, and the scheduler.
- Python must not duplicate operational `VehicleState`.
- Offline and online must converge on a compatible canonical domain
  representation.
- The same Feature Builder must eventually be used offline and online.
- Organizer datasets must not be committed or published.
- Buses are the current implementation target; domain naming stays
  vehicle-generic (`vehicle`/`route`/`trip`/`stop`).
- External enrichment is allowed but is not part of the pre-hackathon
  critical path.

### Integration sync points

Points where fz, Valeria, and Andrey's work must agree, in the order they'll
matter:

1. **ML Task Spec** — target/horizon/metric semantics, once released.
2. **Canonical Schema** — the shared domain representation both offline and
   online map into.
3. **Feature Contract** — the shared `features/` interface both training and
   serving call.
4. **Artifact Contract** — the shape of the ML Artifact Bundle the serving
   layer loads.

## 15. TBD after full task release

- Actual CSV column names and types.
- Actual delay/target definition and unit.
- Actual prediction horizon.
- Emulator event schema and its mapping to the CSV schema.
- Whether purge/embargo windows or walk-forward CV are needed for temporal
  evaluation (depends on horizon/leakage characteristics of the real data).
- Feature set (Valeria, post-target).
- Model choice (Valeria).
- Artifact Bundle concrete schema (Valeria).

## 16. Explicit non-goals (for this offline repo)

Not implemented here, on purpose:

- FastAPI / serving service
- CatBoost/XGBoost/LightGBM or any training pipeline
- Feature engineering or an invented feature set
- Emulator client or emulator schema
- A second `VehicleState` store or a Python scheduler
- PostgreSQL/Prisma, Redis, Kafka layers
- MLflow, DVC, Airflow/Prefect/Ray, Optuna/Hydra
- CI/CD, dashboards, Docker/Compose for this repo
- Invented target, invented telemetry fields, invented canonical schema

## 17. First-hour hackathon checklist

Once the organizer CSV and task spec are released:

1. `uv run python scripts/inspect_csv.py --path <csv>` — first look at schema,
   missingness, duplicates.
2. Build a `DatasetManifest` for the received file (`data/manifest.py`).
3. Define the real `CanonicalMapping` for the organizer schema and, once
   given, the emulator schema.
4. Read the full task spec; fill in a real `TargetSpec` and implement target
   construction.
5. Pick `train_end` / `validation_end` boundaries and run
   `split_by_time_boundaries`.
6. Fit `MedianBaselineRegressor`, compute baseline MAE with `mae()`, log it
   via `experiments/log.py`.
7. Hand off canonical data + target + split boundaries to Valeria for
   feature/model work.
8. Keep this document updated as TBDs resolve.

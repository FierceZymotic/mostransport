# ML Serving Contract (provisional v0)

**THIS IS A PROVISIONAL PRE-HACKATHON CONTRACT.** It exists only to unblock
HTTP integration between the backend (Andrey) and the Python ML service
(Valeria) before the organizer's task spec, CSV schema, and CSV ↔ emulator
mapping are released. Once those land, this contract will be tightened —
see §8. The current HTTP layer and app structure are sufficient for
mock/pre-hackathon serving; the request/response *shapes* will change, and
the Artifact Contract sync point may still extend the app/Predictor
boundary itself once real model metadata semantics exist (see
`docs/ARCHITECTURE.md` §7).

## 1. Purpose

Let the backend call the Python ML service over HTTP today, using a
deterministic mock predictor, so both sides can build and test their
integration ahead of a real model.

## 2. Current provisional status

- Implementation: `src/mostransport_ml/serving/`.
- Predictor in use: `MockPredictor` (`serving/mock.py`) — deterministic,
  always ready, `model_version = "mock-v0"`, `predicted_delay = 0.0` for
  every vehicle.
- No real feature engineering, no real model, no artifact loading.
- Run it explicitly — there is no hidden default-to-mock behavior:

  ```bash
  uv sync --extra dev --extra serving
  uv run uvicorn mostransport_ml.serving.mock_app:app --host 127.0.0.1 --port 8000
  ```

## 3. Endpoint list

| Method | Path                        | Purpose                          |
|--------|-----------------------------|-----------------------------------|
| GET    | `/health`                   | Process liveness                  |
| GET    | `/ready`                    | Predictor/runtime readiness       |
| POST   | `/api/v1/predict/batch`     | Batch delay prediction            |

No other endpoints exist. `/openapi.json` and `/docs` are available (FastAPI
defaults) for interactive/schema inspection. **OpenAPI is kept aligned with
the actual runtime contract** — every response status an endpoint can really
return is documented against the schema it really returns (see §6); it is
meant to be usable by the backend as the integration contract, not a stale
default.

## 4. PredictionBatchRequest v0

```jsonc
{
  "prediction_time": "2026-01-01T00:00:00Z",   // ISO 8601 datetime, any timezone or none
  "horizon_minutes": 15,                        // optional; None, or finite and > 0
  "vehicles": [
    {
      "vehicle_id": "synthetic-1",              // string, matches backend's current vehicle_id type
      "context": { "synthetic": true }          // opaque JSON object — see below
    }
  ]
}
```

Rules:

- Top-level unknown fields are **rejected** (`extra="forbid"`).
- Each vehicle envelope's unknown fields are **rejected**.
- `context` **must** be a JSON object (not a string, number, array, or null)
  and is currently required on every vehicle (send `{}` if you have nothing
  to put there yet).
- `vehicles` must have **at least 1** entry. No maximum is imposed yet —
  the emulator's realistic batch size isn't known.
- `vehicle_id` is a plain string, matching the type already used in the
  backend's current `TelemetryEventDto.vehicle_id`.
- `horizon_minutes`, when present, must be a **finite number strictly
  greater than 0** — `0`, negative values, `NaN`, and `Infinity`/`-Infinity`
  are all rejected with `422`. There is no hardcoded default, no assumed
  unit, and no maximum yet — this is a generic sanity bound, not a guess at
  the organizer's real horizon.

**`context` is intentionally opaque.** Nothing on the Python side reads any
key out of it — it exists purely to unblock HTTP integration. It **will be
replaced or tightened into a real typed schema** once the official CSV ↔
emulator field mapping is released. Do not build backend logic that depends
on specific keys inside it surviving unchanged.

## 5. PredictionBatchResponse v0

```jsonc
{
  "prediction_time": "2026-01-01T00:00:00Z",
  "horizon_minutes": 15,
  "model_version": "mock-v0",
  "target_name": null,          // optional; TBD
  "target_unit": null,          // optional; units TBD
  "predictions": [
    {
      "vehicle_id": "synthetic-1",
      "predicted_delay": 0.0,   // finite number when status == "ok"; unit TBD
      "status": "ok"
    }
  ]
}
```

- **Invariant:** `status == "ok"` if and only if `predicted_delay` is a
  present, finite number. Any other status (`insufficient_data`, `error`)
  always carries `predicted_delay: null`. This is enforced at the schema
  level (`VehiclePrediction`), not just by convention — a prediction
  violating it is rejected before the response is ever built.
- `predicted_delay` is a plain numeric value — **never** `NaN`/`Infinity`.
  A predictor that produces a non-finite value causes a controlled 500, not
  a malformed response (see §6).
- The field is deliberately **not** named `predicted_delay_sec` or
  `predicted_delay_minutes` — the unit is unknown until the organizer
  releases target semantics. `target_unit` will carry that once it's known.
- `status` is one of: `ok`, `insufficient_data`, `error`. `MockPredictor`
  always returns `ok`.
- The response contains **no** risk level, probability, SHAP values, or
  dispatcher recommendation. Turning a delay into a LOW/MEDIUM/HIGH risk
  classification or an actionable recommendation is backend/product logic,
  not this service's job.

## 6. Error semantics

| Situation                                    | Response                                  | Schema |
|-----------------------------------------------|--------------------------------------------|--------|
| Invalid request body                          | `422`, sanitized detail (see below)       | `ValidationErrorResponse` |
| Predictor not ready                           | `503`, generic `detail`                    | `ErrorResponse` |
| Predictor violates the identity/order/mutation/readiness-type contract, or otherwise produces invalid output | `500`, generic `detail`, no traceback | `ErrorResponse` |
| Predictor raises for any other reason         | `500`, generic `detail`, no traceback     | `ErrorResponse` |

`ValidationErrorResponse` is `{"detail": [ErrorDetail, ...]}` where each
`ErrorDetail` is exactly `{"loc": [...], "msg": "...", "type": "..."}` — no
`input`, no `ctx`. `ErrorResponse` is exactly `{"detail": "<short message>"}`.
Both are real Pydantic models (`serving/schemas.py`) used to build the
response body itself, so the published OpenAPI schema and the actual
runtime body cannot drift apart; FastAPI's own default `HTTPValidationError`
schema (which would otherwise advertise `input`/`ctx`) is not published.

The backend should treat any non-2xx response as "prediction unavailable
right now" and continue operating — this service does not implement retries
or circuit breaking; that policy belongs to the backend (TBD there).

**Predictor output contract:** a predictor must return exactly one
prediction per input vehicle, in the same order, with a matching
`vehicle_id` at each position — checked against the *original* request, not
whatever the predictor's copy of it looks like afterward (see next
paragraph). A predictor that reorders, drops, adds, or mismatches an id is
treated as producing invalid output (`500`) — this service never silently
re-sorts or re-maps predictor output to "fix" it.

**A predictor must not mutate the request it is given.** `predict_batch`
receives `PredictionBatchRequest` by reference; this service snapshots
`prediction_time`, `horizon_minutes`, and the ordered vehicle ids *before*
calling the predictor, and re-checks them afterward. Any change — including
clearing `vehicles` and returning an empty prediction list — is a contract
violation (`500`), not a way to short-circuit validation. The response is
always built from the original snapshot, never from a possibly-mutated
request.

**Predictor readiness/model_version have a runtime-checked type contract,**
not just a type hint: `is_ready()` must return exactly `bool` (a truthy
string or `1` is rejected, not coerced), and once ready, `model_version()`
must return a non-empty `str`. A predictor that violates this is treated as
"not ready" for `/ready` (`503`) and as invalid output for `/predict`
(`500`) — never an uncontrolled crash either way.

**422 responses are sanitized**, on two axes. FastAPI's default validation
error body includes the offending `input` value verbatim, which would echo
request data (including `context`) straight back to the client — this
service's handler returns only `{"detail": [{"loc": [...], "msg": "...",
"type": "..."}]}`. Additionally, `loc` itself is sanitized: an unknown,
client-supplied field name (e.g. a bogus top-level key) is itself request
data, so any `loc` segment that isn't one of this schema's known field names
(`prediction_time`, `horizon_minutes`, `vehicles`, `vehicle_id`, `context`)
or a structural marker/list index is replaced with `<field>`.

**Server-side logs never carry request content or exception messages.**
Nothing sent to the client, and nothing written to logs, includes a stack
trace, an internal file path, an exception's message, or an echo of the
request's `context` — a predictor exception's message could itself be
derived from request data. Logs record only metadata: batch size, model
version, error *type* (e.g. `RuntimeError`), and success/failure.

## 7. Health / readiness

- `GET /health` — always `200 {"status": "ok"}` if the process is up. Does
  not check the predictor, the organizer emulator, or any database.
- `GET /ready` — `200 {"ready": true, "model_version": "..."}` when the
  predictor can serve inference, otherwise `503 {"ready": false,
  "model_version": null}`. This also covers the predictor's readiness or
  `model_version` check itself raising — that is treated as "not ready"
  (`503`), never an uncontrolled `500`; only the exception type is logged.
  `MockPredictor` is always ready.

## 8. Explicit TBD after organizer release

- The real shape of `context` (replacing the opaque object with typed
  fields once the CSV ↔ emulator mapping exists).
- Exact `delay`/target semantics, unit, and `horizon_minutes` semantics.
- Whether a maximum batch size is needed (depends on the emulator).
- The real predictor (artifact-backed, replacing `MockPredictor`).
- `target_name` / `target_unit` concrete values.

## 9. What backend owns

Telemetry ingestion, `VehicleState`, recent telemetry window,
`PredictionScheduler`, `PredictionService`, the ML Client that calls this
API, risk rules/alerts, WebSocket/REST to the frontend, PostgreSQL. See
`docs/ARCHITECTURE.md` §5.

## 10. What Python ML owns

This HTTP service, the `Predictor` boundary, feature logic (once it exists),
model training/inference, artifact metadata/bundle. Stateless across
requests — no per-vehicle history is stored here; recent telemetry, if
needed, travels in the request itself.

## 11. Example curl request

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "prediction_time": "2026-01-01T00:00:00Z",
    "horizon_minutes": 15,
    "vehicles": [
      {"vehicle_id": "synthetic-1", "context": {"synthetic": true}},
      {"vehicle_id": "synthetic-2", "context": {"synthetic": true}}
    ]
  }'
```

## 12. Example response

```json
{
  "prediction_time": "2026-01-01T00:00:00Z",
  "horizon_minutes": 15.0,
  "model_version": "mock-v0",
  "target_name": null,
  "target_unit": null,
  "predictions": [
    {"vehicle_id": "synthetic-1", "predicted_delay": 0.0, "status": "ok"},
    {"vehicle_id": "synthetic-2", "predicted_delay": 0.0, "status": "ok"}
  ]
}
```

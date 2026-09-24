"""External API DTOs for the provisional ML serving contract.

THIS IS A PROVISIONAL PRE-HACKATHON CONTRACT — see
docs/ML_SERVING_CONTRACT.md for the full write-up. The official CSV ↔
emulator field mapping does not exist yet, so `VehicleRequest.context` is an
intentionally opaque JSON object. Nothing in this module assumes any real
transport field (no lat/lon/speed/route_id/etc.) — those arrive only after
the official mapping is released, at which point `context` gets replaced or
tightened into a real domain schema.

This module holds DTOs only — no inference logic, no predictor calls.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A plain float that rejects NaN/+Infinity/-Infinity, so a non-finite value
# can never survive into a serialized response (see §13 of the contract doc).
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

# horizon_minutes has no known upper bound or default yet, but a horizon can
# never be zero, negative, or non-finite — this is a generic invariant, not
# a guess at the organizer's real horizon.
PositiveFiniteMinutes = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class PredictionStatus(StrEnum):
    """Per-vehicle outcome of a prediction attempt."""

    OK = "ok"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


class VehicleRequest(BaseModel):
    """One vehicle's provisional inference input.

    `context` is intentionally opaque: MockPredictor (and this schema) never
    interprets its keys. It exists only to unblock HTTP integration ahead of
    the official schema.
    """

    model_config = ConfigDict(extra="forbid")

    vehicle_id: str
    context: dict[str, Any]


class PredictionBatchRequest(BaseModel):
    """Request envelope for `POST /api/v1/predict/batch` (provisional v0)."""

    model_config = ConfigDict(extra="forbid")

    prediction_time: datetime
    horizon_minutes: PositiveFiniteMinutes | None = None
    vehicles: list[VehicleRequest] = Field(min_length=1)


class VehiclePrediction(BaseModel):
    """One vehicle's prediction result.

    Invariant, enforced here rather than trusted from callers:
    `status == "ok"` iff a finite `predicted_delay` is present; any other
    status carries no delay at all. Non-finite predictor output must never
    reach a client (see `serving/service.py`).
    """

    model_config = ConfigDict(extra="forbid")

    vehicle_id: str
    predicted_delay: FiniteFloat | None = None
    status: PredictionStatus

    @model_validator(mode="after")
    def _check_status_delay_invariant(self) -> VehiclePrediction:
        if self.status == PredictionStatus.OK:
            if self.predicted_delay is None:
                raise ValueError("predicted_delay is required when status is 'ok'")
        elif self.predicted_delay is not None:
            raise ValueError(f"predicted_delay must be None when status is {self.status.value!r}")
        return self


class PredictionBatchResponse(BaseModel):
    """Response envelope for `POST /api/v1/predict/batch` (provisional v0).

    Centred on a numeric delay (the confirmed official metric is MAE of
    actual delay), not a classification probability. `target_unit` is
    optional because units are not yet defined by the organizers.
    """

    model_config = ConfigDict(extra="forbid")

    prediction_time: datetime
    horizon_minutes: PositiveFiniteMinutes | None = None
    model_version: str
    target_name: str | None = None
    target_unit: str | None = None
    predictions: list[VehiclePrediction]


class HealthResponse(BaseModel):
    """`GET /health` — process liveness only."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class ReadyResponse(BaseModel):
    """`GET /ready` — predictor/runtime readiness."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    model_version: str | None = None


class ErrorDetail(BaseModel):
    """One sanitized validation error entry.

    Deliberately narrower than FastAPI/Pydantic's default per-error shape:
    no `input` (the offending value) and no `ctx`, since request data —
    including a vehicle's `context` — must never be echoed back. `loc`
    segments that aren't a known field name are replaced; see the sanitizer
    in `serving/app.py`.
    """

    model_config = ConfigDict(extra="forbid")

    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """`422` body shape actually returned by this service's validation
    error handler — used both to build that response and to document it in
    OpenAPI, so the two can't drift apart."""

    model_config = ConfigDict(extra="forbid")

    detail: list[ErrorDetail]


class ErrorResponse(BaseModel):
    """Generic safe error body for `500`/`503` responses: a short, fixed,
    non-sensitive message — never an exception message or a traceback."""

    model_config = ConfigDict(extra="forbid")

    detail: str

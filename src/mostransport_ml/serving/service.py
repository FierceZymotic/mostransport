"""Inference orchestration and the replaceable Predictor boundary.

    FastAPI endpoint -> InferenceService -> Predictor -> prediction results

`InferenceService` owns no transport-specific logic and no per-vehicle
history — it validates readiness, calls the predictor, and turns raw output
into a strictly-validated response. `Predictor` is the seam a future
artifact-backed implementation plugs into without touching the FastAPI layer
or endpoint code (see `mock.py` for the only implementation that exists
today).

Two things a `Predictor` implementation must never be able to do, even by
accident: mutate the request it was handed, or lie about its own
readiness/model_version type. Both are enforced here, not merely documented
— see `_RequestSnapshot` and the validation in `is_ready`/`model_version`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from mostransport_ml.serving.schemas import (
    PredictionBatchRequest,
    PredictionBatchResponse,
    PredictionStatus,
    VehiclePrediction,
)


class PredictorNotReadyError(RuntimeError):
    """Raised when inference is requested before the predictor is ready."""


class PredictorContractError(RuntimeError):
    """Raised when a predictor violates its contract with `InferenceService`:
    wrong prediction count, wrong order, a `vehicle_id` that doesn't match
    the input at that position, mutation of the request it was handed, or an
    `is_ready`/`model_version` return value of the wrong type or shape. The
    predictor must satisfy this contract itself — this is fail-fast
    detection, not silent correction (no re-sorting, no mapping by id, no
    coercion of a truthy/falsy value into a bool)."""


@dataclass(frozen=True)
class _RequestSnapshot:
    """Immutable snapshot of the request fields the serving contract depends
    on, taken before the predictor runs.

    Deliberately narrow: this is not a deep copy of the whole request (no
    duplicating `context` payloads) — only the fields needed to detect a
    predictor mutating the envelope out from under `InferenceService`, and
    to build the response from values a predictor can't have touched.
    """

    prediction_time: datetime
    horizon_minutes: float | None
    vehicle_ids: tuple[str, ...]

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicle_ids)

    @classmethod
    def of(cls, request: PredictionBatchRequest) -> _RequestSnapshot:
        return cls(
            prediction_time=request.prediction_time,
            horizon_minutes=request.horizon_minutes,
            vehicle_ids=tuple(vehicle.vehicle_id for vehicle in request.vehicles),
        )


@dataclass(frozen=True)
class RawPrediction:
    """What a `Predictor` returns for one vehicle, before response validation.

    Deliberately not validated on construction — `InferenceService` is
    responsible for turning this into a strictly-validated `VehiclePrediction`
    (finite `predicted_delay` or None), so a buggy predictor can't put a
    malformed value directly on the wire.
    """

    vehicle_id: str
    predicted_delay: float | None
    status: PredictionStatus = PredictionStatus.OK


class Predictor(Protocol):
    """Minimal, replaceable boundary between serving and a model implementation.

    Stateless by contract: no method here takes or returns anything tied to
    a previous request, and `predict_batch` must not mutate the `request` it
    is given. This boundary is sufficient for mock/pre-hackathon serving; the
    Artifact Contract sync point (see docs/ARCHITECTURE.md §7) may still
    extend it once real model metadata semantics exist, but a future
    artifact-backed predictor is expected to implement these same methods
    without the FastAPI layer changing.

    `typing.Protocol` documents this contract but can't enforce it at
    runtime — `InferenceService` is what actually checks it (see
    `is_ready`/`model_version` below, and the mutation check in
    `predict_batch`).
    """

    def is_ready(self) -> bool:
        """Whether the predictor can currently serve `predict_batch`.
        Must return exactly `bool` — not merely a truthy/falsy value."""
        ...

    def model_version(self) -> str:
        """A short, stable, non-empty identifier for the currently loaded
        model. Only meaningful once `is_ready()` is `True`."""
        ...

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        """Return exactly one `RawPrediction` per `request.vehicles`, in
        order, without mutating `request`."""
        ...


class InferenceService:
    """Validates readiness, calls the predictor, and builds a safe response."""

    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor

    def is_ready(self) -> bool:
        """Runtime-validated readiness.

        `typing.Protocol` can't stop a predictor from returning e.g. the
        string `"false"` (truthy!) or `1` instead of a real `bool` — that
        would otherwise make `/ready` and `/predict` disagree. Reject
        anything that isn't exactly `bool` instead of coercing it.
        """
        value = self._predictor.is_ready()
        if type(value) is not bool:
            raise PredictorContractError(
                f"predictor.is_ready() must return exactly bool, got "
                f"{type(value).__name__} ({value!r})"
            )
        return value

    def model_version(self) -> str:
        """Runtime-validated model version: a non-empty `str`.

        Only call this once the predictor has reported itself ready — a
        not-ready predictor is not required to have a meaningful version.
        """
        value = self._predictor.model_version()
        if not isinstance(value, str) or value == "":
            raise PredictorContractError(
                f"predictor.model_version() must return a non-empty str, got {value!r}"
            )
        return value

    def predict_batch(self, request: PredictionBatchRequest) -> PredictionBatchResponse:
        if not self.is_ready():
            raise PredictorNotReadyError("predictor is not ready to serve inference")

        snapshot = _RequestSnapshot.of(request)

        raw_predictions = self._predictor.predict_batch(request)

        # The predictor was handed `request` by reference and could have
        # mutated it. Re-check every protected field against the snapshot
        # taken *before* the call — a predictor must never be able to
        # redefine what "the request" was after the fact.
        if request.prediction_time != snapshot.prediction_time:
            raise PredictorContractError("predictor mutated request.prediction_time")
        if request.horizon_minutes != snapshot.horizon_minutes:
            raise PredictorContractError("predictor mutated request.horizon_minutes")
        if len(request.vehicles) != snapshot.vehicle_count:
            raise PredictorContractError("predictor mutated the number of request.vehicles")
        if tuple(vehicle.vehicle_id for vehicle in request.vehicles) != snapshot.vehicle_ids:
            raise PredictorContractError("predictor mutated request.vehicles ids and/or order")

        if len(raw_predictions) != snapshot.vehicle_count:
            raise PredictorContractError(
                f"predictor returned {len(raw_predictions)} predictions for "
                f"{snapshot.vehicle_count} input vehicles"
            )

        # Exact 1:1 identity/order, checked against the ORIGINAL ids (not
        # whatever `request.vehicles` looks like now) — a predictor that
        # reorders or swaps IDs must fail loudly here. This never re-sorts
        # or maps by id to "fix" it.
        for index, (expected_id, raw) in enumerate(
            zip(snapshot.vehicle_ids, raw_predictions, strict=True)
        ):
            if raw.vehicle_id != expected_id:
                raise PredictorContractError(
                    f"predictor violated the 1:1 identity/order contract at index "
                    f"{index}: expected vehicle_id {expected_id!r}, got "
                    f"{raw.vehicle_id!r}"
                )

        # Constructing VehiclePrediction validates finiteness of
        # predicted_delay; a non-finite predictor output raises here, before
        # any response is built or sent.
        predictions = [
            VehiclePrediction(
                vehicle_id=raw.vehicle_id,
                predicted_delay=raw.predicted_delay,
                status=raw.status,
            )
            for raw in raw_predictions
        ]

        # Built from the ORIGINAL snapshot, never from `request` — even if
        # every check above somehow passed, the response can't carry a
        # mutated prediction_time/horizon_minutes.
        return PredictionBatchResponse(
            prediction_time=snapshot.prediction_time,
            horizon_minutes=snapshot.horizon_minutes,
            model_version=self.model_version(),
            predictions=predictions,
        )

"""FastAPI app factory and endpoints for the provisional ML serving contract.

THIS IS A PROVISIONAL PRE-HACKATHON CONTRACT — see
docs/ML_SERVING_CONTRACT.md. Endpoints contain no model/feature logic; they
validate the request, delegate to `InferenceService`, and translate failures
into safe HTTP errors. There is no hidden default predictor — `create_app`
always requires one explicitly, so a real deployment can never silently fall
back to a mock (see `mock_app.py` for the explicit development app).

Confidentiality, on both sides of the wire:

- Incoming request payloads (especially each vehicle's `context`, which will
  hold organizer data once the hackathon starts) are never logged here —
  only metadata such as batch size, model version, and success/failure.
- Server-side error logs record only the *type* of an exception
  (`type(exc).__name__`), never `str(exc)` or a traceback — a predictor
  exception's message could itself contain data derived from the request.
- 422 validation responses are sanitized: FastAPI's default body includes
  the offending `input` value verbatim, which would echo request data
  (including `context`) straight back to the client. The handler below
  strips that down to `loc`/`msg`/`type` — and also replaces any `loc`
  segment that isn't a known field name of this schema, since an unknown
  *field name* (e.g. a bogus top-level key) is itself request-controlled
  data (see `_sanitize_loc`).
- OpenAPI is kept aligned with what each endpoint actually returns (see the
  `responses=` on each route below) — it is meant to be usable by Andrey's
  backend as the real integration contract, not a stale default.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from mostransport_ml.serving.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    PredictionBatchRequest,
    PredictionBatchResponse,
    ReadyResponse,
    ValidationErrorResponse,
)
from mostransport_ml.serving.service import (
    InferenceService,
    Predictor,
    PredictorContractError,
    PredictorNotReadyError,
)

logger = logging.getLogger("mostransport_ml.serving")

# The only field names this provisional schema actually has. Any other `loc`
# segment is request-controlled (a client-supplied key, valid or not) and
# must not be echoed back — see `_sanitize_loc`. "body" is the structural
# marker FastAPI/Pydantic use for a body-sourced error; every endpoint here
# takes its input as a single body model, so it's the only one that appears.
_KNOWN_LOC_FIELDS = frozenset(
    {
        "body",
        "prediction_time",
        "horizon_minutes",
        "vehicles",
        "vehicle_id",
        "context",
    }
)
_REDACTED_LOC_SEGMENT = "<field>"


def _sanitize_loc(loc: Sequence[str | int]) -> list[str | int]:
    """Keep list indexes and known field names; redact everything else.

    A validation error's `loc` can contain an arbitrary client-supplied key
    (e.g. an unexpected top-level field, or — in principle — a path pointing
    inside the opaque `context` object). Those are request data too.
    """
    return [
        segment
        if isinstance(segment, int) or segment in _KNOWN_LOC_FIELDS
        else _REDACTED_LOC_SEGMENT
        for segment in loc
    ]


def create_app(predictor: Predictor) -> FastAPI:
    """Build a FastAPI app wired to the given predictor.

    No default: the caller must always supply a `Predictor` explicitly.
    """
    service = InferenceService(predictor)
    app = FastAPI(title="mostransport-ml serving (provisional)")

    @app.exception_handler(RequestValidationError)
    async def _sanitized_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Same 422 status as FastAPI's default, but strips `input`/`ctx`
        (so invalid request data — e.g. a vehicle's `context` — is never
        echoed back) and sanitizes `loc` (so an arbitrary client-supplied
        *field name* isn't echoed back either). Built from the same
        `ValidationErrorResponse` model documented in OpenAPI below, so the
        runtime body and the published schema can't drift apart."""
        errors = [
            ErrorDetail(loc=_sanitize_loc(error["loc"]), msg=error["msg"], type=error["type"])
            for error in exc.errors()
        ]
        body = ValidationErrorResponse(detail=errors)
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Process liveness only. Does not touch the predictor, the
        organizer emulator, or a database."""
        return HealthResponse()

    @app.get(
        "/ready",
        responses={
            200: {"model": ReadyResponse},
            503: {"model": ReadyResponse, "description": "Predictor is not ready"},
        },
    )
    def ready() -> JSONResponse:
        """Predictor/runtime readiness. Returns 503, not 200, both when the
        predictor reports itself not ready AND when the readiness check
        itself raises — a broken `is_ready`/`model_version` must never
        surface as an uncontrolled 500."""
        try:
            is_ready = service.is_ready()
            model_version = service.model_version() if is_ready else None
        except Exception as exc:
            logger.error("readiness_check_failed error_type=%s", type(exc).__name__)
            is_ready = False
            model_version = None

        payload = ReadyResponse(ready=is_ready, model_version=model_version)
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content=payload.model_dump(mode="json"),
        )

    @app.post(
        "/api/v1/predict/batch",
        response_model=PredictionBatchResponse,
        responses={
            422: {
                "model": ValidationErrorResponse,
                "description": "Invalid request (sanitized — no input echo)",
            },
            500: {
                "model": ErrorResponse,
                "description": "Predictor failed or produced invalid output",
            },
            503: {"model": ErrorResponse, "description": "Predictor is not ready"},
        },
    )
    def predict_batch(request: PredictionBatchRequest) -> PredictionBatchResponse:
        batch_size = len(request.vehicles)
        try:
            response = service.predict_batch(request)
        except PredictorNotReadyError:
            logger.warning("prediction_failed event=not_ready batch_size=%d", batch_size)
            raise HTTPException(status_code=503, detail="predictor is not ready") from None
        except (ValidationError, ValueError, PredictorContractError) as exc:
            # Only the exception's TYPE is logged — never str(exc) or a
            # traceback. A predictor's exception message or invalid output
            # could itself be derived from (or embed) request data.
            logger.error(
                "prediction_failed event=invalid_output error_type=%s batch_size=%d",
                type(exc).__name__,
                batch_size,
            )
            raise HTTPException(
                status_code=500, detail="predictor produced invalid output"
            ) from None
        except Exception as exc:
            logger.error(
                "prediction_failed event=unexpected error_type=%s batch_size=%d",
                type(exc).__name__,
                batch_size,
            )
            raise HTTPException(status_code=500, detail="internal error during inference") from None

        logger.info(
            "predict_batch ok (batch_size=%d, model_version=%s)",
            batch_size,
            response.model_version,
        )
        return response

    return app

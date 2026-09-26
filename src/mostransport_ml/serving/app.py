"""FastAPI app factory: Backend → ML Contract v1 (`POST /api/v1/predict`).

Эндпоинт не содержит model/feature-логики: Pydantic-валидация Contract v1 →
`InferenceService` (runtime adapter → CanonicalBatch → Predictor) →
ответ. Predictor передаётся явно: `create_app` никогда не подставляет mock
(см. `mock_app.py`) и не загружает artifact сам (см. `artifact_app.py`).

Конфиденциальность:
- тела запросов (telemetry, контекст) не логируются — только тип события,
  тип исключения, число пакетов и версия модели;
- 422 санитизирован: без `input`/`ctx`, неизвестные сегменты `loc` скрыты;
- 500/503 — фиксированные сообщения без текста исключений и traceback.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mostransport_ml.serving.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    PredictRequestV1,
    PredictResponseV1,
    ReadyResponse,
    ScheduleContextV1,
    TelemetryPacketV1,
    ValidationErrorResponse,
    VehicleContextV1,
)
from mostransport_ml.serving.service import (
    InferenceService,
    InvalidPredictionContextError,
    Predictor,
    PredictorNotReadyError,
    utc_now,
)

logger = logging.getLogger("mostransport_ml.serving")

# Только имена полей Contract v1; любой иной сегмент `loc` задан клиентом и скрывается.
_KNOWN_LOC_FIELDS = frozenset(
    {"body"}
    | set(PredictRequestV1.model_fields)
    | set(VehicleContextV1.model_fields)
    | set(ScheduleContextV1.model_fields)
    | set(TelemetryPacketV1.model_fields)
)
_REDACTED_LOC_SEGMENT = "<field>"


def _sanitize_loc(loc: Sequence[str | int]) -> list[str | int]:
    return [
        segment
        if isinstance(segment, int) or segment in _KNOWN_LOC_FIELDS
        else _REDACTED_LOC_SEGMENT
        for segment in loc
    ]


def _validation_response(errors: list[ErrorDetail]) -> JSONResponse:
    body = ValidationErrorResponse(detail=errors)
    return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


def create_app(predictor: Predictor, *, clock: Callable[[], datetime] = utc_now) -> FastAPI:
    """Собрать приложение для явно переданного predictor'а (и часов для `generated_at`)."""
    service = InferenceService(predictor, clock=clock)
    app = FastAPI(title="mostransport-ml serving (Backend → ML Contract v1)")

    @app.exception_handler(RequestValidationError)
    async def _sanitized_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _validation_response(
            [
                ErrorDetail(loc=_sanitize_loc(error["loc"]), msg=error["msg"], type=error["type"])
                for error in exc.errors()
            ]
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Только process liveness."""
        return HealthResponse()

    @app.get(
        "/ready",
        responses={
            200: {"model": ReadyResponse},
            503: {"model": ReadyResponse, "description": "Predictor/artifact is not ready"},
        },
    )
    def ready() -> JSONResponse:
        try:
            is_ready = service.is_ready()
            model_version = service.model_version() if is_ready else None
            schema_version = service.feature_schema_version() if is_ready else None
        except Exception as exc:
            logger.error("readiness_check_failed error_type=%s", type(exc).__name__)
            is_ready, model_version, schema_version = False, None, None
        payload = ReadyResponse(
            ready=is_ready, model_version=model_version, feature_schema_version=schema_version
        )
        return JSONResponse(
            status_code=200 if is_ready else 503, content=payload.model_dump(mode="json")
        )

    @app.post(
        "/api/v1/predict",
        response_model=PredictResponseV1,
        responses={
            422: {"model": ValidationErrorResponse, "description": "Invalid request (sanitized)"},
            500: {"model": ErrorResponse, "description": "Inference failed"},
            503: {"model": ErrorResponse, "description": "Predictor/artifact is not ready"},
        },
    )
    def predict(request: PredictRequestV1) -> PredictResponseV1 | JSONResponse:
        n_packets = len(request.telemetry)
        try:
            response = service.predict(request)
        except PredictorNotReadyError:
            logger.warning("prediction_failed event=not_ready n_packets=%d", n_packets)
            raise HTTPException(status_code=503, detail="predictor is not ready") from None
        except InvalidPredictionContextError:
            logger.warning("prediction_failed event=invalid_context n_packets=%d", n_packets)
            return _validation_response(
                [ErrorDetail(loc=["body"], msg="invalid prediction context", type="context_error")]
            )
        except Exception as exc:
            logger.error(
                "prediction_failed event=error error_type=%s n_packets=%d",
                type(exc).__name__,
                n_packets,
            )
            raise HTTPException(status_code=500, detail="internal error during inference") from None
        logger.info("predict ok n_packets=%d model_version=%s", n_packets, response.model_version)
        return response

    return app

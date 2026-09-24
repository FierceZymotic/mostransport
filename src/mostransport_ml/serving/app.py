"""FastAPI app factory и эндпоинты для provisional ML serving contract.

ЭТО PROVISIONAL PRE-HACKATHON CONTRACT — см. docs/ML_SERVING_CONTRACT.md.
Эндпоинты не содержат model/feature-логики; они валидируют запрос,
делегируют в `InferenceService` и превращают падения в безопасные
HTTP-ошибки. Скрытого predictor'а по умолчанию нет — `create_app` всегда
требует его явно, поэтому реальный деплой никогда не может тихо
скатиться на mock (см. `mock_app.py` — explicit development app).

Конфиденциальность, по обе стороны провода:

- Входящие payload'ы запроса (особенно `context` каждого vehicle, который
  после старта хакатона будет нести организаторские данные) здесь никогда
  не логируются — только метаданные вроде размера батча, версии модели и
  успеха/неудачи.
- Server-side логи ошибок фиксируют только *тип* исключения
  (`type(exc).__name__`), никогда не `str(exc)` и не traceback — сообщение
  исключения predictor'а само может содержать данные, производные от
  запроса.
- Ответы 422 санитизированы: стандартное тело FastAPI включает
  проблемное значение `input` целиком, что вернуло бы данные запроса
  (включая `context`) обратно клиенту. Обработчик ниже сокращает это до
  `loc`/`msg`/`type` — а также заменяет любой сегмент `loc`, не являющийся
  известным именем поля этой схемы, поскольку неизвестное *имя поля*
  (например, случайный лишний top-level ключ) — тоже данные,
  контролируемые запросом (см. `_sanitize_loc`).
- OpenAPI держится в согласии с тем, что реально возвращает каждый
  эндпоинт (см. `responses=` на каждом роуте ниже) — рассчитан на то, что
  backend Андрея использует его как реальный интеграционный контракт, а
  не устаревшую заглушку.
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

# Единственные имена полей, которые реально есть в этой provisional-схеме.
# Любой другой сегмент `loc` контролируется запросом (ключ, заданный
# клиентом, валидный или нет) и не должен эхом возвращаться — см.
# `_sanitize_loc`. "body" — структурный маркер, которым FastAPI/Pydantic
# помечают ошибку, пришедшую из тела запроса; каждый эндпоинт здесь
# принимает вход как одну body-модель, поэтому он единственный, кто
# встречается.
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
    """Оставить индексы списков и известные имена полей; всё остальное — скрыть.

    `loc` ошибки валидации может содержать произвольный ключ, заданный
    клиентом (например, неожиданное top-level поле, или — в принципе —
    путь внутрь непрозрачного объекта `context`). Это тоже данные запроса.
    """
    return [
        segment
        if isinstance(segment, int) or segment in _KNOWN_LOC_FIELDS
        else _REDACTED_LOC_SEGMENT
        for segment in loc
    ]


def create_app(predictor: Predictor) -> FastAPI:
    """Собрать FastAPI-приложение, подключённое к заданному predictor'у.

    Без значения по умолчанию: вызывающий обязан всегда явно передать
    `Predictor`.
    """
    service = InferenceService(predictor)
    app = FastAPI(title="mostransport-ml serving (provisional)")

    @app.exception_handler(RequestValidationError)
    async def _sanitized_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Тот же статус 422, что и по умолчанию у FastAPI, но без
        `input`/`ctx` (поэтому невалидные данные запроса — например,
        `context` vehicle — никогда не эхо́ятся обратно) и с
        санитизированным `loc` (поэтому произвольное клиентское *имя
        поля* тоже не эхо́ится). Строится из той же модели
        `ValidationErrorResponse`, что документирована в OpenAPI ниже,
        поэтому runtime-тело и опубликованная схема не могут разойтись."""
        errors = [
            ErrorDetail(loc=_sanitize_loc(error["loc"]), msg=error["msg"], type=error["type"])
            for error in exc.errors()
        ]
        body = ValidationErrorResponse(detail=errors)
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Только process liveness. Не трогает predictor, организаторский
        эмулятор или базу данных."""
        return HealthResponse()

    @app.get(
        "/ready",
        responses={
            200: {"model": ReadyResponse},
            503: {"model": ReadyResponse, "description": "Predictor is not ready"},
        },
    )
    def ready() -> JSONResponse:
        """Готовность predictor'а/рантайма. Возвращает 503, а не 200,
        как когда predictor сообщает, что не готов, ТАК И когда сама
        проверка готовности бросает исключение — сломанный
        `is_ready`/`model_version` никогда не должен всплыть как
        неконтролируемый 500."""
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
            # В лог пишется только ТИП исключения — никогда не str(exc) и
            # не traceback. Сообщение исключения predictor'а или его
            # невалидный output сами могут быть производными от данных
            # запроса (или содержать их).
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

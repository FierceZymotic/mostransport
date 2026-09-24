"""External API DTO для provisional ML serving contract.

ЭТО PROVISIONAL PRE-HACKATHON CONTRACT — полное описание в
docs/ML_SERVING_CONTRACT.md. Официального mapping CSV ↔ emulator fields
пока не существует, поэтому `VehicleRequest.context` — намеренно
непрозрачный JSON-объект. Ничто в этом модуле не предполагает реального
transport-поля (никаких lat/lon/speed/route_id и т.п.) — они появятся
только после публикации официального mapping, и тогда `context` будет
заменён или ужесточён в реальную domain-схему.

Этот модуль содержит только DTO — никакой inference-логики, никаких
вызовов predictor'а.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Обычный float, отклоняющий NaN/+Infinity/-Infinity, чтобы нефинитное
# значение никогда не попало в сериализованный ответ (см. §6 контракта).
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

# У horizon_minutes пока нет известной верхней границы или значения по
# умолчанию, но horizon никогда не может быть нулевым, отрицательным или
# нефинитным — это generic-инвариант, а не догадка о реальном horizon
# организаторов.
PositiveFiniteMinutes = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class PredictionStatus(StrEnum):
    """Исход попытки прогноза для одного vehicle."""

    OK = "ok"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


class VehicleRequest(BaseModel):
    """Provisional inference-вход для одного vehicle.

    `context` намеренно непрозрачен: ни `MockPredictor`, ни эта схема
    никогда не интерпретируют его ключи. Существует только чтобы
    разблокировать HTTP-интеграцию до появления официальной схемы.
    """

    model_config = ConfigDict(extra="forbid")

    vehicle_id: str
    context: dict[str, Any]


class PredictionBatchRequest(BaseModel):
    """Конверт запроса для `POST /api/v1/predict/batch` (provisional v0)."""

    model_config = ConfigDict(extra="forbid")

    prediction_time: datetime
    horizon_minutes: PositiveFiniteMinutes | None = None
    vehicles: list[VehicleRequest] = Field(min_length=1)


class VehiclePrediction(BaseModel):
    """Результат прогноза для одного vehicle.

    Инвариант проверяется здесь, а не просто ожидается от вызывающих:
    `status == "ok"` тогда и только тогда, когда присутствует конечный
    `predicted_delay`; любой другой status вообще не несёт delay.
    Нефинитный output predictor'а никогда не должен дойти до клиента (см.
    `serving/service.py`).
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
    """Конверт ответа для `POST /api/v1/predict/batch` (provisional v0).

    Строится вокруг численной задержки (подтверждённая официальная
    метрика — MAE фактической задержки), а не вероятности классификации.
    `target_unit` опционален, поскольку единицы измерения организаторами
    ещё не определены.
    """

    model_config = ConfigDict(extra="forbid")

    prediction_time: datetime
    horizon_minutes: PositiveFiniteMinutes | None = None
    model_version: str
    target_name: str | None = None
    target_unit: str | None = None
    predictions: list[VehiclePrediction]


class HealthResponse(BaseModel):
    """`GET /health` — только process liveness."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class ReadyResponse(BaseModel):
    """`GET /ready` — готовность predictor'а/рантайма."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    model_version: str | None = None


class ErrorDetail(BaseModel):
    """Одна санитизированная запись ошибки валидации.

    Намеренно уже, чем стандартная форма ошибки FastAPI/Pydantic: без
    `input` (проблемного значения) и без `ctx`, поскольку данные запроса
    — включая `context` vehicle — никогда не должны эхом возвращаться
    клиенту. Сегменты `loc`, не являющиеся известным именем поля,
    заменяются — см. sanitizer в `serving/app.py`.
    """

    model_config = ConfigDict(extra="forbid")

    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """Реальная форма тела `422`, которую возвращает validation-error
    handler этого сервиса — используется и для построения самого ответа,
    и для его документирования в OpenAPI, поэтому они не могут разойтись."""

    model_config = ConfigDict(extra="forbid")

    detail: list[ErrorDetail]


class ErrorResponse(BaseModel):
    """Generic безопасное тело ошибки для ответов `500`/`503`: короткое,
    фиксированное, не-чувствительное сообщение — никогда сообщение
    исключения или traceback."""

    model_config = ConfigDict(extra="forbid")

    detail: str

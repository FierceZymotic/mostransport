"""Backend → ML Contract v1: DTO запроса/ответа `POST /api/v1/predict`.

Единственная активная схема инференса (provisional v0 batch/`context`
удалена). Только DTO и контрактные проверки входа — никаких признаков,
вызовов predictor'а или доменной логики Backend'а.

Время: только timezone-aware ISO-8601 строки (предпочтительно `...Z`);
naive-время неоднозначно и отклоняется. Внутри ML время приводится к naive
UTC (так представлены официальные данные) runtime-адаптером.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    model_validator,
)

from mostransport_ml.features.schema import (
    FORBIDDEN_INPUT_COLUMNS,
    HORIZON_MAX_INCLUSIVE_MINUTES,
    HORIZON_MIN_EXCLUSIVE_MINUTES,
)


def _require_iso_string(value: Any) -> Any:
    if not isinstance(value, str):
        raise ValueError("must be an ISO-8601 datetime string")
    return value


ContractDatetime = Annotated[AwareDatetime, BeforeValidator(_require_iso_string)]
NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Latitude = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-90, le=90)]
Longitude = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-180, le=180)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class VehicleContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_id: NonEmptyStr
    tr_id: NonEmptyStr
    route_id: NonEmptyStr


class ScheduleContextV1(BaseModel):
    """Доменные факты Backend'а о целевом плановом событии."""

    model_config = ConfigDict(extra="forbid")

    target_action_id: NonEmptyStr
    target_time_begin: ContractDatetime
    target_lat: Latitude
    target_lon: Longitude
    current_deviation_seconds: FiniteNumber
    manual_fill: StrictBool


class TelemetryPacketV1(BaseModel):
    """Сырой telemetry-пакет (G6CellNav00: timestamp/latitude/longitude/locationValid/speedAvg).

    Все поля обязательны. `speed` (= `speedAvg`) — всегда конечное число: это
    wire-инвариант Contract v1 (Backend всегда присылает числовую скорость).
    `lat`/`lon`/`location_valid` могут быть `null` (нет GPS — реальное
    состояние). Дополнительные сырые поля (например `heading`) допускаются и
    игнорируются моделью, кроме запрещённых factual/target-имён.

    Внутренний канонический контекст (`features/`) остаётся общим и допускает
    отсутствующую скорость исторических данных; ужесточение — только здесь,
    на HTTP-границе.
    """

    model_config = ConfigDict(extra="allow")

    event_time: ContractDatetime
    lat: FiniteNumber | None
    lon: FiniteNumber | None
    location_valid: StrictBool | None
    speed: FiniteNumber

    @model_validator(mode="after")
    def _reject_forbidden_fields(self) -> TelemetryPacketV1:
        if FORBIDDEN_INPUT_COLUMNS & set(self.model_extra or {}):
            raise ValueError("telemetry packet contains forbidden factual/target fields")
        return self


class PredictRequestV1(BaseModel):
    """Одна prediction point `(tr_id, T)` с историей telemetry.

    История: все пакеты в `(T-15m, T]` + последний пакет `<= T` + последний
    strict-valid GPS пакет `<= T` (якоря могут быть старше окна). Пакеты
    `> T` не влияют на прогноз.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyStr
    prediction_time: ContractDatetime
    vehicle_context: VehicleContextV1
    schedule_context: ScheduleContextV1
    telemetry: list[TelemetryPacketV1]

    @model_validator(mode="after")
    def _check_horizon(self) -> PredictRequestV1:
        minutes = (
            self.schedule_context.target_time_begin - self.prediction_time
        ).total_seconds() / 60.0
        if not HORIZON_MIN_EXCLUSIVE_MINUTES < minutes <= HORIZON_MAX_INCLUSIVE_MINUTES:
            raise ValueError(
                "target_time_begin must be within (10, 15] minutes after prediction_time"
            )
        return self


class PredictionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delay_seconds: FiniteFloat
    target_time: AwareDatetime
    reason: None = None


class PredictResponseV1(BaseModel):
    """Успешный ответ. `prediction.target_time = target_time_begin + delay_seconds`."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["success"] = "success"
    prediction: PredictionV1
    generated_at: AwareDatetime
    model_version: str
    feature_schema_version: str


class HealthResponse(BaseModel):
    """`GET /health` — только process liveness."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"


class ReadyResponse(BaseModel):
    """`GET /ready` — готовность predictor'а/artifact."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    model_version: str | None = None
    feature_schema_version: str | None = None


class ErrorDetail(BaseModel):
    """Одна санитизированная ошибка валидации: без `input`/`ctx`, `loc` санитизирован."""

    model_config = ConfigDict(extra="forbid")

    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """Реальная форма тела `422` этого сервиса (она же в OpenAPI)."""

    model_config = ConfigDict(extra="forbid")

    detail: list[ErrorDetail]


class ErrorResponse(BaseModel):
    """Безопасное тело `500`/`503`: фиксированное сообщение, без исключений/traceback."""

    model_config = ConfigDict(extra="forbid")

    detail: str

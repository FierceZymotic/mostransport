"""Inference orchestration и заменяемая граница Predictor.

    FastAPI endpoint -> InferenceService -> Predictor -> prediction results

`InferenceService` не владеет никакой transport-specific логикой и
никакой историей по vehicle — она проверяет readiness, вызывает
predictor и превращает сырой output в строго провалидированный ответ.
`Predictor` — та точка, куда будущая artifact-backed реализация
подключится, не трогая FastAPI-слой или код эндпоинтов (см. `mock.py` —
единственную реализацию, существующую сегодня).

Есть две вещи, которые реализация `Predictor` никогда не должна суметь
сделать, даже случайно: мутировать переданный ей request, либо соврать о
типе своей готовности/model_version. Обе проверяются здесь именно в
рантайме, а не просто описаны в документации — см. `_RequestSnapshot` и
валидацию в `is_ready`/`model_version`.
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
    """Бросается, когда инференс запрошен до готовности predictor'а."""


class PredictorContractError(RuntimeError):
    """Бросается, когда predictor нарушает свой контракт с
    `InferenceService`: неверное число predictions, неверный порядок,
    `vehicle_id`, не совпадающий со входом на этой позиции, мутация
    переданного ему request, либо возвращаемое значение `is_ready`/
    `model_version` неверного типа или формы. Predictor обязан
    самостоятельно соблюдать этот контракт — это fail-fast обнаружение, а
    не тихое исправление (никакой пересортировки, никакого маппинга по
    id, никакого приведения truthy/falsy значения к bool)."""


@dataclass(frozen=True)
class _RequestSnapshot:
    """Неизменяемый снимок полей request'а, от которых зависит serving
    contract, снятый до запуска predictor'а.

    Намеренно узкий: это не deep copy всего request'а (без дублирования
    payload'ов `context`) — только поля, нужные, чтобы обнаружить, что
    predictor подменил конверт из-под `InferenceService`, и чтобы строить
    ответ из значений, которых predictor не мог коснуться.
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
    """То, что `Predictor` возвращает для одного vehicle, до валидации ответа.

    Намеренно не валидируется при конструировании — превратить это в
    строго провалидированный `VehiclePrediction` (конечный
    `predicted_delay` либо None) обязана `InferenceService`, чтобы
    сломанный predictor не мог напрямую отправить в ответ некорректное
    значение.
    """

    vehicle_id: str
    predicted_delay: float | None
    status: PredictionStatus = PredictionStatus.OK


class Predictor(Protocol):
    """Минимальная, заменяемая граница между serving и реализацией модели.

    Stateless по контракту: ни один метод здесь не принимает и не
    возвращает ничего, связанного с предыдущим запросом, а `predict_batch`
    не должен мутировать переданный ему `request`. Этой границы
    достаточно для mock/pre-hackathon serving; точка синхронизации
    Artifact Contract (см. docs/ARCHITECTURE.md §7) может ещё расширить
    её, когда появится реальная семантика метаданных модели, но от
    будущего artifact-backed predictor'а ожидается реализация тех же
    методов без изменения FastAPI-слоя.

    `typing.Protocol` документирует этот контракт, но не может проверить
    его в рантайме — реально проверяет его `InferenceService` (см.
    `is_ready`/`model_version` ниже и проверку мутации в `predict_batch`).
    """

    def is_ready(self) -> bool:
        """Может ли predictor сейчас обслуживать `predict_batch`.
        Обязан вернуть ровно `bool` — не просто truthy/falsy значение."""
        ...

    def model_version(self) -> str:
        """Короткий, стабильный, непустой идентификатор текущей
        загруженной модели. Имеет смысл только когда `is_ready()` вернул
        `True`."""
        ...

    def predict_batch(self, request: PredictionBatchRequest) -> list[RawPrediction]:
        """Вернуть ровно один `RawPrediction` на каждый `request.vehicles`,
        в том же порядке, не мутируя `request`."""
        ...


class InferenceService:
    """Проверяет readiness, вызывает predictor и строит безопасный ответ."""

    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor

    def is_ready(self) -> bool:
        """Readiness, провалидированная в рантайме.

        `typing.Protocol` не может помешать predictor'у вернуть, например,
        строку `"false"` (truthy!) или `1` вместо настоящего `bool` — иначе
        это рассогласовало бы `/ready` и `/predict`. Отклоняем всё, что не
        является ровно `bool`, вместо того чтобы приводить к типу.
        """
        value = self._predictor.is_ready()
        if type(value) is not bool:
            raise PredictorContractError(
                f"predictor.is_ready() must return exactly bool, got "
                f"{type(value).__name__} ({value!r})"
            )
        return value

    def model_version(self) -> str:
        """Model version, провалидированная в рантайме: непустая `str`.

        Вызывать только после того, как predictor сам сообщил о своей
        готовности — от неготового predictor'а не требуется осмысленная
        версия.
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

        # Predictor'у передали `request` по ссылке, и он мог его
        # мутировать. Перепроверяем каждое защищённое поле относительно
        # снимка, снятого *до* вызова — predictor никогда не должен суметь
        # задним числом переопределить, чем был "request".
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

        # Точное 1:1 соответствие identity/order, сверяемое с ИСХОДНЫМИ id
        # (а не с тем, как `request.vehicles` выглядит сейчас) — predictor,
        # переставивший порядок или подменивший id, обязан здесь громко
        # упасть. Это никогда не пересортирует и не перемаппит по id, чтобы
        # "исправить".
        for index, (expected_id, raw) in enumerate(
            zip(snapshot.vehicle_ids, raw_predictions, strict=True)
        ):
            if raw.vehicle_id != expected_id:
                raise PredictorContractError(
                    f"predictor violated the 1:1 identity/order contract at index "
                    f"{index}: expected vehicle_id {expected_id!r}, got "
                    f"{raw.vehicle_id!r}"
                )

        # Конструирование VehiclePrediction проверяет конечность
        # predicted_delay; нефинитный output predictor'а бросит исключение
        # здесь, до того как ответ будет построен или отправлен.
        predictions = [
            VehiclePrediction(
                vehicle_id=raw.vehicle_id,
                predicted_delay=raw.predicted_delay,
                status=raw.status,
            )
            for raw in raw_predictions
        ]

        # Строится из ИСХОДНОГО снимка, никогда из `request` — даже если бы
        # каждая проверка выше как-то прошла, ответ не может нести
        # мутированные prediction_time/horizon_minutes.
        return PredictionBatchResponse(
            prediction_time=snapshot.prediction_time,
            horizon_minutes=snapshot.horizon_minutes,
            model_version=self.model_version(),
            predictions=predictions,
        )

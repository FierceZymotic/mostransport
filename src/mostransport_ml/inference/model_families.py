"""Поддерживаемые model family: сериализация, загрузка и проверка задачи модели.

Сейчас исполняемый family ровно один — `catboost`. Новый family добавляется
отдельной записью в `MODEL_FAMILIES` без изменения bundle/predictor/serving.
Загрузка идёт из уже проверенных по SHA-256 байтов (см. `artifacts.bundle`),
без pickle. CatBoost импортируется лениво, только при реальной работе с моделью.

`model_family == "catboost"` означает: обученная CatBoost-модель скалярной
регрессии, выход которой — одно значение задержки на строку. Проверка
выполняется по самой модели (её сохранённому objective и форме выхода), а
не по Python-классу обёртки: байты классификатора загружаются и через
`CatBoostRegressor`, поэтому класс обёртки ничего не доказывает.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class ModelFamilyError(ValueError):
    """Бинарник модели не удалось сериализовать или загрузить."""


class ModelTaskError(ModelFamilyError):
    """Модель не является скалярной регрессией задержки (классификация, ранжирование и т.п.)."""


class LoadedModel(Protocol):
    def predict(self, data: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelFamily:
    name: str
    model_filename: str
    serialize: Callable[[Any], bytes]
    load: Callable[[bytes], LoadedModel]
    feature_names: Callable[[Any], Sequence[str] | None]


# CatBoost считает эти objective регрессионными, но их прогноз — не значение на
# шкале target: Cox даёт относительный риск (hazard), RMSEWithUncertainty —
# пару (среднее, неопределённость) вместо одного значения.
_CATBOOST_NON_TARGET_SCALE_OBJECTIVES = frozenset({"Cox", "RMSEWithUncertainty"})
_PROBE_ROWS = 2


def validate_catboost_delay_regressor(model: Any) -> None:
    """Проверить, что загруженная CatBoost-модель — скалярная регрессия задержки.

    Правило (по данным самой модели, не по классу обёртки):
    1. модель обучена (`is_fitted()`);
    2. сохранённый в модели objective (`get_all_params()["loss_function"]`)
       — регрессионный по собственной классификации CatBoost
       (`catboost.core.is_regression_objective`) и не классификационный,
       не multi-regression и не survival;
    3. objective не входит в `_CATBOOST_NON_TARGET_SCALE_OBJECTIVES`;
    4. пробный прогноз на двух нулевых строках (по собственным именам
       признаков модели) возвращает ровно одно конечное значение на строку.

    Любой legitimate скалярный регрессионный objective (RMSE, MAE, Quantile,
    Huber, Expectile, MAPE, Poisson, Tweedie, …) проходит; custom-objective,
    для которого нельзя установить семантику, отклоняется.
    """
    from catboost.core import (
        is_classification_objective,
        is_multiregression_objective,
        is_regression_objective,
        is_survivalregression_objective,
    )

    is_fitted = getattr(model, "is_fitted", None)
    if not callable(is_fitted) or not is_fitted():
        raise ModelTaskError("catboost model is not fitted")
    try:
        objective = model.get_all_params().get("loss_function")
    except Exception:
        raise ModelTaskError("catboost model objective is unavailable") from None
    if not isinstance(objective, str) or not objective:
        raise ModelTaskError("catboost model objective is unavailable")
    objective_name = objective.split(":", 1)[0]
    try:
        scalar_regression = bool(is_regression_objective(objective)) and not (
            is_classification_objective(objective)
            or is_multiregression_objective(objective)
            or is_survivalregression_objective(objective)
        )
    except Exception:
        scalar_regression = False
    if not scalar_regression or objective_name in _CATBOOST_NON_TARGET_SCALE_OBJECTIVES:
        raise ModelTaskError(
            f"catboost objective {objective_name!r} is not a scalar delay regression"
        )

    names = _catboost_feature_names(model)
    if not names:
        raise ModelTaskError("catboost model has no feature names")
    probe = pd.DataFrame(np.zeros((_PROBE_ROWS, len(names))), columns=list(names))
    try:
        output = np.asarray(model.predict(probe), dtype=float)
    except Exception:
        raise ModelTaskError("catboost model could not produce a probe prediction") from None
    if output.shape != (_PROBE_ROWS,) or not np.all(np.isfinite(output)):
        raise ModelTaskError("catboost model output is not exactly one finite value per row")


def _catboost_load(blob: bytes) -> LoadedModel:
    from catboost import CatBoostError, CatBoostRegressor

    model = CatBoostRegressor()
    try:
        model.load_model(blob=blob)
    except CatBoostError:
        raise ModelFamilyError("catboost model binary could not be loaded") from None
    validate_catboost_delay_regressor(model)
    return model


def _catboost_serialize(model: Any) -> bytes:
    """Только обученный `CatBoostRegressor`; сериализованные байты проверяются
    тем же валидатором, что и при загрузке (что записано — то и проверено)."""
    from catboost import CatBoostRegressor

    if not isinstance(model, CatBoostRegressor):
        raise ModelTaskError(
            "catboost family exports only fitted CatBoostRegressor models "
            f"(got {type(model).__name__})"
        )
    if not model.is_fitted():
        raise ModelTaskError("catboost model is not fitted")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.cbm"
        model.save_model(str(path), format="cbm")
        blob = path.read_bytes()
    _catboost_load(blob)
    return blob


def _catboost_feature_names(model: Any) -> Sequence[str] | None:
    names = getattr(model, "feature_names_", None)
    return list(names) if names is not None else None


MODEL_FAMILIES: dict[str, ModelFamily] = {
    "catboost": ModelFamily(
        name="catboost",
        model_filename="model.cbm",
        serialize=_catboost_serialize,
        load=_catboost_load,
        feature_names=_catboost_feature_names,
    ),
}

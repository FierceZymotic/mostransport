"""Поддерживаемые model family: сериализация, загрузка и проверка задачи модели.

Исполняемые family: `catboost` (legacy M1) и `hist_gradient_boosting`
(production-кандидат). Новый family добавляется отдельной записью в
`MODEL_FAMILIES` без изменения bundle/predictor/serving. Загрузка идёт из уже
проверенных по SHA-256 байтов (см. `artifacts.bundle`), без pickle/joblib.
Библиотеки моделей импортируются лениво, только при реальной работе с моделью.

`model_family == "catboost"` означает: обученная CatBoost-модель скалярной
регрессии, выход которой — одно значение задержки на строку. Проверка
выполняется по самой модели (её сохранённому objective и форме выхода), а
не по Python-классу обёртки: байты классификатора загружаются и через
`CatBoostRegressor`, поэтому класс обёртки ничего не доказывает.

`model_family == "hist_gradient_boosting"` означает: обученный объект ровно
класса `sklearn.ensemble.HistGradientBoostingRegressor` (не подкласс, не
классификатор), только числовые признаки с именами, одно конечное значение на
строку. Формат — `skops.io` (`model.skops`), без pickle. Из типов, которые skops
не доверяет по умолчанию, разрешён ровно один — `TreePredictor` (см.
`HGB_TRUSTED_SKOPS_TYPES`); любой другой недоверенный тип — отказ до загрузки.
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


# ------------------------------------------------------------------ hist_gradient_boosting

# skops по умолчанию не доверяет `TreePredictor`: его массив `nodes` хранит сырые
# индексы (`left`, `right`, `feature_idx`), по которым sklearn читает без проверки
# границ, — злонамеренный файл может уронить процесс или прочитать чужую память
# при `predict` (выполнения кода skops не допускает). Без этого типа HGB не
# загрузить, поэтому он — единственное исключение, и оно заморожено здесь, а не
# берётся из `get_untrusted_types()`. Компенсация: до любого `predict`
# `_check_hgb_tree` проверяет каждое дерево — дети строго после родителя и в
# пределах массива, `feature_idx` в пределах числа признаков, без категориальных
# сплитов, конечные значения листьев. Имя типа привязано к версии sklearn из
# `uv.lock`; другая сериализация модели будет отклонена, а не молча доверена.
HGB_TRUSTED_SKOPS_TYPES: tuple[str, ...] = (
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
)


def _check_hgb_tree(tree: Any, n_features: int) -> None:
    """Структурные инварианты одного `TreePredictor`, которые делают `predict` безопасным."""
    from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE

    nodes = getattr(tree, "nodes", None)
    if (
        not isinstance(nodes, np.ndarray)
        or nodes.dtype != PREDICTOR_RECORD_DTYPE
        or nodes.ndim != 1
        or nodes.size == 0
    ):
        raise ModelFamilyError("hist_gradient_boosting tree nodes are malformed")
    n_nodes = nodes.size
    is_leaf = nodes["is_leaf"]
    if not np.isin(is_leaf, (0, 1)).all() or nodes["is_categorical"].any():
        raise ModelFamilyError("hist_gradient_boosting tree nodes are malformed")
    split = np.flatnonzero(is_leaf == 0)
    position = split.astype(np.int64)
    left = nodes["left"][split].astype(np.int64)
    right = nodes["right"][split].astype(np.int64)
    feature = nodes["feature_idx"][split].astype(np.int64)
    bad_children = (left <= position) | (right <= position) | (left == right)
    bad_children |= (left >= n_nodes) | (right >= n_nodes)
    if bad_children.any() or ((feature < 0) | (feature >= n_features)).any():
        raise ModelFamilyError("hist_gradient_boosting tree references out-of-range nodes")
    if np.isnan(nodes["num_threshold"][split]).any():
        raise ModelFamilyError("hist_gradient_boosting tree has undefined split thresholds")
    if not np.isin(nodes["missing_go_to_left"][split], (0, 1)).all():
        raise ModelFamilyError("hist_gradient_boosting tree nodes are malformed")
    if not np.isfinite(nodes["value"][is_leaf == 1]).all():
        raise ModelFamilyError("hist_gradient_boosting tree has non-finite leaf values")


def validate_hgb_delay_regressor(model: Any) -> None:
    """Проверить, что объект — обученный HGB скалярной регрессии задержки.

    По самому объекту: ровно класс `HistGradientBoostingRegressor`, обучен, одно
    дерево на итерацию, уникальные строковые имена признаков, без категориальных
    признаков, структурно корректные деревья (до любого `predict`), конечный
    baseline и пробный прогноз (с NaN) — ровно одно конечное значение на строку.
    H0-параметры здесь не проверяются: family — это инфраструктура.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor

    if type(model) is not HistGradientBoostingRegressor:
        raise ModelTaskError(
            "hist_gradient_boosting family accepts only HistGradientBoostingRegressor "
            f"(got {type(model).__name__})"
        )
    predictors = getattr(model, "_predictors", None)
    if not isinstance(predictors, list) or not predictors:
        raise ModelTaskError("hist_gradient_boosting model is not fitted")
    raw_names = getattr(model, "feature_names_in_", None)
    names = [] if raw_names is None else list(raw_names)
    if (
        not names
        or not all(isinstance(n, str) and n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ModelTaskError("hist_gradient_boosting model has no valid feature names")
    if getattr(model, "n_features_in_", None) != len(names):
        raise ModelTaskError("hist_gradient_boosting model feature count is inconsistent")
    if getattr(model, "n_trees_per_iteration_", None) != 1:
        raise ModelTaskError("hist_gradient_boosting model is not a scalar regression")
    if getattr(model, "is_categorical_", None) is not None or getattr(model, "_preprocessor", None):
        raise ModelTaskError("hist_gradient_boosting model must use numeric features only")
    for iteration in predictors:
        if not isinstance(iteration, list) or len(iteration) != 1:
            raise ModelTaskError("hist_gradient_boosting model is not a scalar regression")
        if type(iteration[0]) is not TreePredictor:
            raise ModelFamilyError("hist_gradient_boosting model contains a non-tree predictor")
        _check_hgb_tree(iteration[0], len(names))
    try:
        baseline = np.asarray(model._baseline_prediction, dtype=float)
    except (TypeError, ValueError):
        raise ModelFamilyError("hist_gradient_boosting baseline is malformed") from None
    if baseline.shape != (1, 1) or not np.all(np.isfinite(baseline)):
        raise ModelFamilyError("hist_gradient_boosting baseline is malformed")

    probe = pd.DataFrame(np.zeros((_PROBE_ROWS, len(names))), columns=names)
    probe.iloc[1, :] = np.nan
    try:
        output = np.asarray(model.predict(probe), dtype=float)
    except Exception:
        raise ModelTaskError("hist_gradient_boosting model could not produce a probe") from None
    if output.shape != (_PROBE_ROWS,) or not np.all(np.isfinite(output)):
        raise ModelTaskError(
            "hist_gradient_boosting model output is not exactly one finite value per row"
        )


def _hgb_load(blob: bytes) -> LoadedModel:
    import skops.io as sio

    try:
        untrusted = sio.get_untrusted_types(data=blob)
    except Exception:
        raise ModelFamilyError("hist_gradient_boosting model binary could not be read") from None
    unexpected = sorted(set(untrusted) - set(HGB_TRUSTED_SKOPS_TYPES))
    if unexpected:
        raise ModelFamilyError(
            f"hist_gradient_boosting model binary contains untrusted types: {unexpected[:5]}"
        )
    try:
        model = sio.loads(blob, trusted=list(HGB_TRUSTED_SKOPS_TYPES))
    except Exception:
        raise ModelFamilyError("hist_gradient_boosting model binary could not be loaded") from None
    validate_hgb_delay_regressor(model)
    return model


def _hgb_serialize(model: Any) -> bytes:
    """Только обученный `HistGradientBoostingRegressor`; байты проверяются загрузкой."""
    import skops.io as sio

    validate_hgb_delay_regressor(model)
    blob = sio.dumps(model)
    _hgb_load(blob)
    return blob


def _hgb_feature_names(model: Any) -> Sequence[str] | None:
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else None


MODEL_FAMILIES: dict[str, ModelFamily] = {
    "catboost": ModelFamily(
        name="catboost",
        model_filename="model.cbm",
        serialize=_catboost_serialize,
        load=_catboost_load,
        feature_names=_catboost_feature_names,
    ),
    "hist_gradient_boosting": ModelFamily(
        name="hist_gradient_boosting",
        model_filename="model.skops",
        serialize=_hgb_serialize,
        load=_hgb_load,
        feature_names=_hgb_feature_names,
    ),
}

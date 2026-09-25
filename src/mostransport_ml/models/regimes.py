"""Структурные train-группы, режимы обучения и фиксированная сетка M1.

Группа A (shared-real): `tr_id` из официального real-набора (выводится
вызывающим из planned-only источника — в M1 это плановые поля официального
test schedule).
Группа B (train-only / synthetic-candidate): все остальные строки train.
Принадлежность к B определяется только отсутствием `tr_id` в real-наборе —
никакого семантического диапазона ID здесь нет, и группа B не
называется доказанно synthetic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mostransport_ml.target.formulation import FORMULATIONS, Formulation

TrainRegime = Literal["real_only", "all_train", "balanced_group_weight"]
TRAIN_REGIMES: tuple[TrainRegime, ...] = ("real_only", "all_train", "balanced_group_weight")

GROUP_A = "A_shared_real"
GROUP_B = "B_train_only_candidate"


@dataclass(frozen=True)
class TrainSelection:
    """Какие строки train используются и с какими весами (None = без весов)."""

    regime: TrainRegime
    mask: np.ndarray
    sample_weight: np.ndarray | None
    n_group_a: int
    n_group_b: int
    group_b_weight: float | None
    weight_policy: str


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    train_regime: TrainRegime
    formulation: Formulation


M1_EXPERIMENTS: tuple[ExperimentSpec, ...] = tuple(
    ExperimentSpec(f"m1_{regime}_{formulation}", regime, formulation)
    for regime in TRAIN_REGIMES
    for formulation in FORMULATIONS
)


def assign_train_groups(tr_ids: Any, real_vehicle_ids: Iterable[Any]) -> np.ndarray:
    """Вернуть массив меток групп (`GROUP_A`/`GROUP_B`) для каждой train-строки."""
    real = set(real_vehicle_ids)
    if not real:
        raise ValueError("real_vehicle_ids must not be empty")
    is_real = np.array([tr in real for tr in np.asarray(tr_ids).tolist()], dtype=bool)
    return np.where(is_real, GROUP_A, GROUP_B)


def select_training_rows(groups: Any, regime: TrainRegime) -> TrainSelection:
    """Маска строк и sample weights для режима обучения."""
    groups = np.asarray(groups)
    unknown = set(groups.tolist()) - {GROUP_A, GROUP_B}
    if unknown:
        raise ValueError(f"Unknown group labels: {sorted(unknown)}")
    is_a = groups == GROUP_A
    n_a, n_b = int(is_a.sum()), int((~is_a).sum())

    if regime == "real_only":
        if n_a == 0:
            raise ValueError("real_only regime requires at least one group A row")
        return TrainSelection(regime, is_a, None, n_a, 0, None, "none (group A only)")
    if regime == "all_train":
        return TrainSelection(regime, np.ones_like(is_a), None, n_a, n_b, None, "none (uniform)")
    if regime == "balanced_group_weight":
        if n_a == 0 or n_b == 0:
            raise ValueError("balanced_group_weight requires both groups to be non-empty")
        weight_b = n_a / n_b
        weights = np.where(is_a, 1.0, weight_b)
        return TrainSelection(
            regime,
            np.ones_like(is_a),
            weights,
            n_a,
            n_b,
            weight_b,
            "group A = 1.0, group B = n_group_A / n_group_B",
        )
    raise ValueError(f"Unknown train regime {regime!r}; expected one of {TRAIN_REGIMES}")

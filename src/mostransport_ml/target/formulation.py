"""Формулировки обучающего target: DIRECT и RESIDUAL.

Target всегда берётся из официальной разметки (`target_delay_s`,
секунды, `time_fact_begin − time_begin`); здесь он не строится, а только
преобразуется в обучающую цель модели и обратно в итоговый прогноз.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from mostransport_ml.target.spec import TargetSpec

Formulation = Literal["direct", "residual"]
FORMULATIONS: tuple[Formulation, ...] = ("direct", "residual")

TARGET_SPEC = TargetSpec(
    name="target_delay_s",
    unit="seconds",
    description=(
        "Official delay at the first stop with planned arrival in (T+10min, T+15min]: "
        "time_fact_begin - time_begin; positive = late, negative = early."
    ),
    version="official-v1",
)


def _finite(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _check(formulation: str) -> None:
    if formulation not in FORMULATIONS:
        raise ValueError(f"Unknown formulation {formulation!r}; expected one of {FORMULATIONS}")


def training_target(target_delay_s: Any, cur_dev_s: Any, formulation: Formulation) -> np.ndarray:
    """DIRECT: y = target; RESIDUAL: y = target − cur_dev_s."""
    _check(formulation)
    target = _finite(target_delay_s, "target_delay_s")
    if formulation == "direct":
        return target
    cur_dev = _finite(cur_dev_s, "cur_dev_s")
    if cur_dev.shape != target.shape:
        raise ValueError("target_delay_s and cur_dev_s must have the same shape")
    return target - cur_dev


def final_prediction(model_output: Any, cur_dev_s: Any, formulation: Formulation) -> np.ndarray:
    """DIRECT: ŷ = output; RESIDUAL: ŷ = cur_dev_s + output. Без clipping/калибровки."""
    _check(formulation)
    output = _finite(model_output, "model_output")
    if formulation == "direct":
        return output
    cur_dev = _finite(cur_dev_s, "cur_dev_s")
    if cur_dev.shape != output.shape:
        raise ValueError("model_output and cur_dev_s must have the same shape")
    return cur_dev + output

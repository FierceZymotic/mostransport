"""Стабильное метаданное представление prediction target.

Организаторы подтвердили primary-метрику (MAE), но пока не точное
определение delay, единицы измерения или horizon. `TargetSpec` даёт
экспериментам и логам стабильное место, чтобы *описать* target, когда он
появится — намеренно не строит его сам.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TargetSpec:
    """Метаданные, описывающие prediction target.

    Ничего не захардкожено: name, unit и horizon задаёт вызывающий, как
    только станет известен task specification организатора.
    """

    name: str
    unit: str
    description: str
    version: str
    horizon_minutes: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
            "version": self.version,
            "horizon_minutes": self.horizon_minutes,
        }

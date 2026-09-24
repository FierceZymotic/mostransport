"""Stable metadata representation for a prediction target.

The organizers have confirmed the primary metric (MAE) but not yet the exact
delay definition, units, or horizon. `TargetSpec` gives experiments and logs
a stable place to *describe* a target once one exists — it intentionally
does not construct one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TargetSpec:
    """Metadata describing a prediction target.

    Nothing is hardcoded: name, unit, and horizon are all supplied by the
    caller once the organizer's task specification is known.
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

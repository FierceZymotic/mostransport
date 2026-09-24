"""Минимальный, стабильный контракт метаданных артефакта.

Это НЕ настоящий загрузчик Artifact Bundle — реальный формат модельного
файла (CatBoost или другой) неизвестен, пока не появится реальная модель.
`ArtifactMetadata` даёт экспериментам и будущему artifact-backed
predictor'у только стабильный, строго-JSON-сериализуемый способ описать
версию обученной модели. Никогда не хранит сырые данные; вызывающие не
должны класть в любое из его полей абсолютный, user-specific путь.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactMetadata:
    """Метаданные, описывающие один артефакт обученной модели.

    `model_version`, `created_at`, `model_type` и `target_name` обязательны;
    всё остальное опционально и заполняется по мере появления. Если
    `validation_mae` задан, он обязан быть конечным и неотрицательным.
    """

    model_version: str
    created_at: str
    model_type: str
    target_name: str
    target_unit: str | None = None
    target_version: str | None = None
    feature_schema_version: str | None = None
    validation_mae: float | None = None

    def __post_init__(self) -> None:
        if self.validation_mae is None:
            return
        if not math.isfinite(self.validation_mae):
            raise ValueError(f"validation_mae must be finite, got {self.validation_mae!r}")
        if self.validation_mae < 0:
            raise ValueError(f"validation_mae must be >= 0, got {self.validation_mae!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        return cls(
            model_version=data["model_version"],
            created_at=data["created_at"],
            model_type=data["model_type"],
            target_name=data["target_name"],
            target_unit=data.get("target_unit"),
            target_version=data.get("target_version"),
            feature_schema_version=data.get("feature_schema_version"),
            validation_mae=data.get("validation_mae"),
        )

    @classmethod
    def from_json(cls, text: str) -> ArtifactMetadata:
        return cls.from_dict(json.loads(text))

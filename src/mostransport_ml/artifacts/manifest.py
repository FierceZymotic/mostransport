"""Artifact Manifest v1: модельно-независимые метаданные будущего артефакта модели.

Только инфраструктура: manifest описывает обученную модель, но не
загружает, не выбирает и не применяет её. Модуль намеренно зависит только
от stdlib — никаких CatBoost/PyTorch, `serving/`, `features/` или Backend.

Идентичность manifest = его каноническое JSON-представление:
`sort_keys`, компактные разделители, UTF-8, без NaN/Infinity, `-0.0` → `0.0`,
`created_at` в UTC вида `YYYY-MM-DDTHH:MM:SS.ffffffZ`. Равенство manifest'ов,
hash и SHA-256 fingerprint определяются этими байтами, поэтому семантически
одинаковый manifest всегда даёт одинаковые байты и fingerprint. Путь к файлу
manifest'а в идентичность не входит.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

ARTIFACT_SCHEMA_VERSION = "artifact-manifest-v1"
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS: frozenset[str] = frozenset({ARTIFACT_SCHEMA_VERSION})

# Manifest — это метаданные; большой payload означает встроенные данные.
MAX_CANONICAL_BYTES = 64 * 1024

_CREATED_AT_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.\-]*$")
_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|~|\\\\|[A-Za-z]:[\\/])")

_MANIFEST_FIELDS = (
    "artifact_schema_version",
    "model_version",
    "model_family",
    "feature_schema_version",
    "target_formulation",
    "train_regime",
    "model_params",
    "training_data_provenance",
    "code_provenance",
    "evaluation",
    "created_at",
)
_EVALUATION_FIELDS = ("split", "metric", "value", "n_rows")


class ManifestValidationError(ValueError):
    """Manifest некорректен: отсутствуют/лишние поля, неверные типы, NaN/Infinity и т.п."""


class ArtifactCompatibilityError(ValueError):
    """Manifest корректен, но несовместим с ожиданиями потребителя."""


class UnsupportedArtifactSchemaError(ArtifactCompatibilityError):
    """Версия схемы manifest'а не поддерживается этим кодом."""


# --------------------------------------------------------------------------- JSON values


def _check_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ManifestValidationError(
            f"{path}: must be a non-empty string without outer whitespace"
        )
    if _ABSOLUTE_PATH_RE.match(value):
        raise ManifestValidationError(f"{path}: absolute/machine-specific paths are not allowed")
    return value


def _check_identifier(value: Any, path: str) -> str:
    _check_string(value, path)
    if not _IDENTIFIER_RE.match(value):
        raise ManifestValidationError(
            f"{path}: must be a lowercase identifier ([a-z0-9][a-z0-9_.-]*)"
        )
    return value


def _freeze_json(value: Any, path: str) -> Any:
    """Проверить JSON-значение и вернуть его неизменяемую нормализованную форму.

    Допустимо: None, bool, int, конечный float, str, list/tuple, dict со str-ключами.
    Объект → MappingProxyType, массив → tuple, float → float (`-0.0` → `0.0`).
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestValidationError(f"{path}: must be finite (no NaN/Infinity)")
        return float(value) + 0.0  # -0.0 + 0.0 == 0.0
    if isinstance(value, str):
        if _ABSOLUTE_PATH_RE.match(value):
            raise ManifestValidationError(
                f"{path}: absolute/machine-specific paths are not allowed"
            )
        return value
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item, f"{path}[{i}]") for i, item in enumerate(value))
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ManifestValidationError(f"{path}: object keys must be non-empty strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    raise ManifestValidationError(f"{path}: {type(value).__name__} is not a JSON value")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _check_object(value: Any, path: str) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{path}: must be a JSON object")
    return _freeze_json(value, path)


def _check_created_at(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ManifestValidationError("created_at: must be an ISO-8601 datetime") from exc
    if not isinstance(value, datetime):
        raise ManifestValidationError("created_at: must be a timezone-aware datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ManifestValidationError("created_at: must be timezone-aware (UTC)")
    return value.astimezone(UTC)


# --------------------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class EvaluationSummary:
    """Одна фиксированная оценка артефакта: на каком split, какой метрикой, сколько строк.

    Только запись факта: никаких порогов «достаточно хорошо».
    """

    split: str
    metric: str
    value: float
    n_rows: int

    def __post_init__(self) -> None:
        _check_identifier(self.split, "evaluation.split")
        _check_identifier(self.metric, "evaluation.metric")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise ManifestValidationError("evaluation.value: must be a number")
        object.__setattr__(self, "value", _freeze_json(float(self.value), "evaluation.value"))
        if isinstance(self.n_rows, bool) or not isinstance(self.n_rows, int) or self.n_rows < 1:
            raise ManifestValidationError("evaluation.n_rows: must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "metric": self.metric,
            "value": self.value,
            "n_rows": self.n_rows,
        }

    @classmethod
    def from_dict(cls, data: Any) -> EvaluationSummary:
        if not isinstance(data, Mapping):
            raise ManifestValidationError("evaluation: must be a JSON object")
        _check_exact_keys(data, _EVALUATION_FIELDS, "evaluation")
        return cls(**{key: data[key] for key in _EVALUATION_FIELDS})


def _check_exact_keys(data: Mapping[str, Any], expected: tuple[str, ...], path: str) -> None:
    missing = [k for k in expected if k not in data]
    unknown = sorted(set(data) - set(expected))
    if missing or unknown:
        raise ManifestValidationError(f"{path}: missing fields {missing}, unknown fields {unknown}")


@dataclass(frozen=True, eq=False)
class ArtifactManifest:
    """Artifact Manifest v1. Все поля обязательны; `created_at` задаётся явно.

    `target_formulation`, `train_regime`, `model_family` — просто
    идентификаторы-метаданные (например `residual`, `all_train`, `catboost`),
    а не логика выбора модели. `model_params`, `training_data_provenance`,
    `code_provenance` — JSON-объекты (хранятся в неизменяемой форме).
    """

    artifact_schema_version: str
    model_version: str
    model_family: str
    feature_schema_version: str
    target_formulation: str
    train_regime: str
    model_params: Mapping[str, Any]
    training_data_provenance: Mapping[str, Any]
    code_provenance: Mapping[str, Any]
    evaluation: EvaluationSummary
    created_at: datetime

    def __post_init__(self) -> None:
        _check_string(self.artifact_schema_version, "artifact_schema_version")
        if self.artifact_schema_version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
            raise UnsupportedArtifactSchemaError(
                f"artifact_schema_version {self.artifact_schema_version!r} is not supported; "
                f"supported: {sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)}"
            )
        _check_string(self.model_version, "model_version")
        _check_string(self.feature_schema_version, "feature_schema_version")
        for name in ("model_family", "target_formulation", "train_regime"):
            _check_identifier(getattr(self, name), name)
        for name in ("model_params", "training_data_provenance", "code_provenance"):
            object.__setattr__(self, name, _check_object(getattr(self, name), name))
        if not isinstance(self.evaluation, EvaluationSummary):
            raise ManifestValidationError("evaluation: must be an EvaluationSummary")
        object.__setattr__(self, "created_at", _check_created_at(self.created_at))
        size = len(self.canonical_bytes())
        if size > MAX_CANONICAL_BYTES:
            raise ManifestValidationError(
                f"manifest is {size} bytes; metadata must not exceed {MAX_CANONICAL_BYTES} bytes "
                "(dataset rows/predictions must not be embedded)"
            )

    # -------------------------------------------------------------- representation

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "model_version": self.model_version,
            "model_family": self.model_family,
            "feature_schema_version": self.feature_schema_version,
            "target_formulation": self.target_formulation,
            "train_regime": self.train_regime,
            "model_params": _thaw_json(self.model_params),
            "training_data_provenance": _thaw_json(self.training_data_provenance),
            "code_provenance": _thaw_json(self.code_provenance),
            "evaluation": self.evaluation.to_dict(),
            "created_at": self.created_at.strftime(_CREATED_AT_FORMAT),
        }

    def canonical_bytes(self) -> bytes:
        text = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return text.encode("utf-8")

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    def fingerprint(self) -> str:
        """SHA-256 канонических байтов (hex)."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArtifactManifest):
            return NotImplemented
        return self.canonical_bytes() == other.canonical_bytes()

    def __hash__(self) -> int:
        return hash(self.canonical_bytes())

    # -------------------------------------------------------------- parsing

    @classmethod
    def from_dict(cls, data: Any) -> ArtifactManifest:
        """Строгий разбор: сначала версия схемы, затем ровно поля v1."""
        if not isinstance(data, Mapping):
            raise ManifestValidationError("manifest: must be a JSON object")
        version = data.get("artifact_schema_version")
        if not isinstance(version, str) or version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
            raise UnsupportedArtifactSchemaError(
                f"artifact_schema_version {version!r} is not supported; "
                f"supported: {sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)}"
            )
        _check_exact_keys(data, _MANIFEST_FIELDS, "manifest")
        fields = {key: data[key] for key in _MANIFEST_FIELDS}
        fields["evaluation"] = EvaluationSummary.from_dict(data["evaluation"])
        return cls(**fields)

    @classmethod
    def from_json(cls, text: str | bytes) -> ArtifactManifest:
        def reject_constant(name: str) -> Any:
            raise ManifestValidationError(
                f"manifest: non-finite JSON constant {name} is not allowed"
            )

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            # Вызывается для каждого JSON-объекта на любом уровне вложенности.
            obj: dict[str, Any] = {}
            for key, value in pairs:
                if key in obj:
                    raise ManifestValidationError(f"manifest: duplicate JSON key {key!r}")
                obj[key] = value
            return obj

        try:
            data = json.loads(
                text, parse_constant=reject_constant, object_pairs_hook=reject_duplicate_keys
            )
        except json.JSONDecodeError as exc:
            raise ManifestValidationError("manifest: invalid JSON") from exc
        return cls.from_dict(data)


# --------------------------------------------------------------------------- file + compatibility


def write_manifest(manifest: ArtifactManifest, path: str | Path) -> None:
    """Записать ровно канонические байты (путь в идентичность не входит)."""
    Path(path).write_bytes(manifest.canonical_bytes())


def read_manifest(path: str | Path) -> ArtifactManifest:
    return ArtifactManifest.from_json(Path(path).read_bytes())


def _accepted_feature_schemas(
    expected: str | None, supported: Iterable[str] | None
) -> tuple[frozenset[str], str]:
    """Ровно один режим: точная схема (`expected`) или набор поддерживаемых (`supported`)."""
    if (expected is None) == (supported is None):
        raise ValueError(
            "pass exactly one of expected_feature_schema_version or "
            "supported_feature_schema_versions"
        )
    if expected is not None:
        if not isinstance(expected, str) or not expected:
            raise ValueError("expected_feature_schema_version must be a non-empty string")
        return frozenset({expected}), repr(expected)
    if isinstance(supported, str | bytes):
        raise ValueError("supported_feature_schema_versions must be a collection, not a string")
    accepted = frozenset(supported)
    if not accepted or not all(isinstance(v, str) and v for v in accepted):
        raise ValueError("supported_feature_schema_versions must be non-empty strings")
    return accepted, f"one of {sorted(accepted)}"


def validate_compatibility(
    manifest: ArtifactManifest | Mapping[str, Any],
    *,
    expected_feature_schema_version: str | None = None,
    supported_feature_schema_versions: Iterable[str] | None = None,
) -> ArtifactManifest:
    """Проверить, что manifest поддерживаемой схемы и собран под допустимую схему признаков.

    Вызывающий задаёт ровно один режим: `expected_feature_schema_version`
    (точная схема) или `supported_feature_schema_versions` (любая из
    поддерживаемых им схем). Проверяется: версия схемы manifest'а, корректность
    и конечность всех метаданных (при разборе/конструировании), принадлежность
    `feature_schema_version` допустимому набору. Не проверяется и не решается:
    загрузка модели, inference, Backend-контракт, выбор модели/режима,
    «достаточно ли хорош» MAE.
    """
    accepted, described = _accepted_feature_schemas(
        expected_feature_schema_version, supported_feature_schema_versions
    )
    if not isinstance(manifest, ArtifactManifest):
        manifest = ArtifactManifest.from_dict(manifest)
    if manifest.artifact_schema_version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise UnsupportedArtifactSchemaError(
            f"artifact_schema_version {manifest.artifact_schema_version!r} is not supported"
        )
    if manifest.feature_schema_version not in accepted:
        raise ArtifactCompatibilityError(
            f"feature schema mismatch: artifact built for {manifest.feature_schema_version!r}, "
            f"expected {described}"
        )
    return manifest

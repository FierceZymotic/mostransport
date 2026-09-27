"""Artifact Bundle v1: криптографическая привязка manifest'а к конкретному бинарнику модели.

Каталог bundle содержит ровно три файла:

    manifest.json   — канонические байты ArtifactManifest v1
    <model file>    — бинарник модели (для catboost: model.cbm)
    bundle.json     — канонический JSON с SHA-256 обоих файлов

Идентичность bundle (SHA-256 канонического `bundle.json`) отделена от
идентичности manifest'а: manifest v1 не хранит хеш модели.

Загрузка проверяет всё до того, как кто-либо сможет загрузить модель:
строгий `bundle.json` (без дубликатов/лишних ключей), безопасные имена
файлов (без путей, traversal, symlink), точный состав каталога, оба хеша,
строгий разбор manifest'а, совместимость `feature_schema_version` и
поддержку `model_family` вызывающим. Наружу отдаются ровно те байты модели,
чьи хеши проверены, — повторного чтения файла между проверкой и загрузкой
нет. Модуль зависит только от stdlib и не знает ни одного model family.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mostransport_ml.artifacts.manifest import ArtifactManifest, validate_compatibility

BUNDLE_SCHEMA_VERSION = "artifact-bundle-v1"
BUNDLE_FILENAME = "bundle.json"
MANIFEST_FILENAME = "manifest.json"

_BUNDLE_FIELDS = (
    "bundle_schema_version",
    "manifest_filename",
    "manifest_sha256",
    "model_filename",
    "model_sha256",
)
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BUNDLE_JSON_BYTES = 4096


class ArtifactBundleError(ValueError):
    """Bundle отсутствует, повреждён, небезопасен или не соответствует ожиданиям.

    Сообщения содержат только относительные имена файлов bundle, но не
    абсолютные пути.
    """


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_filename(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_FILENAME_RE.match(value) or value in {".", ".."}:
        raise ArtifactBundleError(f"{field}: must be a plain safe filename (no paths)")
    return value


@dataclass(frozen=True)
class BundleDescriptor:
    """Содержимое `bundle.json`."""

    bundle_schema_version: str
    manifest_filename: str
    manifest_sha256: str
    model_filename: str
    model_sha256: str

    def __post_init__(self) -> None:
        if self.bundle_schema_version != BUNDLE_SCHEMA_VERSION:
            raise ArtifactBundleError(
                f"unsupported bundle_schema_version; supported: {BUNDLE_SCHEMA_VERSION}"
            )
        if self.manifest_filename != MANIFEST_FILENAME:
            raise ArtifactBundleError(f"manifest_filename must be exactly {MANIFEST_FILENAME!r}")
        _check_filename(self.model_filename, "model_filename")
        if self.model_filename in {BUNDLE_FILENAME, MANIFEST_FILENAME}:
            raise ArtifactBundleError("model_filename collides with a reserved bundle filename")
        for field in ("manifest_sha256", "model_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _SHA256_RE.match(value):
                raise ArtifactBundleError(f"{field}: must be a lowercase hex SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in _BUNDLE_FIELDS}

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")

    def fingerprint(self) -> str:
        """Идентичность bundle: SHA-256 канонического `bundle.json`."""
        return sha256_bytes(self.canonical_bytes())

    @classmethod
    def from_json(cls, data: bytes) -> BundleDescriptor:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            obj: dict[str, Any] = {}
            for key, value in pairs:
                if key in obj:
                    raise ArtifactBundleError(f"{BUNDLE_FILENAME}: duplicate JSON key {key!r}")
                obj[key] = value
            return obj

        def reject_constant(name: str) -> Any:
            raise ArtifactBundleError(f"{BUNDLE_FILENAME}: non-finite JSON constant {name}")

        if len(data) > _MAX_BUNDLE_JSON_BYTES:
            raise ArtifactBundleError(f"{BUNDLE_FILENAME}: too large")
        try:
            parsed = json.loads(
                data, object_pairs_hook=reject_duplicates, parse_constant=reject_constant
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ArtifactBundleError(f"{BUNDLE_FILENAME}: invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ArtifactBundleError(f"{BUNDLE_FILENAME}: must be a JSON object")
        missing = [k for k in _BUNDLE_FIELDS if k not in parsed]
        unknown = sorted(set(parsed) - set(_BUNDLE_FIELDS))
        if missing or unknown:
            raise ArtifactBundleError(
                f"{BUNDLE_FILENAME}: missing fields {missing}, unknown fields {unknown}"
            )
        return cls(**{k: parsed[k] for k in _BUNDLE_FIELDS})


@dataclass(frozen=True)
class VerifiedBundle:
    """Результат успешной проверки: manifest, дескриптор и проверенные байты модели."""

    manifest: ArtifactManifest
    descriptor: BundleDescriptor
    model_bytes: bytes

    @property
    def bundle_sha256(self) -> str:
        return self.descriptor.fingerprint()


def write_bundle(
    directory: str | Path, manifest: ArtifactManifest, model_bytes: bytes, model_filename: str
) -> BundleDescriptor:
    """Записать bundle в новый (несуществующий или пустой) каталог."""
    if not isinstance(manifest, ArtifactManifest):
        raise TypeError("manifest must be an ArtifactManifest")
    if not isinstance(model_bytes, bytes) or not model_bytes:
        raise ArtifactBundleError("model_bytes must be non-empty bytes")
    manifest_bytes = manifest.canonical_bytes()
    descriptor = BundleDescriptor(
        bundle_schema_version=BUNDLE_SCHEMA_VERSION,
        manifest_filename=MANIFEST_FILENAME,
        manifest_sha256=sha256_bytes(manifest_bytes),
        model_filename=model_filename,
        model_sha256=sha256_bytes(model_bytes),
    )
    root = Path(directory)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ArtifactBundleError("bundle directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    (root / model_filename).write_bytes(model_bytes)
    (root / MANIFEST_FILENAME).write_bytes(manifest_bytes)
    (root / BUNDLE_FILENAME).write_bytes(descriptor.canonical_bytes())
    return descriptor


def _read_regular_file(root: Path, name: str) -> bytes:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ArtifactBundleError(f"{name}: missing or not a regular file")
    return path.read_bytes()


def load_bundle(
    directory: str | Path,
    *,
    supported_model_families: Iterable[str],
    expected_feature_schema_version: str | None = None,
    supported_feature_schema_versions: Iterable[str] | None = None,
) -> VerifiedBundle:
    """Проверить bundle целиком и вернуть проверенные manifest и байты модели.

    Порядок: bundle.json → состав каталога → хеш manifest → хеш модели →
    строгий manifest → совместимость схемы признаков → поддержка model_family.
    Схема признаков задаётся ровно одним режимом (см. `validate_compatibility`):
    точная `expected_feature_schema_version` или набор
    `supported_feature_schema_versions`. Модель здесь не загружается.
    """
    supported = frozenset(supported_model_families)
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactBundleError("bundle directory is missing or not a directory")

    descriptor = BundleDescriptor.from_json(_read_regular_file(root, BUNDLE_FILENAME))

    expected = {BUNDLE_FILENAME, descriptor.manifest_filename, descriptor.model_filename}
    present = {entry.name for entry in root.iterdir()}
    if present != expected:
        raise ArtifactBundleError(
            f"bundle must contain exactly {sorted(expected)}; "
            f"missing {sorted(expected - present)}, unexpected entries: {len(present - expected)}"
        )

    manifest_bytes = _read_regular_file(root, descriptor.manifest_filename)
    if sha256_bytes(manifest_bytes) != descriptor.manifest_sha256:
        raise ArtifactBundleError(f"{descriptor.manifest_filename}: SHA-256 mismatch")
    model_bytes = _read_regular_file(root, descriptor.model_filename)
    if sha256_bytes(model_bytes) != descriptor.model_sha256:
        raise ArtifactBundleError(f"{descriptor.model_filename}: SHA-256 mismatch")

    manifest = ArtifactManifest.from_json(manifest_bytes)
    validate_compatibility(
        manifest,
        expected_feature_schema_version=expected_feature_schema_version,
        supported_feature_schema_versions=supported_feature_schema_versions,
    )
    if manifest.model_family not in supported:
        raise ArtifactBundleError(
            f"model_family {manifest.model_family!r} is not supported; "
            f"supported: {sorted(supported)}"
        )
    return VerifiedBundle(manifest=manifest, descriptor=descriptor, model_bytes=model_bytes)

"""Artifact Manifest v1: валидация, каноническая сериализация, fingerprint, совместимость.

Только синтетические метаданные; ни одна модель не обучается и не загружается.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from mostransport_ml.artifacts import manifest as manifest_module
from mostransport_ml.artifacts.manifest import (
    ARTIFACT_SCHEMA_VERSION,
    MAX_CANONICAL_BYTES,
    ArtifactCompatibilityError,
    ArtifactManifest,
    EvaluationSummary,
    ManifestValidationError,
    UnsupportedArtifactSchemaError,
    read_manifest,
    validate_compatibility,
    write_manifest,
)

CREATED_AT = datetime(2026, 9, 25, 12, 30, 0, 123456, tzinfo=UTC)


def make_fields(**overrides):
    fields = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": "synthetic-model-0001",
        "model_family": "catboost",
        "feature_schema_version": "tabular-v1",
        "target_formulation": "residual",
        "train_regime": "all_train",
        "model_params": {
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "MAE",
            "verbose": False,
        },
        "training_data_provenance": {
            "data_version": "official-planned-sha256:0000000000000000",
            "splits": ["train"],
            "n_rows": 4434,
        },
        "code_provenance": {"git_commit": "e59c48a", "git_tag": "m1-tabular-baseline-v1"},
        "evaluation": EvaluationSummary(split="labels_test", metric="mae", value=57.19, n_rows=353),
        "created_at": CREATED_AT,
    }
    fields.update(overrides)
    return fields


def make(**overrides) -> ArtifactManifest:
    return ArtifactManifest(**make_fields(**overrides))


# ------------------------------------------------------------------ construction + roundtrip


def test_valid_manifest_construction():
    manifest = make()
    assert manifest.artifact_schema_version == "artifact-manifest-v1"
    assert manifest.created_at == CREATED_AT
    assert manifest.model_params["depth"] == 6
    assert manifest.training_data_provenance["splits"] == ("train",)
    assert manifest.evaluation.value == 57.19


def test_json_roundtrip_is_exact():
    manifest = make()
    restored = ArtifactManifest.from_json(manifest.canonical_json())
    assert restored == manifest
    assert restored.canonical_bytes() == manifest.canonical_bytes()
    assert restored.to_dict() == manifest.to_dict()
    assert ArtifactManifest.from_dict(manifest.to_dict()) == manifest


def test_file_roundtrip_writes_exact_canonical_bytes(tmp_path):
    manifest = make()
    first, second = tmp_path / "a" / "manifest.json", tmp_path / "b.json"
    first.parent.mkdir()
    write_manifest(manifest, first)
    write_manifest(manifest, second)
    assert first.read_bytes() == manifest.canonical_bytes() == second.read_bytes()
    assert read_manifest(first) == manifest
    assert str(tmp_path) not in manifest.canonical_json()


def test_canonical_json_is_strict_json_and_expected_shape():
    parsed = json.loads(make().canonical_json())
    assert list(parsed) == sorted(parsed)
    assert parsed["created_at"] == "2026-09-25T12:30:00.123456Z"
    assert parsed["evaluation"] == {
        "metric": "mae",
        "n_rows": 353,
        "split": "labels_test",
        "value": 57.19,
    }
    canonical = make().canonical_bytes()
    assert b", " not in canonical and b": " not in canonical and b"\n" not in canonical


# ------------------------------------------------------------------ determinism


def test_canonical_serialization_is_deterministic():
    a = make()
    b = make(
        model_params={"verbose": False, "loss_function": "MAE", "learning_rate": 0.05, "depth": 6}
    )
    assert a.canonical_bytes() == b.canonical_bytes()
    assert a.fingerprint() == b.fingerprint()
    assert a == b and hash(a) == hash(b)


def test_fingerprint_is_sha256_of_canonical_bytes_and_stable():
    manifest = make()
    assert manifest.fingerprint() == hashlib.sha256(manifest.canonical_bytes()).hexdigest()
    assert manifest.fingerprint() == make().fingerprint()
    assert len(manifest.fingerprint()) == 64


def test_field_order_does_not_affect_deserialization():
    data = make().to_dict()
    reordered = {key: data[key] for key in reversed(list(data))}
    reordered["evaluation"] = dict(reversed(list(data["evaluation"].items())))
    reordered["model_params"] = dict(reversed(list(data["model_params"].items())))
    text = json.dumps(reordered, indent=4)
    restored = ArtifactManifest.from_json(text)
    assert restored == make()
    assert restored.fingerprint() == make().fingerprint()


def test_equivalent_created_at_instants_share_identity():
    moscow = CREATED_AT.astimezone(timezone(timedelta(hours=3)))
    assert make(created_at=moscow) == make()
    assert make(created_at="2026-09-25T15:30:00.123456+03:00").fingerprint() == make().fingerprint()


def test_negative_zero_and_numpy_float_normalize():
    assert make(model_params={"x": -0.0}) == make(model_params={"x": 0.0})
    assert make(model_params={"x": np.float64(0.5)}) == make(model_params={"x": 0.5})


@pytest.mark.parametrize(
    "mutation",
    [
        {"model_version": "synthetic-model-0002"},
        {"model_family": "pytorch"},
        {"feature_schema_version": "tabular-v2"},
        {"target_formulation": "direct"},
        {"train_regime": "real_only"},
        {
            "model_params": {
                "depth": 7,
                "learning_rate": 0.05,
                "loss_function": "MAE",
                "verbose": False,
            }
        },
        {
            "model_params": {
                "depth": 6.0,
                "learning_rate": 0.05,
                "loss_function": "MAE",
                "verbose": False,
            }
        },
        {"training_data_provenance": {"data_version": "official-planned-sha256:1111111111111111"}},
        {"code_provenance": {"git_commit": "facc2dc"}},
        {
            "evaluation": EvaluationSummary(
                split="labels_test", metric="mae", value=57.2, n_rows=353
            )
        },
        {
            "evaluation": EvaluationSummary(
                split="labels_test", metric="mae", value=57.19, n_rows=352
            )
        },
        {"evaluation": EvaluationSummary(split="holdout", metric="mae", value=57.19, n_rows=353)},
        {
            "evaluation": EvaluationSummary(
                split="labels_test", metric="rmse", value=57.19, n_rows=353
            )
        },
        {"created_at": CREATED_AT + timedelta(microseconds=1)},
    ],
)
def test_fingerprint_changes_on_meaningful_mutation(mutation):
    mutated = make(**mutation)
    assert mutated.fingerprint() != make().fingerprint()
    assert mutated != make()


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_rejected(bad):
    with pytest.raises(ManifestValidationError):
        EvaluationSummary(split="labels_test", metric="mae", value=bad, n_rows=353)
    with pytest.raises(ManifestValidationError, match="finite"):
        make(model_params={"lr": bad})
    with pytest.raises(ManifestValidationError, match="finite"):
        make(training_data_provenance={"nested": [1.0, {"x": bad}]})


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_constants_rejected_on_parse(constant):
    text = make().canonical_json().replace('"value":57.19', f'"value":{constant}')
    with pytest.raises(ManifestValidationError, match="non-finite"):
        ArtifactManifest.from_json(text)


@pytest.mark.parametrize(
    "params",
    [
        {"callback": object()},
        {"fn": print},
        {"array": np.array([1.0, 2.0])},
        {"int64": np.int64(3)},
        {"set": {1, 2}},
        {"bytes": b"raw"},
        {"when": datetime(2026, 1, 1)},
        {1: "non-string key"},
        {"nested": {"deep": [1, {"bad": object()}]}},
    ],
)
def test_non_json_model_params_rejected(params):
    with pytest.raises(ManifestValidationError):
        make(model_params=params)


@pytest.mark.parametrize("field", ["model_params", "training_data_provenance", "code_provenance"])
def test_json_object_fields_must_be_objects(field):
    with pytest.raises(ManifestValidationError):
        make(**{field: ["not", "an", "object"]})


def test_timezone_naive_created_at_rejected():
    with pytest.raises(ManifestValidationError, match="timezone-aware"):
        make(created_at=datetime(2026, 9, 25, 12, 30))
    with pytest.raises(ManifestValidationError, match="timezone-aware"):
        make(created_at="2026-09-25T12:30:00")


def test_utc_aware_created_at_accepted():
    assert make(created_at=datetime(2026, 9, 25, tzinfo=UTC)).created_at.tzinfo is UTC
    parsed = make(created_at="2026-09-25T12:30:00.123456Z")
    assert parsed.created_at == CREATED_AT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_version", ""),
        ("model_version", " padded "),
        ("model_family", "CatBoost"),
        ("target_formulation", "residual with spaces"),
        ("train_regime", 3),
        ("feature_schema_version", None),
    ],
)
def test_malformed_string_fields_rejected(field, value):
    with pytest.raises(ManifestValidationError):
        make(**{field: value})


@pytest.mark.parametrize(
    "location",
    [
        {"model_version": "/home/fz/models/m1"},
        {"code_provenance": {"repo": "/home/fz/projects/mostransport-ml"}},
        {"training_data_provenance": {"root": "C:\\data\\dataset"}},
        {"model_params": {"train_dir": "~/catboost_info"}},
    ],
)
def test_machine_specific_absolute_paths_rejected(location):
    with pytest.raises(ManifestValidationError, match="path"):
        make(**location)


@pytest.mark.parametrize("n_rows", [0, -1, 3.0, True])
def test_evaluation_n_rows_must_be_positive_integer(n_rows):
    with pytest.raises(ManifestValidationError):
        EvaluationSummary(split="labels_test", metric="mae", value=1.0, n_rows=n_rows)


def test_evaluation_value_is_not_judged():
    # Совместимость не решает, «хорош» ли результат: любое конечное значение допустимо.
    for value in (0.0, 1e9, -3.5):
        make(evaluation=EvaluationSummary(split="labels_test", metric="r2", value=value, n_rows=1))


def test_embedded_payload_is_rejected():
    predictions = list(range(MAX_CANONICAL_BYTES // 4))
    with pytest.raises(ManifestValidationError, match="must not be embedded"):
        make(training_data_provenance={"predictions": predictions})


def test_missing_and_unknown_fields_rejected():
    data = make().to_dict()
    del data["train_regime"]
    with pytest.raises(ManifestValidationError, match="train_regime"):
        ArtifactManifest.from_dict(data)
    data = make().to_dict()
    data["request_id"] = "runtime-field"
    with pytest.raises(ManifestValidationError, match="request_id"):
        ArtifactManifest.from_dict(data)
    data = make().to_dict()
    data["evaluation"]["predictions"] = [1.0]
    with pytest.raises(ManifestValidationError, match="predictions"):
        ArtifactManifest.from_dict(data)


def test_manifest_is_immutable():
    manifest = make()
    with pytest.raises(AttributeError):
        manifest.model_version = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.model_params["depth"] = 8  # type: ignore[index]


# ------------------------------------------------------------------ compatibility


def test_compatible_manifest_passes_and_is_returned():
    manifest = make()
    assert (
        validate_compatibility(manifest, expected_feature_schema_version="tabular-v1") is manifest
    )
    parsed = validate_compatibility(
        manifest.to_dict(), expected_feature_schema_version="tabular-v1"
    )
    assert parsed == manifest


def test_feature_schema_mismatch_rejected():
    with pytest.raises(ArtifactCompatibilityError, match="feature schema mismatch"):
        validate_compatibility(make(), expected_feature_schema_version="tabular-v2")


@pytest.mark.parametrize("version", ["artifact-manifest-v2", "artifact-manifest-v0", None, ["v1"]])
def test_unsupported_artifact_schema_rejected(version):
    data = make().to_dict()
    data["artifact_schema_version"] = version
    with pytest.raises(UnsupportedArtifactSchemaError):
        validate_compatibility(data, expected_feature_schema_version="tabular-v1")
    with pytest.raises(UnsupportedArtifactSchemaError):
        ArtifactManifest.from_dict(data)
    if isinstance(version, str):
        with pytest.raises(UnsupportedArtifactSchemaError):
            make(artifact_schema_version=version)


def test_unsupported_schema_is_checked_before_other_fields():
    # Будущая схема может иметь другие поля: сначала отказ по версии.
    with pytest.raises(UnsupportedArtifactSchemaError):
        ArtifactManifest.from_dict({"artifact_schema_version": "artifact-manifest-v2", "x": 1})


def test_malformed_metadata_rejected_by_compatibility_boundary():
    data = make().to_dict()
    data["evaluation"]["value"] = "57.19"
    with pytest.raises(ManifestValidationError):
        validate_compatibility(data, expected_feature_schema_version="tabular-v1")
    data = make().to_dict()
    data["model_params"]["lr"] = float("nan")
    with pytest.raises(ManifestValidationError):
        validate_compatibility(data, expected_feature_schema_version="tabular-v1")


def test_expected_feature_schema_version_is_required():
    with pytest.raises(ValueError):
        validate_compatibility(make(), expected_feature_schema_version="")


def test_manifest_is_model_family_agnostic():
    for family in ("catboost", "pytorch", "lightgbm", "constant-baseline"):
        manifest = make(model_family=family)
        assert validate_compatibility(manifest, expected_feature_schema_version="tabular-v1")


# ------------------------------------------------------------------ architecture


FORBIDDEN_MODULES = (
    "catboost",
    "torch",
    "fastapi",
    "pydantic",
    "uvicorn",
    "mostransport_ml.serving",
    "mostransport_ml.features",
    "mostransport_ml.models",
)


def test_manifest_module_imports_only_stdlib():
    tree = ast.parse(Path(manifest_module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}


def newly_loaded_forbidden(target: str, *, preload: tuple[str, ...] = ()) -> list[str]:
    """Запрещённые модули, которые впервые загружает именно `import target`.

    Свежий subprocess; снимок `sys.modules` берётся непосредственно перед
    импортом, поэтому модули, предзагруженные окружением запуска (или
    `preload`), не считаются загруженными импортом.
    """
    code = (
        "import importlib, json, sys\n"
        f"for name in {list(preload)!r}:\n"
        "    importlib.import_module(name)\n"
        "before = set(sys.modules)\n"
        f"importlib.import_module({target!r})\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=True, text=True
    )
    loaded_by_import = json.loads(result.stdout)
    return [
        m
        for m in loaded_by_import
        if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)
    ]


@pytest.mark.parametrize(
    "target", ["mostransport_ml.artifacts", "mostransport_ml.artifacts.manifest"]
)
def test_importing_artifacts_does_not_newly_load_forbidden_modules(target):
    assert newly_loaded_forbidden(target) == []


def test_import_isolation_check_ignores_preloaded_modules():
    # Окружение может заранее загрузить, например, pydantic: это не должно
    # засчитываться как загрузка со стороны artifacts.
    assert newly_loaded_forbidden("mostransport_ml.artifacts", preload=("pydantic",)) == []


def test_import_isolation_check_detects_forbidden_imports():
    # Позитивный контроль: serving.app тянет serving и fastapi, и проверка это видит.
    # pydantic не требуется: окружение запуска могло загрузить его заранее.
    leaked = newly_loaded_forbidden("mostransport_ml.serving.app")
    assert "mostransport_ml.serving" in leaked
    assert any(module == "fastapi" or module.startswith("fastapi.") for module in leaked)


# ------------------------------------------------------------------ duplicate JSON keys


def duplicated(text: str, original: str, duplicate: str) -> str:
    assert text.count(original) == 1, original
    return text.replace(original, f"{duplicate},{original}")


@pytest.mark.parametrize(
    ("original", "duplicate"),
    [
        # верхний уровень: при last-value-wins итог был бы валидным tabular-v1
        ('"feature_schema_version":"tabular-v1"', '"feature_schema_version":"x"'),
        ('"artifact_schema_version":"artifact-manifest-v1"', '"artifact_schema_version":"v0"'),
        ('"metric":"mae"', '"metric":"mae"'),
        ('"depth":6', '"depth":7'),
        ('"data_version":"official-planned-sha256:0000000000000000"', '"data_version":"other"'),
        ('"git_commit":"e59c48a"', '"git_commit":"e59c48a"'),
    ],
)
def test_duplicate_json_keys_rejected(original, duplicate):
    text = duplicated(make().canonical_json(), original, duplicate)
    with pytest.raises(ManifestValidationError, match="duplicate JSON key"):
        ArtifactManifest.from_json(text)


def test_duplicate_keys_rejected_at_deeper_nesting():
    manifest = make(model_params={"nested": {"inner": {"b": 1}}, "list": [{"c": 2}]})
    text = manifest.canonical_json()
    for original, duplicate in (('"b":1', '"b":3'), ('"c":2', '"c":2')):
        with pytest.raises(ManifestValidationError, match="duplicate JSON key"):
            ArtifactManifest.from_json(duplicated(text, original, duplicate))
    assert ArtifactManifest.from_json(text) == manifest


def test_duplicate_keys_rejected_when_reading_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(duplicated(make().canonical_json(), '"split":"labels_test"', '"split":"x"'))
    with pytest.raises(ManifestValidationError, match="duplicate JSON key"):
        read_manifest(path)


# Зафиксировано до добавления проверки дубликатов: валидный manifest обязан
# давать те же канонические байты и fingerprint.
GOLDEN_FINGERPRINT = "4fdab00d92fa6a601cfe3c6a0cc0fd0fe004db78fae009fbf1f9b13ec94fde89"


def test_valid_manifest_bytes_and_fingerprint_unchanged():
    manifest = make()
    assert manifest.fingerprint() == GOLDEN_FINGERPRINT
    restored = ArtifactManifest.from_json(manifest.canonical_json())
    assert restored.canonical_bytes() == manifest.canonical_bytes()
    assert restored.fingerprint() == GOLDEN_FINGERPRINT
    assert ArtifactManifest.from_json(json.dumps(manifest.to_dict(), indent=2)) == manifest


def test_canonical_bytes_are_float_repr_exact():
    value = 0.1 + 0.2
    manifest = make(model_params={"x": value})
    assert json.loads(manifest.canonical_json())["model_params"]["x"] == value
    assert not math.isnan(ArtifactManifest.from_json(manifest.canonical_bytes()).model_params["x"])

"""Artifact Bundle v1: хеши, целостность, безопасность путей, совместимость."""

from __future__ import annotations

import json
import os

import pytest
from conftest import make_manifest

from mostransport_ml.artifacts.bundle import (
    BUNDLE_FILENAME,
    BUNDLE_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    ArtifactBundleError,
    BundleDescriptor,
    load_bundle,
    sha256_bytes,
    write_bundle,
)
from mostransport_ml.artifacts.manifest import (
    ArtifactCompatibilityError,
    ManifestValidationError,
    UnsupportedArtifactSchemaError,
)

MODEL_BYTES = b"synthetic-model-binary-\x00\x01\x02"


def write(tmp_path, name="bundle", manifest=None, model_filename="model.cbm"):
    directory = tmp_path / name
    write_bundle(directory, manifest or make_manifest(), MODEL_BYTES, model_filename)
    return directory


def load(directory, *, schema="tabular-v1", families=("catboost",)):
    return load_bundle(
        directory, expected_feature_schema_version=schema, supported_model_families=families
    )


def rewrite_descriptor(directory, **changes):
    data = json.loads((directory / BUNDLE_FILENAME).read_bytes())
    data.update(changes)
    (directory / BUNDLE_FILENAME).write_text(json.dumps(data))


def test_valid_roundtrip(tmp_path):
    directory = write(tmp_path)
    assert sorted(p.name for p in directory.iterdir()) == [
        "bundle.json",
        "manifest.json",
        "model.cbm",
    ]
    verified = load(directory)
    assert verified.manifest == make_manifest()
    assert verified.model_bytes == MODEL_BYTES
    assert verified.descriptor.model_sha256 == sha256_bytes(MODEL_BYTES)
    assert verified.descriptor.manifest_sha256 == sha256_bytes(make_manifest().canonical_bytes())
    assert (directory / MANIFEST_FILENAME).read_bytes() == make_manifest().canonical_bytes()


def test_bundle_metadata_is_deterministic(tmp_path):
    a, b = write(tmp_path, "a"), write(tmp_path, "b")
    assert (a / BUNDLE_FILENAME).read_bytes() == (b / BUNDLE_FILENAME).read_bytes()
    raw = (a / BUNDLE_FILENAME).read_bytes()
    assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")).encode()
    assert load(a).bundle_sha256 == load(b).bundle_sha256 == sha256_bytes(raw)
    assert json.loads(raw)["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION


def test_bundle_identity_separate_from_manifest_identity(tmp_path):
    base = load(write(tmp_path, "a"))
    other_model = tmp_path / "b"
    write_bundle(other_model, make_manifest(), MODEL_BYTES + b"x", "model.cbm")
    changed = load(other_model)
    assert changed.manifest.fingerprint() == base.manifest.fingerprint()
    assert changed.bundle_sha256 != base.bundle_sha256


def test_modified_manifest_rejected(tmp_path):
    directory = write(tmp_path)
    tampered = make_manifest(model_version="tampered").canonical_bytes()
    (directory / MANIFEST_FILENAME).write_bytes(tampered)
    with pytest.raises(ArtifactBundleError, match="manifest.json: SHA-256 mismatch"):
        load(directory)


def test_modified_model_binary_rejected(tmp_path):
    directory = write(tmp_path)
    (directory / "model.cbm").write_bytes(MODEL_BYTES[:-1] + b"\x03")
    with pytest.raises(ArtifactBundleError, match="model.cbm: SHA-256 mismatch"):
        load(directory)


@pytest.mark.parametrize("name", [BUNDLE_FILENAME, MANIFEST_FILENAME, "model.cbm"])
def test_missing_file_rejected(tmp_path, name):
    directory = write(tmp_path)
    (directory / name).unlink()
    with pytest.raises(ArtifactBundleError):
        load(directory)


def test_missing_directory_rejected(tmp_path):
    with pytest.raises(ArtifactBundleError):
        load(tmp_path / "absent")


def test_extra_file_rejected(tmp_path):
    directory = write(tmp_path)
    (directory / "notes.txt").write_text("extra")
    with pytest.raises(ArtifactBundleError, match="exactly"):
        load(directory)


@pytest.mark.parametrize(
    "model_filename",
    ["../model.cbm", "/tmp/model.cbm", "sub/model.cbm", "..", ".hidden", "", "a\\b", "bundle.json"],
)
def test_unsafe_model_filename_rejected(tmp_path, model_filename):
    directory = write(tmp_path)
    rewrite_descriptor(directory, model_filename=model_filename)
    with pytest.raises(ArtifactBundleError):
        load(directory)
    with pytest.raises(ArtifactBundleError):
        write_bundle(tmp_path / "new", make_manifest(), MODEL_BYTES, model_filename)


def test_manifest_filename_must_be_exact(tmp_path):
    directory = write(tmp_path)
    rewrite_descriptor(directory, manifest_filename="../manifest.json")
    with pytest.raises(ArtifactBundleError, match="manifest_filename"):
        load(directory)


def test_symlinked_model_rejected(tmp_path):
    directory = write(tmp_path)
    outside = tmp_path / "outside.cbm"
    outside.write_bytes(MODEL_BYTES)
    (directory / "model.cbm").unlink()
    os.symlink(outside, directory / "model.cbm")
    with pytest.raises(ArtifactBundleError, match="regular file"):
        load(directory)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"bundle_schema_version":"artifact-bundle-v1","bundle_schema_version":"x"}',
        b"not json",
        b"[]",
        b'{"bundle_schema_version":"artifact-bundle-v1"}',
    ],
)
def test_malformed_bundle_json_rejected(tmp_path, raw):
    directory = write(tmp_path)
    (directory / BUNDLE_FILENAME).write_bytes(raw)
    with pytest.raises(ArtifactBundleError):
        load(directory)


@pytest.mark.parametrize(
    "changes",
    [
        {"bundle_schema_version": "artifact-bundle-v2"},
        {"model_sha256": "ABC"},
        {"extra_field": 1},
    ],
)
def test_invalid_descriptor_values_rejected(tmp_path, changes):
    directory = write(tmp_path)
    rewrite_descriptor(directory, **changes)
    with pytest.raises(ArtifactBundleError):
        load(directory)


def test_unsupported_model_family_rejected(tmp_path):
    directory = write(tmp_path, manifest=make_manifest(model_family="pytorch"))
    with pytest.raises(ArtifactBundleError, match="model_family 'pytorch' is not supported"):
        load(directory)


def test_feature_schema_mismatch_rejected(tmp_path):
    directory = write(tmp_path, manifest=make_manifest(feature_schema_version="tabular-v2"))
    with pytest.raises(ArtifactCompatibilityError, match="feature schema mismatch"):
        load(directory)


def test_unsupported_manifest_schema_rejected_even_with_matching_hash(tmp_path):
    directory = write(tmp_path)
    data = make_manifest().to_dict()
    data["artifact_schema_version"] = "artifact-manifest-v9"
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    (directory / MANIFEST_FILENAME).write_bytes(raw)
    rewrite_descriptor(directory, manifest_sha256=sha256_bytes(raw))
    with pytest.raises(UnsupportedArtifactSchemaError):
        load(directory)


def test_malformed_manifest_with_matching_hash_rejected(tmp_path):
    directory = write(tmp_path)
    raw = b'{"artifact_schema_version":"artifact-manifest-v1","model_version":"x"}'
    (directory / MANIFEST_FILENAME).write_bytes(raw)
    rewrite_descriptor(directory, manifest_sha256=sha256_bytes(raw))
    with pytest.raises(ManifestValidationError):
        load(directory)


def test_write_requires_new_or_empty_directory(tmp_path):
    directory = write(tmp_path)
    with pytest.raises(ArtifactBundleError, match="new or empty"):
        write_bundle(directory, make_manifest(), MODEL_BYTES, "model.cbm")
    with pytest.raises(ArtifactBundleError):
        write_bundle(tmp_path / "empty-model", make_manifest(), b"", "model.cbm")


def test_errors_do_not_contain_absolute_paths(tmp_path):
    directory = write(tmp_path)
    (directory / "model.cbm").write_bytes(b"corrupted")
    with pytest.raises(ArtifactBundleError) as excinfo:
        load(directory)
    assert str(tmp_path) not in str(excinfo.value)


def test_descriptor_parsing_is_strict_and_roundtrips(tmp_path):
    descriptor = load(write(tmp_path)).descriptor
    assert BundleDescriptor.from_json(descriptor.canonical_bytes()) == descriptor

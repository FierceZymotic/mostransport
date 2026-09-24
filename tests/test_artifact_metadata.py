import json

import pytest

from mostransport_ml.artifacts.metadata import ArtifactMetadata


def _make(**overrides) -> ArtifactMetadata:
    defaults = dict(
        model_version="mock-v0",
        created_at="2026-01-01T00:00:00+00:00",
        model_type="mock",
        target_name="delay",
    )
    defaults.update(overrides)
    return ArtifactMetadata(**defaults)


def test_round_trip_via_json():
    metadata = _make(target_unit="minutes", target_version="v1", validation_mae=1.5)
    restored = ArtifactMetadata.from_json(metadata.to_json())
    assert restored == metadata


def test_round_trip_via_dict():
    metadata = _make()
    restored = ArtifactMetadata.from_dict(metadata.to_dict())
    assert restored == metadata


def test_validation_mae_none_is_allowed():
    metadata = _make(validation_mae=None)
    assert metadata.validation_mae is None


def test_finite_non_negative_validation_mae_accepted():
    metadata = _make(validation_mae=0.0)
    assert metadata.validation_mae == 0.0


def test_negative_validation_mae_rejected():
    with pytest.raises(ValueError):
        _make(validation_mae=-1.0)


def test_nan_validation_mae_rejected():
    with pytest.raises(ValueError):
        _make(validation_mae=float("nan"))


def test_positive_infinity_validation_mae_rejected():
    with pytest.raises(ValueError):
        _make(validation_mae=float("inf"))


def test_negative_infinity_validation_mae_rejected():
    with pytest.raises(ValueError):
        _make(validation_mae=float("-inf"))


def test_serialized_json_is_strict_standard_json():
    metadata = _make(validation_mae=2.0)
    parsed = json.loads(metadata.to_json())  # must parse as standard JSON
    assert parsed["model_version"] == "mock-v0"
    assert parsed["validation_mae"] == 2.0


def test_no_raw_data_fields_present():
    metadata = _make()
    assert set(metadata.to_dict().keys()) == {
        "model_version",
        "created_at",
        "model_type",
        "target_name",
        "target_unit",
        "target_version",
        "feature_schema_version",
        "validation_mae",
    }

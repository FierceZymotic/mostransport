"""ArtifactPredictor на крошечной синтетической CatBoost-модели из Artifact Bundle v1."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_manifest, synthetic_feature_frame, train_tiny_catboost
from test_ml_context import offline_frames, packet, runtime_request

from mostransport_ml.artifacts.bundle import BUNDLE_FILENAME, ArtifactBundleError, sha256_bytes
from mostransport_ml.features.adapters import offline_context, runtime_context
from mostransport_ml.features.context import CanonicalBatch, build_features_from_context
from mostransport_ml.inference import model_families
from mostransport_ml.inference.predictor import (
    ArtifactLoadError,
    ArtifactPredictor,
    PredictionOutputError,
    export_bundle,
)


def batch_one():
    return runtime_context(
        runtime_request([packet(-s, speed=10.0 + s % 5) for s in range(600, -1, -20)])
    )


def multi_point_batch() -> CanonicalBatch:
    points, telemetry, plan = offline_frames(
        [packet(-s, speed=float(s % 9)) for s in range(900, -1, -15)]
    )
    shifted = [
        points.assign(sample_id=f"p{i}", T=points["T"] - pd.Timedelta(minutes=i)).assign(
            target_time_begin=points["target_time_begin"] - pd.Timedelta(minutes=i)
        )
        for i in (3, 0, 7)
    ]
    return offline_context(pd.concat(shifted, ignore_index=True), telemetry, plan)


def test_catboost_bundle_roundtrip_matches_in_memory_model_direct(
    catboost_bundle, tiny_catboost_model
):
    predictor = ArtifactPredictor.load(catboost_bundle(target_formulation="direct"))
    batch = batch_one()
    expected = tiny_catboost_model.predict(build_features_from_context(batch))
    np.testing.assert_array_equal(predictor.predict(batch), expected)


def test_catboost_bundle_roundtrip_residual_reconstruction(catboost_bundle, tiny_catboost_model):
    predictor = ArtifactPredictor.load(catboost_bundle(target_formulation="residual"))
    batch = batch_one()
    features = build_features_from_context(batch)
    expected = features["cur_dev_s"].to_numpy() + tiny_catboost_model.predict(features)
    np.testing.assert_array_equal(predictor.predict(batch), expected)
    assert batch.points[0].current_deviation_s == 45.0


def test_direct_and_residual_differ_exactly_by_current_deviation(catboost_bundle):
    batch = batch_one()
    direct = ArtifactPredictor.load(catboost_bundle(target_formulation="direct")).predict(batch)
    residual = ArtifactPredictor.load(catboost_bundle(target_formulation="residual")).predict(batch)
    np.testing.assert_allclose(residual - direct, [45.0], rtol=0, atol=1e-9)


def test_prediction_order_and_finiteness(catboost_bundle, tiny_catboost_model):
    predictor = ArtifactPredictor.load(catboost_bundle())
    batch = multi_point_batch()
    predictions = predictor.predict(batch)
    features = build_features_from_context(batch)
    assert list(features.index) == ["p3", "p0", "p7"]
    np.testing.assert_array_equal(predictions, tiny_catboost_model.predict(features))
    assert predictions.shape == (3,) and np.all(np.isfinite(predictions))
    assert len(set(predictions.tolist())) > 1  # разные точки → разные прогнозы, порядок значим


def test_metadata_exposure(catboost_bundle):
    directory = catboost_bundle()
    predictor = ArtifactPredictor.load(directory)
    assert predictor.is_ready() is True
    assert predictor.model_version() == "synthetic-infra-0001"
    assert predictor.feature_schema_version() == "tabular-v1"
    assert predictor.manifest.model_family == "catboost"
    assert predictor.bundle_sha256 == sha256_bytes((directory / BUNDLE_FILENAME).read_bytes())


def test_empty_batch_and_wrong_type():
    predictor = ArtifactPredictor.__new__(ArtifactPredictor)
    with pytest.raises(TypeError):
        predictor.predict("not a batch")  # type: ignore[arg-type]


# ------------------------------------------------------------------ controlled load failures


def reason(directory) -> str:
    with pytest.raises(ArtifactLoadError) as excinfo:
        ArtifactPredictor.load(directory)
    return excinfo.value.reason


def rehash_model(directory, new_bytes: bytes) -> None:
    (directory / "model.cbm").write_bytes(new_bytes)
    data = json.loads((directory / BUNDLE_FILENAME).read_bytes())
    data["model_sha256"] = sha256_bytes(new_bytes)
    (directory / BUNDLE_FILENAME).write_text(json.dumps(data))


def test_tampered_model_is_bundle_invalid(catboost_bundle):
    directory = catboost_bundle()
    (directory / "model.cbm").write_bytes(b"tampered")
    assert reason(directory) == "bundle_invalid"


def test_corrupted_model_with_valid_hash_is_controlled(catboost_bundle):
    directory = catboost_bundle()
    rehash_model(directory, b"not a catboost model at all")
    assert reason(directory) == "model_load_failed"


def test_incompatible_feature_schema(tmp_path, tiny_catboost_model):
    from mostransport_ml.artifacts.bundle import write_bundle

    directory = tmp_path / "b"
    blob = model_families.MODEL_FAMILIES["catboost"].serialize(tiny_catboost_model)
    write_bundle(directory, make_manifest(feature_schema_version="tabular-v2"), blob, "model.cbm")
    assert reason(directory) == "incompatible_feature_schema"


def test_unsupported_formulation_and_family(tmp_path, tiny_catboost_model):
    from mostransport_ml.artifacts.bundle import write_bundle

    blob = model_families.MODEL_FAMILIES["catboost"].serialize(tiny_catboost_model)
    write_bundle(tmp_path / "f", make_manifest(target_formulation="quantile"), blob, "model.cbm")
    assert reason(tmp_path / "f") == "unsupported_target_formulation"
    write_bundle(tmp_path / "m", make_manifest(model_family="pytorch"), blob, "model.pt")
    assert reason(tmp_path / "m") == "bundle_invalid"


def test_model_trained_on_other_features_rejected(tmp_path):
    from mostransport_ml.artifacts.bundle import write_bundle

    other = train_tiny_catboost(columns=[f"f{i}" for i in range(37)])
    blob = model_families.MODEL_FAMILIES["catboost"].serialize(other)
    write_bundle(tmp_path / "b", make_manifest(), blob, "model.cbm")
    assert reason(tmp_path / "b") == "model_feature_mismatch"
    with pytest.raises(ArtifactBundleError, match="tabular-v1"):
        export_bundle(tmp_path / "c", other, make_manifest())


def test_model_is_never_loaded_before_compatibility_check(
    tmp_path, tiny_catboost_model, monkeypatch
):
    from mostransport_ml.artifacts.bundle import write_bundle

    calls = []
    family = model_families.MODEL_FAMILIES["catboost"]
    blob = family.serialize(tiny_catboost_model)
    write_bundle(tmp_path / "b", make_manifest(feature_schema_version="other"), blob, "model.cbm")
    patched = type(family)(
        name=family.name,
        model_filename=family.model_filename,
        serialize=family.serialize,
        load=lambda data: calls.append(data),
        feature_names=family.feature_names,
    )
    monkeypatch.setitem(model_families.MODEL_FAMILIES, "catboost", patched)
    assert reason(tmp_path / "b") == "incompatible_feature_schema"
    assert calls == []


def test_non_finite_model_output_is_rejected(catboost_bundle):
    predictor = ArtifactPredictor.load(catboost_bundle())

    class NanModel:
        def predict(self, data):
            return np.full(len(data), np.nan)

    predictor._model = NanModel()
    with pytest.raises(PredictionOutputError):
        predictor.predict(batch_one())


def test_export_requires_compatible_manifest(tmp_path, tiny_catboost_model):
    from mostransport_ml.artifacts.manifest import ArtifactCompatibilityError

    with pytest.raises(ArtifactCompatibilityError):
        export_bundle(
            tmp_path / "x", tiny_catboost_model, make_manifest(feature_schema_version="v0")
        )
    with pytest.raises(ArtifactBundleError):
        export_bundle(tmp_path / "y", object(), make_manifest(model_family="pytorch"))


def test_synthetic_frame_matches_schema():
    assert tuple(synthetic_feature_frame().columns) == tuple(
        build_features_from_context(batch_one()).columns
    )

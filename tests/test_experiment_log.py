import json

import pytest

from mostransport_ml.experiments.log import ExperimentLogger, ExperimentRecord


def test_finite_mae_is_logged_as_valid_json(tmp_path):
    path = tmp_path / "runs.jsonl"
    logger = ExperimentLogger(path)
    logger.log_run(model_name="median-baseline", validation_mae=1.2345)

    line = path.read_text().strip()
    parsed = json.loads(line)  # обязан парситься как стандартный JSON
    assert parsed["model_name"] == "median-baseline"
    assert parsed["validation_mae"] == pytest.approx(1.2345)


def test_none_mae_is_allowed_and_round_trips_as_null(tmp_path):
    path = tmp_path / "runs.jsonl"
    logger = ExperimentLogger(path)
    logger.log_run(model_name="median-baseline", validation_mae=None)

    parsed = json.loads(path.read_text().strip())
    assert parsed["validation_mae"] is None


def test_record_rejects_nan_mae():
    with pytest.raises(ValueError):
        ExperimentRecord(model_name="x", validation_mae=float("nan"))


def test_record_rejects_positive_infinity_mae():
    with pytest.raises(ValueError):
        ExperimentRecord(model_name="x", validation_mae=float("inf"))


def test_record_rejects_negative_infinity_mae():
    with pytest.raises(ValueError):
        ExperimentRecord(model_name="x", validation_mae=float("-inf"))


def test_log_run_rejects_nan_before_writing_anything(tmp_path):
    path = tmp_path / "runs.jsonl"
    logger = ExperimentLogger(path)

    with pytest.raises(ValueError):
        logger.log_run(model_name="x", validation_mae=float("nan"))

    assert not path.exists()


def test_logged_line_is_parseable_standard_json(tmp_path):
    path = tmp_path / "runs.jsonl"
    logger = ExperimentLogger(path)
    logger.log_run(
        model_name="median-baseline",
        validation_mae=2.5,
        data_version="v1",
        target_version="v1",
        feature_version="v1",
        params={"window": 5},
        notes="smoke test",
    )

    line = path.read_text().strip()
    parsed = json.loads(line)
    assert parsed["params"] == {"window": 5}
    assert parsed["notes"] == "smoke test"

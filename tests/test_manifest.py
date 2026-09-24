import pandas as pd
import pytest

from mostransport_ml.data.manifest import DatasetManifest, build_manifest


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "t": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_build_manifest_basic(sample_csv):
    manifest = build_manifest(sample_csv, label="test-v1")

    assert manifest.label == "test-v1"
    assert len(manifest.source_files) == 1

    source = manifest.source_files[0]
    assert source.columns == ("a", "b", "t")
    assert source.row_count == 3
    assert source.size_bytes > 0
    assert len(source.sha256) == 64


def test_build_manifest_with_time_column(sample_csv):
    manifest = build_manifest(sample_csv, label="test-v1", time_column="t")
    source = manifest.source_files[0]
    assert source.time_column == "t"
    assert source.time_range is not None
    assert source.time_range[0] <= source.time_range[1]


def test_build_manifest_missing_time_column_raises(sample_csv):
    with pytest.raises(KeyError):
        build_manifest(sample_csv, label="test-v1", time_column="does_not_exist")


def test_build_manifest_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path / "missing.csv", label="x")


def test_manifest_json_roundtrip(sample_csv):
    manifest = build_manifest(sample_csv, label="test-v1")
    restored = DatasetManifest.from_json(manifest.to_json())
    assert restored == manifest


def test_manifest_never_stores_row_values(sample_csv):
    manifest = build_manifest(sample_csv, label="test-v1")
    # Only metadata fields at the dataset level — no place for actual cell
    # values to hide.
    assert set(manifest.to_dict().keys()) == {"label", "source_files", "created_at"}
    # ...and none at the per-file level either.
    assert set(manifest.to_dict()["source_files"][0].keys()) == {
        "path",
        "size_bytes",
        "sha256",
        "columns",
        "dtypes",
        "row_count",
        "time_column",
        "time_range",
    }


def test_build_manifest_multi_file_with_different_schemas(tmp_path):
    telemetry_path = tmp_path / "telemetry.csv"
    schedule_path = tmp_path / "schedule.csv"

    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(telemetry_path, index=False)
    pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]}).to_csv(
        schedule_path, index=False
    )

    manifest = build_manifest([telemetry_path, schedule_path], label="mixed-v1")

    assert len(manifest.source_files) == 2
    telemetry_source, schedule_source = manifest.source_files

    # Each file's own schema and row count is preserved independently —
    # neither file's schema is silently applied to the other.
    assert telemetry_source.columns == ("a", "b")
    assert telemetry_source.row_count == 2
    assert schedule_source.columns == ("x", "y", "z")
    assert schedule_source.row_count == 3
    assert telemetry_source.sha256 != schedule_source.sha256


def test_build_manifest_time_column_applies_only_where_present(tmp_path):
    # One file has the time column, the other doesn't — this must not force
    # a shared schema or raise, since the time_column exists in at least one
    # file.
    with_time = tmp_path / "with_time.csv"
    without_time = tmp_path / "without_time.csv"

    pd.DataFrame({"t": ["2026-01-01", "2026-01-02"], "v": [1, 2]}).to_csv(with_time, index=False)
    pd.DataFrame({"other": [1, 2, 3]}).to_csv(without_time, index=False)

    manifest = build_manifest([with_time, without_time], label="mixed-v2", time_column="t")

    with_time_source, without_time_source = manifest.source_files
    assert with_time_source.time_column == "t"
    assert with_time_source.time_range is not None
    assert without_time_source.time_column is None
    assert without_time_source.time_range is None

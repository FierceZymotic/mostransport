import pandas as pd
import pytest

from mostransport_ml.data.inspection import inspect_dataframe


def test_inspect_dataframe_basic_profile():
    df = pd.DataFrame(
        {
            "num": [1, 2, 3, None],
            "cat": ["a", "a", "b", "b"],
        }
    )
    report = inspect_dataframe(df)

    assert report.rows_inspected == 4
    assert report.n_columns == 2

    num_profile = next(c for c in report.columns if c.name == "num")
    assert num_profile.missing_count == 1
    assert num_profile.missing_ratio == pytest.approx(0.25)
    assert num_profile.numeric_summary is not None
    assert num_profile.numeric_summary["min"] == pytest.approx(1.0)

    cat_profile = next(c for c in report.columns if c.name == "cat")
    assert cat_profile.numeric_summary is None
    assert cat_profile.unique_count == 2


def test_inspect_dataframe_duplicate_row_count():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    report = inspect_dataframe(df)
    assert report.duplicate_row_count == 1


def test_inspect_dataframe_time_column_range():
    df = pd.DataFrame({"t": ["2026-01-01", "2026-01-03", "2026-01-02"]})
    report = inspect_dataframe(df, time_column="t")
    assert report.time_range is not None
    assert report.time_range[0] < report.time_range[1]


def test_inspect_dataframe_missing_time_column_raises():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError):
        inspect_dataframe(df, time_column="does_not_exist")


def test_inspect_csv_missing_file_raises(tmp_path):
    from mostransport_ml.data.inspection import inspect_csv

    with pytest.raises(FileNotFoundError):
        inspect_csv(tmp_path / "missing.csv")

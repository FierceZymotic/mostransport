import pandas as pd
import pytest

from mostransport_ml.evaluation.temporal import split_by_time_boundaries


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": pd.date_range("2026-01-01", periods=10, freq="h"),
            "value": range(10),
        }
    )


def test_split_basic_boundaries():
    df = _sample_df()
    split = split_by_time_boundaries(
        df, "t", train_end="2026-01-01T05:00:00", validation_end="2026-01-01T08:00:00"
    )
    assert len(split.train) == 5
    assert len(split.validation) == 3
    assert len(split.test) == 2
    assert len(split.train) + len(split.validation) + len(split.test) == len(df)


def test_split_missing_time_column_raises():
    df = _sample_df()
    with pytest.raises(KeyError):
        split_by_time_boundaries(df, "does_not_exist", "2026-01-01", "2026-01-02")


def test_split_invalid_boundary_order_raises():
    df = _sample_df()
    with pytest.raises(ValueError):
        split_by_time_boundaries(
            df, "t", train_end="2026-01-01T08:00:00", validation_end="2026-01-01T05:00:00"
        )


def test_split_is_deterministic():
    df = _sample_df()
    split_a = split_by_time_boundaries(df, "t", "2026-01-01T05:00:00", "2026-01-01T08:00:00")
    split_b = split_by_time_boundaries(df, "t", "2026-01-01T05:00:00", "2026-01-01T08:00:00")
    pd.testing.assert_frame_equal(split_a.train, split_b.train)
    pd.testing.assert_frame_equal(split_a.validation, split_b.validation)
    pd.testing.assert_frame_equal(split_a.test, split_b.test)


def test_split_timezone_mismatch_raises():
    df = _sample_df()
    df["t"] = df["t"].dt.tz_localize("UTC")
    with pytest.raises(ValueError):
        split_by_time_boundaries(df, "t", "2026-01-01T05:00:00", "2026-01-01T08:00:00")


def test_split_timezone_aware_both_sides_works():
    df = _sample_df()
    df["t"] = df["t"].dt.tz_localize("UTC")
    split = split_by_time_boundaries(
        df,
        "t",
        train_end="2026-01-01T05:00:00+00:00",
        validation_end="2026-01-01T08:00:00+00:00",
    )
    assert len(split.train) == 5

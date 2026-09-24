import pandas as pd
import pytest

from mostransport_ml.data.canonical import CanonicalMapping, apply_canonical_mapping


def test_rename_and_required_fields_pass():
    df = pd.DataFrame({"veh_id": [1, 2], "ts": ["a", "b"], "extra": [0, 0]})
    mapping = CanonicalMapping(
        source_name="synthetic-v1",
        rename={"veh_id": "vehicle_id", "ts": "event_time"},
        required_fields=("vehicle_id", "event_time"),
    )
    result = apply_canonical_mapping(df, mapping)
    assert list(result.columns) == ["vehicle_id", "event_time", "extra"]


def test_unknown_source_column_raises():
    df = pd.DataFrame({"veh_id": [1]})
    mapping = CanonicalMapping(source_name="synthetic-v1", rename={"does_not_exist": "vehicle_id"})
    with pytest.raises(ValueError):
        apply_canonical_mapping(df, mapping)


def test_missing_required_field_raises():
    df = pd.DataFrame({"veh_id": [1]})
    mapping = CanonicalMapping(
        source_name="synthetic-v1",
        rename={"veh_id": "vehicle_id"},
        required_fields=("vehicle_id", "event_time"),
    )
    with pytest.raises(ValueError):
        apply_canonical_mapping(df, mapping)


def test_rename_colliding_with_existing_unrenamed_column_raises():
    # columns = ["a", "b"], rename = {"a": "b"} — "b" already exists and is
    # not itself being renamed away, so this must fail fast.
    df = pd.DataFrame({"a": [1], "b": [2]})
    mapping = CanonicalMapping(source_name="synthetic-v1", rename={"a": "b"})
    with pytest.raises(ValueError):
        apply_canonical_mapping(df, mapping)


def test_multiple_sources_to_same_destination_raises():
    # columns = ["a", "c"], rename = {"a": "x", "c": "x"} — ambiguous target.
    df = pd.DataFrame({"a": [1], "c": [2]})
    mapping = CanonicalMapping(source_name="synthetic-v1", rename={"a": "x", "c": "x"})
    with pytest.raises(ValueError):
        apply_canonical_mapping(df, mapping)


def test_disjoint_rename_targets_pass():
    # columns = ["a", "b"], rename = {"a": "x", "b": "y"} — no collision.
    df = pd.DataFrame({"a": [1], "b": [2]})
    mapping = CanonicalMapping(source_name="synthetic-v1", rename={"a": "x", "b": "y"})
    result = apply_canonical_mapping(df, mapping)
    assert list(result.columns) == ["x", "y"]
    assert not result.columns.duplicated().any()


def test_swap_style_rename_is_not_a_false_collision():
    # a -> b, b -> c: "b" is itself being renamed away, so the destination
    # "b" for the first mapping does not collide with anything left behind.
    df = pd.DataFrame({"a": [1], "b": [2]})
    mapping = CanonicalMapping(source_name="synthetic-v1", rename={"a": "b", "b": "c"})
    result = apply_canonical_mapping(df, mapping)
    assert sorted(result.columns) == ["b", "c"]
    assert not result.columns.duplicated().any()


def test_no_feature_engineering_only_rename():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    mapping = CanonicalMapping(source_name="synthetic-v1", rename={"a": "renamed_a"})
    result = apply_canonical_mapping(df, mapping)
    pd.testing.assert_series_equal(
        result["renamed_a"], df["a"].rename("renamed_a"), check_names=True
    )
    assert "b" in result.columns

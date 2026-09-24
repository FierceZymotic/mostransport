"""Canonicalization boundary: an explicit, caller-supplied rename/validate step.

No canonical field list is hardcoded here. The official CSV schema and the
emulator schema are not known yet, so this module only provides the
*mechanism* that will later turn whatever the organizers give us into a
shared domain representation, once the actual field mapping is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class CanonicalMapping:
    """An explicit rename + required-field contract for one source schema.

    Nothing is inferred: every renamed column and every required field must
    be listed explicitly by the caller.
    """

    source_name: str
    rename: dict[str, str] = field(default_factory=dict)
    required_fields: tuple[str, ...] = ()


def apply_canonical_mapping(df: pd.DataFrame, mapping: CanonicalMapping) -> pd.DataFrame:
    """Rename columns per `mapping.rename` and validate `required_fields`.

    Only explicitly listed columns are renamed. No feature engineering, no
    type coercion, no fuzzy/inferred column matching. Fails fast on any
    rename that would make the resulting column set ambiguous.

    Raises
    ------
    ValueError
        If `mapping.rename` references a source column that doesn't exist,
        if two source columns would rename to the same destination, if a
        destination name collides with an existing column that isn't itself
        being renamed away, or if a required field is missing after
        renaming.
    """
    unknown_sources = set(mapping.rename) - set(df.columns)
    if unknown_sources:
        raise ValueError(
            f"CanonicalMapping {mapping.source_name!r} references source columns "
            f"not present in the dataframe: {sorted(unknown_sources)}. "
            f"Available columns: {list(df.columns)}"
        )

    destination_sources: dict[str, list[str]] = {}
    for source, destination in mapping.rename.items():
        destination_sources.setdefault(destination, []).append(source)
    colliding_destinations = {
        destination: sources
        for destination, sources in destination_sources.items()
        if len(sources) > 1
    }
    if colliding_destinations:
        details = ", ".join(
            f"{destination!r} <- {sorted(sources)}"
            for destination, sources in sorted(colliding_destinations.items())
        )
        raise ValueError(
            f"CanonicalMapping {mapping.source_name!r} maps multiple source columns to "
            f"the same destination name: {details}"
        )

    # A destination name is only safe if it's not shared with a column that
    # keeps its original name (columns being renamed away don't count).
    unrenamed_columns = set(df.columns) - set(mapping.rename)
    colliding_with_unrenamed = set(mapping.rename.values()) & unrenamed_columns
    if colliding_with_unrenamed:
        raise ValueError(
            f"CanonicalMapping {mapping.source_name!r} renames a column to a name that "
            f"already exists and is not itself being renamed: {sorted(colliding_with_unrenamed)}"
        )

    renamed = df.rename(columns=mapping.rename)

    duplicate_columns = renamed.columns[renamed.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        raise ValueError(
            f"CanonicalMapping {mapping.source_name!r} produces duplicate columns after "
            f"renaming: {sorted(duplicate_columns)}"
        )

    missing_required = [f for f in mapping.required_fields if f not in renamed.columns]
    if missing_required:
        raise ValueError(
            f"CanonicalMapping {mapping.source_name!r} is missing required fields "
            f"after renaming: {missing_required}. Available columns after rename: "
            f"{list(renamed.columns)}"
        )

    return renamed

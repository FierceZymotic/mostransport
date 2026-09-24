"""Canonicalization boundary: явный шаг rename/validate, задаваемый вызывающим.

Здесь не захардкожен ни один canonical список полей. Официальная схема
CSV и схема эмулятора пока не известны, поэтому модуль даёт только
*механизм*, который позже превратит то, что дадут организаторы, в общее
доменное представление — как только реальный field mapping станет
известен.

Важное уточнение (см. docs/PROJECT_KNOWLEDGE.md §10.10): это
offline-механизм работы с `pandas.DataFrame`, а не обязательный
runtime-конвертер между backend'ом и Python-сервисом.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class CanonicalMapping:
    """Явный контракт rename + required-field для одной source-схемы.

    Ничего не выводится автоматически: каждая переименовываемая колонка и
    каждое обязательное поле должны быть явно перечислены вызывающим.
    """

    source_name: str
    rename: dict[str, str] = field(default_factory=dict)
    required_fields: tuple[str, ...] = ()


def apply_canonical_mapping(df: pd.DataFrame, mapping: CanonicalMapping) -> pd.DataFrame:
    """Переименовать колонки по `mapping.rename` и проверить `required_fields`.

    Переименовываются только явно перечисленные колонки. Никакого feature
    engineering, никакого приведения типов, никакого нечёткого/выведенного
    сопоставления колонок. Fail-fast на любом rename, который сделал бы
    итоговый набор колонок неоднозначным.

    Исключения
    ----------
    ValueError
        Если `mapping.rename` ссылается на несуществующую source-колонку,
        если две source-колонки переименовываются в одно и то же
        назначение, если имя назначения совпадает с уже существующей
        колонкой, которая сама не переименовывается, либо если после
        переименования отсутствует обязательное поле.
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

    # Имя назначения безопасно, только если оно не совпадает с колонкой,
    # сохраняющей своё исходное имя (колонки, переименовываемые прочь, не в счёт).
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

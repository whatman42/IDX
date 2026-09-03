"""
Feature validation helpers.

Ensures ML-ready frames contain no unexpected null/inf and that
feature columns match the declared schema.
"""

from __future__ import annotations

from typing import Sequence

import polars as pl

from src.python.features.schema import ALL_FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION


def validate_features(
    df: pl.DataFrame,
    required_features: Sequence[str] | None = None,
    allow_null: bool = False,
) -> list[str]:
    """
    Validate a feature frame.

    Returns a list of human-readable issue strings (empty = OK).
    """
    issues: list[str] = []
    cols = required_features or ALL_FEATURE_COLUMNS

    missing = [c for c in cols if c not in df.columns]
    if missing:
        issues.append(f"Missing feature columns: {missing}")

    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        n_null = s.null_count()
        if n_null and not allow_null:
            issues.append(f"Column '{c}' has {n_null} nulls")
        try:
            n_inf = int((~s.is_null() & ~s.is_finite()).sum())
        except Exception:
            n_inf = 0
        if n_inf:
            issues.append(f"Column '{c}' has {n_inf} non-finite values")

    if "feature_schema_version" in df.columns:
        vers = df["feature_schema_version"].unique().to_list()
        if vers != [FEATURE_SCHEMA_VERSION]:
            issues.append(
                f"Schema version mismatch: got {vers}, expected [{FEATURE_SCHEMA_VERSION}]"
            )

    return issues


def assert_features_valid(df: pl.DataFrame, **kwargs) -> None:
    issues = validate_features(df, **kwargs)
    if issues:
        raise ValueError("Feature validation failed:\n  - " + "\n  - ".join(issues))

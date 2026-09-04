"""
Signal contract — unified inference output for Primary + Meta models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.python.features.schema import FEATURE_SCHEMA_VERSION
from src.python.ml.meta_labeling import MetaLabelGovernor
from src.python.ml.primary_side import PrimarySideModel


def generate_signals(
    X: pd.DataFrame,
    primary: PrimarySideModel,
    meta: MetaLabelGovernor,
    timestamps: Optional[pd.Series] = None,
    symbols: Optional[pd.Series] = None,
    side_threshold: float = 0.5,
    meta_threshold: Optional[float] = None,
    sizing_method: str = "sigmoid",
    max_weight: float = 0.20,
) -> pd.DataFrame:
    """
    Produce the standard signal table.

    Columns:
      timestamp, symbol, side, primary_probability, meta_probability,
      accepted, confidence, suggested_size, model_version, feature_schema_version
    """
    primary_proba = primary.predict_proba(X)
    side = np.where(primary_proba >= side_threshold, 1, -1).astype(int)
    meta_proba = meta.predict_proba(X)
    thr = meta.threshold if meta_threshold is None else float(meta_threshold)
    accepted = meta_proba >= thr
    size = meta.size_from_proba(meta_proba, method=sizing_method, max_weight=max_weight)
    size = np.where(accepted, size, 0.0)

    n = len(X)
    ts = timestamps if timestamps is not None else pd.Series(range(n), name="timestamp")
    sym = symbols if symbols is not None else pd.Series(["UNKNOWN"] * n, name="symbol")

    return pd.DataFrame(
        {
            "timestamp": ts.values if hasattr(ts, "values") else ts,
            "symbol": sym.values if hasattr(sym, "values") else sym,
            "side": side,
            "primary_probability": primary_proba,
            "meta_probability": meta_proba,
            "accepted": accepted.astype(bool),
            "confidence": meta_proba,
            "suggested_size": size,
            "model_version": f"{primary.model_version}+{meta.model_version}",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }
    )

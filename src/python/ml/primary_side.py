"""
Primary Side Model – LightGBM regression / probability of direction.
Produces raw alpha signal (side).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import Any


class PrimarySideModel:
    """LightGBM-based directional model."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "verbose": -1,
        }
        self.model: lgb.Booster | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PrimarySideModel":
        dtrain = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, dtrain, num_boost_round=200)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def export_onnx(self, path: str) -> None:
        """Export to ONNX for Rust consumption (placeholder)."""
        # Actual conversion via onnxmltools / skl2onnx in next iteration
        print(f"[PrimarySide] ONNX export stub → {path}")

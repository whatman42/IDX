"""
Meta-Labeling (Size) Governor.

Binary classifier that filters false positives from the primary side model
and produces calibrated position weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from typing import Literal


class MetaLabelGovernor:
    """Secondary model that decides whether to take the primary signal and how large."""

    def __init__(
        self,
        base_estimator: str = "rf",
        calibration: Literal["platt", "isotonic"] = "platt",
    ):
        if base_estimator == "rf":
            self.base = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                n_jobs=-1,
                class_weight="balanced",
            )
        else:
            raise ValueError("Only 'rf' supported in boilerplate")

        self.calibration = calibration
        self.calibrated: CalibratedClassifierCV | None = None
        self.iso: IsotonicRegression | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MetaLabelGovernor":
        """
        y should be 1 if primary side was profitable after barriers, else 0.
        """
        method = "sigmoid" if self.calibration == "platt" else "isotonic"
        self.calibrated = CalibratedClassifierCV(
            self.base, method=method, cv=3
        )
        self.calibrated.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrated is None:
            raise RuntimeError("Governor not fitted")
        return self.calibrated.predict_proba(X)[:, 1]

    def size_from_proba(
        self,
        proba: np.ndarray,
        method: Literal["sigmoid", "kelly"] = "sigmoid",
        max_weight: float = 0.25,
    ) -> np.ndarray:
        """Convert calibrated probability into portfolio weight."""
        if method == "sigmoid":
            # Soft scaling around 0.5
            w = 1.0 / (1.0 + np.exp(-10 * (proba - 0.5)))
        else:
            # Simplified Kelly fraction (p - q) / b  with b=1
            w = np.clip(2 * proba - 1, 0, 1)

        return np.clip(w, 0.0, max_weight)

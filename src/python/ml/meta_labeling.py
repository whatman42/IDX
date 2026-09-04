"""
Meta-Labeling Governor.

After Primary Side produces a candidate signal, the meta-model answers:
  "Is this signal worth taking?"

y_meta = 1 if primary side was correct, else 0.

Calibration (Platt / isotonic) uses a time-ordered holdout — never random K-fold.
Sizing is a recommendation only; Rust Risk Engine remains final authority.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.python.features.schema import ALL_FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION

DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": 1,
    "random_state": 42,
}


@dataclass
class MetaModelMeta:
    model_version: str
    created_at: str
    feature_schema_version: str
    feature_columns: list[str]
    calibration: str
    threshold: float
    training_rows: int = 0
    validation_rows: int = 0
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class MetaLabelGovernor:
    MODEL_PREFIX = "meta_rf"

    def __init__(
        self,
        base_estimator: str = "rf",
        calibration: Literal["platt", "isotonic", "none"] = "platt",
        threshold: float = 0.55,
        rf_params: Optional[dict[str, Any]] = None,
        feature_columns: Optional[Sequence[str]] = None,
        model_version: Optional[str] = None,
    ):
        if base_estimator != "rf":
            raise ValueError("Only 'rf' supported in this stage")
        self.base_estimator_name = base_estimator
        self.calibration = calibration
        self.threshold = float(threshold)
        self.rf_params = {**DEFAULT_RF_PARAMS, **(rf_params or {})}
        self.feature_columns: list[str] = list(feature_columns or ALL_FEATURE_COLUMNS)
        self.model_version = model_version or f"{self.MODEL_PREFIX}_v001"
        self.base = RandomForestClassifier(**self.rf_params)
        self.calibrated: Optional[Any] = None
        self.iso: Optional[IsotonicRegression] = None
        self._fitted = False
        self.meta = MetaModelMeta(
            model_version=self.model_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_columns=self.feature_columns,
            calibration=calibration,
            threshold=self.threshold,
            hyperparameters=self.rf_params,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calib_X: Optional[pd.DataFrame] = None,
        calib_y: Optional[pd.Series] = None,
    ) -> "MetaLabelGovernor":
        if X.empty or len(y) == 0:
            raise ValueError("Cannot train MetaLabelGovernor on empty dataset")
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")

        X = self._select_features(X)
        y = pd.Series(y).astype(int)
        if set(y.unique()) - {0, 1}:
            raise ValueError("Meta labels must be binary {0, 1}")

        if calib_X is None or calib_y is None:
            n = len(X)
            if n < 20:
                self.base.fit(X, y)
                self.calibrated = self.base
                self._fitted = True
                self.meta.training_rows = n
                return self
            cut = int(n * 0.8)
            X_fit, y_fit = X.iloc[:cut], y.iloc[:cut]
            X_cal, y_cal = X.iloc[cut:], y.iloc[cut:]
        else:
            X_fit, y_fit = X, y
            X_cal = self._select_features(calib_X)
            y_cal = pd.Series(calib_y).astype(int)

        self.base.fit(X_fit, y_fit)
        self.meta.training_rows = len(X_fit)
        self.meta.validation_rows = len(X_cal)

        if len(set(y_cal.tolist())) < 2 or self.calibration == "none":
            self.calibrated = self.base
        elif self.calibration == "platt":
            proba = self.base.predict_proba(X_cal)
            col = 1 if proba.shape[1] > 1 else 0
            raw = proba[:, col].reshape(-1, 1)
            platt = LogisticRegression(max_iter=1000, random_state=42)
            platt.fit(raw, y_cal)
            self.calibrated = ("platt", self.base, platt)
        elif self.calibration == "isotonic":
            proba = self.base.predict_proba(X_cal)
            col = 1 if proba.shape[1] > 1 else 0
            raw = proba[:, col]
            self.iso = IsotonicRegression(out_of_bounds="clip")
            self.iso.fit(raw, y_cal)
            self.calibrated = ("isotonic", self.base, self.iso)
        else:
            raise ValueError(f"Unknown calibration: {self.calibration}")

        self._fitted = True
        self.feature_columns = list(X.columns)
        self.meta.feature_columns = self.feature_columns
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self.calibrated is None:
            raise RuntimeError("Governor not fitted")
        X = self._select_features(X)

        def _pos_proba(model, X_):
            proba = model.predict_proba(X_)
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        if isinstance(self.calibrated, RandomForestClassifier):
            return _pos_proba(self.calibrated, X)
        kind, base, calibrator = self.calibrated
        raw = _pos_proba(base, X)
        if kind == "platt":
            return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return calibrator.predict(raw)

    def accept(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        thr = self.threshold if threshold is None else float(threshold)
        return self.predict_proba(X) >= thr

    def size_from_proba(
        self,
        proba: np.ndarray,
        method: Literal["sigmoid", "kelly"] = "sigmoid",
        max_weight: float = 0.20,
        kelly_fraction: float = 0.5,
    ) -> np.ndarray:
        proba = np.asarray(proba, dtype=float)
        if method == "sigmoid":
            w = 1.0 / (1.0 + np.exp(-10.0 * (proba - 0.5)))
        elif method == "kelly":
            raw = np.clip(2.0 * proba - 1.0, 0.0, 1.0)
            w = raw * kelly_fraction
        else:
            raise ValueError(f"Unknown sizing method: {method}")
        return np.clip(w, 0.0, max_weight)

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        out = X[self.feature_columns].copy()
        if out.isna().any().any():
            raise ValueError("NaN in feature matrix")
        if not np.isfinite(out.to_numpy(dtype=float)).all():
            raise ValueError("Inf in feature matrix")
        return out

    def save(self, directory: Union[str, Path]) -> Path:
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted governor")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / f"{self.model_version}.joblib"
        meta_path = directory / f"{self.model_version}.meta.json"
        joblib.dump({
            "calibrated": self.calibrated, "iso": self.iso, "base": self.base,
            "feature_columns": self.feature_columns, "threshold": self.threshold,
            "calibration": self.calibration,
        }, model_path)
        meta_path.write_text(json.dumps(asdict(self.meta), indent=2, default=str))
        return model_path

    @classmethod
    def load(cls, directory: Union[str, Path], model_version: str) -> "MetaLabelGovernor":
        directory = Path(directory)
        model_path = directory / f"{model_version}.joblib"
        meta_path = directory / f"{model_version}.meta.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        blob = joblib.load(model_path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        obj = cls(
            calibration=blob.get("calibration", meta.get("calibration", "platt")),
            threshold=blob.get("threshold", meta.get("threshold", 0.55)),
            feature_columns=blob.get("feature_columns", meta.get("feature_columns")),
            model_version=model_version,
        )
        obj.calibrated = blob["calibrated"]
        obj.iso = blob.get("iso")
        obj.base = blob["base"]
        obj._fitted = True
        obj.meta = MetaModelMeta(
            model_version=model_version,
            created_at=meta.get("created_at", ""),
            feature_schema_version=meta.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            feature_columns=blob.get("feature_columns", ALL_FEATURE_COLUMNS),
            calibration=blob.get("calibration", "platt"),
            threshold=blob.get("threshold", 0.55),
            training_rows=meta.get("training_rows", 0),
            validation_rows=meta.get("validation_rows", 0),
            hyperparameters=meta.get("hyperparameters", {}),
            metrics=meta.get("metrics", {}),
        )
        return obj

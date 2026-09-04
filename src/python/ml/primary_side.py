"""
Primary Side Model – LightGBM directional probability.

X = causal features (historical/current only)
y = directional label from Triple-Barrier (+1 PT, -1 SL, 0 vertical)

Binary map: label > 0 → 1, else 0. Outputs P(up); side = +1 if P>=0.5 else -1.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.python.features.schema import ALL_FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "is_unbalance": True,
    "verbosity": -1,
    "n_jobs": 1,
    "random_state": 42,
}


@dataclass
class PrimaryModelMeta:
    model_version: str
    created_at: str
    feature_schema_version: str
    feature_columns: list[str]
    training_start: Optional[str] = None
    training_end: Optional[str] = None
    training_rows: int = 0
    validation_rows: int = 0
    test_rows: int = 0
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class PrimarySideModel:
    MODEL_PREFIX = "primary_lgbm"

    def __init__(
        self,
        params: Optional[dict[str, Any]] = None,
        feature_columns: Optional[Sequence[str]] = None,
        model_version: Optional[str] = None,
    ):
        self.params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
        self._n_estimators = int(self.params.pop("n_estimators", 200))
        self.feature_columns: list[str] = list(feature_columns or ALL_FEATURE_COLUMNS)
        self.model: Optional[lgb.Booster] = None
        self.model_version = model_version or f"{self.MODEL_PREFIX}_v001"
        self.meta = PrimaryModelMeta(
            model_version=self.model_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_columns=self.feature_columns,
            hyperparameters={**self.params, "n_estimators": self._n_estimators},
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "PrimarySideModel":
        if X.empty or len(y) == 0:
            raise ValueError("Cannot train PrimarySideModel on empty dataset")
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if y.isna().any():
            raise ValueError("Target contains NaN")

        X = self._select_features(X, strict=True)
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Feature matrix contains NaN/Inf")

        y_bin = (pd.Series(y).astype(float) > 0).astype(int)
        dtrain = lgb.Dataset(X, label=y_bin, feature_name=list(X.columns), free_raw_data=False)
        valid_sets, valid_names = [dtrain], ["train"]
        if eval_set is not None:
            Xv, yv = eval_set
            Xv = self._select_features(Xv, strict=True)
            yv_bin = (pd.Series(yv).astype(float) > 0).astype(int)
            dval = lgb.Dataset(Xv, label=yv_bin, reference=dtrain, free_raw_data=False)
            valid_sets.append(dval)
            valid_names.append("valid")
            self.meta.validation_rows = len(Xv)

        self.model = lgb.train(
            self.params, dtrain, num_boost_round=self._n_estimators,
            valid_sets=valid_sets, valid_names=valid_names,
        )
        self.meta.training_rows = len(X)
        self.feature_columns = list(X.columns)
        self.meta.feature_columns = self.feature_columns
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        X = self._select_features(X, strict=True)
        return self.model.predict(X)

    def predict_side(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.where(proba >= threshold, 1, -1).astype(int)

    def feature_importance(self, importance_type: str = "gain", top_n: int = 20) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        imp = self.model.feature_importance(importance_type=importance_type)
        names = self.model.feature_name()
        return pd.Series(imp, index=names).sort_values(ascending=False).head(top_n)

    def _select_features(self, X: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        out = X[self.feature_columns].copy()
        if strict and out.isna().any().any():
            raise ValueError("NaN in feature matrix after selection")
        return out

    def check_schema(self, expected_version: str = FEATURE_SCHEMA_VERSION) -> None:
        if self.meta.feature_schema_version != expected_version:
            raise ValueError(
                f"Feature schema mismatch: model={self.meta.feature_schema_version}, "
                f"expected={expected_version}"
            )

    def save(self, directory: Union[str, Path]) -> Path:
        if self.model is None:
            raise RuntimeError("Cannot save unfitted model")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / f"{self.model_version}.txt"
        meta_path = directory / f"{self.model_version}.meta.json"
        self.model.save_model(str(model_path))
        meta_path.write_text(json.dumps(asdict(self.meta), indent=2, default=str))
        return model_path

    @classmethod
    def load(cls, directory: Union[str, Path], model_version: str) -> "PrimarySideModel":
        directory = Path(directory)
        model_path = directory / f"{model_version}.txt"
        meta_path = directory / f"{model_version}.meta.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        obj = cls(
            params=meta.get("hyperparameters"),
            feature_columns=meta.get("feature_columns"),
            model_version=model_version,
        )
        obj.model = lgb.Booster(model_file=str(model_path))
        obj.meta = PrimaryModelMeta(
            model_version=model_version,
            created_at=meta.get("created_at", ""),
            feature_schema_version=meta.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            feature_columns=meta.get("feature_columns", ALL_FEATURE_COLUMNS),
            training_start=meta.get("training_start"),
            training_end=meta.get("training_end"),
            training_rows=meta.get("training_rows", 0),
            validation_rows=meta.get("validation_rows", 0),
            test_rows=meta.get("test_rows", 0),
            hyperparameters=meta.get("hyperparameters", {}),
            metrics=meta.get("metrics", {}),
        )
        return obj

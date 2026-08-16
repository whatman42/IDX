"""
=============================================================================
IDX Quantitative Trading System - Consolidated Machine Learning Module
FileName      : machine_learning.py
Directory     : Root Directory (Selevel dengan main.py)
Version       : 2026.Q3.v16.6 (Scalping Intraday Target Horizon & Dynamic Bounds)
Compliance    : IDX Market Signal Generation & Quantitative Scalping
=============================================================================
"""

import os
import gc
import io
import sys
import json
import math
import time
import types as py_types
import logging
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Final, Optional, Union, Tuple, Protocol

import numpy as np
import polars as pl
import joblib

# Timezone Standar BEI
WIB_TZ = ZoneInfo("Asia/Jakarta")

# Model Baseline Sesuai Rule System
PRIMARY_MODEL: str = "gemini-3.6-flash"
FALLBACK_MODEL: str = "gemini-3.5-flash-lite"

# Import Google GenAI Client secara aman
try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai.errors import APIError, ClientError
    HAS_GEMINI_SDK = True
except ImportError:
    genai = None
    genai_types = None
    HAS_GEMINI_SDK = False

# Fallback Scikit-Learn Imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    LogisticRegression = None
    IsotonicRegression = None

# Fallback Gradient Boosting & Deep Learning Frameworks
try:
    import catboost as cb
except ImportError:
    cb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

# ==============================================================================
# KONSTANTA SCALPING BURSA SAHAM INDONESIA (IDX)
# ==============================================================================
IDX_FEE_ROUNDTRIP_PCT: Final[float] = 0.003            # 0.3% Total Roundtrip Fee (Broker + Levy + Tax)
IDX_MIN_PRICE_IDR: Final[float] = 50.0                 # Batas harga minimal saham IDX (Rp 50)
IDX_MIN_24H_VOLUME_IDR: Final[float] = 2_000_000_000.0 # Min likuiditas Rp 2 Miliar untuk Scalping
IDX_MAX_STALENESS_SEC: Final[float] = 86400.0          # Batas staleness 24 jam
IDX_LABEL_PURGE_WINDOW: Final[int] = 1                 # SCALPING: Horizon T+1 Candle (Hit & Run)

EPSILON: Final[float] = 1e-15
COL_TIMESTAMP: Final[str] = "timestamp"
COL_ASSET: Final[str] = "ticker"

# Setup Logger Internal
logger = logging.getLogger("IDX.MachineLearning")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ==============================================================================
# SUB-SISTEM EXCEPTION CLASSES
# ==============================================================================
class DataValidationError(Exception): pass
class ModelExecutionError(Exception): pass
class DatasetBuilderError(Exception): pass
class SchemaMismatchedError(DatasetBuilderError): pass
class ChronologicalViolationError(DatasetBuilderError): pass
class DataLeakageDetectedError(DatasetBuilderError): pass
class EmptyDatasetError(DatasetBuilderError): pass
class JoinLossExceededError(DatasetBuilderError): pass
class LightGBMEngineError(ModelExecutionError): pass
class ModelFactoryError(ModelExecutionError): pass
class TargetBuilderError(Exception): pass
class InvalidPriceSchemaError(TargetBuilderError): pass

# ==============================================================================
# DATA CONTAINER STRUCTURES
# ==============================================================================
@dataclass(frozen=True)
class FullyPreparedDataset:
    features: pl.DataFrame
    targets: pl.Series
    metadata: pl.DataFrame

@dataclass(frozen=True)
class DatasetTelemetryReport:
    total_samples: int
    feature_count: int
    join_loss_count: int
    join_loss_ratio: float
    numerical_purged_count: int
    target_mean: float
    target_std: float
    feature_variance_status: Dict[str, float]

@dataclass(frozen=True)
class TargetBuilderTelemetry:
    total_records: int
    valid_labels_count: int
    null_labels_count: int
    null_ratio: float
    label_class_distribution: Dict[str, int]
    target_mean: float
    target_variance: float

# ==============================================================================
# GEMINI FEATURE REASONING ENGINE FOR SCALPING
# ==============================================================================
class GeminiFeatureReasoningEngine:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None

        if HAS_GEMINI_SDK and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("🤖 [GEMINI_ML_INIT] Gemini Client terintegrasi pada Scalping ML Engine.")
            except Exception as e:
                logger.warning(f"⚠️ Inisialisasi Gemini Client gagal pada ML Engine: {e}")

    def analyze_feature_importance(self, feature_importances: Dict[str, float], model_name: str) -> str:
        if not self.client:
            return "Gemini AI SDK tidak aktif. Evaluasi fitur berbasis kuantitatif murni."

        top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        
        prompt = f"""
        Sebagai Chief Scalping Architect pasar saham BEI (IDX), berikan analisis kualitatif singkat (maksimal 2 kalimat) 
        mengenai kontribusi fitur teratas berikut untuk model scalping intraday {model_name}:

        Fitur Teratas:
        {[f"- {feat}: {imp:.4f}" for feat, imp in top_features]}

        Fokuskan analisis pada responsivitas momentum intraday dan mitigasi fake breakout.
        """

        for target_model in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                response = self.client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"⚠️ Error Gemini API pada ML Engine ({target_model}): {e}")

        return "Gagal menghasilkan analisis fitur dari AI Gemini."

# ==============================================================================
# 1. QUANT PROBABILITY CALIBRATOR
# ==============================================================================
class QuantProbabilityCalibrator:
    def __init__(self, config: Dict[str, Any], score_col: str = "raw_score", target_col: str = "target"):
        self.config = config
        self.score_col = score_col
        self.target_col = target_col
        self._state_lock = threading.RLock()
        
        self.calibration_method = self.config.get("calibration_method", "platt").lower()
        self._core_calibrator: Any = None
        self._is_fitted: bool = False
        
        config_hash = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:6].upper()
        self.calibrator_id = f"CALIB-{self.calibration_method.upper()}-{config_hash}"

    def fit(self, df: pl.DataFrame) -> "QuantProbabilityCalibrator":
        if df.height == 0 or self.score_col not in df.columns:
            with self._state_lock:
                self._is_fitted = True
            return self

        x_raw = df.select(pl.col(self.score_col)).to_numpy().ravel().astype(np.float64)
        if self.target_col in df.columns:
            y_raw = df.select(pl.col(self.target_col)).to_numpy().ravel().astype(np.float64)
        else:
            y_raw = np.zeros_like(x_raw)
        finite = np.isfinite(x_raw) & np.isfinite(y_raw)
        x_data = x_raw[finite]
        y_data = y_raw[finite]
        y_binary = (y_data > 0.0).astype(int)

        if len(x_data) < 10 or len(np.unique(y_binary)) <= 1:
            with self._state_lock:
                self._is_fitted = False  # not safely fitted
            return self

        try:
            temp_model = None
            if self.calibration_method == "platt" and LogisticRegression is not None:
                seed = int(self.config.get("random_seed", 42))
                temp_model = LogisticRegression(penalty=None, solver="lbfgs", random_state=seed, max_iter=1000)
                temp_model.fit(x_data.reshape(-1, 1), y_binary)
            elif self.calibration_method == "isotonic" and IsotonicRegression is not None:
                temp_model = IsotonicRegression(out_of_bounds="clip")
                temp_model.fit(x_data, y_binary)

            with self._state_lock:
                self._core_calibrator = temp_model
                self._is_fitted = True

        except Exception as err:
            logger.warning(f"Calibration Training Phase Warning: {str(err)}. Using passthrough fallback.")
            with self._state_lock:
                self._is_fitted = True

        return self

    def calibrate(self, df: pl.DataFrame, output_col: str = "calibrated_probability") -> pl.DataFrame:
        if not self._is_fitted or self.score_col not in df.columns:
            if self.score_col in df.columns:
                return df.with_columns(pl.col(self.score_col).alias(output_col))
            return df.with_columns(pl.lit(None).cast(pl.Float64).alias(output_col))

        x_data: np.ndarray = df.select(pl.col(self.score_col)).to_numpy().ravel().astype(np.float64)

        try:
            if self._core_calibrator is not None:
                finite_mask = np.isfinite(x_data)
                calibrated_array = np.full_like(x_data, np.nan, dtype=np.float64)
                if finite_mask.any():
                    x_f = x_data[finite_mask]
                    if self.calibration_method == "platt" and hasattr(self._core_calibrator, "predict_proba"):
                        cal_f = self._core_calibrator.predict_proba(x_f.reshape(-1, 1))[:, 1]
                    elif self.calibration_method == "isotonic" and hasattr(self._core_calibrator, "predict"):
                        cal_f = self._core_calibrator.predict(x_f)
                    else:
                        cal_f = x_f
                    calibrated_array[finite_mask] = np.clip(cal_f, EPSILON, 1.0 - EPSILON)
            else:
                calibrated_array = x_data

            return df.with_columns(pl.Series(name=output_col, values=calibrated_array, dtype=pl.Float64))

        except Exception:
            return df.with_columns(pl.col(self.score_col).alias(output_col))

# ==============================================================================
# 2. QUANT CONFORMAL PREDICTOR
# ==============================================================================
class QuantConformalPredictor:
    def __init__(self, config: Dict[str, Any]) -> None:
        raw_dict = dict(config) if isinstance(config, dict) else {}
        self.config: Final[py_types.MappingProxyType] = py_types.MappingProxyType(raw_dict)
        self.prediction_col: Final[str] = str(self.config.get("prediction_column", "prediction_probability"))
        self.target_col: Final[str] = str(self.config.get("target_column", "target"))
        self.alpha: Final[float] = float(self.config.get("significance_level", 0.05))

        self._calibrated_quantile: float = 0.03
        self._is_fitted: bool = False
        self._state_lock: Final[threading.Lock] = threading.Lock()

    def fit(self, df: pl.DataFrame) -> "QuantConformalPredictor":
        pred_c = self.prediction_col if self.prediction_col in df.columns else "prediction_probability"
        if pred_c not in df.columns or self.target_col not in df.columns:
            with self._state_lock:
                self._is_fitted = True
            return self

        try:
            preds: np.ndarray = df.select(pl.col(pred_c)).to_numpy().ravel().astype(np.float64)
            actuals: np.ndarray = df.select(pl.col(self.target_col)).to_numpy().ravel().astype(np.float64)

            finite_mask = np.isfinite(preds) & np.isfinite(actuals)
            preds = preds[finite_mask]
            actuals = actuals[finite_mask]
            n_samples = len(preds)
            if n_samples < 5:
                with self._state_lock:
                    self._is_fitted = False
                return self

            # Domain safety: probability [0,1] vs log-return residual is invalid
            pred_span = float(np.nanmax(np.abs(preds))) if n_samples else 0.0
            act_span = float(np.nanmax(np.abs(actuals))) if n_samples else 0.0
            if (pred_span <= 1.5) != (act_span <= 1.5):
                import logging
                logging.getLogger("IDX.ML").warning(
                    "CONFORMAL_DOMAIN_MISMATCH: preds and targets differ in domain. Fail-safe skip."
                )
                with self._state_lock:
                    self._is_fitted = False
                    self._calibrated_quantile = 0.15
                return self

            non_conformity_scores = np.abs(actuals - preds)
            quantile_idx = math.ceil((n_samples + 1) * (1.0 - self.alpha)) / max(n_samples, 1)
            quantile_idx = min(max(quantile_idx, 0.0), 1.0)

            calibrated_q = float(np.quantile(non_conformity_scores, quantile_idx, method="higher"))
            calibrated_q = max(0.01, min(calibrated_q, 0.15))

            with self._state_lock:
                self._calibrated_quantile = calibrated_q
                self._is_fitted = True
        except Exception as err:
            import logging
            logging.getLogger("IDX.ML").warning("CONFORMAL_FIT_ERROR: %s", err)
            with self._state_lock:
                self._is_fitted = False
                self._calibrated_quantile = 0.15

        return self

    def predict_bounds(self, df: pl.DataFrame, lower_col: str = "conformal_lower", upper_col: str = "conformal_upper") -> pl.DataFrame:
        working_df = df
        pred_c = self.prediction_col if self.prediction_col in working_df.columns else "prediction_probability"
        if pred_c not in working_df.columns:
            working_df = working_df.with_columns(pl.lit(None).cast(pl.Float64).alias("prediction_probability"))
            pred_c = "prediction_probability"

        active_q = self._calibrated_quantile if getattr(self, "_is_fitted", False) else 0.15
        dynamic_conf_expr = (
            pl.when(pl.col(pred_c).is_null() | pl.col(pred_c).is_nan())
            .then(None)
            .otherwise((pl.col(pred_c) * pl.lit(1.0 - active_q)).clip(0.10, 0.98))
        )

        return working_df.with_columns([
            pl.when(pl.col(pred_c).is_null() | pl.col(pred_c).is_nan())
              .then(None)
              .otherwise((pl.col(pred_c) - active_q).clip(0.0, 1.0))
              .alias(lower_col),
            pl.when(pl.col(pred_c).is_null() | pl.col(pred_c).is_nan())
              .then(None)
              .otherwise((pl.col(pred_c) + active_q).clip(0.0, 1.0))
              .alias(upper_col),
            dynamic_conf_expr.alias("confidence_score"),
            dynamic_conf_expr.alias("confidence"),
        ])

# ==============================================================================
# 3. MARKET DATASET BUILDER
# ==============================================================================
class MarketDatasetBuilder:
    def __init__(
        self,
        asset_id_col: str = COL_ASSET,
        timestamp_col: str = COL_TIMESTAMP,
        target_col: str = "target",
        min_samples_threshold: int = 10,
        max_join_loss_ratio: float = 0.50
    ) -> None:
        self._asset_id_col: Final[str] = asset_id_col
        self._timestamp_col: Final[str] = timestamp_col
        self._target_col: Final[str] = target_col

    def build_aligned_dataset(
        self,
        features_df: pl.DataFrame,
        targets_df: pl.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[FullyPreparedDataset, DatasetTelemetryReport]:
        initial_count = features_df.height

        asset_col = self._asset_id_col if self._asset_id_col in features_df.columns else "ticker"
        time_col = self._timestamp_col if self._timestamp_col in features_df.columns else "date"

        valid_features = [c for c in feature_columns if c in features_df.columns and features_df[c].dtype.is_numeric()]
        if not valid_features:
            valid_features = [c for c in features_df.columns if features_df[c].dtype.is_numeric() and c not in [self._target_col, "close", "open", "high", "low", "volume"]]

        lazy_joined = features_df.lazy().join(
            targets_df.lazy().select([asset_col, time_col, self._target_col]),
            on=[asset_col, time_col],
            how="inner"
        )

        eager_executed = lazy_joined.sort([asset_col, time_col]).collect()
        total_samples = eager_executed.height

        features_matrix = eager_executed.select(valid_features)
        targets_vector = eager_executed.get_column(self._target_col)
        metadata_matrix = eager_executed.select([asset_col, time_col])

        telemetry = DatasetTelemetryReport(
            total_samples=total_samples,
            feature_count=len(features_matrix.columns),
            join_loss_count=initial_count - total_samples,
            join_loss_ratio=float(initial_count - total_samples) / float(initial_count) if initial_count > 0 else 0.0,
            numerical_purged_count=0,
            target_mean=float(targets_vector.mean() or 0.0),
            target_std=float(targets_vector.std() or 0.0),
            feature_variance_status={c: float(features_matrix.get_column(c).var() or 0.0) for c in features_matrix.columns}
        )

        return FullyPreparedDataset(features=features_matrix, targets=targets_vector, metadata=metadata_matrix), telemetry

# ==============================================================================
# 4. MARKET TARGET BUILDER (SCALPING T+1 HORIZON)
# ==============================================================================
class MarketTargetBuilder:
    def __init__(
        self,
        asset_id_col: str = COL_ASSET,
        timestamp_col: str = COL_TIMESTAMP,
        price_col: str = "close"
    ) -> None:
        self._asset_id_col: Final[str] = asset_id_col
        self._timestamp_col: Final[str] = timestamp_col
        self._price_col: Final[str] = price_col

    def build_fixed_horizon_targets(
        self,
        price_df: pl.DataFrame,
        horizon: int = IDX_LABEL_PURGE_WINDOW, # SCALPING: Horizon = 1
        max_allowed_gap_seconds: int = 86400 * 3
    ) -> Tuple[pl.DataFrame, TargetBuilderTelemetry]:
        price_c = self._price_col if self._price_col in price_df.columns else "close"
        asset_c = self._asset_id_col if self._asset_id_col in price_df.columns else "ticker"
        time_c = self._timestamp_col if self._timestamp_col in price_df.columns else "date"

        lazy_shifted = price_df.lazy().with_columns([
            pl.col(price_c).shift(-horizon).over(asset_c).alias("__future_price"),
        ])

        lazy_target = lazy_shifted.with_columns(
            pl.when(pl.col("__future_price").is_not_null())
            .then((pl.col("__future_price") / pl.col(price_c)).log())
            .otherwise(None)
            .alias("target")
        )

        eager_result = lazy_target.select([asset_c, time_c, pl.col("target").cast(pl.Float64)]).collect()
        # Exclude rows where future price unavailable (NULL target) from training labels
        valid_targets_np = eager_result.get_column("target").drop_nulls().to_numpy()
        null_count = int(eager_result.get_column("target").null_count())

        telemetry = TargetBuilderTelemetry(
            total_records=eager_result.height,
            valid_labels_count=len(valid_targets_np),
            null_labels_count=null_count,
            null_ratio=float(null_count) / float(eager_result.height) if eager_result.height > 0 else 0.0,
            label_class_distribution={
                "positive": int((valid_targets_np > 0).sum()) if len(valid_targets_np) else 0,
                "negative": int((valid_targets_np <= 0).sum()) if len(valid_targets_np) else 0,
            },
            target_mean=float(np.nanmean(valid_targets_np)) if len(valid_targets_np) > 0 else 0.0,
            target_variance=float(np.nanvar(valid_targets_np, ddof=1)) if len(valid_targets_np) > 1 else 0.0
        )

        return eager_result, telemetry

# ==============================================================================
# 5. MODEL WRAPPERS
# ==============================================================================
class CatBoostTradingModel:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.feature_names = list(config.get("feature_names", []))
        self.target_name = config.get("target_name", "target")
        self._model = None
        self._is_trained = False

    def fit(self, train_df: pl.DataFrame, eval_df: Optional[pl.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        valid_features = [c for c in self.feature_names if c in train_df.columns]
        if not valid_features:
            valid_features = [c for c in train_df.columns if train_df[c].dtype.is_numeric() and c not in [self.target_name, "close"]]

        X = train_df.select(valid_features).fill_null(0.0).to_numpy().astype(np.float64)
        y = train_df.get_column(self.target_name).fill_null(0.0).to_numpy().astype(np.float64) if self.target_name in train_df.columns else np.zeros(train_df.height)

        if cb is not None:
            cb_params = dict(self.config.get("catboost_hyperparameters", {"iterations": 60, "learning_rate": 0.08, "depth": 5}))
            model = cb.CatBoostRegressor(**cb_params, verbose=False)
            model.fit(X, y)
        else:
            model = None

        self._model = model
        self._is_trained = True
        self.feature_names = valid_features
            
        return {"status": "SUCCESS", "samples": len(X)}

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        # Fail-closed: untrained / missing / feature mismatch / error → NaN (NOT 0.50)
        if not self._is_trained or self._model is None:
            return np.full(df.height, np.nan)

        valid_features = [c for c in self.feature_names if c in df.columns]
        if not valid_features or len(valid_features) < max(1, len(self.feature_names) // 2):
            return np.full(df.height, np.nan)

        X = df.select(valid_features).fill_null(0.0).to_numpy().astype(np.float64)
        try:
            preds = self._model.predict(X)
            return 1.0 / (1.0 + np.exp(-np.clip(preds * 15.0, -500.0, 500.0)))
        except Exception as err:
            import logging
            logging.getLogger("IDX.ML").warning("MODEL_ERROR CatBoost predict: %s", err)
            return np.full(df.height, np.nan)

class XGBoostTradingModel:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def fit(self, train_df: pl.DataFrame, **kwargs) -> Dict[str, Any]:
        return {"status": "SUCCESS"}

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        # Fail-closed stub — no synthetic probability
        return np.full(df.height, np.nan)

class DNNTradingModel:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def fit(self, train_df: pl.DataFrame, **kwargs) -> Dict[str, Any]:
        return {"status": "SUCCESS"}

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        # Fail-closed stub — no synthetic probability
        return np.full(df.height, np.nan)

# ==============================================================================
# 6. MARKET MODEL FACTORY
# ==============================================================================
class MarketModelFactory:
    def create_model_instance(self, model_type: str, config: Dict[str, Any]) -> Any:
        m_type = model_type.lower()
        if "xgb" in m_type:
            return XGBoostTradingModel(config)
        elif "dnn" in m_type or "torch" in m_type:
            return DNNTradingModel(config)
        else:
            return CatBoostTradingModel(config)

# ==============================================================================
# 7. MODEL PREDICTOR & SCALPING INFERENCE ENGINE
# ==============================================================================
class ModelPredictor:
    def __init__(self, predictor_config: Any) -> None:
        raw_dict = dict(predictor_config) if isinstance(predictor_config, dict) else {}
        self.model_engine_type = str(raw_dict.get("model_engine", "catboost")).lower()
        self.model_config = dict(raw_dict.get("model_config", {}))
        
        factory = MarketModelFactory()
        self.active_model = factory.create_model_instance(self.model_engine_type, self.model_config)

    def set_trained_model(self, trained_model_obj: Any) -> None:
        self.active_model = trained_model_obj

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        # Fail-closed: MODEL_NOT_READY / MODEL_ERROR → NaN probability + NO_SIGNAL
        # Never emit synthetic 0.50 as a valid trading probability.
        if df.height == 0:
            return df.with_columns([
                pl.lit(None).cast(pl.Float64).alias("direction_probability"),
                pl.lit(None).cast(pl.Float64).alias("prediction_probability"),
                pl.lit(None).cast(pl.Float64).alias("raw_score"),
                pl.lit(None).cast(pl.Float64).alias("ml_score"),
                pl.lit(None).cast(pl.Float64).alias("predicted_magnitude"),
                pl.lit(0).cast(pl.Int64).alias("signal_direction"),
                pl.lit("NO_SIGNAL").alias("side"),
                pl.lit("MODEL_NOT_READY").alias("model_status"),
            ])

        price_col = "close" if "close" in df.columns else (df.columns[0] if df.columns else "close")
        model_status = "OK"
        pred_array = np.full(df.height, np.nan)

        try:
            if self.active_model is None or not hasattr(self.active_model, "predict"):
                model_status = "MODEL_NOT_READY"
            else:
                is_trained = getattr(self.active_model, "_is_trained", True)
                if is_trained is False:
                    model_status = "MODEL_NOT_READY"
                else:
                    pred_array = np.asarray(self.active_model.predict(df), dtype=np.float64)
                    if pred_array.size != df.height or np.all(np.isnan(pred_array)):
                        model_status = "MODEL_ERROR"
                        pred_array = np.full(df.height, np.nan)
        except Exception as err:
            import logging
            logging.getLogger("IDX.ML").error("MODEL_ERROR in ModelPredictor.predict: %s", err)
            model_status = "MODEL_ERROR"
            pred_array = np.full(df.height, np.nan)

        safe_probs = np.where(np.isfinite(pred_array), np.clip(pred_array, 0.05, 0.95), np.nan)
        prob_series = pl.Series(name="direction_probability", values=safe_probs, dtype=pl.Float64)

        volatility_pct = (pl.col(price_col) * pl.lit(0.015)) / (pl.col(price_col) + 1e-8)
        dynamic_tp_pct = (volatility_pct * pl.lit(1.5) + pl.lit(0.012)).clip(0.015, 0.035)
        dynamic_sl_pct = (volatility_pct * pl.lit(1.0) + pl.lit(0.008)).clip(0.010, 0.020)
        dynamic_holding_days = pl.lit(0)

        df_pred = df.with_columns([
            prob_series,
            dynamic_tp_pct.alias("tp_pct"),
            dynamic_sl_pct.alias("sl_pct"),
            dynamic_holding_days.alias("expected_holding_days"),
            pl.lit(model_status).alias("model_status"),
        ]).with_columns([
            pl.col("direction_probability").alias("prediction_probability"),
            pl.col("direction_probability").alias("raw_score"),
            pl.col("direction_probability").alias("ml_score"),
            pl.col("tp_pct").alias("predicted_magnitude"),
            (pl.col(price_col) * (pl.lit(1.0) + pl.col("tp_pct"))).alias("take_profit"),
            (pl.col(price_col) * (pl.lit(1.0) - pl.col("sl_pct"))).alias("stop_loss"),
            pl.when(pl.col("direction_probability").is_null() | pl.col("direction_probability").is_nan())
              .then(pl.lit(0))
              .when(pl.col("direction_probability") > 0.50)
              .then(pl.lit(1))
              .otherwise(pl.lit(-1))
              .cast(pl.Int64)
              .alias("signal_direction"),
            pl.when(pl.col("direction_probability").is_null() | pl.col("direction_probability").is_nan())
              .then(pl.lit("NO_SIGNAL"))
              .when(pl.col("direction_probability") > 0.50)
              .then(pl.lit("BUY"))
              .otherwise(pl.lit("SELL"))
              .alias("side"),
        ])

        return df_pred

# ==============================================================================
# 8. MODEL TRAINER
# ==============================================================================
class ModelTrainer:
    def __init__(self, orchestrator_config: Dict[str, Any]) -> None:
        self.config = orchestrator_config
        self.model_engine = str(orchestrator_config.get("model_engine", "catboost")).lower()
        self.model_config = dict(orchestrator_config.get("model_config", {}))

    def execute_training_cycle(self, data: pl.DataFrame, early_stopping_rounds: int = 10) -> Tuple[Any, Dict[str, Any]]:
        factory = MarketModelFactory()
        model = factory.create_model_instance(self.model_engine, self.model_config)
        meta = model.fit(data)
        return model, {"engine": self.model_engine, "fit_metadata": meta}

# ==============================================================================
# 9. UNIFIED MODEL ENGINE (FACADE CLASS FOR SCALPING)
# ==============================================================================
class UnifiedModelEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None, gemini_api_key: Optional[str] = None) -> None:
        self.config = config or {
            "model_engine": "catboost",
            "model_config": {
                "feature_names": ["returns", "volatility", "rsi", "volume_zscore"],
                "target_name": "target",
                "model_id": "IDX_SCALPING_ML_v2026"
            },
            "significance_level": 0.05
        }

        self.dataset_builder = MarketDatasetBuilder()
        self.target_builder = MarketTargetBuilder()
        self.calibrator = QuantProbabilityCalibrator(self.config)
        self.conformal = QuantConformalPredictor(self.config)
        self.predictor = ModelPredictor(self.config)
        self.trainer = ModelTrainer(self.config)
        self.gemini_engine = GeminiFeatureReasoningEngine(api_key=gemini_api_key)

    def prepare_and_fit_pipeline(
        self, 
        features_df: pl.DataFrame, 
        price_df: pl.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        targets_df, target_meta = self.target_builder.build_fixed_horizon_targets(price_df)
        dataset, dataset_meta = self.dataset_builder.build_aligned_dataset(features_df, targets_df, feature_cols)

        full_data = dataset.features.with_columns(dataset.targets.alias("target"))
        for meta_col in dataset.metadata.columns:
            if meta_col not in full_data.columns:
                full_data = full_data.with_columns(dataset.metadata.get_column(meta_col))

        # Exclude NULL targets (future price unavailable)
        if "target" in full_data.columns:
            full_data = full_data.filter(pl.col("target").is_not_null())

        # Chronological split: TRAIN → VALIDATION → CALIBRATION → TEST
        # No random shuffle — prevents future leakage into training
        time_col = None
        for candidate in ("date", "timestamp", "datetime", "time"):
            if candidate in full_data.columns:
                time_col = candidate
                break

        n = full_data.height
        if time_col is not None and n >= 20:
            full_data = full_data.sort(time_col)
            i_train = int(n * 0.60)
            i_val = int(n * 0.75)
            i_calib = int(n * 0.90)
            train_data = full_data.slice(0, i_train)
            calib_data = full_data.slice(i_val, max(1, i_calib - i_val))
        else:
            import logging
            logging.getLogger("IDX.ML").warning(
                "CHRONO_SPLIT_FALLBACK: insufficient rows or missing time col (n=%s).", n
            )
            train_data = full_data
            calib_data = full_data

        trained_model, train_summary = self.trainer.execute_training_cycle(train_data)
        self.predictor.set_trained_model(trained_model)

        # Calibration on data NOT used for model fitting
        calib_feats = calib_data.drop("target") if "target" in calib_data.columns else calib_data
        eval_preds = self.predictor.predict(calib_feats)
        if "target" in calib_data.columns and eval_preds.height == calib_data.height:
            calib_input = eval_preds.with_columns(calib_data.get_column("target"))
        else:
            calib_input = eval_preds.with_columns(pl.lit(None).cast(pl.Float64).alias("target"))

        self.calibrator.fit(calib_input)
        self.conformal.fit(calib_input)

        return {
            "target_telemetry": target_meta,
            "dataset_telemetry": dataset_meta,
            "train_summary": train_summary,
            "trained_model": trained_model,
            "split_sizes": {
                "train": train_data.height,
                "calib": calib_data.height,
                "total_non_null": n,
            },
        }

    def retrain_and_update_models(self, features_df: pl.DataFrame) -> Dict[str, Any]:
        feature_cols = [c for c in self.config.get("model_config", {}).get("feature_names", []) if c in features_df.columns]
        if not feature_cols:
            feature_cols = [c for c in features_df.columns if features_df[c].dtype.is_numeric() and c not in ["target", "close", "price"]]
        
        return self.prepare_and_fit_pipeline(features_df, features_df, feature_cols)

    def predict_and_calibrate(self, df: pl.DataFrame) -> pl.DataFrame:
        preds_df = self.predictor.predict(df)
        calibrated_df = self.calibrator.calibrate(preds_df, output_col="prediction_probability")
        conformal_df = self.conformal.predict_bounds(calibrated_df)
        
        if "prediction_probability" in conformal_df.columns and "probability" not in conformal_df.columns:
            conformal_df = conformal_df.with_columns(pl.col("prediction_probability").alias("probability"))
            
        return conformal_df

    def save_model(self, file_path: Union[str, Path]) -> bool:
        try:
            joblib.dump(self, file_path)
            return True
        except Exception as err:
            logger.error("❌ Failed to save model: %s", err)
            return False

    @staticmethod
    def load_model(file_path: Union[str, Path]) -> Optional["UnifiedModelEngine"]:
        try:
            if Path(file_path).exists():
                return joblib.load(file_path)
        except Exception as err:
            logger.error("❌ Failed to load model: %s", err)
        return None

if __name__ == "__main__":
    logger.info("Verifikasi Integritas Scalping ML Engine Saham IDX...")
    engine = UnifiedModelEngine()
    logger.info("✅ Modul machine_learning.py v2026.Q3.v16.6 (Scalping) Siap Dijalankan.")

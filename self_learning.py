"""
=============================================================================
Module      : self_learning.py
Description : IDX Stock Quantitative Analysis & Signal Generation
              Autonomous Self-Learning & Adaptive Feedback Loop Engine v2026.Q3.v16.4
Consolidates 13 sub-engines into a single root-level module for autonomous model learning, 
drift detection, performance tracking, model selection, retraining orchestration,
Google Gemini AI Diagnostics, and GUARANTEED disk state persistence for the Indonesian Stock Market (IDX).
Path        : ./self_learning.py (Root Directory)
=============================================================================
"""

import os
import json
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Final

import joblib
import numpy as np
import polars as pl
from scipy import stats
from scipy.special import erf
from scipy.stats import norm

# Standar Zona Waktu Pasar Saham Indonesia (WIB)
WIB_TZ: Final[ZoneInfo] = ZoneInfo("Asia/Jakarta")

# Model Baseline Gemini SDK
PRIMARY_MODEL: Final[str] = "gemini-2.5-flash"
FALLBACK_MODEL: Final[str] = "gemini-2.0-flash"

# Import Google GenAI Client
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# Optional ML framework imports with graceful fallbacks
try:
    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

try:
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# IDX STOCK MARKET COMPLIANCE CONSTANTS & FALLBACK ALIASES
# =============================================================================

IDX_FEE_ROUNDTRIP_PCT: float = 0.003            # Biaya transaksi roundtrip pasar saham IDX (0.3%)
IDX_MIN_PRICE_IDR: float = 50.0                 # Batas harga minimal saham IDX / Papan Efek (Rp 50)
IDX_MIN_24H_VOLUME_IDR: float = 1_000_000_000.0 # Minimal volume transaksi harian IDX (Rp 1 Miliar)
IDX_MAX_STALENESS_SEC: float = 43200.0          # Batas maksimal usia data candle (12 Jam)
EPSILON: float = 1e-6                           # Safety regularizer

# Backward Compatibility Aliases for Upstream/Downstream Integration
TOKOCRYPTO_FEE_ROUNDTRIP_PCT: float = IDX_FEE_ROUNDTRIP_PCT
TOKOCRYPTO_MIN_PRICE_USDT: float = IDX_MIN_PRICE_IDR
TOKOCRYPTO_MIN_24H_VOLUME_USDT: float = IDX_MIN_24H_VOLUME_IDR
TOKOCRYPTO_MAX_STALENESS_SEC: float = IDX_MAX_STALENESS_SEC


# =============================================================================
# SYNCHRONIZED FALLBACK EXCEPTION CLASSES & LOGGER
# =============================================================================

try:
    from exceptions import (
        ChronologyError,
        ConfidenceEstimatorError,
        DataValidationError,
        DriftDetectorError,
        HyperparameterOptimizerError,
        ModelRankerError,
        ModelSelectorError,
        ModelValidationError,
        OnlineLearningError,
        OptunaManagerError,
        RetrainingSchedulerError,
        UncertaintyEstimatorError,
    )
except ImportError:
    class SelfLearningBaseError(Exception): pass
    class DataValidationError(SelfLearningBaseError): pass
    class ChronologyError(SelfLearningBaseError): pass
    class ModelValidationError(SelfLearningBaseError): pass
    class ConfidenceEstimatorError(SelfLearningBaseError): pass
    class DriftDetectorError(SelfLearningBaseError): pass
    class HyperparameterOptimizerError(SelfLearningBaseError): pass
    class ModelRankerError(SelfLearningBaseError): pass
    class ModelSelectorError(SelfLearningBaseError): pass
    class OnlineLearningError(SelfLearningBaseError): pass
    class OptunaManagerError(SelfLearningBaseError): pass
    class RetrainingSchedulerError(SelfLearningBaseError): pass
    class UncertaintyEstimatorError(SelfLearningBaseError): pass

try:
    from logger import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("IDX.SelfLearningEngine")


# =============================================================================
# DEFENSIVE SANITIZATION HELPERS
# =============================================================================

def _get_wib_timestamp_str() -> str:
    """Mengembalikan timestamp berformat ISO dengan Zona Waktu Indonesia Barat (WIB)."""
    return datetime.now(WIB_TZ).isoformat()


def _ensure_polars_df_sl(data: Any, default_cols: Optional[List[str]] = None) -> pl.DataFrame:
    """Sanitasi input internal untuk menjamin objek bertipe pl.DataFrame (Polars 1.0+ Compliant)."""
    if data is None:
        cols = default_cols or ["asset", "date"]
        return pl.DataFrame(schema={col: pl.String for col in cols})
    if isinstance(data, list):
        if not data:
            cols = default_cols or ["asset", "date"]
            return pl.DataFrame(schema={col: pl.String for col in cols})
        return pl.DataFrame(data)
    if isinstance(data, pl.DataFrame):
        return data
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
    except ImportError:
        pass
    return pl.DataFrame(data)


def sanitize_predictions_for_feedback(
    df: Union[pl.DataFrame, Any],
    default_model_id: str = "champion_idx_v1"
) -> pl.DataFrame:
    """Menyiapkan skema DataFrame agar siap dikonsumsi oleh FeedbackEngine."""
    df = _ensure_polars_df_sl(df)
    if df.height == 0:
        return df

    exprs = []
    
    # Standardisasi Asset Key
    if "asset" not in df.columns and "ticker" in df.columns:
        exprs.append(pl.col("ticker").alias("asset"))
    
    # Standardisasi Date Key
    if "signal_date" not in df.columns and "date" in df.columns:
        exprs.append(pl.col("date").alias("signal_date"))
        
    # Standardisasi Predicted Return
    if "predicted_return" not in df.columns:
        pred_col = next((c for c in ["prediction", "score", "probability", "signal", "predicted_return"] if c in df.columns), None)
        if pred_col:
            exprs.append(pl.col(pred_col).alias("predicted_return"))
        else:
            exprs.append(pl.lit(0.0).alias("predicted_return"))

    # Synthesize Model ID jika belum ada
    if "model_id" not in df.columns:
        exprs.append(pl.lit(default_model_id).alias("model_id"))

    if exprs:
        df = df.with_columns(exprs)

    # Synthesize Unique Prediction ID
    if "prediction_id" not in df.columns:
        asset_col = "asset" if "asset" in df.columns else ("ticker" if "ticker" in df.columns else None)
        date_col = "signal_date" if "signal_date" in df.columns else ("date" if "date" in df.columns else None)
        if asset_col and date_col:
            df = df.with_columns(
                (pl.col("model_id") + pl.lit("_") + pl.col(asset_col).cast(pl.String) + pl.lit("_") + pl.col(date_col).cast(pl.String)).alias("prediction_id")
            )
        else:
            df = df.with_columns(
                (pl.lit("PRED_") + pl.int_range(0, df.height, dtype=pl.Int64).cast(pl.String)).alias("prediction_id")
            )

    return df


# =============================================================================
# GEMINI SELF-LEARNING DIAGNOSTIC ENGINE
# =============================================================================

class GeminiSelfLearningDiagnosticEngine:
    """
    Sub-Engine Google Gemini AI sebagai Meta-Diagnostic & Advisory Layer untuk Self-Learning & Adaptive Loop.
    Mengevaluasi concept drift, performance decay, serta keputusan retraining model ML kuantitatif dengan Dual-Model Fallback.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None

        if HAS_GEMINI_SDK and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("🤖 [GEMINI_SELF_LEARNING_INIT] Gemini Client terhubung untuk Diagnostik Model Learning.")
            except Exception as err:
                logger.warning(f"⚠️ [GEMINI_SELF_LEARNING_INIT_FAILED] Gagal inisialisasi Gemini Client: {err}")

    def run_learning_diagnosis(
        self,
        active_model_id: str,
        drift_report: Dict[str, Any],
        decay_report: Optional[Dict[str, Any]],
        retrain_manifest: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Melakukan analisis kualitatif dan saran tindakan atas kondisi model kuantitatif menggunakan Google Gemini AI."""
        if not self.client:
            return {
                "ai_learning_diagnosis": "Gemini AI Client tidak aktif.",
                "ai_action_recommendation": "Lanjutkan evaluasi retraining berdasarkan threshold kuantitatif standar.",
                "urgency_level": "LOW"
            }

        prompt = f"""
        Sebagai Chief AI Officer / Head of Quantitative Machine Learning di Dana Kelolaan Saham BEI (IDX), berikan diagnosis kualitatif dan rekomendasi tindakan terkait kondisi model ML berikut:

        === METADATA MODEL & METRIK SINKRONISASI ===
        - Target Active Model ID: {active_model_id}
        - Keputusan Retraining Scheduler: {retrain_manifest.get('trigger_retraining', False)}
        - Eksekusi Tuning Hyperparameter: {retrain_manifest.get('execute_hyperparameter_tuning', False)}
        - Justifikasi Otomatis: {retrain_manifest.get('audit_justification', 'N/A')}

        === HASIL LAPORAN CONCEPT DRIFT (FEATURE STREAM) ===
        {json.dumps(drift_report.get('summary_metrics', {}), indent=2, default=str)}

        === HASIL LAPORAN PERFORMANCE DECAY & ROLLING METRICS ===
        {json.dumps(decay_report or {}, indent=2, default=str)}

        Tugas:
        1. Berikan diagnosa kualitatif ringkas (1-2 kalimat) mengenai integritas model saat ini (apakah terjadi Concept Drift berat atau Degraded Alpha).
        2. Berikan arahan tindakan operasional ML (misal: "Keep Model Active", "Trigger Incremental Warm-Start", "Trigger Full Retrain & Optuna Tuning").
        3. Tentukan Urgency Level ("LOW", "MEDIUM", "HIGH", "CRITICAL").

        Format Jawaban (JSON murni):
        {{
            "diagnosis": "...",
            "action_recommendation": "...",
            "urgency_level": "LOW"
        }}
        """

        for model_target in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                response = self.client.models.generate_content(
                    model=model_target,
                    contents=prompt,
                )
                if not response or not hasattr(response, "text") or not response.text:
                    continue

                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif raw_text.startswith("```"):
                    raw_text = raw_text.split("

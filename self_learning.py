"""
=============================================================================
Module      : self_learning.py
Description : IDX Stock Quantitative Analysis & Signal Generation
              Autonomous Self-Learning & Adaptive Feedback Loop Engine v2026.Q3.v16.3
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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import joblib
import numpy as np
import polars as pl
from scipy import stats
from scipy.special import erf
from scipy.stats import norm

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

def _ensure_polars_df_sl(data: Any, default_cols: Optional[List[str]] = None) -> pl.DataFrame:
    """Sanitasi input internal untuk menjamin objek bertipe pl.DataFrame."""
    if data is None:
        cols = default_cols or ["asset", "date"]
        return pl.DataFrame(schema={col: pl.Utf8 for col in cols})
    if isinstance(data, list):
        if not data:
            cols = default_cols or ["asset", "date"]
            return pl.DataFrame(schema={col: pl.Utf8 for col in cols})
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
                (pl.col("model_id") + pl.lit("_") + pl.col(asset_col).cast(pl.Utf8) + pl.lit("_") + pl.col(date_col).cast(pl.Utf8)).alias("prediction_id")
            )
        else:
            df = df.with_columns(
                (pl.lit("PRED_") + pl.int_range(0, df.height, dtype=pl.Int64).cast(pl.Utf8)).alias("prediction_id")
            )

    return df


# =============================================================================
# GEMINI SELF-LEARNING DIAGNOSTIC ENGINE
# =============================================================================

class GeminiSelfLearningDiagnosticEngine:
    """
    Sub-Engine Google Gemini AI sebagai Meta-Diagnostic & Advisory Layer untuk Self-Learning & Adaptive Loop.
    Mengevaluasi concept drift, performance decay, serta keputusan retraining model ML kuantitatif.
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

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            raw_text = response.text.strip()
            
            if raw_text.startswith("```json"):
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
                
            parsed = json.loads(raw_text)
            return {
                "ai_learning_diagnosis": parsed.get("diagnosis", "Diagnostik learning berhasil."),
                "ai_action_recommendation": parsed.get("action_recommendation", "Pertahankan performa sesuai retraining scheduler."),
                "urgency_level": parsed.get("urgency_level", "LOW")
            }
        except Exception as err:
            logger.warning(f"⚠️ [GEMINI_SELF_LEARNING_DIAGNOSIS_FAILED] Error saat memanggil Gemini API: {err}")
            return {
                "ai_learning_diagnosis": "Gagal memperoleh respon AI Gemini.",
                "ai_action_recommendation": "Ikuti protokol retraining kuantitatif bawaan.",
                "urgency_level": "MEDIUM" if retrain_manifest.get("trigger_retraining") else "LOW"
            }


# =============================================================================
# 1. CHAMPION-CHALLENGER EVALUATION ENGINE
# =============================================================================

class ChampionChallengerEngine:
    """
    Automates the statistical validation and promotion flow of machine learning models for IDX stock signals.
    Enforces a multi-layered hurdle matrix (DM-HLN statistical test with Bartlett Kernel + financial metric dominance).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_sharpe_improvement: float = 0.15,
        min_samples_required: int = 30,
    ) -> None:
        self.alpha = alpha
        self.min_sharpe_improvement = min_sharpe_improvement
        self.min_samples_required = min_samples_required
        self._lock = threading.Lock()

    def evaluate_promotion(
        self,
        challenger_id: str,
        challenger_summary: Dict[str, Any],
        champion_id: Optional[str] = None,
        champion_summary: Optional[Dict[str, Any]] = None,
        challenger_series: Optional[pl.DataFrame] = None,
        champion_series: Optional[pl.DataFrame] = None,
        join_keys: Optional[List[str]] = None,
        error_col: str = "absolute_error",
        date_col: str = "signal_date",
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        challenger_series = _ensure_polars_df_sl(challenger_series) if challenger_series is not None else None
        champion_series = _ensure_polars_df_sl(champion_series) if champion_series is not None else None

        if join_keys is not None:
            keys = join_keys
        elif challenger_series is not None and "ticker" in challenger_series.columns:
            keys = ["ticker", date_col]
        elif challenger_series is not None and "asset" in challenger_series.columns:
            keys = ["asset", date_col]
        else:
            keys = ["asset_id", date_col]
        
        logger.info("Initiating promotion evaluation for Challenger: %s vs Champion: %s", challenger_id, champion_id)

        if champion_id is None or champion_summary is None:
            logger.info("Zero active champion state detected. Automatically establishing Cold-Start promotion pathway.")
            return self._build_promotion_report(
                promoted=True,
                reason="Cold-start activation. No active champion model currently registered.",
                challenger_id=challenger_id,
                champion_id=None,
                p_value=0.0,
                t_stat=0.0,
                duration=time.time() - start_time
            )

        self._validate_summary_payloads(challenger_summary, champion_summary)

        challenger_sharpe = float(challenger_summary["sharpe_ratio"])
        champion_sharpe = float(champion_summary["sharpe_ratio"])
        sharpe_delta = challenger_sharpe - champion_sharpe

        logger.info("PIT Metrics Assessment -> Champion Sharpe: %.4f | Challenger Sharpe: %.4f", champion_sharpe, challenger_sharpe)

        if sharpe_delta < self.min_sharpe_improvement:
            return self._build_promotion_report(
                promoted=False,
                reason=f"Challenger failed minimum Sharpe hurdle gate. Delta: {sharpe_delta:.4f} < Hurdle: {self.min_sharpe_improvement:.4f}",
                challenger_id=challenger_id,
                champion_id=champion_id,
                p_value=1.0,
                t_stat=0.0,
                duration=time.time() - start_time
            )

        p_value = 1.0
        t_stat = 0.0
        
        if challenger_series is not None and champion_series is not None and challenger_series.height > 0 and champion_series.height > 0:
            try:
                t_stat, p_value = self._execute_diebold_mariano_hln_test(
                    challenger_series=challenger_series,
                    champion_series=champion_series,
                    join_keys=keys,
                    error_col=error_col
                )
                logger.info("Statistical Significance resolved (DM-HLN) -> DM-Stat: %.4f | P-Value: %.6f", t_stat, p_value)
            except Exception as err:
                logger.error("Statistical significance execution aborted. Falling back to conservative PIT rules: %s", str(err))
                return self._build_promotion_report(
                    promoted=False,
                    reason=f"Longitudinal analysis processing failure: {str(err)}",
                    challenger_id=challenger_id,
                    champion_id=champion_id,
                    p_value=1.0,
                    t_stat=0.0,
                    duration=time.time() - start_time
                )
        else:
            logger.warning("Missing raw performance time-series inputs. Promotion evaluation degraded to PIT metrics only.")
            p_value = 0.0 if sharpe_delta >= (self.min_sharpe_improvement * 2) else 1.0

        with self._lock:
            statistically_superior = (p_value < self.alpha) and (t_stat < 0)
            
            if challenger_series is None or champion_series is None or challenger_series.height == 0 or champion_series.height == 0:
                is_promoted = sharpe_delta >= (self.min_sharpe_improvement * 2)
                reason_msg = "Promoted on outsized point-in-time financial outperformance (No TS data available)." if is_promoted else "Rejected. Insufficient point-in-time margin without statistical backing series."
            else:
                is_promoted = statistically_superior and (sharpe_delta >= self.min_sharpe_improvement)
                if is_promoted:
                    reason_msg = f"Challenger demonstrated verified statistical superiority via DM-HLN (p-val: {p_value:.6f}) and satisfied alpha hurdles."
                else:
                    reason_msg = f"Rejected. Challenger lacks statistical significance dominance (p-val: {p_value:.6f}, dm-stat: {t_stat:.4f}) or Sharpe hurdle unmet."

            report = self._build_promotion_report(
                promoted=is_promoted,
                reason=reason_msg,
                challenger_id=challenger_id,
                champion_id=champion_id,
                p_value=p_value,
                t_stat=t_stat,
                duration=time.time() - start_time
            )
            
            if is_promoted:
                logger.warning("PROMOTION TRIGGERED: Model %s is replacing active production Champion %s.", challenger_id, champion_id)
            else:
                logger.info("PROMOTION DENIED: Champion %s retains operational production hierarchy.", champion_id)
                
            return report

    def _execute_diebold_mariano_hln_test(
        self,
        challenger_series: pl.DataFrame,
        champion_series: pl.DataFrame,
        join_keys: List[str],
        error_col: str,
        h: int = 1
    ) -> Tuple[float, float]:
        if challenger_series.is_empty() or champion_series.is_empty():
            raise DataValidationError("One or both time-series dataframes are empty.")

        for df_name, df in [("Challenger", challenger_series), ("Champion", champion_series)]:
            if error_col not in df.columns:
                raise DataValidationError(f"Target error metric column '{error_col}' missing from {df_name} series.")
            
            anomalies = df.select([
                pl.col(error_col).is_null().sum().alias("nulls"),
                pl.col(error_col).is_nan().sum().alias("nans"),
                pl.col(error_col).is_infinite().sum().alias("infs")
            ])
            if sum([v for v in anomalies.row(0) if v is not None]) > 0:
                raise DataValidationError(f"Numerical anomalies detected within {df_name} tracking series column '{error_col}'.")

        aligned_matrix = challenger_series.select(join_keys + [error_col]).join(
            champion_series.select(join_keys + [error_col]),
            on=join_keys,
            how="inner",
            suffix="_champion"
        )

        N = aligned_matrix.shape[0]
        if N < self.min_samples_required:
            raise DataValidationError(
                f"Aligned samples count ({N}) is below the minimum required statistical threshold ({self.min_samples_required})."
            )

        err_challenger = aligned_matrix.select(error_col).to_numpy().ravel()
        err_champion = aligned_matrix.select(f"{error_col}_champion").to_numpy().ravel()

        d = err_challenger - err_champion
        d_bar = float(np.mean(d))

        gamma_0 = float(np.var(d, ddof=0))
        var_d = gamma_0
        
        for k in range(1, h):
            weight = 1.0 - (k / float(h))
            gamma_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
            var_d += 2.0 * weight * gamma_k

        var_d = var_d / float(N)
        if var_d <= 1e-12:
            return 0.0, 1.0

        dm_stat = d_bar / np.sqrt(var_d)
        hln_factor = np.sqrt((N + 1 - 2 * h + (h / N) * (h - 1)) / N)
        dm_hln_stat = float(dm_stat * hln_factor)
        p_value = float(stats.t.cdf(dm_hln_stat, df=N - 1))

        return dm_hln_stat, p_value

    def _validate_summary_payloads(self, chal_sum: Dict[str, Any], champ_sum: Dict[str, Any]) -> None:
        required_metrics = ["sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate", "mae"]
        for metric in required_metrics:
            if metric not in chal_sum or metric not in champ_sum:
                raise DataValidationError(f"Missing essential tracking metric '{metric}' inside model performance summary payload.")
            if not isinstance(chal_sum[metric], (int, float)) or not isinstance(champ_sum[metric], (int, float)):
                raise DataValidationError(f"Non-numeric metric encountered inside tracking field '{metric}'.")

    def _build_promotion_report(
        self,
        promoted: bool,
        reason: str,
        challenger_id: str,
        champion_id: Optional[str],
        p_value: float,
        t_stat: float,
        duration: float
    ) -> Dict[str, Any]:
        return {
            "promotion_executed": promoted,
            "decision_timestamp": datetime.now(timezone.utc).isoformat(),
            "challenger_id": challenger_id,
            "champion_id": champion_id,
            "statistical_test": {
                "test_method": "Diebold-Mariano (HLN Adjusted with Bartlett Kernel)",
                "t_statistic": t_stat,
                "p_value": p_value,
                "alpha_threshold": self.alpha
            },
            "metrics_hurdle_settings": {
                "min_sharpe_improvement_required": self.min_sharpe_improvement
            },
            "execution_duration_seconds": round(duration, 4),
            "audit_justification": reason
        }


# =============================================================================
# 2. CONFIDENCE ESTIMATOR ENGINE
# =============================================================================

class ConfidenceEstimator:
    """
    Quantifies predictive uncertainty for live stock inference signals.
    Translates rolling model performance analytics into deterministic signal conviction metrics.
    """

    def __init__(
        self,
        default_confidence_level: float = 0.95,
        min_acceptable_win_rate: float = 0.48,
        epsilon: float = EPSILON,
    ) -> None:
        if not (0.0 < default_confidence_level < 1.0):
            raise DataValidationError(f"Confidence level must be bounded between 0 and 1 exclusive: {default_confidence_level}")

        self.default_confidence_level = default_confidence_level
        self.min_acceptable_win_rate = min_acceptable_win_rate
        self.epsilon = epsilon
        self._lock = threading.Lock()

    def validate_estimation_inputs(
        self,
        predictions_df: pl.DataFrame,
        performance_summary: Dict[str, Any],
        model_id: str,
        pred_col: str,
    ) -> Dict[str, Any]:
        if predictions_df.is_empty():
            raise DataValidationError("Incoming inference predictions DataFrame is completely empty.")

        expected_cols = [pred_col, "model_id"]
        missing_cols = [c for c in expected_cols if c not in predictions_df.columns]
        if missing_cols:
            raise DataValidationError(f"Predictions DataFrame is missing required columns: {missing_cols}")

        if model_id in performance_summary:
            model_meta = performance_summary[model_id]
        elif "models" in performance_summary and model_id in performance_summary["models"]:
            model_meta = performance_summary["models"][model_id]
        else:
            raise DataValidationError(f"Target model_id '{model_id}' registry key is missing from performance summary payload.")

        win_rate = model_meta.get("win_rate", model_meta.get("hit_rate"))
        if win_rate is None:
            raise DataValidationError(f"Model metadata profile for '{model_id}' lacks win_rate/hit_rate parameter.")

        error_var = model_meta.get("error_variance")
        if error_var is None:
            err_vol = model_meta.get("error_volatility", model_meta.get("rmse", 0.01))
            error_var = float(err_vol ** 2)

        mae = model_meta.get("mae", 0.0)

        numeric_cols = [c for c in predictions_df.columns if predictions_df[c].dtype.is_numeric()]
        anomalies = predictions_df.select(
            [pl.col(c).is_null().sum().alias(f"{c}_null") for c in numeric_cols] +
            [pl.col(c).is_nan().sum().alias(f"{c}_nan") for c in numeric_cols] +
            [pl.col(c).is_infinite().sum().alias(f"{c}_inf") for c in numeric_cols]
        )
        if sum([v for v in anomalies.row(0) if v is not None]) > 0:
            raise DataValidationError("Inference feature vectors contain corrupted numerical states (Null/NaN/Inf).")

        return {
            "win_rate": float(win_rate),
            "error_variance": float(error_var),
            "mae": float(mae)
        }

    def estimate_uncertainty(
        self,
        predictions_df: pl.DataFrame,
        performance_summary: Dict[str, Any],
        model_id: str,
        pred_col: str = "predicted_return",
    ) -> pl.DataFrame:
        start_time = time.time()
        predictions_df = sanitize_predictions_for_feedback(predictions_df, default_model_id=model_id)

        model_meta = self.validate_estimation_inputs(
            predictions_df=predictions_df,
            performance_summary=performance_summary,
            model_id=model_id,
            pred_col=pred_col
        )

        logger.info("Executing uncertainty quantification over %d records for model: %s", predictions_df.shape[0], model_id)

        try:
            win_rate = model_meta["win_rate"]
            error_var = model_meta["error_variance"]
            error_std = float(np.sqrt(max(error_var, self.epsilon)))

            z_score = float(norm.ppf(1.0 - (1.0 - self.default_confidence_level) / 2.0))
            critical_margin = float(z_score * error_std)

            pred_series = predictions_df.select(pl.col(pred_col)).to_series().to_numpy()
            z_signals = np.abs(pred_series) / error_std
            gaussian_cdfs = 0.5 * (1.0 + erf(z_signals / np.sqrt(2.0)))
            magnitude_convictions = 2.0 * gaussian_cdfs - 1.0

            if win_rate >= self.min_acceptable_win_rate:
                confidence_scores = (magnitude_convictions * 0.4) + (win_rate * 0.6)
            else:
                confidence_scores = ((magnitude_convictions * 0.2) + (win_rate * 0.8)) * (win_rate / self.min_acceptable_win_rate)

            confidence_scores = np.clip(confidence_scores, 0.0, 1.0)

            output_df = predictions_df.with_columns([
                pl.Series("confidence_score", confidence_scores),
                (pl.col(pred_col) - critical_margin).alias("lower_bound"),
                (pl.col(pred_col) + critical_margin).alias("upper_bound")
            ])

            with self._lock:
                execution_duration = time.time() - start_time
                mean_confidence = float(output_df.select(pl.col("confidence_score").mean()).item() or 0.0)
                logger.info("Uncertainty estimation resolved in %.4f seconds. Mean Confidence Score: %.4f", execution_duration, mean_confidence)
                return output_df

        except DataValidationError:
            raise
        except Exception as err:
            logger.critical("Critical operational error occurred within predictive uncertainty loops: %s", str(err))
            raise ConfidenceEstimatorError(f"Uncertainty quantification pipeline aborted: {str(err)}") from err


# =============================================================================
# 3. DRIFT DETECTOR ENGINE
# =============================================================================

class DriftDetector:
    """
    Monitors data distribution variations across historical profiles and recent stock feature streams.
    Provides automated quantitative triggers using PSI and KS-tests.
    """

    def __init__(
        self,
        psi_warning_threshold: float = 0.10,
        psi_action_threshold: float = 0.25,
        ks_alpha: float = 0.05,
        num_bins: int = 10,
        epsilon: float = 1e-4,
    ) -> None:
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_action_threshold = psi_action_threshold
        self.ks_alpha = ks_alpha
        self.num_bins = num_bins
        self.epsilon = epsilon
        self._lock = threading.Lock()

    def validate_drift_inputs(
        self,
        baseline_df: pl.DataFrame,
        target_df: pl.DataFrame,
        feature_cols: List[str],
    ) -> None:
        if baseline_df.is_empty():
            raise DataValidationError("Baseline reference dataset is completely empty.")
        if target_df.is_empty():
            raise DataValidationError("Target assessment batch dataset is completely empty.")
        if not feature_cols:
            raise DataValidationError("Feature columns selection list cannot be empty.")

        for name, df in [("Baseline", baseline_df), ("Target", target_df)]:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise DataValidationError(f"Dataset {name} is missing expected feature paths: {missing}")

        numeric_cols = [c for c in feature_cols if baseline_df[c].dtype.is_numeric()]
        if len(numeric_cols) != len(feature_cols):
            non_numeric = [c for c in feature_cols if c not in numeric_cols]
            raise DataValidationError(f"Drift analysis requires continuous numeric features. Non-numeric columns: {non_numeric}")

        for name, df in [("Baseline", baseline_df), ("Target", target_df)]:
            anomalies = df.select(
                [pl.col(c).is_null().sum().alias(f"{c}_null") for c in numeric_cols] +
                [pl.col(c).is_nan().sum().alias(f"{c}_nan") for c in numeric_cols] +
                [pl.col(c).is_infinite().sum().alias(f"{c}_inf") for c in numeric_cols]
            )
            if sum([v for v in anomalies.row(0) if v is not None]) > 0:
                raise DataValidationError(f"Dataset {name} feature matrix contains corrupted states (Null/NaN/Inf).")

    def analyze_feature_drift(
        self,
        baseline_df: pl.DataFrame,
        target_df: pl.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        start_time = time.time()
        baseline_df = _ensure_polars_df_sl(baseline_df)
        target_df = _ensure_polars_df_sl(target_df)

        if baseline_df.height == 0 or target_df.height == 0 or not feature_cols:
            return {
                "monitoring_timestamp": datetime.now(timezone.utc).isoformat(),
                "summary_metrics": {
                    "total_features_monitored": 0,
                    "warning_features_count": 0,
                    "action_features_count": 0,
                    "system_retrain_recommended": False
                },
                "features_drift_logs": {},
                "execution_time_seconds": 0.0
            }

        self.validate_drift_inputs(baseline_df, target_df, feature_cols)

        logger.info("Initiating drift detection audit across %d features.", len(feature_cols))

        drift_audit_registry: Dict[str, Any] = {}
        global_drift_triggered = False
        warning_features_count = 0
        action_features_count = 0

        try:
            for feature in feature_cols:
                arr_base = baseline_df.select(feature).to_numpy().ravel()
                arr_target = target_df.select(feature).to_numpy().ravel()

                base_std = float(np.std(arr_base))
                target_std = float(np.std(arr_target))
                
                if base_std <= 1e-8 and target_std <= 1e-8:
                    drift_audit_registry[feature] = {
                        "psi_value": 0.0, "ks_statistic": 0.0, "ks_p_value": 1.0,
                        "drift_status": "STABLE", "is_statistically_significant": False
                    }
                    continue

                psi_value = self._compute_vectorized_psi(arr_base, arr_target)

                ks_stat, ks_p_val = stats.ks_2samp(arr_base, arr_target, method="auto")
                ks_p_val = float(ks_p_val) if not np.isnan(ks_p_val) else 1.0
                ks_stat = float(ks_stat) if not np.isnan(ks_stat) else 0.0

                is_ks_significant = bool(ks_p_val < self.ks_alpha)

                if psi_value >= self.psi_action_threshold:
                    status = "CRITICAL_DRIFT"
                    action_features_count += 1
                    if is_ks_significant:
                        global_drift_triggered = True
                elif psi_value >= self.psi_warning_threshold:
                    status = "WARNING_DRIFT"
                    warning_features_count += 1
                else:
                    status = "STABLE"

                drift_audit_registry[feature] = {
                    "psi_value": round(psi_value, 6),
                    "ks_statistic": round(ks_stat, 6),
                    "ks_p_value": round(ks_p_val, 6),
                    "drift_status": status,
                    "is_statistically_significant": is_ks_significant
                }

            with self._lock:
                execution_duration = time.time() - start_time
                return {
                    "monitoring_timestamp": datetime.now(timezone.utc).isoformat(),
                    "summary_metrics": {
                        "total_features_monitored": len(feature_cols),
                        "warning_features_count": warning_features_count,
                        "action_features_count": action_features_count,
                        "system_retrain_recommended": global_drift_triggered or (action_features_count > 0)
                    },
                    "features_drift_logs": drift_audit_registry,
                    "execution_time_seconds": round(execution_duration, 4)
                }

        except DataValidationError:
            raise
        except Exception as err:
            logger.critical("Critical operational error occurred within drift detection loops: %s", str(err))
            raise DriftDetectorError(f"Drift monitoring pipeline execution aborted: {str(err)}") from err

    def _compute_vectorized_psi(self, baseline: np.ndarray, target: np.ndarray) -> float:
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        bins = np.percentile(baseline, percentiles)
        bins = np.unique(bins)
        
        if len(bins) < 2:
            bins = np.linspace(np.min(baseline), np.max(baseline), self.num_bins + 1)
            bins = np.unique(bins)
            if len(bins) < 2:
                return 0.0

        bins[0] = -np.inf
        bins[-1] = np.inf

        base_counts, _ = np.histogram(baseline, bins=bins)
        target_counts, _ = np.histogram(target, bins=bins)

        base_pcts = base_counts / float(len(baseline))
        target_pcts = target_counts / float(len(target))

        base_pcts = np.where(base_pcts == 0, self.epsilon, base_pcts)
        target_pcts = np.where(target_pcts == 0, self.epsilon, target_pcts)

        base_pcts /= np.sum(base_pcts)
        target_pcts /= np.sum(target_pcts)

        psi_vector = (target_pcts - base_pcts) * np.log(target_pcts / base_pcts)
        return float(np.sum(psi_vector))


# =============================================================================
# 4. FEEDBACK ENGINE
# =============================================================================

class FeedbackEngine:
    """
    Executes deterministic processing of historical stock signals against realized market metrics.
    Acts as the single source of truth for generating performance telemetry.
    """

    def __init__(self, execution_date: Optional[Union[str, date, datetime]] = None) -> None:
        if execution_date is None:
            self.execution_date = date.today()
        elif isinstance(execution_date, str):
            self.execution_date = datetime.strptime(execution_date, "%Y-%m-%d").date()
        elif isinstance(execution_date, datetime):
            self.execution_date = execution_date.date()
        else:
            self.execution_date = execution_date

    def validate_feedback_inputs(
        self,
        predictions: pl.DataFrame,
        actuals: pl.DataFrame,
        prediction_schema: List[str],
        actuals_schema: List[str],
        asset_keys: List[str],
    ) -> None:
        if predictions.is_empty():
            raise DataValidationError("Historical predictions DataFrame is empty.")
        if actuals.is_empty():
            raise DataValidationError("Realized market actuals DataFrame is empty.")

        missing_pred = [col for col in prediction_schema if col not in predictions.columns]
        if missing_pred:
            raise DataValidationError(f"Missing required columns in predictions: {missing_pred}")

        missing_act = [col for col in actuals_schema if col not in actuals.columns]
        if missing_act:
            raise DataValidationError(f"Missing required columns in actuals: {missing_act}")

        for key in asset_keys:
            if key not in predictions.columns or key not in actuals.columns:
                raise DataValidationError(f"Specified asset join key '{key}' is missing from input DataFrames.")

        numeric_pred_cols = [c for c in prediction_schema if predictions[c].dtype.is_numeric()]
        numeric_act_cols = [c for c in actuals_schema if actuals[c].dtype.is_numeric()]

        pred_anomalies = predictions.select(
            [pl.col(c).is_null().sum().alias(f"{c}_null") for c in numeric_pred_cols] +
            [pl.col(c).is_nan().sum().alias(f"{c}_nan") for c in numeric_pred_cols] +
            [pl.col(c).is_infinite().sum().alias(f"{c}_inf") for c in numeric_pred_cols]
        )
        
        act_anomalies = actuals.select(
            [pl.col(c).is_null().sum().alias(f"{c}_null") for c in numeric_act_cols] +
            [pl.col(c).is_nan().sum().alias(f"{c}_nan") for c in numeric_act_cols] +
            [pl.col(c).is_infinite().sum().alias(f"{c}_inf") for c in numeric_act_cols]
        )

        if sum([v for v in pred_anomalies.row(0) if v is not None]) > 0:
            raise DataValidationError("Numerical anomalies (NaN/Null/Inf) detected in prediction dataset.")
        
        if sum([v for v in act_anomalies.row(0) if v is not None]) > 0:
            raise DataValidationError("Numerical anomalies (NaN/Null/Inf) detected in actuals dataset.")

    def process_feedback_loop(
        self,
        predictions: pl.DataFrame,
        actuals: pl.DataFrame,
        processed_prediction_ids: Set[str],
        prediction_id_col: str = "prediction_id",
        model_id_col: str = "model_id",
        signal_date_col: str = "signal_date",
        outcome_date_col: str = "outcome_date",
        pred_value_col: str = "predicted_return",
        realized_value_col: str = "realized_return",
        join_keys: Optional[List[str]] = None,
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        start_time = time.time()
        
        predictions = sanitize_predictions_for_feedback(predictions)
        actuals = _ensure_polars_df_sl(actuals)

        if predictions.is_empty() or actuals.is_empty():
            return pl.DataFrame(), {"global_processed_count": 0, "models": {}}

        if join_keys is not None:
            asset_keys = [k for k in join_keys if k not in [signal_date_col, outcome_date_col, "date"]]
        elif "ticker" in predictions.columns and "ticker" in actuals.columns:
            asset_keys = ["ticker"]
        elif "asset" in predictions.columns and "asset" in actuals.columns:
            asset_keys = ["asset"]
        else:
            asset_keys = ["asset"] if "asset" in predictions.columns else ["ticker"]

        pred_schema = [prediction_id_col, model_id_col, signal_date_col, pred_value_col] + asset_keys
        act_schema = [realized_value_col] + asset_keys

        self.validate_feedback_inputs(
            predictions=predictions,
            actuals=actuals,
            prediction_schema=pred_schema,
            actuals_schema=act_schema,
            asset_keys=asset_keys
        )

        filtered_preds = predictions.filter(~pl.col(prediction_id_col).is_in(list(processed_prediction_ids)))
        
        if filtered_preds.select(pl.col(prediction_id_col).is_duplicated().any()).item():
            filtered_preds = filtered_preds.unique(subset=[prediction_id_col], keep="first")

        if filtered_preds.is_empty():
            return pl.DataFrame(), {"global_processed_count": 0, "models": {}}

        actuals_date_col = "date" if "date" in actuals.columns else outcome_date_col
        pred_target_col = outcome_date_col if outcome_date_col in filtered_preds.columns else signal_date_col

        join_on_pred = asset_keys + [pred_target_col]
        join_on_act = asset_keys + [actuals_date_col]

        feedback_matrix = filtered_preds.join(
            actuals,
            left_on=join_on_pred,
            right_on=join_on_act,
            how="inner"
        )

        if feedback_matrix.is_empty():
            return pl.DataFrame(), {"global_processed_count": 0, "models": {}}

        if outcome_date_col in feedback_matrix.columns and signal_date_col in feedback_matrix.columns:
            cast_exprs = []
            if feedback_matrix[signal_date_col].dtype == pl.Utf8:
                cast_exprs.append(pl.col(signal_date_col).str.to_date(strict=False))
            if feedback_matrix[outcome_date_col].dtype == pl.Utf8:
                cast_exprs.append(pl.col(outcome_date_col).str.to_date(strict=False))

            if cast_exprs:
                feedback_matrix = feedback_matrix.with_columns(cast_exprs)

            chronology_violation = feedback_matrix.filter(
                pl.col(outcome_date_col) <= pl.col(signal_date_col)
            )
            if not chronology_violation.is_empty():
                invalid_ids = chronology_violation.select(prediction_id_col).unique().to_series().to_list()
                raise ChronologyError(f"Data leakage detected! Outcome date occurs prior to or on signal date for IDs: {invalid_ids}")

            exec_date_lit = pl.lit(self.execution_date).cast(pl.Date)
            future_leakage = feedback_matrix.filter(
                pl.col(outcome_date_col) > exec_date_lit
            )
            if not future_leakage.is_empty():
                invalid_future_ids = future_leakage.select(prediction_id_col).unique().to_series().to_list()
                raise ChronologyError(f"Look-ahead violation! Outcome date extends into the future relative to execution anchor: {invalid_future_ids}")

        feedback_matrix = feedback_matrix.with_columns([
            (pl.col(realized_value_col) - pl.col(pred_value_col)).alias("residual_error"),
            (pl.col(realized_value_col) - pl.col(pred_value_col)).abs().alias("absolute_error"),
            ((pl.col(realized_value_col) - pl.col(pred_value_col)) ** 2).alias("squared_error"),
            (((pl.col(pred_value_col) >= 0) & (pl.col(realized_value_col) >= 0)) |
             ((pl.col(pred_value_col) < 0) & (pl.col(realized_value_col) < 0))).cast(pl.Int8).alias("directional_hit"),
            pl.lit(datetime.now(timezone.utc).isoformat()).alias("processed_timestamp")
        ])

        summary_payload = self._generate_downstream_telemetry(
            feedback_matrix=feedback_matrix,
            model_id_col=model_id_col,
            prediction_id_col=prediction_id_col
        )

        execution_duration = time.time() - start_time
        logger.info("Feedback loop resolved. Processed Records: %d. Duration: %.4f seconds.", feedback_matrix.shape[0], execution_duration)

        return feedback_matrix, summary_payload

    def _generate_downstream_telemetry(
        self,
        feedback_matrix: pl.DataFrame,
        model_id_col: str,
        prediction_id_col: str
    ) -> Dict[str, Any]:
        metrics_by_model = feedback_matrix.group_by(model_id_col).agg([
            pl.col(prediction_id_col).count().alias("sample_count"),
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("squared_error").mean().sqrt().alias("rmse"),
            pl.col("directional_hit").mean().alias("hit_rate"),
            pl.col("residual_error").std().alias("error_volatility")
        ])

        summary: Dict[str, Any] = {
            "global_processed_count": feedback_matrix.shape[0],
            "models": {}
        }

        for row in metrics_by_model.iter_rows(named=True):
            mid = str(row[model_id_col])
            hit_rate = float(row["hit_rate"]) if row["hit_rate"] is not None else 0.0
            err_vol = float(row["error_volatility"]) if row["error_volatility"] is not None else 0.01

            summary["models"][mid] = {
                "sample_count": row["sample_count"],
                "mae": float(row["mae"]) if row["mae"] is not None else 0.0,
                "rmse": float(row["rmse"]) if row["rmse"] is not None else 0.0,
                "hit_rate": hit_rate,
                "win_rate": hit_rate,
                "error_volatility": err_vol,
                "error_variance": float(err_vol ** 2),
                "requires_retrain_alert": bool(hit_rate < 0.48)
            }

        return summary


# =============================================================================
# 5. HYPERPARAMETER OPTIMIZER ENGINE
# =============================================================================

class HyperparameterOptimizer:
    """
    Executes thread-safe, deterministic, and resource-lean hyperparameter tuning for stock predictive models.
    Persists optimization histories directly to SQLite.
    """

    def __init__(
        self,
        storage_uri: str = "sqlite:///optuna_study.db",
        seed: int = 42,
        epsilon: float = EPSILON,
    ) -> None:
        self.storage_uri = storage_uri
        self.seed = seed
        self.epsilon = epsilon
        self._lock = threading.Lock()

    def validate_optimization_datasets(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str,
        date_col: Optional[str] = "date",
    ) -> None:
        if train_df.is_empty() or val_df.is_empty():
            raise DataValidationError("Training or validation dataset is completely empty.")

        required_cols = feature_cols + [target_col]
        if date_col and date_col in train_df.columns:
            required_cols.append(date_col)

        for name, df in [("Train", train_df), ("Validation", val_df)]:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise DataValidationError(f"Dataset {name} is missing required columns: {missing}")

        numeric_cols = [c for c in feature_cols + [target_col] if train_df[c].dtype.is_numeric()]
        for name, df in [("Train", train_df), ("Validation", val_df)]:
            anomalies = df.select(
                [pl.col(c).is_null().sum().alias(f"{c}_null") for c in numeric_cols] +
                [pl.col(c).is_nan().sum().alias(f"{c}_nan") for c in numeric_cols] +
                [pl.col(c).is_infinite().sum().alias(f"{c}_inf") for c in numeric_cols]
            )
            if sum([v for v in anomalies.row(0) if v is not None]) > 0:
                raise DataValidationError(f"Dataset {name} contains corrupted data points matching Null, NaN, or Inf.")

        if date_col and date_col in train_df.columns and date_col in val_df.columns:
            max_train_date = train_df.select(pl.col(date_col).max()).item()
            min_val_date = val_df.select(pl.col(date_col).min()).item()

            if max_train_date is not None and min_val_date is not None:
                if min_val_date < max_train_date:
                    raise DataValidationError(
                        f"Temporal leak detected! Validation start boundary ({min_val_date}) overlaps training historical sequence end ({max_train_date})."
                    )

    def optimize_gbdt(
        self,
        model_type: str,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_trials: int = 15,
        timeout: Optional[int] = 300,
        date_col: Optional[str] = "date",
    ) -> Tuple[Dict[str, Any], float]:
        if not OPTUNA_AVAILABLE:
            raise HyperparameterOptimizerError("Optuna library is required for hyperparameter tuning.")

        normalized_type = model_type.strip().lower()
        if normalized_type not in ["lightgbm", "xgboost", "catboost"]:
            raise DataValidationError(f"Unsupported model architecture type token: {model_type}")

        train_df = _ensure_polars_df_sl(train_df)
        val_df = _ensure_polars_df_sl(val_df)

        self.validate_optimization_datasets(
            train_df=train_df, val_df=val_df, feature_cols=feature_cols, target_col=target_col, date_col=date_col
        )

        start_time = time.time()
        
        x_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df.select(target_col).to_numpy().ravel()
        x_val = val_df.select(feature_cols).to_numpy()
        y_val = val_df.select(target_col).to_numpy().ravel()

        study_name = f"idx_study_{normalized_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        try:
            with self._lock:
                sampler = optuna.samplers.TPESampler(seed=self.seed)
                pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

                study = optuna.create_study(
                    study_name=study_name,
                    storage=self.storage_uri,
                    direction="minimize",
                    sampler=sampler,
                    pruner=pruner,
                    load_if_exists=True,
                )

            def objective_wrapper(trial: optuna.Trial) -> float:
                return self._objective(trial, normalized_type, x_train, y_train, x_val, y_val)

            logger.info("Starting hyperparameter tuning loop for %s. Limit: %d trials.", normalized_type, n_trials)
            study.optimize(objective_wrapper, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

            best_params = study.best_params
            best_score = study.best_value

            logger.info("Optimization loop resolved successfully for %s in %.4f seconds. Best RMSE: %.6f", normalized_type, time.time() - start_time, best_score)
            return best_params, best_score

        except DataValidationError:
            raise
        except Exception as err:
            logger.critical("Catastrophic error encountered inside hyperparameter optimization: %s", str(err))
            raise HyperparameterOptimizerError(f"Hyperparameter optimization cycle aborted: {str(err)}") from err

    def _objective(
        self,
        trial: optuna.Trial,
        model_type: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        if model_type == "lightgbm" and LGB_AVAILABLE:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "seed": self.seed,
                "deterministic": True,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": 1,
                "verbose": -1,
            }
            train_set = lgb.Dataset(x_train, label=y_train)
            val_set = lgb.Dataset(x_val, label=y_val, reference=train_set)

            callbacks = [lgb.early_stopping(stopping_rounds=15, verbose=False)]
            model = lgb.train(
                params=params,
                train_set=train_set,
                num_boost_round=150,
                valid_sets=[val_set],
                callbacks=callbacks
            )
            preds = model.predict(x_val)

        elif model_type == "xgboost" and XGB_AVAILABLE:
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "random_state": self.seed,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "verbosity": 0,
            }
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dval = xgb.DMatrix(x_val, label=y_val)
            
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=150,
                evals=[(dval, "rmse")],
                early_stopping_rounds=15,
                verbose_eval=False,
            )
            preds = model.predict(dval)

        elif model_type == "catboost" and CB_AVAILABLE:
            params = {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": self.seed,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
                "verbose": False,
            }
            model = cb.CatBoostRegressor(**params, iterations=150)
            model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=15, verbose=False)
            preds = model.predict(x_val)

        else:
            raise HyperparameterOptimizerError(f"Engine framework {model_type} is unavailable or unsupported.")

        score = float(np.sqrt(np.mean((y_val - preds) ** 2)))
        if np.isnan(score) or np.isinf(score):
            return float("inf")
            
        return score


# =============================================================================
# 6. MODEL RANKER ENGINE
# =============================================================================

class ModelRanker:
    """
    Analyzes historical cross-sectional rankings and rank volatility across the stock model population.
    Prevents selection bias by ensuring candidates exhibit performance persistence.
    """

    def __init__(
        self,
        metric_weights: Optional[Dict[str, float]] = None,
        stability_threshold: float = 1.5,
        epsilon: float = EPSILON,
    ) -> None:
        self.stability_threshold = stability_threshold
        self.epsilon = epsilon
        self._lock = threading.Lock()

        if metric_weights is None:
            self.metric_weights = {
                "sharpe_ratio": 0.40,
                "sortino_ratio": 0.30,
                "win_rate": 0.15,
                "max_drawdown": 0.10,
                "mae": -0.05,
            }
        else:
            self._validate_weights(metric_weights)
            self.metric_weights = metric_weights

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        required_keys = ["sharpe_ratio", "sortino_ratio", "win_rate", "max_drawdown", "mae"]
        missing = [k for k in required_keys if k not in weights]
        if missing:
            raise DataValidationError(f"Missing core metric weights parameters in ranker: {missing}")

    def analyze_rank_persistence(
        self,
        historical_metrics_df: pl.DataFrame,
        model_id_col: str = "model_id",
        date_col: str = "signal_date",
    ) -> Dict[str, Any]:
        start_time = time.time()
        historical_metrics_df = _ensure_polars_df_sl(historical_metrics_df)

        if historical_metrics_df.is_empty():
            return {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_observed_models": 0,
                "total_historical_periods": 0,
                "persistence_cohort_audit": [],
                "execution_time_seconds": 0.0
            }

        expected_fields = [model_id_col, date_col, "sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate", "mae"]
        missing_fields = [f for f in expected_fields if f not in historical_metrics_df.columns]
        if missing_fields:
            raise DataValidationError(f"Missing required columns in historical metrics dataframe: {missing_fields}")

        numeric_cols = ["sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate", "mae"]
        anomalies = historical_metrics_df.select(
            [pl.col(c).is_null().sum().alias(f"{c}_null") for c in numeric_cols] +
            [pl.col(c).is_nan().sum().alias(f"{c}_nan") for c in numeric_cols] +
            [pl.col(c).is_infinite().sum().alias(f"{c}_inf") for c in numeric_cols]
        )
        if sum([v for v in anomalies.row(0) if v is not None]) > 0:
            raise DataValidationError("Historical metrics matrix contains corrupted numerical states (Null/NaN/Inf).")

        try:
            scored_df = self._compute_cross_sectional_scores(historical_metrics_df, date_col)

            ranked_df = scored_df.with_columns(
                pl.col("max_drawdown").abs().alias("_abs_drawdown")
            ).sort(
                [date_col, "composite_score", "sharpe_ratio", "_abs_drawdown", "mae", model_id_col],
                descending=[False, True, True, False, False, False]
            ).with_columns(
                pl.int_range(1, pl.len() + 1).over(date_col).alias("daily_rank")
            ).drop("_abs_drawdown")

            unique_dates_count = ranked_df.select(pl.col(date_col).n_unique()).item()

            stability_df = ranked_df.group_by(model_id_col).agg([
                pl.col("daily_rank").mean().alias("mean_rank"),
                pl.col("daily_rank").std().fill_null(0.0).alias("rank_volatility"),
                pl.col("daily_rank").first().alias("_initial_rank"),
                pl.col("daily_rank").last().alias("_latest_rank"),
                pl.col("composite_score").last().alias("latest_composite_score")
            ]).with_columns(
                (pl.col("_latest_rank").cast(pl.Float32) - pl.col("_initial_rank").cast(pl.Float32)).alias("rank_drift")
            )

            total_models = stability_df.shape[0]
            final_ranked_df = stability_df.sort(
                by=["latest_composite_score", "mean_rank", "rank_volatility", model_id_col],
                descending=[True, False, False, False]
            ).with_columns(
                pl.int_range(1, pl.len() + 1).alias("final_persistence_rank")
            ).with_columns(
                (pl.col("final_persistence_rank") / total_models).alias("_percentile")
            ).with_columns(
                pl.when(pl.col("_percentile") <= 0.25).then(pl.lit("TIER_1_ELITE"))
                .when(pl.col("_percentile") <= 0.50).then(pl.lit("TIER_2_ALPHA"))
                .when(pl.col("_percentile") <= 0.75).then(pl.lit("TIER_3_BENCHMARK"))
                .otherwise(pl.lit("TIER_4_SUBSTANDARD"))
                .alias("performance_tier")
            ).drop(["_initial_rank", "_latest_rank", "_percentile"])

            with self._lock:
                cohort_rankings = []
                for row in final_ranked_df.iter_rows(named=True):
                    cohort_rankings.append({
                        "persistence_rank": row["final_persistence_rank"],
                        "model_id": row[model_id_col],
                        "mean_rank": round(row["mean_rank"], 2),
                        "rank_volatility": round(row["rank_volatility"], 4),
                        "rank_drift": int(row["rank_drift"]),
                        "performance_tier": row["performance_tier"],
                        "is_volatile": bool(row["rank_volatility"] > self.stability_threshold)
                    })

                execution_duration = time.time() - start_time
                logger.info("Longitudinal rank persistence resolved over %d unique tracks.", total_models)

                return {
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_observed_models": total_models,
                    "total_historical_periods": unique_dates_count,
                    "persistence_cohort_audit": cohort_rankings,
                    "execution_time_seconds": round(execution_duration, 4)
                }

        except DataValidationError:
            raise
        except Exception as err:
            logger.critical("Critical error encountered inside model ranker processing loop: %s", str(err))
            raise ModelRankerError(f"Longitudinal ranking execution cycle aborted: {str(err)}") from err

    def _compute_cross_sectional_scores(self, df: pl.DataFrame, date_col: str) -> pl.DataFrame:
        bounds_df = df.group_by(date_col).agg([
            pl.col("sharpe_ratio").min().alias("sharpe_min"), pl.col("sharpe_ratio").max().alias("sharpe_max"),
            pl.col("sortino_ratio").min().alias("sortino_min"), pl.col("sortino_ratio").max().alias("sortino_max"),
            pl.col("win_rate").min().alias("win_rate_min"), pl.col("win_rate").max().alias("win_rate_max"),
            pl.col("max_drawdown").abs().min().alias("drawdown_min"), pl.col("max_drawdown").abs().max().alias("drawdown_max"),
            pl.col("mae").min().alias("mae_min"), pl.col("mae").max().alias("mae_max")
        ])

        joined_df = df.join(bounds_df, on=date_col, how="inner")

        def norm_expr(col_expr: pl.Expr, min_c: str, max_c: str) -> pl.Expr:
            diff = pl.col(max_c) - pl.col(min_c)
            return pl.when(diff.abs() <= self.epsilon).then(pl.lit(1.0)).otherwise((col_expr - pl.col(min_c)) / diff)

        scored_df = joined_df.with_columns([
            norm_expr(pl.col("sharpe_ratio"), "sharpe_min", "sharpe_max").alias("_n_sharpe"),
            norm_expr(pl.col("sortino_ratio"), "sortino_min", "sortino_max").alias("_n_sortino"),
            norm_expr(pl.col("win_rate"), "win_rate_min", "win_rate_max").alias("_n_win_rate"),
            (1.0 - norm_expr(pl.col("max_drawdown").abs(), "drawdown_min", "drawdown_max")).alias("_n_drawdown"),
            (1.0 - norm_expr(pl.col("mae"), "mae_min", "mae_max")).alias("_n_mae")
        ])

        w = self.metric_weights
        composite_expr = (
            (pl.col("_n_sharpe") * w["sharpe_ratio"]) +
            (pl.col("_n_sortino") * w["sortino_ratio"]) +
            (pl.col("_n_win_rate") * w["win_rate"]) +
            (pl.col("_n_drawdown") * w["max_drawdown"]) +
            (pl.col("_n_mae") * abs(w["mae"]))
        ).alias("composite_score")

        return scored_df.with_columns(composite_expr)


# =============================================================================
# 7. MODEL SELECTOR ENGINE
# =============================================================================

class ModelSelector:
    """Ranks, filters, and selects the optimal stock predictive model from active and candidate model pools."""

    def __init__(
        self,
        metric_weights: Optional[Dict[str, float]] = None,
        min_win_rate: float = 0.48,
        max_drawdown_limit: float = -0.20,
        epsilon: float = EPSILON,
    ) -> None:
        self.min_win_rate = min_win_rate
        self.max_drawdown_limit = max_drawdown_limit
        self.epsilon = epsilon
        self._lock = threading.Lock()

        if metric_weights is None:
            self.metric_weights = {
                "sharpe_ratio": 0.40,
                "sortino_ratio": 0.30,
                "win_rate": 0.15,
                "max_drawdown": 0.10,
                "mae": -0.05, 
            }
        else:
            self._validate_weights(metric_weights)
            self.metric_weights = metric_weights

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        required_keys = ["sharpe_ratio", "sortino_ratio", "win_rate", "max_drawdown", "mae"]
        missing = [k for k in required_keys if k not in weights]
        if missing:
            raise DataValidationError(f"Missing core metric weights parameters: {missing}")

    def select_best_model(self, metrics_payload: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        start_time = time.time()
        if not metrics_payload:
            return {
                "selection_executed_timestamp": datetime.now(timezone.utc).isoformat(),
                "selection_mode": "EMPTY",
                "recommended_model_id": "champion_idx_v1",
                "is_recommended_model_safe": True,
                "metrics_snapshot": {},
                "full_rankings_audit": [],
                "execution_time_seconds": 0.0
            }

        raw_records = []
        for model_id, core_metrics in metrics_payload.items():
            record = {"model_id": str(model_id)}
            for metric_key, val in core_metrics.items():
                if metric_key != "last_updated_metric_date":
                    float_val = float(val)
                    if np.isnan(float_val) or np.isinf(float_val):
                        float_val = 0.0
                    record[metric_key] = float_val
            raw_records.append(record)

        try:
            pool_df = pl.DataFrame(raw_records)
            expected_fields = ["model_id", "sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate", "mae"]
            missing_fields = [f for f in expected_fields if f not in pool_df.columns]
            if missing_fields:
                raise DataValidationError(f"Missing core metric fields in converted dataframe: {missing_fields}")

            filtered_df = pool_df.with_columns([
                (pl.col("win_rate") >= self.min_win_rate).alias("_pass_win_rate"),
                (pl.col("max_drawdown") >= self.max_drawdown_limit).alias("_pass_drawdown")
            ]).with_columns([
                (pl.col("_pass_win_rate") & pl.col("_pass_drawdown")).alias("is_eligible")
            ])

            eligible_count = filtered_df.filter(pl.col("is_eligible")).shape[0]
            
            selection_mode = "STRICT"
            if eligible_count == 0:
                selection_mode = "FALLBACK"
                logger.warning("Zero models survived risk filters! Forcing allocation fallback track onto full model space.")
                target_df = filtered_df.with_columns(pl.lit(True).alias("is_eligible"))
            else:
                target_df = filtered_df

            scored_df = self._execute_vectorized_scoring(target_df)

            final_ranked_df = scored_df.with_columns(
                pl.col("max_drawdown").abs().alias("_abs_drawdown")
            ).sort(
                by=["composite_score", "sharpe_ratio", "_abs_drawdown", "mae", "model_id"],
                descending=[True, True, False, False, False]
            ).drop("_abs_drawdown")

            if final_ranked_df.is_empty():
                raise ModelSelectorError("Model selection produced empty ranked output.")

            with self._lock:
                top_model_row = final_ranked_df.row(0, named=True)
                rankings_list = [
                    {
                        "rank": idx + 1, 
                        "model_id": row["model_id"],
                        "composite_score": round(row["composite_score"], 6),
                        "is_risk_eligible": row["is_eligible"], 
                        "underlying_sharpe": row["sharpe_ratio"]
                    }
                    for idx, row in enumerate(final_ranked_df.iter_rows(named=True))
                ]

                logger.info("Optimal model selection resolved. Selected Champion: %s", top_model_row["model_id"])

                return {
                    "selection_executed_timestamp": datetime.now(timezone.utc).isoformat(),
                    "selection_mode": selection_mode,
                    "recommended_model_id": top_model_row["model_id"],
                    "is_recommended_model_safe": top_model_row["is_eligible"],
                    "metrics_snapshot": {
                        "sharpe_ratio": top_model_row["sharpe_ratio"], 
                        "sortino_ratio": top_model_row["sortino_ratio"],
                        "max_drawdown": top_model_row["max_drawdown"], 
                        "win_rate": top_model_row["win_rate"], 
                        "mae": top_model_row["mae"]
                    },
                    "full_rankings_audit": rankings_list,
                    "execution_time_seconds": round(time.time() - start_time, 4)
                }

        except Exception as err:
            logger.critical("Critical error occurred within model selection engine: %s", str(err))
            raise ModelSelectorError(f"Model selection processing cycle aborted: {str(err)}") from err

    def _execute_vectorized_scoring(self, df: pl.DataFrame) -> pl.DataFrame:
        bounds = df.select([
            pl.col("sharpe_ratio").min().alias("sharpe_min"), pl.col("sharpe_ratio").max().alias("sharpe_max"),
            pl.col("sortino_ratio").min().alias("sortino_min"), pl.col("sortino_ratio").max().alias("sortino_max"),
            pl.col("win_rate").min().alias("win_rate_min"), pl.col("win_rate").max().alias("win_rate_max"),
            pl.col("max_drawdown").abs().min().alias("drawdown_min"), pl.col("max_drawdown").abs().max().alias("drawdown_max"),
            pl.col("mae").min().alias("mae_min"), pl.col("mae").max().alias("mae_max")
        ]).row(0, named=True)

        def get_norm_expr(col_expr: pl.Expr, min_val: float, max_val: float) -> pl.Expr:
            diff = max_val - min_val
            if abs(diff) <= self.epsilon:
                return pl.lit(1.0)
            return (col_expr - min_val) / diff

        scored_df = df.with_columns([
            get_norm_expr(pl.col("sharpe_ratio"), bounds["sharpe_min"], bounds["sharpe_max"]).alias("_n_sharpe"),
            get_norm_expr(pl.col("sortino_ratio"), bounds["sortino_min"], bounds["sortino_max"]).alias("_n_sortino"),
            get_norm_expr(pl.col("win_rate"), bounds["win_rate_min"], bounds["win_rate_max"]).alias("_n_win_rate"),
            (1.0 - get_norm_expr(pl.col("max_drawdown").abs(), bounds["drawdown_min"], bounds["drawdown_max"])).alias("_n_drawdown"),
            (1.0 - get_norm_expr(pl.col("mae"), bounds["mae_min"], bounds["mae_max"])).alias("_n_mae")
        ])

        w = self.metric_weights
        composite_expr = (
            (pl.col("_n_sharpe") * w["sharpe_ratio"]) +
            (pl.col("_n_sortino") * w["sortino_ratio"]) +
            (pl.col("_n_win_rate") * w["win_rate"]) +
            (pl.col("_n_drawdown") * w["max_drawdown"]) +
            (pl.col("_n_mae") * abs(w["mae"]))
        ).alias("composite_score")

        return scored_df.with_columns(composite_expr)


# =============================================================================
# 8. ONLINE LEARNER ENGINE
# =============================================================================

class OnlineLearner:
    """
    Handles incremental updates (online learning / warm-starting) for trained GBDT models predicting stock signals.
    """

    def __init__(
        self,
        seed: int = 42,
        max_incremental_rounds: int = 10,
        learning_rate_scale: float = 0.5,
    ) -> None:
        self.seed = seed
        self.max_incremental_rounds = max_incremental_rounds
        self.learning_rate_scale = learning_rate_scale

    def _verify_model_schema(self, model: Any, model_type: str, feature_cols: List[str]) -> None:
        normalized_type = model_type.strip().lower()
        model_features: List[str] = []

        try:
            if normalized_type == "lightgbm" and LGB_AVAILABLE:
                booster = model.booster_ if hasattr(model, "booster_") else model
                if isinstance(booster, lgb.Booster):
                    model_features = list(booster.feature_name())

            elif normalized_type == "xgboost" and XGB_AVAILABLE:
                booster = model.get_booster() if hasattr(model, "get_booster") else model
                if isinstance(booster, xgb.Booster):
                    model_features = list(booster.feature_names) if booster.feature_names is not None else []

            elif normalized_type == "catboost" and CB_AVAILABLE:
                if hasattr(model, "feature_names_") and model.feature_names_ is not None:
                    model_features = list(model.feature_names_)

            else:
                raise ModelValidationError(f"Unsupported or uninstalled model framework type: {model_type}")

        except Exception as err:
            raise ModelValidationError(f"Failed to extract metadata fingerprint from model: {str(err)}") from err

        if model_features:
            if len(model_features) != len(feature_cols):
                raise DataValidationError(f"Feature dimension mismatch! Model expects {len(model_features)}, input provided {len(feature_cols)}.")
            if list(model_features) != list(feature_cols):
                raise DataValidationError(f"Feature layout mismatch! Expected ordering: {model_features}, received: {feature_cols}.")

    def validate_incremental_data(
        self,
        new_data: pl.DataFrame,
        feature_cols: List[str],
        target_col: str,
        last_trained_date: Union[str, date, datetime],
        date_col: str = "date",
    ) -> None:
        if new_data.is_empty():
            raise DataValidationError("Incremental batch dataset is completely empty.")

        required_cols = feature_cols + [target_col, date_col]
        missing_cols = [col for col in required_cols if col not in new_data.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required structural columns: {missing_cols}")

        if not new_data.select(pl.col(date_col).is_sorted()).item():
            raise ChronologyError("Incremental data stream layout violates chronological sorting rules.")

        if isinstance(last_trained_date, str):
            parsed_cutoff = datetime.strptime(last_trained_date, "%Y-%m-%d").date()
        elif isinstance(last_trained_date, datetime):
            parsed_cutoff = last_trained_date.date()
        else:
            parsed_cutoff = last_trained_date

        min_new_date = new_data.select(pl.col(date_col).min()).item()
        if min_new_date is not None:
            if isinstance(min_new_date, datetime):
                min_new_date = min_new_date.date()
            
            if min_new_date <= parsed_cutoff:
                raise ChronologyError(f"Look-Ahead Bias detected! Batch minimum date ({min_new_date}) overlaps historical cutoff ({parsed_cutoff}).")

        numeric_cols = [c for c in required_cols if c != date_col and new_data[c].dtype.is_numeric()]
        anomalies = new_data.select(
            [pl.col(c).is_null().sum().alias(f"{c}_null") for c in numeric_cols] +
            [pl.col(c).is_nan().sum().alias(f"{c}_nan") for c in numeric_cols] +
            [pl.col(c).is_infinite().sum().alias(f"{c}_inf") for c in numeric_cols]
        )
        if sum([val for val in anomalies.row(0) if val is not None]) > 0:
            raise DataValidationError("Data batch contains corrupted points matching Null, NaN, or Inf states.")

    def fit_incremental(
        self,
        model: Any,
        model_type: str,
        new_data: pl.DataFrame,
        feature_cols: List[str],
        target_col: str,
        last_trained_date: Union[str, date, datetime],
        date_col: str = "date",
    ) -> Any:
        start_time = time.time()
        new_data = _ensure_polars_df_sl(new_data)
        
        self._verify_model_schema(model=model, model_type=model_type, feature_cols=feature_cols)
        self.validate_incremental_data(
            new_data=new_data,
            feature_cols=feature_cols,
            target_col=target_col,
            last_trained_date=last_trained_date,
            date_col=date_col,
        )

        normalized_type = model_type.strip().lower()
        x_new = new_data.select(feature_cols).to_numpy()
        y_new = new_data.select(target_col).to_numpy().ravel()

        logger.info("Executing model incremental updates on %d samples.", x_new.shape[0])

        try:
            if normalized_type == "lightgbm" and LGB_AVAILABLE:
                updated_model = self._update_lightgbm(model, x_new, y_new)
            elif normalized_type == "xgboost" and XGB_AVAILABLE:
                updated_model = self._update_xgboost(model, x_new, y_new)
            elif normalized_type == "catboost" and CB_AVAILABLE:
                updated_model = self._update_catboost(model, x_new, y_new)
            else:
                raise ModelValidationError(f"Invalid or uninstalled model framework provided: {model_type}")

            duration = time.time() - start_time
            logger.info("Incremental update completed successfully in %.4f seconds.", duration)
            return updated_model

        except Exception as err:
            logger.critical("Catastrophic operational failure during incremental update: %s", str(err))
            raise OnlineLearningError(f"Incremental learning pipeline aborted: {str(err)}") from err

    def _update_lightgbm(self, model: Any, x_new: np.ndarray, y_new: np.ndarray) -> Any:
        is_wrapper = hasattr(model, "booster_")
        booster = model.booster_ if is_wrapper else model

        train_data = lgb.Dataset(data=x_new, label=y_new, free_raw_data=True)
        current_params = getattr(booster, "params", {}) or {}
        current_lr = current_params.get("learning_rate", 0.05)
        
        update_params = {
            "seed": self.seed,
            "deterministic": True,
            "learning_rate": float(current_lr * self.learning_rate_scale),
            "verbose": -1,
        }

        updated_booster = lgb.train(
            params=update_params,
            train_set=train_data,
            num_boost_round=self.max_incremental_rounds,
            init_model=booster,
            keep_training_booster=True,
        )

        if is_wrapper:
            model.booster_ = updated_booster
            return model
        return updated_booster

    def _update_xgboost(self, model: Any, x_new: np.ndarray, y_new: np.ndarray) -> Any:
        if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            current_n = model.n_estimators
            model.set_params(n_estimators=int(current_n + self.max_incremental_rounds), random_state=self.seed)
            model.fit(X=x_new, y=y_new, xgb_model=model.get_booster())
            return model

        if isinstance(model, xgb.Booster):
            dtrain = xgb.DMatrix(data=x_new, label=y_new)
            return xgb.train(
                params={"seed": self.seed, "validate_parameters": True},
                dtrain=dtrain,
                num_boost_round=self.max_incremental_rounds,
                xgb_model=model,
            )

        raise ModelValidationError("Invalid structure passed for XGBoost model continuation.")

    def _update_catboost(self, model: Any, x_new: np.ndarray, y_new: np.ndarray) -> Any:
        if not isinstance(model, (cb.CatBoostRegressor, cb.CatBoostClassifier)):
            raise ModelValidationError("Target object does not match CatBoost signatures.")
        model.fit(X=x_new, y=y_new, init_model=model, verbose=False)
        return model


# =============================================================================
# 9. OPTUNA MANAGER ENGINE
# =============================================================================

class OptunaManager:
    """
    Manages the persistent storage, optimization history extraction, and structural 
    maintenance of Optuna hyperparameter studies via SQLite in the root directory.
    """

    def __init__(self, db_path: str = "optuna_study.db", seed: int = 42) -> None:
        if not db_path:
            raise DataValidationError("Database path string configuration cannot be empty.")

        self.db_path = db_path
        self.storage_uri = f"sqlite:///{db_path}"
        self.seed = seed
        self._lock = threading.Lock()

    def get_or_create_study(self, study_name: str, direction: str = "minimize") -> Any:
        if not OPTUNA_AVAILABLE:
            raise OptunaManagerError("Optuna library is required for OptunaManager operations.")
        if not study_name:
            raise DataValidationError("Study name identifier cannot be empty.")
        if direction not in ["minimize", "maximize"]:
            raise DataValidationError(f"Invalid optimization direction parameter: {direction}")

        with self._lock:
            try:
                sampler = optuna.samplers.TPESampler(seed=self.seed)
                pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

                study = optuna.create_study(
                    study_name=study_name,
                    storage=self.storage_uri,
                    direction=direction,
                    sampler=sampler,
                    pruner=pruner,
                    load_if_exists=True,
                )
                logger.info("Optuna study resource established safely: %s [%s]", study_name, direction)
                return study
            except Exception as err:
                logger.critical("Failed to instantiate or load target Optuna study %s: %s", study_name, str(err))
                raise OptunaManagerError(f"Study acquisition cycle terminated: {str(err)}") from err

    def delete_study(self, study_name: str) -> None:
        if not OPTUNA_AVAILABLE:
            return
        if not study_name:
            raise DataValidationError("Target study name for deletion cannot be empty.")

        with self._lock:
            try:
                optuna.delete_study(study_name=study_name, storage=self.storage_uri)
                logger.warning("Optuna study profile purged completely from database: %s", study_name)
            except KeyError:
                pass
            except Exception as err:
                raise OptunaManagerError(f"Failed to complete study purge command for {study_name}: {str(err)}")

    def list_all_studies(self) -> List[str]:
        if not OPTUNA_AVAILABLE:
            return []
        with self._lock:
            try:
                summaries = optuna.study.get_all_study_summaries(storage=self.storage_uri)
                return [summary.study_name for summary in summaries]
            except Exception as err:
                raise OptunaManagerError(f"Failed to extract study summary registry indices: {str(err)}")

    def load_study_trials_to_dataframe(self, study_name: str) -> pl.DataFrame:
        if not OPTUNA_AVAILABLE:
            return pl.DataFrame()
        try:
            study = self.get_or_create_study(study_name=study_name)
            pandas_df = study.trials_dataframe()

            if pandas_df.empty:
                return pl.DataFrame()

            return pl.from_pandas(pandas_df)
        except Exception as err:
            raise OptunaManagerError(f"Failed to parse and map trials database records to Polars: {str(err)}")

    def prune_and_vacuum_database(self) -> Dict[str, Any]:
        if not os.path.exists(self.db_path):
            return {"status": "SKIPPED", "message": "Database file does not exist yet."}

        start_time = time.time()
        initial_size = os.path.getsize(self.db_path)

        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                target_states = (2, 3, "FAIL", "PRUNED")

                cursor.execute("DELETE FROM trial_user_attributes WHERE trial_id IN (SELECT trial_id FROM trials WHERE state IN (?, ?, ?, ?))", target_states)
                cursor.execute("DELETE FROM trial_system_attributes WHERE trial_id IN (SELECT trial_id FROM trials WHERE state IN (?, ?, ?, ?))", target_states)
                cursor.execute("DELETE FROM trial_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE state IN (?, ?, ?, ?))", target_states)
                cursor.execute("DELETE FROM trial_params WHERE trial_id IN (SELECT trial_id FROM trials WHERE state IN (?, ?, ?, ?))", target_states)
                cursor.execute("DELETE FROM trials WHERE state IN (?, ?, ?, ?)", target_states)
                
                rows_purged = cursor.rowcount
                conn.commit()

                cursor.execute("VACUUM")
                conn.close()

                final_size = os.path.getsize(self.db_path)
                bytes_saved = initial_size - final_size

                logger.info("Database sanitation gate complete. Rows Purged: %d. Reclaimed Space: %.2f KB", rows_purged, bytes_saved / 1024.0)

                return {
                    "status": "SUCCESS",
                    "rows_purged": rows_purged,
                    "initial_size_bytes": initial_size,
                    "final_size_bytes": final_size,
                    "space_saved_kb": round(bytes_saved / 1024.0, 2),
                    "execution_duration_seconds": round(time.time() - start_time, 4)
                }

            except Exception as err:
                if conn:
                    conn.rollback()
                    conn.close()
                raise OptunaManagerError(f"Database pruning transaction failure: {str(err)}") from err


# =============================================================================
# 10. PERFORMANCE TRACKER ENGINE
# =============================================================================

class PerformanceTracker:
    """
    Engineers advanced cross-sectional and longitudinal performance tracking metrics for stock models.
    Annualized over 252 IDX daily trading sessions per year.
    """

    def __init__(self, annualization_factor: int = 252, min_samples_for_stats: int = 30) -> None:
        self.annualization_factor = annualization_factor
        self.min_samples_for_stats = min_samples_for_stats

    def validate_feedback_matrix(
        self, 
        feedback_df: pl.DataFrame, 
        required_cols: List[str]
    ) -> None:
        if feedback_df.is_empty():
            raise DataValidationError("Input feedback dataframe is empty.")

        missing_cols = [col for col in required_cols if col not in feedback_df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required feedback columns: {missing_cols}")

        numeric_cols = [c for c in required_cols if feedback_df[c].dtype.is_numeric()]
        anomalies = feedback_df.select(
            [pl.col(c).is_null().sum().alias(f"{c}_null") for c in numeric_cols] +
            [pl.col(c).is_nan().sum().alias(f"{c}_nan") for c in numeric_cols] +
            [pl.col(c).is_infinite().sum().alias(f"{c}_inf") for c in numeric_cols]
        )
        if sum([v for v in anomalies.row(0) if v is not None]) > 0:
            raise DataValidationError("Feedback matrix contains corrupted numerical states (Null/NaN/Inf).")

    def compute_rolling_metrics(
        self,
        feedback_df: pl.DataFrame,
        window_size: int = 60,
        model_id_col: str = "model_id",
        signal_date_col: str = "signal_date",
        realized_return_col: str = "realized_return",
        predicted_return_col: str = "predicted_return",
        residual_error_col: str = "residual_error",
        directional_hit_col: str = "directional_hit",
    ) -> pl.DataFrame:
        start_time = time.time()
        feedback_df = _ensure_polars_df_sl(feedback_df)
        
        if feedback_df.is_empty():
            return pl.DataFrame()

        required_fields = [
            model_id_col, signal_date_col, realized_return_col, 
            predicted_return_col, residual_error_col, directional_hit_col
        ]
        
        self.validate_feedback_matrix(feedback_df, required_fields)
        sorted_df = feedback_df.sort([model_id_col, signal_date_col])
        sqrt_ann = np.sqrt(self.annualization_factor)

        processed_df = sorted_df.with_columns([
            pl.col(realized_return_col).rolling_mean(window_size=window_size, min_samples=window_size).over(model_id_col).alias("rolling_mean_return"),
            pl.col(realized_return_col).rolling_std(window_size=window_size, min_samples=window_size).over(model_id_col).alias("rolling_std_return"),
            pl.col(directional_hit_col).rolling_mean(window_size=window_size, min_samples=window_size).over(model_id_col).alias("rolling_win_rate"),
            pl.col(residual_error_col).rolling_mean(window_size=window_size, min_samples=window_size).over(model_id_col).alias("rolling_mae"),
            
            pl.when(pl.col(realized_return_col) < 0)
            .then(pl.col(realized_return_col) ** 2)
            .otherwise(0)
            .rolling_mean(window_size=window_size, min_samples=window_size)
            .over(model_id_col)
            .sqrt()
            .alias("rolling_downside_deviation"),
            
            (1.0 + pl.col(realized_return_col)).cum_prod().over(model_id_col).alias("_cum_return")
        ]).with_columns([
            pl.col("_cum_return").cum_max().over(model_id_col).alias("_max_cum_return")
        ]).with_columns([
            pl.when(pl.col("_max_cum_return") > 0)
            .then((pl.col("_cum_return") - pl.col("_max_cum_return")) / pl.col("_max_cum_return"))
            .otherwise(0.0)
            .alias("rolling_drawdown")
        ]).with_columns([
            pl.when(pl.col("rolling_std_return").is_not_null() & (pl.col("rolling_std_return") > 0))
            .then((pl.col("rolling_mean_return") / pl.col("rolling_std_return")) * sqrt_ann)
            .otherwise(None)
            .alias("rolling_sharpe_ratio"),

            pl.when(pl.col("rolling_downside_deviation").is_not_null() & (pl.col("rolling_downside_deviation") > 0))
            .then((pl.col("rolling_mean_return") / pl.col("rolling_downside_deviation")) * sqrt_ann)
            .otherwise(None)
            .alias("rolling_sortino_ratio"),

            pl.col("rolling_drawdown").rolling_min(window_size=window_size, min_samples=window_size).over(model_id_col).alias("rolling_max_drawdown")
        ]).drop(["_cum_return", "_max_cum_return"])

        final_df = processed_df.filter(pl.col("rolling_sharpe_ratio").is_not_null())
        logger.info("Rolling performance metrics computed in %.4f seconds.", time.time() - start_time)
        return final_df

    def analyze_performance_decay(
        self,
        rolling_metrics_df: pl.DataFrame,
        model_id: str,
        model_id_col: str = "model_id",
        sharpe_col: str = "rolling_sharpe_ratio",
        win_rate_col: str = "rolling_win_rate",
        evaluation_window: int = 30
    ) -> Dict[str, Any]:
        rolling_metrics_df = _ensure_polars_df_sl(rolling_metrics_df)
        if rolling_metrics_df.is_empty():
            return {
                "model_id": model_id,
                "decay_detected": False,
                "status_message": "Empty rolling metrics matrix.",
                "t_statistic": 0.0,
                "p_value": 1.0,
                "requires_immediate_retrain": False
            }

        model_data = rolling_metrics_df.filter(pl.col(model_id_col) == model_id)
        total_samples = model_data.shape[0]

        if total_samples < (evaluation_window * 2) or total_samples < self.min_samples_for_stats:
            return {
                "model_id": model_id,
                "decay_detected": False,
                "status_message": "Insufficient history length to execute decay statistical testing.",
                "t_statistic": 0.0,
                "p_value": 1.0,
                "requires_immediate_retrain": False
            }

        sharpe_series = model_data.select(sharpe_col).to_numpy().ravel()
        baseline_pop = sharpe_series[:-evaluation_window]
        recent_pop = sharpe_series[-evaluation_window:]

        t_stat, p_val = stats.ttest_ind(recent_pop, baseline_pop, equal_var=False, nan_policy="omit")

        mean_baseline = float(np.nanmean(baseline_pop))
        mean_recent = float(np.nanmean(recent_pop))
        
        is_decaying = bool(mean_recent < mean_baseline and p_val < 0.05)
        recent_win_rate = float(model_data.select(win_rate_col).to_numpy().ravel()[-1])

        return {
            "model_id": model_id,
            "decay_detected": is_decaying,
            "mean_baseline_sharpe": mean_baseline,
            "mean_recent_sharpe": mean_recent,
            "current_win_rate": recent_win_rate,
            "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
            "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
            "requires_immediate_retrain": bool(is_decaying or recent_win_rate < 0.45)
        }

    def generate_selector_payload(
        self,
        rolling_metrics_df: pl.DataFrame,
        model_id_col: str = "model_id",
    ) -> Dict[str, Any]:
        rolling_metrics_df = _ensure_polars_df_sl(rolling_metrics_df)
        if rolling_metrics_df.is_empty():
            return {}

        latest_metrics = rolling_metrics_df.group_by(model_id_col).last()
        selector_payload: Dict[str, Any] = {}
        
        for row in latest_metrics.iter_rows(named=True):
            mid = str(row[model_id_col])
            raw_std = row.get("rolling_std_return")
            error_variance = float(raw_std ** 2) if raw_std is not None else 0.0001
            
            selector_payload[mid] = {
                "sharpe_ratio": float(row.get("rolling_sharpe_ratio") or 0.0),
                "sortino_ratio": float(row.get("rolling_sortino_ratio") or 0.0),
                "max_drawdown": float(row.get("rolling_max_drawdown") or 0.0),
                "win_rate": float(row.get("rolling_win_rate") or 0.0),
                "mae": float(row.get("rolling_mae") or 0.0),
                "error_variance": error_variance,
                "last_updated_metric_date": str(row.get("signal_date"))
            }

        return selector_payload


# =============================================================================
# 11. RETRAINING SCHEDULER ENGINE
# =============================================================================

class RetrainingScheduler:
    """
    Orchestrates retraining triggers across the stock model population.
    Utilizes SQLite storage in root directory to enforce temporal cooldown barriers.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        cooldown_period_days: int = 7,
        min_samples_to_retrain: int = 100,
        hysteresis_buffer: float = 0.03,
        db_path: str = "storage.db"
    ) -> None:
        self.config = config
        self.cooldown_period_days = cooldown_period_days
        self.min_samples_to_retrain = min_samples_to_retrain
        self.hysteresis_buffer = hysteresis_buffer
        self.db_path = Path(db_path)
        self._lock = threading.Lock()

        self._init_sqlite_state_store()

    def _init_sqlite_state_store(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_retrain_history (
                        model_id TEXT PRIMARY KEY,
                        last_retrained_at TEXT NOT NULL,
                        trigger_reason TEXT NOT NULL
                    )
                """)
                conn.commit()
        except Exception as err:
            logger.error(f"Failed to initialize SQLite state store for RetrainingScheduler: {str(err)}")

    def _get_last_retrain_time(self, model_id: str) -> Optional[datetime]:
        try:
            if not self.db_path.exists():
                return None
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT last_retrained_at FROM model_retrain_history WHERE model_id = ?", (model_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    parsed = datetime.fromisoformat(row[0])
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed
        except Exception as err:
            logger.warning(f"Failed to read retraining history from SQLite: {str(err)}")
        return None

    def _update_last_retrain_time(self, model_id: str, timestamp: datetime, reason: str) -> None:
        try:
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO model_retrain_history (model_id, last_retrained_at, trigger_reason)
                    VALUES (?, ?, ?)
                    ON CONFLICT(model_id) DO UPDATE SET
                        last_retrained_at = excluded.last_retrained_at,
                        trigger_reason = excluded.trigger_reason
                """, (model_id, timestamp.isoformat(), reason))
                conn.commit()
        except Exception as err:
            logger.error(f"Failed to synchronize retrain status to SQLite: {str(err)}")

    def is_retraining_required(
        self, 
        has_drift: bool, 
        decay_report: Optional[Dict[str, Any]] = None,
        confidence_mean: Optional[float] = None
    ) -> bool:
        drift_report = {
            "summary_metrics": {
                "system_retrain_recommended": has_drift,
                "warning_features_count": 1 if has_drift else 0
            }
        }
        
        manifest = self.evaluate_retraining_need(
            model_id="default_production_model",
            current_data_size=self.min_samples_to_retrain + 50,
            drift_report=drift_report,
            decay_report=decay_report,
            confidence_mean=confidence_mean
        )
        
        return manifest.get("trigger_retraining", False)

    def evaluate_retraining_need(
        self,
        model_id: str,
        current_data_size: int,
        drift_report: Optional[Dict[str, Any]] = None,
        decay_report: Optional[Dict[str, Any]] = None,
        selector_report: Optional[Dict[str, Any]] = None,
        confidence_mean: Optional[float] = None,
        empirical_conviction_mean: Optional[float] = None,
        current_pipeline_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        eval_time = current_pipeline_time if current_pipeline_time is not None else datetime.now(timezone.utc)
        if eval_time.tzinfo is None:
            eval_time = eval_time.replace(tzinfo=timezone.utc)

        if not model_id:
            raise DataValidationError("Model identifier string parameter cannot be empty.")

        with self._lock:
            last_executed = self._get_last_retrain_time(model_id)
            if last_executed is not None:
                elapsed_time = eval_time - last_executed
                if elapsed_time < timedelta(days=self.cooldown_period_days):
                    return self._build_scheduler_manifest(
                        action_required=False, optimize_hyperparameters=False,
                        reason=f"Blocked by cooldown lock. Next window after: {last_executed + timedelta(days=self.cooldown_period_days)}",
                        model_id=model_id, timestamp=eval_time, duration=time.time() - start_time
                    )

        if current_data_size < self.min_samples_to_retrain:
            return self._build_scheduler_manifest(
                action_required=False, optimize_hyperparameters=False,
                reason=f"Insufficient samples count. Available: {current_data_size} < Required: {self.min_samples_to_retrain}",
                model_id=model_id, timestamp=eval_time, duration=time.time() - start_time
            )

        try:
            trigger_matrix = self._evaluate_trigger_matrix(
                drift_report=drift_report,
                decay_report=decay_report,
                selector_report=selector_report,
                confidence_mean=confidence_mean,
                empirical_conviction_mean=empirical_conviction_mean
            )

            action_required = False
            optimize_hyperparameters = False
            justification = "System operational states confirmed stable. Retraining unnecessary."

            if trigger_matrix["critical_drift_triggered"] or trigger_matrix["severe_decay_triggered"]:
                action_required = True
                optimize_hyperparameters = True
                justification = f"Emergency trigger initiated: Critical Drift={trigger_matrix['critical_drift_triggered']} | Decay={trigger_matrix['severe_decay_triggered']}"

            elif trigger_matrix["warning_drift_triggered"] or trigger_matrix["mild_decay_triggered"] or trigger_matrix["conviction_collapse_triggered"]:
                action_required = True
                optimize_hyperparameters = False
                justification = f"Standard optimization trigger initiated: Drift Warn={trigger_matrix['warning_drift_triggered']} | Decay Warn={trigger_matrix['mild_decay_triggered']}"

            with self._lock:
                if action_required:
                    self._update_last_retrain_time(model_id, eval_time, justification)

                manifest = self._build_scheduler_manifest(
                    action_required=action_required,
                    optimize_hyperparameters=optimize_hyperparameters,
                    reason=justification,
                    model_id=model_id,
                    timestamp=eval_time,
                    duration=time.time() - start_time
                )

                return manifest

        except DataValidationError:
            raise
        except Exception as err:
            logger.critical("Critical error within scheduling matrix logic loop: %s", str(err))
            raise RetrainingSchedulerError(f"Retraining scheduling assessment aborted: {str(err)}") from err

    def _evaluate_trigger_matrix(
        self,
        drift_report: Optional[Dict[str, Any]],
        decay_report: Optional[Dict[str, Any]],
        selector_report: Optional[Dict[str, Any]],
        confidence_mean: Optional[float],
        empirical_conviction_mean: Optional[float],
    ) -> Dict[str, bool]:
        matrix = {
            "critical_drift_triggered": False,
            "warning_drift_triggered": False,
            "severe_decay_triggered": False,
            "mild_decay_triggered": False,
            "conviction_collapse_triggered": False
        }

        if drift_report and "summary_metrics" in drift_report:
            summary = drift_report["summary_metrics"]
            matrix["critical_drift_triggered"] = bool(summary.get("system_retrain_recommended", False))
            matrix["warning_drift_triggered"] = bool(summary.get("warning_features_count", 0) > 0)

        if decay_report:
            matrix["severe_decay_triggered"] = bool(decay_report.get("requires_immediate_retrain", False))

            mean_base = float(decay_report.get("mean_baseline_sharpe", 0.0))
            mean_recent = float(decay_report.get("mean_recent_sharpe", 0.0))
            if mean_recent < (mean_base - self.hysteresis_buffer):
                matrix["mild_decay_triggered"] = True

        if selector_report and selector_report.get("selection_mode") == "FALLBACK":
            matrix["mild_decay_triggered"] = True

        if confidence_mean is not None and confidence_mean < 0.50:
            matrix["conviction_collapse_triggered"] = True
        if empirical_conviction_mean is not None and empirical_conviction_mean < 0.40:
            matrix["conviction_collapse_triggered"] = True

        return matrix

    def _build_scheduler_manifest(
        self,
        action_required: bool,
        optimize_hyperparameters: bool,
        reason: str,
        model_id: str,
        timestamp: datetime,
        duration: float
    ) -> Dict[str, Any]:
        return {
            "trigger_retraining": action_required,
            "execute_hyperparameter_tuning": optimize_hyperparameters,
            "decision_timestamp": timestamp.isoformat(),
            "target_model_id": model_id,
            "scheduler_metrics": {
                "evaluation_latency_seconds": round(duration, 4),
                "anti_thrashing_cooldown_setting_days": self.cooldown_period_days
            },
            "audit_justification": reason
        }


# =============================================================================
# 12. UNCERTAINTY ESTIMATOR ENGINE
# =============================================================================

class UncertaintyEstimator:
    """
    Quantifies epistemic and aleatoric uncertainty using conformal prediction intervals for stock signals.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_historical_samples: int = 50,
        epsilon: float = EPSILON,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise DataValidationError(f"Miscoverage rate alpha must be tightly bounded between 0 and 1 exclusive: {alpha}")

        self.alpha = alpha
        self.min_historical_samples = min_historical_samples
        self.epsilon = epsilon
        self._lock = threading.Lock()

    def validate_uq_inputs(
        self,
        live_preds_df: pl.DataFrame,
        historical_errors_df: pl.DataFrame,
        pred_col: str,
        error_col: str,
        model_id_col: str = "model_id",
    ) -> None:
        if live_preds_df.is_empty():
            raise DataValidationError("Live inference predictions DataFrame is empty.")
        if historical_errors_df.is_empty():
            raise DataValidationError("Historical calibration error matrix DataFrame is empty.")

        if pred_col not in live_preds_df.columns:
            raise DataValidationError(f"Target forecast column '{pred_col}' missing from inference DataFrame.")
        if error_col not in historical_errors_df.columns:
            raise DataValidationError(f"Target residual column '{error_col}' missing from historical DataFrame.")
        if model_id_col not in live_preds_df.columns or model_id_col not in historical_errors_df.columns:
            raise DataValidationError(f"Model tracking coordinate column '{model_id_col}' must be present in both sets.")

        for name, df, cols in [("Live Inference", live_preds_df, [pred_col]), ("Historical Error", historical_errors_df, [error_col])]:
            numeric_cols = [c for c in cols if df[c].dtype.is_numeric()]
            if not numeric_cols:
                raise DataValidationError(f"Target measurement parameters in {name} must be continuous numeric types.")
            
            anomalies = df.select(
                [pl.col(c).is_null().sum().alias(f"{c}_null") for c in numeric_cols] +
                [pl.col(c).is_nan().sum().alias(f"{c}_nan") for c in numeric_cols] +
                [pl.col(c).is_infinite().sum().alias(f"{c}_inf") for c in numeric_cols]
            )
            if sum([v for v in anomalies.row(0) if v is not None]) > 0:
                raise DataValidationError(f"{name} dataset contains corrupted numerical anomalies (Null/NaN/Inf).")

    def estimate_empirical_bounds(
        self,
        live_preds_df: pl.DataFrame,
        historical_errors_df: pl.DataFrame,
        model_id: str,
        pred_col: str = "predicted_return",
        error_col: str = "absolute_error",
        model_id_col: str = "model_id",
    ) -> pl.DataFrame:
        start_time = time.time()
        live_preds_df = sanitize_predictions_for_feedback(live_preds_df, default_model_id=model_id)
        historical_errors_df = _ensure_polars_df_sl(historical_errors_df)

        if live_preds_df.is_empty() or historical_errors_df.is_empty():
            return live_preds_df.with_columns([
                pl.lit(0.0).alias("empirical_lower_bound"),
                pl.lit(0.0).alias("empirical_upper_bound"),
                pl.lit(0.0).alias("total_uncertainty_score"),
                pl.lit(0.0).alias("empirical_conviction_score")
            ])

        self.validate_uq_inputs(
            live_preds_df=live_preds_df,
            historical_errors_df=historical_errors_df,
            pred_col=pred_col,
            error_col=error_col,
            model_id_col=model_id_col
        )

        model_errors_df = historical_errors_df.filter(pl.col(model_id_col) == model_id)
        sample_size = model_errors_df.shape[0]

        if sample_size < self.min_historical_samples:
            raise DataValidationError(
                f"Historical error partition size ({sample_size}) for model {model_id} is insufficient (Min required: {self.min_historical_samples})."
            )

        try:
            abs_errors = model_errors_df.select(error_col).to_numpy().ravel()
            target_quantile = 1.0 - self.alpha
            conformal_multiplier = float(np.percentile(abs_errors, target_quantile * 100))
            conformal_multiplier = max(conformal_multiplier, self.epsilon)

            raw_skew = model_errors_df.select(pl.col(error_col).skew()).item()
            skewness_factor = abs(float(raw_skew)) if raw_skew is not None and not np.isnan(raw_skew) else 0.0
            
            output_df = live_preds_df.with_columns([
                (pl.col(pred_col) - conformal_multiplier).alias("empirical_lower_bound"),
                (pl.col(pred_col) + conformal_multiplier).alias("empirical_upper_bound"),
                (pl.lit(conformal_multiplier) * (1.0 + pl.lit(skewness_factor))).alias("total_uncertainty_score")
            ]).with_columns([
                pl.when((pl.col("empirical_lower_bound") < 0.0) & (pl.col("empirical_upper_bound") > 0.0))
                .then(pl.lit(0.0))
                .otherwise((pl.col(pred_col).abs() / (pl.col("total_uncertainty_score") + self.epsilon)).clip(0.0, 1.0))
                .alias("empirical_conviction_score")
            ])

            with self._lock:
                execution_duration = time.time() - start_time
                logger.info("Empirical Uncertainty calibration resolved in %.4f seconds for model %s.", execution_duration, model_id)
                return output_df

        except DataValidationError:
            raise
        except Exception as err:
            logger.critical("Critical operational error within empirical uncertainty logic loops: %s", str(err))
            raise UncertaintyEstimatorError(f"Uncertainty validation engine cycle aborted: {str(err)}") from err


# =============================================================================
# 13. UNIFIED SELF-LEARNING ENGINE (FACADE CLASS WITH GEMINI & AUTO-PERSISTENCE)
# =============================================================================

class UnifiedSelfLearningEngine:
    """
    Facade class that orchestrates the entire quantitative self-learning lifecycle for stock prediction.
    Provides a single entry point for feedback evaluation, performance tracking, drift analysis,
    confidence & uncertainty scoring, candidate model ranking/selection, Google Gemini AI Diagnostics,
    GUARANTEED AUTOMATIC model persistence, and retraining triggers.
    """

    def __init__(
        self,
        annualization_factor: int = 252,
        cooldown_period_days: int = 7,
        min_samples_to_retrain: int = 100,
        optuna_db_path: str = "optuna_study.db",
        scheduler_db_path: str = "storage.db",
        default_model_save_path: str = "models/active_idx_model.pkl",
        gemini_api_key: Optional[str] = None
    ) -> None:
        self.feedback_engine = FeedbackEngine()
        self.performance_tracker = PerformanceTracker(annualization_factor=annualization_factor)
        self.drift_detector = DriftDetector()
        self.confidence_estimator = ConfidenceEstimator()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.model_ranker = ModelRanker()
        self.model_selector = ModelSelector()
        self.champion_challenger = ChampionChallengerEngine()
        self.online_learner = OnlineLearner()
        self.hyperparameter_optimizer = HyperparameterOptimizer(storage_uri=f"sqlite:///{optuna_db_path}")
        self.optuna_manager = OptunaManager(db_path=optuna_db_path)
        self.retraining_scheduler = RetrainingScheduler(
            cooldown_period_days=cooldown_period_days,
            min_samples_to_retrain=min_samples_to_retrain,
            db_path=scheduler_db_path
        )
        self.gemini_engine = GeminiSelfLearningDiagnosticEngine(api_key=gemini_api_key)
        
        self.default_model_save_path = Path(default_model_save_path)
        self._active_model: Optional[Any] = None
        self._lock = threading.Lock()
        
        self.default_model_save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("UnifiedSelfLearningEngine v2026.Q3.v16.3 initialized with Gemini AI Diagnostics & Auto-Persistence enabled.")

    def set_active_model(self, model: Any, auto_save: bool = True, save_filepath: Optional[Union[str, Path]] = None) -> None:
        """Assigns model object to engine state and automatically triggers disk persistence."""
        with self._lock:
            self._active_model = model
            logger.info("Active model registered successfully inside UnifiedSelfLearningEngine.")
        
        if auto_save:
            target_path = save_filepath or self.default_model_save_path
            self.save_model(filepath=target_path)

    def save_model(self, filepath: Optional[Union[str, Path]] = None) -> bool:
        """Saves current active model engine state to disk atomically using joblib."""
        target_path = Path(filepath) if filepath else self.default_model_save_path
        with self._lock:
            if self._active_model is None:
                logger.warning("[SAVE_SKIPPED] Cannot persist model: Active model state is None.")
                return False
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = target_path.with_suffix(".tmp")
                joblib.dump(self._active_model, tmp_path, compress=3)
                os.replace(tmp_path, target_path)
                logger.info("[MODEL_PERSISTED_SUCCESS] Self-learning model state written to: %s", target_path)
                return True
            except Exception as err:
                logger.error("[MODEL_PERSIST_FAILED] Could not write model state to disk: %s", str(err))
                return False

    def load_model(self, filepath: Optional[Union[str, Path]] = None) -> bool:
        """Loads active model engine state from disk using joblib."""
        target_path = Path(filepath) if filepath else self.default_model_save_path
        with self._lock:
            if not target_path.exists():
                logger.warning("[LOAD_SKIPPED] Target model state path missing: %s", target_path)
                return False
            try:
                self._active_model = joblib.load(target_path)
                logger.info("[MODEL_LOADED_SUCCESS] Self-learning model state restored from: %s", target_path)
                return True
            except Exception as err:
                logger.error("[MODEL_LOAD_FAILED] Failed to restore self-learning model state: %s", str(err))
                return False

    def run_full_feedback_cycle(
        self,
        predictions_df: Union[pl.DataFrame, Any],
        actuals_df: Union[pl.DataFrame, Any],
        baseline_features_df: Union[pl.DataFrame, Any],
        current_features_df: Union[pl.DataFrame, Any],
        feature_cols: List[str],
        processed_prediction_ids: Set[str],
        active_model_id: str,
        model_instance: Optional[Any] = None,
        auto_persist_model: bool = True
    ) -> Dict[str, Any]:
        """
        Executes an integrated feedback cycle in one single call with Google Gemini AI diagnostics and automatic model state persistence.
        """
        start_time = time.time()
        results: Dict[str, Any] = {}

        predictions_df = sanitize_predictions_for_feedback(predictions_df, default_model_id=active_model_id)
        actuals_df = _ensure_polars_df_sl(actuals_df)
        baseline_features_df = _ensure_polars_df_sl(baseline_features_df)
        current_features_df = _ensure_polars_df_sl(current_features_df)

        if model_instance is not None:
            self.set_active_model(model_instance, auto_save=auto_persist_model)

        # 1. Process Feedback Loop
        feedback_matrix, feedback_summary = self.feedback_engine.process_feedback_loop(
            predictions=predictions_df,
            actuals=actuals_df,
            processed_prediction_ids=processed_prediction_ids
        )
        results["feedback_summary"] = feedback_summary

        # 2. Feature Drift Audit
        drift_report = self.drift_detector.analyze_feature_drift(
            baseline_df=baseline_features_df,
            target_df=current_features_df,
            feature_cols=feature_cols
        )
        results["drift_report"] = drift_report

        # 3. Performance Metrics & Decay Analysis
        if not feedback_matrix.is_empty():
            rolling_df = self.performance_tracker.compute_rolling_metrics(feedback_df=feedback_matrix)
            decay_report = self.performance_tracker.analyze_performance_decay(
                rolling_metrics_df=rolling_df,
                model_id=active_model_id
            )
            selector_payload = self.performance_tracker.generate_selector_payload(rolling_metrics_df=rolling_df)
            results["decay_report"] = decay_report
            results["selector_payload"] = selector_payload

            # 4. Model Selection Evaluation
            if selector_payload:
                selection_report = self.model_selector.select_best_model(metrics_payload=selector_payload)
                results["selection_report"] = selection_report

        # 5. Evaluate Retraining Requirements
        retrain_manifest = self.retraining_scheduler.evaluate_retraining_need(
            model_id=active_model_id,
            current_data_size=predictions_df.shape[0],
            drift_report=drift_report,
            decay_report=results.get("decay_report"),
            selector_report=results.get("selection_report")
        )
        results["retrain_manifest"] = retrain_manifest

        # 6. Integrasi Gemini AI Diagnostic Layer
        ai_diag = self.gemini_engine.run_learning_diagnosis(
            active_model_id=active_model_id,
            drift_report=drift_report,
            decay_report=results.get("decay_report"),
            retrain_manifest=retrain_manifest
        )
        results["ai_diagnosis"] = ai_diag

        # 7. Trigger Auto-Save Model Checkpoint ke Disk
        if auto_persist_model and self._active_model is not None:
            save_success = self.save_model()
            results["model_persisted_to_disk"] = save_success

        results["total_execution_seconds"] = round(time.time() - start_time, 4)
        return results

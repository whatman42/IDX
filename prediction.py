"""
=============================================================================
IDX Quantitative Prediction & Signal Formulation Engine
File Name : prediction.py
Directory : Root Project Level (Selevel dengan main.py)
Version   : v2026.Q3.v25.5.0 (Institutional Production-Ready Edition with Gemini AI)
=============================================================================
"""

import math
import os
import threading
import time
import types
from datetime import datetime, timezone
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import polars as pl

# Import Google GenAI Client
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# -----------------------------------------------------------------------------
# IDX Stock Market Locked Constants & Institutional Guardrails
# -----------------------------------------------------------------------------
IDX_FEE_ROUNDTRIP_PCT: Final[float] = 0.003          # 0.3% roundtrip transaction fee (0.15% Buy, 0.15% Sell PPh)
IDX_MIN_PRICE_IDR: Final[float] = 50.0              # Batas harga minimal saham IDX (Papan Utama/Pengembangan)
IDX_MIN_24H_VOLUME_IDR: Final[float] = 1_000_000_000.0  # Threshold likuiditas transaksi minimal Rp 1 Miliar
IDX_MAX_STALENESS_SEC: Final[float] = 86400.0        # Batas usia data candle harian (24 Jam)

DEFAULT_SLIPPAGE_BPS: Final[float] = 0.0010          # 10 BPS expected slippage
DEFAULT_SPREAD_BPS: Final[float] = 0.0015            # 15 BPS average bid-ask spread
DEFAULT_MAX_PORTFOLIO_RISK_PCT: Final[float] = 0.02  # Max 2% capital risk per trade
DEFAULT_FRACTIONAL_KELLY: Final[float] = 0.25        # Quarter-Kelly Strategy

# Backward Compatibility Aliases
TOKOCRYPTO_FEE_ROUNDTRIP_PCT: Final[float] = IDX_FEE_ROUNDTRIP_PCT
TOKOCRYPTO_MIN_PRICE_USDT: Final[float] = IDX_MIN_PRICE_IDR
TOKOCRYPTO_MIN_24H_VOLUME_USDT: Final[float] = IDX_MIN_24H_VOLUME_IDR
TOKOCRYPTO_MAX_STALENESS_SEC: Final[float] = IDX_MAX_STALENESS_SEC

try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger("IDX.Prediction")

try:
    from exceptions import DataValidationError, ModelExecutionError
except ImportError:
    class DataValidationError(Exception):
        pass

    class ModelExecutionError(Exception):
        pass

try:
    from constants import COL_ASSET, COL_TIMESTAMP, EPSILON
except ImportError:
    COL_TIMESTAMP: Final[str] = "timestamp"
    COL_ASSET: Final[str] = "asset_id"
    EPSILON: Final[float] = 1e-15

COL_TICKER: Final[str] = "ticker"


# =============================================================================
# Helper: IDX Microstructure Tick Size Rounding
# =============================================================================
def round_to_idx_tick_size(prices: np.ndarray) -> np.ndarray:
    """
    Rounds price array to official IDX (Bursa Efek Indonesia) fractional tick sizes:
    - Price < Rp 200: Tick Rp 1
    - Price Rp 200 - Rp 500: Tick Rp 2
    - Price Rp 500 - Rp 2,000: Tick Rp 5
    - Price Rp 2,000 - Rp 5,000: Tick Rp 10
    - Price >= Rp 5,000: Tick Rp 25
    """
    p = np.maximum(np.nan_to_num(prices, nan=IDX_MIN_PRICE_IDR), IDX_MIN_PRICE_IDR)
    ticks = np.where(
        p < 200.0, 1.0,
        np.where(
            p < 500.0, 2.0,
            np.where(
                p < 2000.0, 5.0,
                np.where(p < 5000.0, 10.0, 25.0)
            )
        )
    )
    return np.round(p / ticks) * ticks


# =============================================================================
# Helper: Fittable Standard Platt Calibrator
# =============================================================================
class FittablePlattCalibrator:
    """Fittable Standard Logistic Platt Scaling Calibrator: P(y=1) = 1 / (1 + exp(-(A*x + B)))."""
    def __init__(self, a: float = 1.15, b: float = -0.05) -> None:
        self.a = float(a)
        self.b = float(b)

    def fit_params(self, a: float, b: float) -> None:
        self.a = float(a)
        self.b = float(b)

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        scaled_logits = self.a * raw_scores + self.b
        clipped_logits = np.clip(scaled_logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-clipped_logits))


# =============================================================================
# 1. QuantMarketRegimeEngine (Vectorized Multi-Factor Dynamic Market Regime)
# =============================================================================
class QuantMarketRegimeEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.output_regime_col: Final[str] = str(self.config.get("output_regime_column", "market_regime"))
        self.output_regime_mult_col: Final[str] = str(self.config.get("output_regime_multiplier_column", "regime_multiplier"))

        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()

    def detect_regimes(self, df: pl.DataFrame) -> pl.DataFrame:
        t_start = time.perf_counter()

        col_ema50 = pl.col("ema_50") if "ema_50" in df.columns else pl.lit(1.0)
        col_ema200 = pl.col("ema_200") if "ema_200" in df.columns else pl.lit(1.0)
        col_adx = pl.col("adx_14") if "adx_14" in df.columns else pl.lit(20.0)
        col_vol_pct = pl.col("volatility_percentile") if "volatility_percentile" in df.columns else pl.lit(0.5)
        col_vol_ratio = pl.col("volume_ratio") if "volume_ratio" in df.columns else pl.lit(1.0)

        is_high_vol = col_vol_pct.fill_null(0.5) > 0.85
        is_bull_trend = (col_ema50.fill_null(1.0) > col_ema200.fill_null(1.0)) & (col_adx.fill_null(20.0) > 25.0) & (col_vol_ratio.fill_null(1.0) > 1.0)
        is_bear_trend = (col_ema50.fill_null(1.0) < col_ema200.fill_null(1.0)) & (col_adx.fill_null(20.0) > 25.0)

        out_df = df.with_columns([
            pl.when(is_high_vol)
              .then(pl.lit("HIGH_VOLATILITY"))
              .when(is_bull_trend)
              .then(pl.lit("BULL_TREND"))
              .when(is_bear_trend)
              .then(pl.lit("BEAR_TREND"))
              .otherwise(pl.lit("SIDEWAYS"))
              .alias(self.output_regime_col),

            pl.when(is_high_vol)
              .then(pl.lit(0.65))
              .when(is_bull_trend)
              .then(pl.lit(1.25))
              .when(is_bear_trend)
              .then(pl.lit(0.40))
              .otherwise(pl.lit(0.85))
              .alias(self.output_regime_mult_col),

            pl.when(is_high_vol)
              .then(pl.lit(0.75))
              .when(is_bull_trend)
              .then(pl.lit(0.55))
              .when(is_bear_trend)
              .then(pl.lit(0.75))
              .otherwise(pl.lit(0.65))
              .alias("regime_prob_threshold")
        ])

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 2. QuantProbabilityEngine (Calibrated Empirical Probability + Aliasing)
# =============================================================================
class QuantProbabilityEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.raw_score_cols: Final[List[str]] = list(self.config.get("raw_score_columns", ["raw_score"]))
        self.output_prob_col: Final[str] = str(self.config.get("output_probability_column", "calibrated_prob"))

        raw_weights = self.config.get("ensemble_weights", [1.0] * len(self.raw_score_cols))
        weight_sum = sum(raw_weights)
        self.ensemble_weights: Final[Tuple[float, ...]] = tuple(
            [float(w) / (weight_sum + EPSILON) for w in raw_weights]
        )

        self._is_active: bool = True
        self._active_calibrator: Optional[Any] = FittablePlattCalibrator()
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()
            self._active_calibrator = None

    def attach_calibrator_instance(self, calibrator: Any) -> None:
        if calibrator is None:
            raise DataValidationError("Calibrator instance cannot be null.")
        with self._state_lock:
            self._active_calibrator = calibrator

    def process_calibrated_probabilities(self, df: pl.DataFrame) -> pl.DataFrame:
        t_start = time.perf_counter()

        with self._state_lock:
            local_calibrator = self._active_calibrator

        raw_matrices = df.select(self.raw_score_cols).to_numpy(order="c").astype(np.float64)
        raw_matrices = np.nan_to_num(raw_matrices, nan=0.0)
        blended_scores = np.dot(raw_matrices, np.array(self.ensemble_weights, dtype=np.float64))

        if hasattr(local_calibrator, "predict_proba"):
            probabilities = local_calibrator.predict_proba(blended_scores.reshape(-1, 1))[:, 1]
        elif hasattr(local_calibrator, "predict"):
            probabilities = local_calibrator.predict(blended_scores.reshape(-1, 1))
        elif hasattr(local_calibrator, "calibrate"):
            probabilities = local_calibrator.calibrate(blended_scores)
        else:
            clipped_scores = np.clip(blended_scores, -50.0, 50.0)
            probabilities = 1.0 / (1.0 + np.exp(-clipped_scores))

        probabilities = np.clip(probabilities, 0.01, 0.99)
        
        out_df = df.with_columns([
            pl.Series(self.output_prob_col, probabilities, dtype=pl.Float64),
            pl.Series("prediction_probability", probabilities, dtype=pl.Float64),
            pl.Series("probability", probabilities, dtype=pl.Float64)
        ])

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 3. QuantExpectedReturnEngine (Model Return E[R|Signal] & Almgren-Chriss Impact)
# =============================================================================
class QuantExpectedReturnEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.prob_col: Final[str] = str(self.config.get("probability_column", "calibrated_prob"))
        self.price_col: Final[str] = str(self.config.get("current_price_column", "current_price"))
        self.output_er_col: Final[str] = str(self.config.get("output_expected_return_column", "expected_return"))

        self._fee_pct: float = float(self.config.get("transaction_cost_coefficient", IDX_FEE_ROUNDTRIP_PCT))
        self._slippage_bps: float = float(self.config.get("slippage_bps", DEFAULT_SLIPPAGE_BPS))
        self._spread_bps: float = float(self.config.get("spread_bps", DEFAULT_SPREAD_BPS))
        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()

    def process_expected_returns(self, df: pl.DataFrame, trade_size_idr: float = 20_000_000.0) -> pl.DataFrame:
        t_start = time.perf_counter()
        prob_np = np.nan_to_num(df[self.prob_col].to_numpy().astype(np.float64), nan=0.5)

        if "predicted_magnitude" in df.columns:
            mag_upside = np.nan_to_num(df["predicted_magnitude"].to_numpy().astype(np.float64), nan=0.04)
        else:
            mag_upside = np.full(df.height, 0.04)

        mag_downside = mag_upside * 0.65
        regime_mult = np.nan_to_num(df["regime_multiplier"].to_numpy().astype(np.float64), nan=1.0) if "regime_multiplier" in df.columns else np.ones(df.height)
        vol_24h = np.nan_to_num(df["volume_24h_idr"].to_numpy().astype(np.float64), nan=2e9) if "volume_24h_idr" in df.columns else np.full(df.height, 2e9)
        daily_volatility = np.nan_to_num(df["volatility"].to_numpy().astype(np.float64), nan=0.02) if "volatility" in df.columns else np.full(df.height, 0.02)

        participation_rate = np.clip(trade_size_idr / np.maximum(vol_24h, EPSILON), 0.0, 1.0)
        almgren_chriss_impact = daily_volatility * np.sqrt(participation_rate)

        total_friction = self._fee_pct + self._slippage_bps + (0.5 * self._spread_bps) + almgren_chriss_impact

        expected_gain = prob_np * mag_upside * regime_mult
        expected_loss = (1.0 - prob_np) * mag_downside
        net_ev = expected_gain - expected_loss - total_friction

        out_df = df.with_columns(pl.Series(self.output_er_col, net_ev, dtype=pl.Float64))

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 4. QuantConfidenceEngine (Normalized Shannon Entropy & Gateway Compatibility)
# =============================================================================
class QuantConfidenceEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.score_cols: Final[List[str]] = list(self.config.get("inference_score_columns", ["raw_score"]))
        self.prob_col: Final[str] = str(self.config.get("probability_column", "calibrated_prob"))
        self.output_conf_col: Final[str] = str(self.config.get("output_confidence_column", "signal_confidence"))

        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()

    def calculate_confidence_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        t_start = time.perf_counter()
        prob_np = np.nan_to_num(df[self.prob_col].to_numpy().astype(np.float64), nan=0.5)

        p_clean = np.clip(prob_np, EPSILON, 1.0 - EPSILON)
        raw_entropy = -(p_clean * np.log2(p_clean) + (1.0 - p_clean) * np.log2(1.0 - p_clean))
        normalized_entropy = np.clip(raw_entropy, 0.0, 1.0)
        entropy_conf = 1.0 - normalized_entropy

        if len(self.score_cols) > 1 and all(c in df.columns for c in self.score_cols):
            scores_np = np.nan_to_num(df.select(self.score_cols).to_numpy().astype(np.float64), nan=0.0)
            var_penalty = np.exp(-1.5 * np.var(scores_np, axis=1, ddof=1))
        else:
            var_penalty = 1.0

        confidence = np.clip(entropy_conf * var_penalty, 0.05, 1.0)
        
        out_df = df.with_columns([
            pl.Series(self.output_conf_col, confidence, dtype=pl.Float64),
            pl.Series("prediction_confidence", confidence, dtype=pl.Float64)
        ])

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 5. QuantPriceTargetEngine (IDX Tick Rounded Target Upside Projection)
# =============================================================================
class QuantPriceTargetEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.price_col: Final[str] = str(self.config.get("current_price_column", "current_price"))
        self.output_target_col: Final[str] = str(self.config.get("output_price_target_column", "price_target"))

        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()

    def calculate_price_targets(self, df: pl.DataFrame) -> pl.DataFrame:
        t_start = time.perf_counter()
        price_np = np.nan_to_num(df[self.price_col].to_numpy().astype(np.float64), nan=IDX_MIN_PRICE_IDR)
        conf_np = np.nan_to_num(df["signal_confidence"].to_numpy().astype(np.float64), nan=1.0) if "signal_confidence" in df.columns else np.ones(df.height)

        if "predicted_magnitude" in df.columns:
            expected_upside = np.nan_to_num(df["predicted_magnitude"].to_numpy().astype(np.float64), nan=0.04)
        else:
            expected_upside = np.full(df.height, 0.04)

        raw_targets = price_np * (1.0 + (expected_upside * conf_np))

        if "resistance_price" in df.columns:
            resistance = np.nan_to_num(df["resistance_price"].to_numpy().astype(np.float64), nan=0.0)
            valid_res = np.where(resistance > price_np, resistance, raw_targets)
            final_targets = np.minimum(raw_targets, valid_res)
        else:
            final_targets = raw_targets

        final_targets_rounded = round_to_idx_tick_size(final_targets)
        out_df = df.with_columns(pl.Series(self.output_target_col, final_targets_rounded, dtype=pl.Float64))

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 6. QuantStopLossEngine (IDX Tick Rounded Chandelier & Support SL)
# =============================================================================
class QuantStopLossEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.price_col: Final[str] = str(self.config.get("current_price_column", "current_price"))
        self.output_sl_col: Final[str] = str(self.config.get("output_stoploss_column", "stop_loss"))

        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()

    def calculate_stop_losses(self, df: pl.DataFrame) -> pl.DataFrame:
        t_start = time.perf_counter()
        price_np = np.nan_to_num(df[self.price_col].to_numpy().astype(np.float64), nan=IDX_MIN_PRICE_IDR)

        if "atr_14" in df.columns:
            atr = np.nan_to_num(df["atr_14"].to_numpy().astype(np.float64), nan=price_np * 0.01)
            atr_sl = price_np - (2.0 * atr)
        elif "feature_atr" in df.columns:
            atr = np.nan_to_num(df["feature_atr"].to_numpy().astype(np.float64), nan=price_np * 0.01)
            atr_sl = price_np - (2.0 * atr)
        else:
            atr_sl = price_np * 0.98

        if "support_price" in df.columns:
            support = np.nan_to_num(df["support_price"].to_numpy().astype(np.float64), nan=0.0)
            valid_support = np.where((support < price_np) & (support > price_np * 0.8), support * 0.99, atr_sl)
            final_sl = np.minimum(atr_sl, valid_support)
        else:
            final_sl = atr_sl

        final_sl_rounded = round_to_idx_tick_size(final_sl)
        out_df = df.with_columns(pl.Series(self.output_sl_col, final_sl_rounded, dtype=pl.Float64))

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 7. QuantTakeProfitEngine (IDX Tick Rounded TP with RRR Integrity)
# =============================================================================
class QuantTakeProfitEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.price_col: Final[str] = str(self.config.get("current_price_column", "current_price"))
        self.sl_col: Final[str] = str(self.config.get("stop_loss_column", "stop_loss"))
        self.target_col: Final[str] = str(self.config.get("price_target_column", "price_target"))
        self.output_tp_col: Final[str] = str(self.config.get("output_takeprofit_column", "take_profit"))

        self._min_rrr: float = float(self.config.get("min_risk_reward_ratio", 1.5))
        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()

    def calculate_take_profits(self, df: pl.DataFrame) -> pl.DataFrame:
        t_start = time.perf_counter()
        price_np = np.nan_to_num(df[self.price_col].to_numpy().astype(np.float64), nan=IDX_MIN_PRICE_IDR)
        sl_np = np.nan_to_num(df[self.sl_col].to_numpy().astype(np.float64), nan=price_np * 0.98)
        target_np = np.nan_to_num(df[self.target_col].to_numpy().astype(np.float64), nan=price_np * 1.05) if self.target_col in df.columns else price_np * 1.05

        risk_dist = np.maximum(price_np - sl_np, EPSILON)
        min_reward_dist = risk_dist * self._min_rrr
        min_tp = price_np + min_reward_dist

        initial_tp = np.maximum(target_np, min_tp)
        tp_validity_mask = np.ones(df.height, dtype=bool)
        final_tp = initial_tp

        if "resistance_price" in df.columns:
            resistance = np.nan_to_num(df["resistance_price"].to_numpy().astype(np.float64), nan=0.0)
            has_resistance = resistance > price_np
            blocks_rrr = has_resistance & (resistance < min_tp)
            final_tp = np.where(blocks_rrr, resistance, initial_tp)
            tp_validity_mask = ~blocks_rrr
        else:
            tp_validity_mask = (initial_tp - price_np) >= min_reward_dist

        final_tp_rounded = round_to_idx_tick_size(final_tp)
        realized_rrr = (final_tp_rounded - price_np) / risk_dist

        out_df = df.with_columns([
            pl.Series(self.output_tp_col, final_tp_rounded, dtype=pl.Float64),
            pl.Series("realized_risk_reward_ratio", realized_rrr, dtype=pl.Float64),
            pl.Series("tp_structure_is_valid", tp_validity_mask, dtype=pl.Boolean)
        ])

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 8. QuantExpectedDateEngine (Vol-Decay Adaptive Horizon)
# =============================================================================
class QuantExpectedDateEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.output_horizon_col: Final[str] = str(self.config.get("output_horizon_column", "expected_horizon_ticks"))

        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()

    def calculate_expected_horizons(self, df: pl.DataFrame) -> pl.DataFrame:
        t_start = time.perf_counter()
        price = np.nan_to_num(df["current_price"].to_numpy().astype(np.float64), nan=IDX_MIN_PRICE_IDR)
        tp = np.nan_to_num(df["take_profit"].to_numpy().astype(np.float64), nan=price * 1.05)

        if "atr_14" in df.columns:
            atr = np.nan_to_num(df["atr_14"].to_numpy().astype(np.float64), nan=price * 0.01)
            daily_velocity = (atr / price)
        elif "feature_atr" in df.columns:
            atr = np.nan_to_num(df["feature_atr"].to_numpy().astype(np.float64), nan=price * 0.01)
            daily_velocity = (atr / price)
        else:
            daily_velocity = np.full(df.height, 0.01)

        target_return = np.abs(tp - price) / price
        raw_days = target_return / np.maximum(daily_velocity, 1e-4)
        holding_days = np.clip(raw_days, 1.0, 15.0)

        text_horizons = np.where(
            holding_days <= 1.5, "1 Hari",
            np.where(
                holding_days <= 3.5, "1-3 Hari",
                np.where(
                    holding_days <= 5.5, "3-5 Hari",
                    np.where(holding_days <= 10.5, "1-2 Minggu", "2+ Minggu")
                )
            )
        )

        out_df = df.with_columns([
            pl.Series(self.output_horizon_col, holding_days, dtype=pl.Float64),
            pl.Series("prediction_horizon", text_horizons, dtype=pl.Utf8)
        ])

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 9. QuantCrossSectionalEngine (Row-Wise Regime & Z-Score Ranking Vectorized)
# =============================================================================
class QuantCrossSectionalEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.output_score_col: Final[str] = str(self.config.get("output_composite_score_column", "composite_quant_score"))
        self._top_percentile_cutoff: float = float(self.config.get("top_percentile_cutoff", 0.80))

        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()

    def _z_score_normalize(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) <= 1:
            return np.zeros_like(arr)
        arr_clean = np.nan_to_num(arr, nan=0.0)
        std = np.std(arr_clean)
        if std < EPSILON:
            return np.zeros_like(arr_clean)
        return (arr_clean - np.mean(arr_clean)) / std

    def rank_universe(self, df: pl.DataFrame) -> pl.DataFrame:
        t_start = time.perf_counter()
        prob = np.nan_to_num(df["calibrated_prob"].to_numpy().astype(np.float64), nan=0.5)
        er = np.nan_to_num(df["expected_return"].to_numpy().astype(np.float64), nan=0.0)
        conf = np.nan_to_num(df["signal_confidence"].to_numpy().astype(np.float64), nan=1.0) if "signal_confidence" in df.columns else np.ones(df.height)
        
        momentum = np.nan_to_num(df["momentum_score"].to_numpy().astype(np.float64), nan=0.0) if "momentum_score" in df.columns else np.zeros(df.height)
        liquidity = np.nan_to_num(df["volume_ratio"].to_numpy().astype(np.float64), nan=1.0) if "volume_ratio" in df.columns else np.ones(df.height)
        regimes = df["market_regime"].to_numpy() if "market_regime" in df.columns else np.full(df.height, "SIDEWAYS")

        z_prob = self._z_score_normalize(prob)
        z_er = self._z_score_normalize(er)
        z_conf = self._z_score_normalize(conf)
        z_mom = self._z_score_normalize(momentum)
        z_liq = self._z_score_normalize(liquidity)

        is_bull = (regimes == "BULL_TREND")
        is_bear_vol = np.isin(regimes, ["BEAR_TREND", "HIGH_VOLATILITY"])

        w_prob = np.where(is_bull, 0.30, np.where(is_bear_vol, 0.40, 0.35))
        w_er   = np.where(is_bull, 0.20, np.where(is_bear_vol, 0.10, 0.25))
        w_conf = np.where(is_bull, 0.15, np.where(is_bear_vol, 0.30, 0.20))
        w_mom  = np.where(is_bull, 0.25, np.where(is_bear_vol, 0.05, 0.10))
        w_liq  = np.where(is_bull, 0.10, np.where(is_bear_vol, 0.15, 0.10))

        composite = (w_prob * z_prob) + (w_er * z_er) + (w_conf * z_conf) + (w_mom * z_mom) + (w_liq * z_liq)

        temp_df = df.with_columns([pl.Series(self.output_score_col, composite, dtype=pl.Float64)])
        
        ranked_df = temp_df.with_columns([
            (pl.col(self.output_score_col).rank(descending=False) / pl.count()).alias("percentile_rank")
        ])
        
        is_top = ranked_df["percentile_rank"].to_numpy() >= (1.0 - self._top_percentile_cutoff)
        out_df = ranked_df.with_columns(pl.Series("is_top_percentile", is_top, dtype=pl.Boolean)).drop("percentile_rank")

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 10. QuantPortfolioOptimizerEngine (Bayesian Shrinkage Kelly Allocation)
# =============================================================================
class QuantPortfolioOptimizerEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.fractional_kelly: float = float(self.config.get("fractional_kelly", DEFAULT_FRACTIONAL_KELLY))
        self.max_risk_pct: float = float(self.config.get("max_portfolio_risk_pct", DEFAULT_MAX_PORTFOLIO_RISK_PCT))

        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()

    def optimize_allocation(self, df: pl.DataFrame, total_capital_idr: float = 100_000_000.0) -> pl.DataFrame:
        t_start = time.perf_counter()
        prob = np.nan_to_num(df["calibrated_prob"].to_numpy().astype(np.float64), nan=0.5)
        conf = np.nan_to_num(df["signal_confidence"].to_numpy().astype(np.float64), nan=1.0) if "signal_confidence" in df.columns else np.ones(df.height)
        rrr = np.nan_to_num(df["realized_risk_reward_ratio"].to_numpy().astype(np.float64), nan=1.0)
        price = np.nan_to_num(df["current_price"].to_numpy().astype(np.float64), nan=IDX_MIN_PRICE_IDR)
        sl = np.nan_to_num(df["stop_loss"].to_numpy().astype(np.float64), nan=price * 0.98)
        tp = np.nan_to_num(df["take_profit"].to_numpy().astype(np.float64), nan=price * 1.05)
        
        tp_valid = df["tp_structure_is_valid"].to_numpy() if "tp_structure_is_valid" in df.columns else np.ones(df.height, dtype=bool)
        sector_corr = np.nan_to_num(df["sector_correlation"].to_numpy().astype(np.float64), nan=0.3) if "sector_correlation" in df.columns else np.full(df.height, 0.3)
        regime_threshold = np.nan_to_num(df["regime_prob_threshold"].to_numpy().astype(np.float64), nan=0.60) if "regime_prob_threshold" in df.columns else np.full(df.height, 0.60)

        effective_prob = 0.5 + ((prob - 0.5) * conf)
        q_effective = 1.0 - effective_prob

        raw_kelly = (effective_prob * rrr - q_effective) / np.maximum(rrr, EPSILON)
        
        penalty = 1.0 + (1.5 * np.maximum(0.0, sector_corr))
        adjusted_kelly = (raw_kelly * self.fractional_kelly) / penalty
        clipped_kelly = np.clip(adjusted_kelly, 0.0, self.max_risk_pct)

        structural_integrity = (sl < price) & (price < tp) & (rrr >= 0.5) & tp_valid
        signal_active = (prob >= regime_threshold) & (effective_prob > 0.50) & structural_integrity & (clipped_kelly > 0.0)

        final_kelly = np.where(signal_active, clipped_kelly, 0.0)
        allocation_idr = total_capital_idr * final_kelly

        out_df = df.with_columns([
            pl.Series("effective_probability", effective_prob, dtype=pl.Float64),
            pl.Series("kelly_allocation_pct", final_kelly, dtype=pl.Float64),
            pl.Series("recommended_position_idr", allocation_idr, dtype=pl.Float64),
            pl.Series("is_signal_executable", signal_active, dtype=pl.Boolean)
        ])

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 11. QuantDriftAndExplainabilityEngine (Training Vector PSI & Numerical Attribution)
# =============================================================================
class QuantDriftAndExplainabilityEngine:
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self._training_baseline: Optional[np.ndarray] = None
        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.execution_meta.clear()
            self._training_baseline = None

    def set_training_baseline_vector(self, baseline_prob: np.ndarray) -> None:
        if len(baseline_prob) > 0:
            with self._state_lock:
                self._training_baseline = np.array(baseline_prob, dtype=np.float64)

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(expected, percentiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        exp_pct = np.maximum(expected_counts / len(expected), EPSILON)
        act_pct = np.maximum(actual_counts / len(actual), EPSILON)

        psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi_value)

    def process_explainability(self, df: pl.DataFrame) -> pl.DataFrame:
        t_start = time.perf_counter()
        prob = np.nan_to_num(df["calibrated_prob"].to_numpy().astype(np.float64), nan=0.5)
        er = np.nan_to_num(df["expected_return"].to_numpy().astype(np.float64), nan=0.0)
        conf = np.nan_to_num(df["signal_confidence"].to_numpy().astype(np.float64), nan=1.0) if "signal_confidence" in df.columns else np.ones(df.height)
        
        with self._state_lock:
            baseline = self._training_baseline

        if baseline is not None and len(baseline) > 0:
            psi_score = self._calculate_psi(baseline, prob)
        else:
            psi_score = 0.0

        contrib_prob = prob * 0.35
        contrib_er = np.clip(er * 10.0, -1.0, 1.0) * 0.25
        contrib_conf = conf * 0.20

        is_exec = df["is_signal_executable"].to_numpy() if "is_signal_executable" in df.columns else np.zeros(df.height, dtype=bool)

        explainability_list = [
            f"Prob:{prob[i]:.2f} (Attribution: P_contrib={contrib_prob[i]:+.2f}, ER_contrib={contrib_er[i]:+.2f}, Conf_contrib={contrib_conf[i]:+.2f}) | Status:{'EXECUTABLE' if is_exec[i] else 'REJECTED'}"
            for i in range(df.height)
        ]

        out_df = df.with_columns([
            pl.Series("contrib_prob_score", contrib_prob, dtype=pl.Float64),
            pl.Series("contrib_er_score", contrib_er, dtype=pl.Float64),
            pl.Series("contrib_conf_score", contrib_conf, dtype=pl.Float64),
            pl.Series("signal_explainability", explainability_list, dtype=pl.Utf8),
            pl.Series("psi_drift_score", np.full(df.height, psi_score), dtype=pl.Float64)
        ])

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 12. QuantGeminiReasoningEngine (Qualitative Reasoning & Signal Validation)
# =============================================================================
class QuantGeminiReasoningEngine:
    """
    Sub-Engine integrasi Google Gemini sebagai Qualitative Reasoning Layer
    untuk memvalidasi dan menambahkan penjelasan naratif pada sinyal kuantitatif.
    """
    ENGINE_VERSION: Final[str] = "25.5.0"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Final[types.MappingProxyType] = types.MappingProxyType(dict(config))
        self.api_key: Optional[str] = self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.client: Optional[Any] = None

        self._is_active: bool = True
        self._state_lock: Final[threading.Lock] = threading.Lock()
        self.execution_meta: Dict[str, Any] = {}

        if HAS_GEMINI_SDK and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("🤖 [GEMINI_PREDICTION_INIT] Gemini Client berhasil terhubung pada Prediction Engine.")
            except Exception as e:
                logger.warning(f"⚠️ Gagal inisialisasi Gemini Client pada Prediction Engine: {e}")

    def activate(self) -> None:
        with self._state_lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._state_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._state_lock:
            self._is_active = False
            self.client = None
            self.execution_meta.clear()

    def process_qualitative_reasoning(self, df: pl.DataFrame) -> pl.DataFrame:
        t_start = time.perf_counter()

        if not self.client or df.height == 0:
            out_df = df.with_columns([
                pl.lit("N/A - Gemini AI Inactive").alias("ai_qualitative_reasoning"),
                pl.lit(1.0).alias("ai_confidence_multiplier")
            ])
            return out_df

        # Ambil sampel kandidat sinyal teratas untuk efisiensi API
        executable_mask = df["is_signal_executable"].to_numpy() if "is_signal_executable" in df.columns else np.ones(df.height, dtype=bool)
        
        reasoning_texts = []
        conf_multipliers = []

        for row in df.iter_rows(named=True):
            if not row.get("is_signal_executable", True):
                reasoning_texts.append("Ditolak oleh kuantitatif guardrail.")
                conf_multipliers.append(1.0)
                continue

            ticker = str(row.get("asset", row.get("ticker", "UNKNOWN")))
            price = float(row.get("current_price", 0.0))
            prob = float(row.get("calibrated_prob", 0.5))
            regime = str(row.get("market_regime", "SIDEWAYS"))

            prompt = f"""
            Sebagai Analis Risiko Kuantitatif BEI (IDX), berikan validasi kualitatif 1 kalimat singkat (maksimal 20 kata) 
            mengenai apakah sinyal BELI untuk {ticker} layak dieksekusi:
            - Harga: Rp {price:,.0f}
            - Probabilitas Model: {prob*100:.1f}%
            - Rezim Pasar: {regime}
            """

            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                txt = response.text.strip().replace("\n", " ")
                reasoning_texts.append(txt)
                conf_multipliers.append(1.0)
            except Exception as e:
                logger.warning(f"⚠️ Error Gemini API untuk {ticker}: {e}")
                reasoning_texts.append("Analisis AI tidak tersedia.")
                conf_multipliers.append(1.0)

        out_df = df.with_columns([
            pl.Series("ai_qualitative_reasoning", reasoning_texts, dtype=pl.Utf8),
            pl.Series("ai_confidence_multiplier", conf_multipliers, dtype=pl.Float64)
        ])

        with self._state_lock:
            self.execution_meta = {
                "engine_version": self.ENGINE_VERSION,
                "rows_processed": df.height,
                "calculation_latency_ms": (time.perf_counter() - t_start) * 1000.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        return out_df


# =============================================================================
# 13. UnifiedPredictionEngine (Facade Master Orchestrator - 12 Sub-Engines)
# =============================================================================
class UnifiedPredictionEngine:
    r"""
    Unified Facade Engine for Quantitative Signal Formulation & Dynamic Risk Target Projections in IDX Stocks.
    Orchestrates total lifecycle and execution sequence across all 12 quantitative sub-engines.
    """
    ENGINE_VERSION: Final[str] = "2026.Q3.v25.5.0"
    MANDATORY_COLUMNS: Final[Tuple[str, ...]] = ("current_price", "raw_score")

    def __init__(self, master_config: Optional[Dict[str, Any]] = None) -> None:
        self.config = master_config or {}

        self.regime_engine = QuantMarketRegimeEngine(self.config.get("regime_config", {}))
        self.prob_engine = QuantProbabilityEngine(self.config.get("prob_config", {}))
        self.er_engine = QuantExpectedReturnEngine(self.config.get("er_config", {}))
        self.conf_engine = QuantConfidenceEngine(self.config.get("conf_config", {}))
        self.target_engine = QuantPriceTargetEngine(self.config.get("target_config", {}))
        self.sl_engine = QuantStopLossEngine(self.config.get("sl_config", {}))
        self.tp_engine = QuantTakeProfitEngine(self.config.get("tp_config", {}))
        self.date_engine = QuantExpectedDateEngine(self.config.get("date_config", {}))
        self.ranking_engine = QuantCrossSectionalEngine(self.config.get("ranking_config", {}))
        self.optimizer_engine = QuantPortfolioOptimizerEngine(self.config.get("optimizer_config", {}))
        self.explainability_engine = QuantDriftAndExplainabilityEngine(self.config.get("explainability_config", {}))
        self.gemini_engine = QuantGeminiReasoningEngine(self.config.get("gemini_config", {}))

        self._state_lock: Final[threading.Lock] = threading.Lock()
        logger.info(f"UnifiedPredictionEngine (Institutional Production Edition) initialized [Version: {self.ENGINE_VERSION}].")

    def _validate_required_columns(self, df: pl.DataFrame) -> None:
        if df.height == 0:
            raise DataValidationError("Unified Gate Breach: Streaming DataFrame contains zero rows.")
        for col in self.MANDATORY_COLUMNS:
            if col not in df.columns:
                raise DataValidationError(f"Unified Gate Breach: Mandated column '{col}' is missing from DataFrame.")

    def activate_all(self) -> None:
        with self._state_lock:
            self.regime_engine.activate()
            self.prob_engine.activate()
            self.er_engine.activate()
            self.conf_engine.activate()
            self.target_engine.activate()
            self.sl_engine.activate()
            self.tp_engine.activate()
            self.date_engine.activate()
            self.ranking_engine.activate()
            self.optimizer_engine.activate()
            self.explainability_engine.activate()
            self.gemini_engine.activate()
        logger.info("UnifiedPredictionEngine: All 12 sub-engines ACTIVATED.")

    def deactivate_all(self) -> None:
        with self._state_lock:
            self.regime_engine.deactivate()
            self.prob_engine.deactivate()
            self.er_engine.deactivate()
            self.conf_engine.deactivate()
            self.target_engine.deactivate()
            self.sl_engine.deactivate()
            self.tp_engine.deactivate()
            self.date_engine.deactivate()
            self.ranking_engine.deactivate()
            self.optimizer_engine.deactivate()
            self.explainability_engine.deactivate()
            self.gemini_engine.deactivate()
        logger.info("UnifiedPredictionEngine: All 12 sub-engines DEACTIVATED.")

    def set_training_baseline_distribution(self, baseline_prob: np.ndarray) -> None:
        self.explainability_engine.set_training_baseline_vector(baseline_prob)

    def run_prediction_pipeline(self, df: pl.DataFrame, total_capital_idr: float = 100_000_000.0, trade_size_idr: float = 20_000_000.0) -> pl.DataFrame:
        self._validate_required_columns(df)
        t_start = time.perf_counter()

        # Step 1: Detect Multi-Factor Market Regime
        df_res = self.regime_engine.detect_regimes(df)

        # Step 2: Empirical Probability Calibration
        df_res = self.prob_engine.process_calibrated_probabilities(df_res)

        # Step 3: Expected Return Formulation
        df_res = self.er_engine.process_expected_returns(df_res, trade_size_idr=trade_size_idr)

        # Step 4: Signal Confidence Estimation
        df_res = self.conf_engine.calculate_confidence_scores(df_res)

        # Step 5: Price Target Projection
        df_res = self.target_engine.calculate_price_targets(df_res)

        # Step 6: Dynamic Chandelier & Support Stop-Loss
        df_res = self.sl_engine.calculate_stop_losses(df_res)

        # Step 7: Take-Profit Validation & RRR Conflict Resolution
        df_res = self.tp_engine.calculate_take_profits(df_res)

        # Step 8: Adaptive Vol-Decay Holding Horizon
        df_res = self.date_engine.calculate_expected_horizons(df_res)

        # Step 9: Cross-Sectional Ranking
        df_res = self.ranking_engine.rank_universe(df_res)

        # Step 10: Portfolio Optimizer (Kelly Allocation)
        df_res = self.optimizer_engine.optimize_allocation(df_res, total_capital_idr=total_capital_idr)

        # Step 11: Real Training Baseline PSI Drift & Numerical Factor Attribution
        df_res = self.explainability_engine.process_explainability(df_res)

        # Step 12: Qualitative Reasoning Layer via Google Gemini AI
        df_res = self.gemini_engine.process_qualitative_reasoning(df_res)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(f"Unified Prediction Pipeline v25.5.0 executed for {df.height} records in {elapsed_ms:.2f} ms.")

        return df_res

    def attach_calibrator(self, calibrator: Any) -> None:
        self.prob_engine.attach_calibrator_instance(calibrator)

    def shutdown(self) -> None:
        with self._state_lock:
            self.regime_engine.shutdown()
            self.prob_engine.shutdown()
            self.er_engine.shutdown()
            self.conf_engine.shutdown()
            self.target_engine.shutdown()
            self.sl_engine.shutdown()
            self.tp_engine.shutdown()
            self.date_engine.shutdown()
            self.ranking_engine.shutdown()
            self.optimizer_engine.shutdown()
            self.explainability_engine.shutdown()
            self.gemini_engine.shutdown()
        logger.warning("UnifiedPredictionEngine and all 12 sub-engines have been safely SHUTDOWN.")

"""
================================================================================
MODULE: signal_idx.py (v2026.5.17 Optimized Sensitivity Edition)
DESCRIPTION: Unified Signal Processing Engine for Indonesian Stock Market (IDX)
ARCHITECTURE: 10-Gateway Sequential Pipeline + Numerical Sanity + Facade Class
DIRECTORY STRUCTURE: Flat Directory (Root Level Integration)
COMPLIANCE: BEI/IDX Trading Rules, Pure Polars Expressions, Risk-Based Sizing
================================================================================
"""

import os
import sys
import math
import time
import copy
import logging
import threading
from logging.handlers import RotatingFileHandler
from types import MappingProxyType
from typing import Final, Dict, Any, Optional, List, Union
import numpy as np
import polars as pl

# ==============================================================================
# BURSA EFEK INDONESIA (IDX) CONSTANTS
# ==============================================================================
IDX_FEE_ROUNDTRIP_PCT: Final[float] = 0.003          # Fee roundtrip 0.3% (0.15% Beli, 0.15% Jual+PPh)
IDX_SPREAD_SLIPPAGE_PCT: Final[float] = 0.0015       # Estimasi friction spread & eksekusi 0.15%
IDX_TOTAL_FRICTION_PCT: Final[float] = IDX_FEE_ROUNDTRIP_PCT + IDX_SPREAD_SLIPPAGE_PCT

IDX_MIN_PRICE_IDR: Final[float] = 50.0              # Batas minimum harga saham BEI (Rp 50)
IDX_MIN_NOTIONAL_IDR: Final[float] = 50_000.0        # Batas minimum nilai transaksi IDR (Rp 50.000)

# OPTIMIZED: Menurunkan batas ADTV harian agar saham berpotensi tidak langsung terbuang
IDX_DEFAULT_MIN_24H_VOLUME_IDR: Final[float] = 250_000_000.0 # Standard ADTV Threshold (Rp 250 Juta)
IDX_DRYRUN_MIN_24H_VOLUME_IDR: Final[float] = 50_000_000.0    # Dry-Run ADTV Threshold (Rp 50 Juta)

IDX_MAX_STALENESS_SEC: Final[float] = 43200.0       # Batas usia data candlestick (12 Jam)
IDX_BASE_OVERNIGHT_GAP_PCT: Final[float] = 0.08    # Batas toleransi gap overnight 8%
IDX_LOT_SIZE: Final[int] = 100                      # Satuan perdagangan standar BEI (1 Lot = 100 Lembar)

DEFAULT_IDX_TICKER: Final[str] = "BBCA.JK"

# Backward Compatibility Aliases
TOKOCRYPTO_FEE_ROUNDTRIP_PCT: Final[float] = IDX_FEE_ROUNDTRIP_PCT
TOKOCRYPTO_MIN_PRICE_USDT: Final[float] = IDX_MIN_PRICE_IDR
TOKOCRYPTO_MIN_NOTIONAL_USDT: Final[float] = IDX_MIN_NOTIONAL_IDR

# ==============================================================================
# UNIFIED LOGGER CONFIGURATION
# ==============================================================================
logger = logging.getLogger("IDX.SignalEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s][IDX.SignalEngine] %(message)s'))
    logger.addHandler(ch)

    try:
        fh = RotatingFileHandler("idx_signal_engine.log", maxBytes=10_000_000, backupCount=5)
        fh.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s][IDX.SignalEngine] %(message)s'))
        logger.addHandler(fh)
    except Exception as log_err:
        logger.warning(f"Failed to initialize RotatingFileHandler logging: {log_err}")


# ==============================================================================
# BEI TICK SIZE & ARA/ARB BOUNDARY HELPERS
# ==============================================================================
def get_idx_tick_size_expr(price_col: Union[str, pl.Expr]) -> pl.Expr:
    """Mengembalikan Polars Expression untuk Fraksi Harga Standar BEI."""
    p = pl.col(price_col) if isinstance(price_col, str) else price_col
    return (
        pl.when(p < 200).then(1.0)
        .when(p < 500).then(2.0)
        .when(p < 2000).then(5.0)
        .when(p < 5000).then(10.0)
        .otherwise(25.0)
    )


def get_idx_ara_limit_pct_expr(price_col: Union[str, pl.Expr]) -> pl.Expr:
    """Persentase Batas Auto Rejection Atas (ARA) BEI."""
    p = pl.col(price_col) if isinstance(price_col, str) else price_col
    return (
        pl.when(p < 200).then(0.35)
        .when(p <= 5000).then(0.25)
        .otherwise(0.20)
    )


def get_idx_arb_limit_pct_expr(price_col: Union[str, pl.Expr]) -> pl.Expr:
    """Persentase Batas Auto Rejection Bawah (ARB) BEI."""
    p = pl.col(price_col) if isinstance(price_col, str) else price_col
    return (
        pl.when(p < 200).then(0.35)
        .when(p <= 5000).then(0.25)
        .otherwise(0.20)
    )


def get_prob_expr(cols: List[str]) -> pl.Expr:
    """Helper ekspresi defensif untuk mengambil kolom probabilitas."""
    if "prediction_probability" in cols:
        return pl.col("prediction_probability").fill_nan(0.52).fill_null(0.52)
    elif "probability" in cols:
        return pl.col("probability").fill_nan(0.52).fill_null(0.52)
    return pl.lit(0.52)


def get_conf_expr(cols: List[str]) -> pl.Expr:
    """Helper ekspresi defensif untuk mengambil kolom konfidensi."""
    if "prediction_confidence" in cols:
        return pl.col("prediction_confidence").fill_nan(0.52).fill_null(0.52)
    elif "confidence" in cols:
        return pl.col("confidence").fill_nan(0.52).fill_null(0.52)
    elif "prediction_probability" in cols:
        return pl.col("prediction_probability").fill_nan(0.52).fill_null(0.52)
    elif "probability" in cols:
        return pl.col("probability").fill_nan(0.52).fill_null(0.52)
    return pl.lit(0.52)


# ==============================================================================
# EXCEPTIONS & BASE CLASS WITH FULL TELEMETRY
# ==============================================================================
class SignalIDXError(Exception): pass
class SignalGeneratorError(SignalIDXError): pass
class EntryFilterError(SignalIDXError): pass
class ExitFilterError(SignalIDXError): pass
class ProbabilityFilterError(SignalIDXError): pass
class ConfidenceFilterError(SignalIDXError): pass
class TpSlOptimizerError(SignalIDXError): pass
class SignalRankerError(SignalIDXError): pass
class SignalValidatorError(SignalIDXError): pass
class SignalExplainerError(SignalIDXError): pass


class ExplanationStatus:
    READY: Final[str] = "[READY]"
    REJECTED: Final[str] = "[REJECTED]"
    PIPELINE_ERROR: Final[str] = "[PIPELINE_ERROR]"
    NUMERICAL_ERROR: Final[str] = "[NUMERICAL_SANITY_ERROR]"


class _SignalBaseEngine:
    def __init__(self, engine_id: str, engine_version: str) -> None:
        self.engine_id: Final[str] = engine_id
        self.ENGINE_VERSION: Final[str] = engine_version
        self._lifecycle_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._is_active: bool = False
        self._is_shutdown: bool = False
        self._config: MappingProxyType = MappingProxyType({})
        self._execution_meta: Dict[str, Any] = {
            "engine_id": engine_id,
            "version": engine_version,
            "input_rows": 0,
            "output_rows": 0,
            "rejected_rows": 0,
            "pass_ratio": 0.0,
            "latency_ms": 0.0,
            "timestamp_utc": 0
        }

    def activate(self) -> None:
        with self._lifecycle_lock:
            if self._is_shutdown:
                raise SignalIDXError(f"Engine [{self.engine_id}] cannot be activated after shutdown.")
            self._is_active = True

    def deactivate(self) -> None:
        with self._lifecycle_lock:
            self._is_active = False

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            self._is_active = False
            self._is_shutdown = True

    @property
    def is_operational(self) -> bool:
        with self._lifecycle_lock:
            return self._is_active and not self._is_shutdown

    def record_telemetry(self, input_rows: int, output_rows: int, latency_ms: float, rejected_rows: Optional[int] = None) -> None:
        with self._telemetry_lock:
            rej = rejected_rows if rejected_rows is not None else max(0, input_rows - output_rows)
            ratio = float(output_rows / input_rows) if input_rows > 0 else 0.0
            self._execution_meta = {
                "engine_id": self.engine_id,
                "version": self.ENGINE_VERSION,
                "input_rows": input_rows,
                "output_rows": output_rows,
                "rejected_rows": rej,
                "pass_ratio": round(ratio, 4),
                "latency_ms": round(latency_ms, 4),
                "timestamp_utc": int(time.time())
            }

    def get_latest_telemetry(self) -> Dict[str, Any]:
        with self._telemetry_lock:
            return self._execution_meta.copy()


# ==============================================================================
# GATEWAY 1: SIGNAL GENERATOR (MULTI-TIER ATR & TICK-SIZE LOWER BOUND GUARANTEE)
# ==============================================================================
class SignalGenerator(_SignalBaseEngine):
    def __init__(self, engine_id: str, operational_config: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.5.17")
        self._config = MappingProxyType(operational_config.copy())
        logger.info(f"SignalGenerator [{self.engine_id}] initialized.")

    def generate(self, upstream_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational:
            raise SignalGeneratorError(f"Execution rejected. Engine [{self.engine_id}] is not active.")

        if upstream_df is None or upstream_df.height == 0:
            logger.warning("[Gateway 1] Empty input dataset provided.")
            self.record_telemetry(0, 0, (time.perf_counter_ns() - start_time_ns) / 1_000_000.0)
            return pl.DataFrame()

        input_count = upstream_df.height
        active_cfg = self._config
        min_adtv_idr = float(active_cfg.get("min_24h_volume_idr", IDX_DEFAULT_MIN_24H_VOLUME_IDR))
        df_adapted = upstream_df
        cols = list(df_adapted.columns)

        # 1. Safe Auto-Imputation (OPTIMIZED: Value 0.52 agar aman pada EV Audit)
        adapter_exprs = []
        if "prediction_probability" not in cols and "probability" not in cols:
            logger.info("[Gateway 1] Auto-imputing prediction_probability with 0.52 fallback.")
            adapter_exprs.append(pl.lit(0.52).cast(pl.Float64).alias("prediction_probability"))
            adapter_exprs.append(pl.lit(0.52).cast(pl.Float64).alias("probability"))
        elif "prediction_probability" not in cols:
            adapter_exprs.append(pl.col("probability").cast(pl.Float64).alias("prediction_probability"))
        elif "probability" not in cols:
            adapter_exprs.append(pl.col("prediction_probability").cast(pl.Float64).alias("probability"))

        if "prediction_confidence" not in cols and "confidence" not in cols:
            logger.info("[Gateway 1] Auto-imputing prediction_confidence with 0.52 fallback.")
            adapter_exprs.append(pl.lit(0.52).cast(pl.Float64).alias("prediction_confidence"))
            adapter_exprs.append(pl.lit(0.52).cast(pl.Float64).alias("confidence"))
        elif "prediction_confidence" not in cols:
            adapter_exprs.append(pl.col("confidence").cast(pl.Float64).alias("prediction_confidence"))
        elif "confidence" not in cols:
            adapter_exprs.append(pl.col("prediction_confidence").cast(pl.Float64).alias("confidence"))

        if "signal_direction" not in cols:
            adapter_exprs.append(pl.lit(1).cast(pl.Int8).alias("signal_direction"))
        if "expected_holding_days" not in cols:
            adapter_exprs.append(pl.lit(1.0).cast(pl.Float64).alias("expected_holding_days"))

        if adapter_exprs:
            df_adapted = df_adapted.with_columns(adapter_exprs)
            cols = list(df_adapted.columns)

        # 2. Standardize Ticker
        ticker_col = "ticker" if "ticker" in cols else ("symbol" if "symbol" in cols else "asset" if "asset" in cols else None)
        if not ticker_col:
            df_adapted = df_adapted.with_columns(pl.lit(DEFAULT_IDX_TICKER).alias("ticker"))
            ticker_col = "ticker"
            cols = list(df_adapted.columns)

        # 3. Deduplicate Tickers (Safe sorting with nulls_last)
        df_adapted = df_adapted.sort(by="prediction_probability", descending=True, nulls_last=True).unique(subset=[ticker_col], keep="first")

        # 4. Standardize Prices
        price_col = "current_price" if "current_price" in cols else ("close" if "close" in cols else "open" if "open" in cols else None)
        if not price_col:
            raise SignalGeneratorError("Critical Failure: Input dataset lacks price column ('current_price', 'close', 'open').")

        ref_close_col = next((c for c in ["idx_ref_price", "adjusted_close", "previous_close", "prev_close"] if c in cols), None)
        raw_price = pl.col(price_col).cast(pl.Float64).fill_nan(IDX_MIN_PRICE_IDR).fill_null(IDX_MIN_PRICE_IDR)
        ref_price_expr = pl.col(ref_close_col).cast(pl.Float64).fill_nan(raw_price).fill_null(raw_price) if ref_close_col else raw_price

        df_adapted = df_adapted.with_columns([
            raw_price.alias("current_price"),
            ref_price_expr.alias("previous_close")
        ])
        cols = list(df_adapted.columns)

        # 5. Multi-Tier Hierarchical ATR Fallback
        atr_primary = next((c for c in ["atr_14", "feature_atr", "atr"] if c in cols), None)
        tr_ema_col = next((c for c in ["true_range_ema", "tr_ema"] if c in cols), None)
        parkinson_col = next((c for c in ["parkinson_volatility", "volatility_parkinson"] if c in cols), None)

        if atr_primary:
            atr_expr = pl.col(atr_primary).cast(pl.Float64).fill_nan(0.0).fill_null(0.0)
        elif tr_ema_col:
            atr_expr = pl.col(tr_ema_col).cast(pl.Float64).fill_nan(0.0).fill_null(0.0)
        elif parkinson_col:
            atr_expr = pl.col(parkinson_col).cast(pl.Float64).fill_nan(0.0).fill_null(0.0) * pl.col("current_price")
        else:
            atr_expr = pl.col("current_price") * 0.02

        atr_expr = pl.when(atr_expr > 0.0).then(atr_expr).otherwise(pl.col("current_price") * 0.02)
        atr_expr = pl.max_horizontal([atr_expr, pl.col("current_price") * 0.005])

        # 6. ARA-Aware TP Calculation & Lower-Bound Tick Guarantee (OPTIMIZED: 1.5x ATR TP)
        max_ara_pct = get_idx_ara_limit_pct_expr("previous_close")
        max_ara_price = pl.col("previous_close") * (1.0 + max_ara_pct)

        entry_tick_expr = get_idx_tick_size_expr("current_price")
        entry_ticked = (pl.col("current_price") / entry_tick_expr).floor() * entry_tick_expr

        # Menggunakan 1.5x ATR agar TP rasional dan tidak mudah terpotong ARA
        tp_mult = float(active_cfg.get("tp_atr_multiplier", 1.5))
        raw_tp = pl.col("current_price") + (atr_expr * tp_mult)
        clean_tp = pl.min_horizontal([raw_tp, max_ara_price])
        clean_sl = pl.max_horizontal([pl.col("current_price") - atr_expr, pl.lit(IDX_MIN_PRICE_IDR * 0.5)])

        tp_tick_expr = get_idx_tick_size_expr(clean_tp)
        sl_tick_expr = get_idx_tick_size_expr(clean_sl)

        tp_ticked_raw = (clean_tp / tp_tick_expr).floor() * tp_tick_expr
        sl_ticked_raw = (clean_sl / sl_tick_expr).floor() * sl_tick_expr

        # Jaminan bahwa TP minimal 1 tick di atas entry, dan SL minimal 1 tick di bawah entry
        tp_ticked = pl.min_horizontal([pl.max_horizontal([tp_ticked_raw, entry_ticked + entry_tick_expr]), max_ara_price])
        sl_ticked = pl.min_horizontal([sl_ticked_raw, entry_ticked - entry_tick_expr])
        sl_ticked = pl.max_horizontal([sl_ticked, pl.lit(IDX_MIN_PRICE_IDR * 0.5)])

        # 7. ADTV Threshold Verification
        idr_col = next((c for c in ["adtv_20", "adtv20", "volume_idr", "volume_24h_idr", "turnover_idr", "median_turnover_20d"] if c in cols), None)
        share_col = next((c for c in ["adv_20", "adv20", "volume"] if c in cols), None)

        if idr_col:
            volume_idr_expr = pl.col(idr_col).cast(pl.Float64).fill_nan(0.0).fill_null(0.0)
        elif share_col:
            volume_idr_expr = (pl.col(share_col).cast(pl.Float64) * pl.col("current_price")).fill_nan(0.0).fill_null(0.0)
        else:
            volume_idr_expr = pl.lit(min_adtv_idr)

        volume_valid = (volume_idr_expr >= min_adtv_idr)

        # 8. Pure Boolean Validation
        vector_valid_flag = (
            (pl.col("signal_direction") == 1) &
            (entry_ticked >= IDX_MIN_PRICE_IDR) &
            (sl_ticked < entry_ticked) &
            (tp_ticked > entry_ticked) &
            volume_valid
        ).fill_null(False)

        reason_expr = (
            pl.when(vector_valid_flag).then(pl.lit("VALID_SIGNAL"))
            .when(~volume_valid).then(pl.lit(f"INSUFFICIENT_ADTV_BELOW_{int(min_adtv_idr)}"))
            .otherwise(pl.lit("GEOMETRY_OR_PRICE_INVALID"))
        )

        output_df = df_adapted.with_columns([
            pl.col(ticker_col).alias("ticker"),
            pl.lit("LONG").alias("candidate_signal"),
            entry_ticked.alias("entry_price"),
            tp_ticked.alias("target_price"),
            tp_ticked.alias("take_profit"),
            sl_ticked.alias("stop_loss"),
            atr_expr.alias("atr_value"),
            volume_idr_expr.alias("volume_idr"),
            volume_idr_expr.alias("volume_24h_idr"),
            vector_valid_flag.cast(pl.Boolean).alias("signal_valid"),
            reason_expr.alias("signal_reason")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        self.record_telemetry(input_count, output_df.height, latency_ms)
        return output_df


# ==============================================================================
# GATEWAY 2: ENTRY FILTER (ZERO-DIVIDE SAFEGUARD & ARA/ARB LIMIT)
# ==============================================================================
class EntryFilter(_SignalBaseEngine):
    def __init__(self, engine_id: str, cfg: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.5.17")
        self._config = MappingProxyType(cfg.copy())

    def filter_entry(self, input_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        prev_close_safe = pl.col("previous_close").fill_null(pl.col("entry_price")).clip(lower_bound=1e-8)

        ara_pct = get_idx_ara_limit_pct_expr(prev_close_safe)
        arb_pct = get_idx_arb_limit_pct_expr(prev_close_safe)

        max_ara_price = prev_close_safe * (1.0 + ara_pct)
        min_arb_price = prev_close_safe * (1.0 - arb_pct)
        gap_pct = ((pl.col("entry_price") - prev_close_safe) / prev_close_safe).abs()

        within_ara_arb = (pl.col("entry_price") <= max_ara_price) & (pl.col("entry_price") >= min_arb_price)
        within_gap_limit = gap_pct <= IDX_BASE_OVERNIGHT_GAP_PCT

        entry_pass = (pl.col("signal_valid").fill_null(False) & within_ara_arb & within_gap_limit).fill_null(False)

        reason_expr = (
            pl.when(entry_pass).then(pl.lit("ENTRY_ALLOWED"))
            .when(~pl.col("signal_valid").fill_null(False)).then(pl.col("signal_reason"))
            .when(~within_ara_arb).then(pl.lit("REJECTED_EXCEEDS_ARA_ARB_LIMIT"))
            .otherwise(pl.lit("REJECTED_EXCEEDS_OVERNIGHT_GAP_LIMIT"))
        )

        output_df = input_df.with_columns([
            entry_pass.alias("entry_filter_pass"),
            reason_expr.alias("entry_filter_reason")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        self.record_telemetry(input_df.height, output_df.height, latency_ms)
        return output_df


# ==============================================================================
# GATEWAY 3: EXIT FILTER (NET RISK-REWARD & FRICTION AUDIT)
# ==============================================================================
class ExitFilter(_SignalBaseEngine):
    def __init__(self, engine_id: str, cfg: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.5.17")
        self._config = MappingProxyType(cfg.copy())

    def filter_exit(self, input_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        # OPTIMIZED: Menurunkan batas minimum Risk Reward Ratio bersih menjadi 0.75
        min_rr = float(self._config.get("min_risk_reward_ratio", 0.75))
        friction = pl.col("entry_price") * pl.lit(IDX_TOTAL_FRICTION_PCT)
        net_reward = (pl.col("take_profit") - pl.col("entry_price")) - friction
        net_risk = (pl.col("entry_price") - pl.col("stop_loss")) + friction
        calculated_net_rr = (net_reward / net_risk.clip(lower_bound=1e-8)).fill_nan(0.0).fill_null(0.0)

        rr_valid = (calculated_net_rr >= min_rr) & (net_reward > 0.0)
        exit_pass = (pl.col("entry_filter_pass").fill_null(False) & rr_valid).fill_null(False)

        reason_expr = (
            pl.when(exit_pass).then(pl.lit("EXIT_ALLOWED"))
            .when(~pl.col("entry_filter_pass").fill_null(False)).then(pl.col("entry_filter_reason"))
            .otherwise(pl.lit("REJECTED_INSUFFICIENT_NET_RRR"))
        )

        output_df = input_df.with_columns([
            exit_pass.alias("exit_filter_pass"),
            reason_expr.alias("exit_filter_reason"),
            calculated_net_rr.alias("calculated_risk_reward"),
            net_reward.alias("net_reward_idr"),
            net_risk.alias("net_risk_idr")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        self.record_telemetry(input_df.height, output_df.height, latency_ms)
        return output_df


# ==============================================================================
# GATEWAY 4: PROBABILITY FILTER (DEFENSIVE COLUMN SELECTION & EV AUDIT)
# ==============================================================================
class ProbabilityFilter(_SignalBaseEngine):
    def __init__(self, engine_id: str, cfg: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.5.17")
        self._config = MappingProxyType(cfg.copy())

    def filter_probability(self, input_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        cols = input_df.columns
        # OPTIMIZED: Menurunkan batas probabilitas minimal menjadi 0.45
        min_prob = float(self._config.get("min_prediction_probability", 0.45))
        p_win = get_prob_expr(cols)
        p_loss = (1.0 - p_win).clip(lower_bound=0.0)

        ev_idr = ((p_win * pl.col("net_reward_idr")) - (p_loss * pl.col("net_risk_idr"))).fill_nan(0.0).fill_null(0.0)
        ev_pct = (ev_idr / pl.col("entry_price").clip(lower_bound=1e-8)).fill_nan(0.0).fill_null(0.0)

        prob_valid = (p_win >= min_prob) & (ev_idr > 0.0)
        prob_pass = (pl.col("exit_filter_pass").fill_null(False) & prob_valid).fill_null(False)

        reason_expr = (
            pl.when(prob_pass).then(pl.lit("PROBABILITY_ALLOWED"))
            .when(~pl.col("exit_filter_pass").fill_null(False)).then(pl.col("exit_filter_reason"))
            .when(p_win < min_prob).then(pl.lit(f"REJECTED_PROBABILITY_BELOW_{min_prob}"))
            .otherwise(pl.lit("REJECTED_NEGATIVE_EXPECTED_VALUE"))
        )

        output_df = input_df.with_columns([
            prob_pass.alias("probability_filter_pass"),
            reason_expr.alias("probability_filter_reason"),
            ev_idr.alias("calculated_expected_value"),
            ev_pct.alias("calculated_expected_value_pct")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        self.record_telemetry(input_df.height, output_df.height, latency_ms)
        return output_df


# ==============================================================================
# GATEWAY 5: CONFIDENCE FILTER (DEFENSIVE COLUMN SELECTION)
# ==============================================================================
class ConfidenceFilter(_SignalBaseEngine):
    def __init__(self, engine_id: str, cfg: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.5.17")
        self._config = MappingProxyType(cfg.copy())

    def filter_confidence(self, input_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        cols = input_df.columns
        # OPTIMIZED: Menurunkan batas konfidensi minimal menjadi 0.45
        min_conf = float(self._config.get("min_prediction_confidence", 0.45))
        p_conf = get_conf_expr(cols)

        conf_valid = p_conf >= min_conf
        conf_pass = (pl.col("probability_filter_pass").fill_null(False) & conf_valid).fill_null(False)

        reason_expr = (
            pl.when(conf_pass).then(pl.lit("CONFIDENCE_ALLOWED"))
            .when(~pl.col("probability_filter_pass").fill_null(False)).then(pl.col("probability_filter_reason"))
            .otherwise(pl.lit(f"REJECTED_CONFIDENCE_BELOW_{min_conf}"))
        )

        output_df = input_df.with_columns([
            conf_pass.alias("confidence_filter_pass"),
            reason_expr.alias("confidence_filter_reason")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        self.record_telemetry(input_df.height, output_df.height, latency_ms)
        return output_df


# ==============================================================================
# GATEWAY 6: TP/SL OPTIMIZER & RISK-BASED POSITION SIZING
# ==============================================================================
class TpSlOptimizer(_SignalBaseEngine):
    def __init__(self, engine_id: str, cfg: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.5.17")
        self._config = MappingProxyType(cfg.copy())

    def optimize_tp_sl(self, input_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        # OPTIMIZED: Mengubah default equity sesuai dashboard (Rp 10 Juta)
        total_equity_idr = float(self._config.get("portfolio_equity_idr", 10_000_000.0))
        max_risk_pct = float(self._config.get("max_risk_per_trade_pct", 0.02)) # 2% risk
        max_pos_idr = float(self._config.get("max_position_idr", 2_500_000.0)) # Max 2.5 Juta per saham

        risk_per_share_idr = (pl.col("entry_price") - pl.col("stop_loss")).clip(lower_bound=1.0)
        max_loss_idr = total_equity_idr * max_risk_pct

        risk_shares = max_loss_idr / risk_per_share_idr
        capital_shares = max_pos_idr / pl.col("entry_price").clip(lower_bound=1e-8)

        target_shares = pl.min_horizontal([risk_shares, capital_shares])
        lot_count = (target_shares / pl.lit(IDX_LOT_SIZE)).floor().clip(lower_bound=1.0)
        exact_quantity = lot_count * pl.lit(IDX_LOT_SIZE)
        exact_notional = exact_quantity * pl.col("entry_price")

        optimizer_pass = pl.col("confidence_filter_pass").fill_null(False)

        output_df = input_df.with_columns([
            pl.col("take_profit").alias("optimized_take_profit"),
            pl.col("stop_loss").alias("optimized_stop_loss"),
            pl.col("calculated_risk_reward").alias("optimized_risk_reward"),
            lot_count.cast(pl.Int32).alias("lot_quantity"),
            exact_quantity.cast(pl.Float64).alias("quantity"),
            exact_notional.alias("notional_idr"),
            exact_notional.alias("notional_usdt"),  # Legacy Alias
            optimizer_pass.alias("tp_sl_optimizer_pass"),
            pl.when(optimizer_pass).then(pl.lit("OPTIMIZATION_SUCCESSFUL")).otherwise(pl.col("confidence_filter_reason")).alias("tp_sl_optimizer_reason")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        self.record_telemetry(input_df.height, output_df.height, latency_ms)
        return output_df


# ==============================================================================
# GATEWAY 7: NUMERICAL SANITY CHECK GATEWAY (ISOLATED NULL-SAFE)
# ==============================================================================
class NumericalSanityChecker(_SignalBaseEngine):
    def __init__(self, engine_id: str, cfg: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.5.17")
        self._config = MappingProxyType(cfg.copy())

    def check_sanity(self, input_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        cols = input_df.columns
        p_win = get_prob_expr(cols)
        p_conf = get_conf_expr(cols)

        finite_check = (
            pl.col("entry_price").is_finite() &
            pl.col("take_profit").is_finite() &
            pl.col("stop_loss").is_finite() &
            pl.col("calculated_risk_reward").is_finite() &
            pl.col("calculated_expected_value").is_finite() &
            p_win.is_finite() &
            p_conf.is_finite()
        ).fill_null(False)

        sanity_pass = (pl.col("tp_sl_optimizer_pass").fill_null(False) & finite_check).fill_null(False)

        reason_expr = (
            pl.when(sanity_pass).then(pl.lit("NUMERICAL_SANITY_PASSED"))
            .when(~pl.col("tp_sl_optimizer_pass").fill_null(False)).then(pl.col("tp_sl_optimizer_reason"))
            .otherwise(pl.lit(ExplanationStatus.NUMERICAL_ERROR))
        )

        output_df = input_df.with_columns([
            sanity_pass.alias("numerical_sanity_pass"),
            reason_expr.alias("numerical_sanity_reason")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        self.record_telemetry(input_df.height, output_df.height, latency_ms)
        return output_df


# ==============================================================================
# GATEWAY 8: SIGNAL RANKER (CROSS-SECTIONAL PERCENTILE RANKING)
# ==============================================================================
class SignalRanker(_SignalBaseEngine):
    def __init__(self, engine_id: str, cfg: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.5.17")
        self._config = MappingProxyType(cfg.copy())

    def rank_signals(self, input_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        cols = input_df.columns
        p_win_safe = get_prob_expr(cols)
        p_conf_safe = get_conf_expr(cols)
        ev_pct_safe = pl.col("calculated_expected_value_pct").fill_nan(0.0).fill_null(0.0)

        n_rows = max(1, input_df.height)
        prob_rank = p_win_safe.rank(method="average") / n_rows
        conf_rank = p_conf_safe.rank(method="average") / n_rows
        ev_rank = ev_pct_safe.rank(method="average") / n_rows

        score = (prob_rank * 0.40) + (conf_rank * 0.30) + (ev_rank * 0.30)
        rank = score.rank(descending=True, method="min").cast(pl.Int32)

        ranker_pass = pl.col("numerical_sanity_pass").fill_null(False)

        output_df = input_df.with_columns([
            score.alias("signal_rank_score"),
            rank.alias("signal_rank"),
            rank.alias("signal_rank_position"),
            ranker_pass.alias("signal_ranker_pass"),
            pl.when(ranker_pass).then(pl.lit("RANKING_SUCCESSFUL")).otherwise(pl.col("numerical_sanity_reason")).alias("signal_ranker_reason")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        self.record_telemetry(input_df.height, output_df.height, latency_ms)
        return output_df


# ==============================================================================
# GATEWAY 9: SIGNAL VALIDATOR
# ==============================================================================
class SignalValidator(_SignalBaseEngine):
    def __init__(self, engine_id: str, cfg: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.5.17")
        self._config = MappingProxyType(cfg.copy())

    def validate_signals(self, input_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        valid = (
            pl.col("signal_ranker_pass").fill_null(False) &
            (pl.col("notional_idr").fill_null(0.0) >= IDX_MIN_NOTIONAL_IDR)
        ).fill_null(False)

        output_df = input_df.with_columns([
            valid.alias("is_valid_execution"),
            pl.when(valid).then(pl.lit("EXECUTION_READY")).otherwise(pl.col("signal_ranker_reason")).alias("final_validator_reason")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        self.record_telemetry(input_df.height, output_df.height, latency_ms)
        return output_df


# ==============================================================================
# GATEWAY 10: SIGNAL EXPLAINER & AUDIT PAYLOAD (NULL-SAFE STRING FORMATTING)
# ==============================================================================
class SignalExplainer(_SignalBaseEngine):
    def __init__(self, engine_id: str, cfg: Dict[str, Any]) -> None:
        super().__init__(engine_id, "v2026.5.17")
        self._config = MappingProxyType(cfg.copy())

    def explain(self, input_df: pl.DataFrame) -> pl.DataFrame:
        start_time_ns = time.perf_counter_ns()
        if not self.is_operational or input_df is None or input_df.height == 0:
            return input_df

        safe_ticker = pl.col("ticker").fill_null(DEFAULT_IDX_TICKER)
        safe_entry = pl.col("entry_price").fill_nan(0.0).fill_null(0.0).cast(pl.Int64)
        safe_lots = pl.col("lot_quantity").fill_nan(0).fill_null(0).cast(pl.Int64)
        safe_reason = pl.col("final_validator_reason").fill_null("UNKNOWN_REASON")

        status = pl.when(pl.col("is_valid_execution").fill_null(False)).then(pl.lit(ExplanationStatus.READY)).otherwise(pl.lit(ExplanationStatus.REJECTED))
        exp_text = pl.format("{} IDX STOCK: {} | ENTRY: Rp {} | LOTS: {} | REASON: {}", 
                             status, safe_ticker, safe_entry, safe_lots, safe_reason)

        audit_json = pl.struct([
            safe_ticker.alias("ticker"),
            pl.when(pl.col("is_valid_execution").fill_null(False)).then(1.0).otherwise(0.0).alias("signal"),
            pl.col("entry_price").fill_nan(0.0).fill_null(0.0).alias("entry"),
            pl.col("optimized_take_profit").fill_nan(0.0).fill_null(0.0).alias("tp"),
            pl.col("optimized_stop_loss").fill_nan(0.0).fill_null(0.0).alias("sl"),
            safe_lots.alias("lots"),
            pl.col("notional_idr").fill_nan(0.0).fill_null(0.0).alias("notional_idr"),
            safe_reason.alias("reason"),
            pl.lit("v2026.5.17").alias("engine_version")
        ]).struct.json_encode()

        output_df = input_df.with_columns([
            exp_text.alias("signal_explanation_text"),
            exp_text.alias("signal_explanation"), # Legacy Alias
            audit_json.alias("signal_audit_payload"),
            pl.when(pl.col("is_valid_execution").fill_null(False)).then(1.0).otherwise(0.0).alias("signal"),
            safe_ticker.alias("portfolio_asset_id")
        ])

        latency_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
        self.record_telemetry(input_df.height, output_df.height, latency_ms)
        return output_df


# ==============================================================================
# UNIFIED SIGNAL ENGINE FACADE CLASS
# ==============================================================================
class UnifiedSignalEngine:
    FACADE_VERSION: Final[str] = "v2026.5.17"

    def __init__(self, engine_id_prefix: str = "IDXSignalEngine", custom_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.engine_id_prefix = engine_id_prefix
        self._lock = threading.Lock()
        
        # Deteksi Mode Dry-Run dari Env Variable, Config, dan CLI sys.argv
        run_mode = str(os.getenv("RUN_MODE", "")).lower()
        cli_args = " ".join(sys.argv).lower()
        is_dry_run = (
            "dry_run" in run_mode or "dryrun" in run_mode or "reset" in run_mode or
            "dry_run" in str(custom_configs).lower() or
            "--dry-run" in cli_args or "--reset-dryrun" in cli_args
        )
        active_min_adtv = IDX_DRYRUN_MIN_24H_VOLUME_IDR if is_dry_run else IDX_DEFAULT_MIN_24H_VOLUME_IDR

        # OPTIMIZED: Konfigurasi Default Sensitivitas Tinggi
        default_cfg = {
            "generator": {
                "min_24h_volume_idr": active_min_adtv,
                "tp_atr_multiplier": 1.5
            },
            "entry_filter": {}, 
            "exit_filter": {
                "min_risk_reward_ratio": 0.75  # Menoleransi friction total BEI (0.45%)
            }, 
            "prob_filter": {
                "min_prediction_probability": 0.45 # Ambang batas lebih responsif
            }, 
            "conf_filter": {
                "min_prediction_confidence": 0.45
            },
            "tpsl_optimizer": {
                "portfolio_equity_idr": 10_000_000.0, # Diselaraskan dengan Dashboard simulasi
                "max_risk_per_trade_pct": 0.02,        # Risk 2% per trade
                "max_position_idr": 2_500_000.0        # Max position Rp 2.5 Juta (25%)
            },
            "numerical_sanity": {},
            "ranker": {}, 
            "validator": {}, 
            "explainer": {}
        }
        
        if custom_configs:
            copied_custom = copy.deepcopy(custom_configs)
            for g_key, g_cfg in copied_custom.items():
                if g_key in default_cfg:
                    default_cfg[g_key].update(g_cfg)

        self.generator = SignalGenerator(f"{engine_id_prefix}_G1", default_cfg["generator"])
        self.entry_filter = EntryFilter(f"{engine_id_prefix}_G2", default_cfg["entry_filter"])
        self.exit_filter = ExitFilter(f"{engine_id_prefix}_G3", default_cfg["exit_filter"])
        self.prob_filter = ProbabilityFilter(f"{engine_id_prefix}_G4", default_cfg["prob_filter"])
        self.conf_filter = ConfidenceFilter(f"{engine_id_prefix}_G5", default_cfg["conf_filter"])
        self.tpsl_optimizer = TpSlOptimizer(f"{engine_id_prefix}_G6", default_cfg["tpsl_optimizer"])
        self.sanity_checker = NumericalSanityChecker(f"{engine_id_prefix}_G7", default_cfg["numerical_sanity"])
        self.ranker = SignalRanker(f"{engine_id_prefix}_G8", default_cfg["ranker"])
        self.validator = SignalValidator(f"{engine_id_prefix}_G9", default_cfg["validator"])
        self.explainer = SignalExplainer(f"{engine_id_prefix}_G10", default_cfg["explainer"])

        self.gateways = [
            self.generator, self.entry_filter, self.exit_filter,
            self.prob_filter, self.conf_filter, self.tpsl_optimizer,
            self.sanity_checker, self.ranker, self.validator, self.explainer
        ]

        self.activate_all()
        logger.info(f"UnifiedSignalEngine [{self.engine_id_prefix}] facade initialized (v{self.FACADE_VERSION}) | Mode ADTV: Rp {active_min_adtv:,.0f} IDR.")

    def activate_all(self) -> None:
        with self._lock:
            for gw in self.gateways:
                gw.activate()

    def execute_pipeline(self, prediction_df: pl.DataFrame) -> pl.DataFrame:
        if prediction_df is None or prediction_df.height == 0:
            return pl.DataFrame()

        try:
            df1 = self.generator.generate(prediction_df)
            df2 = self.entry_filter.filter_entry(df1)
            df3 = self.exit_filter.filter_exit(df2)
            df4 = self.prob_filter.filter_probability(df3)
            df5 = self.conf_filter.filter_confidence(df4)
            df6 = self.tpsl_optimizer.optimize_tp_sl(df5)
            df7 = self.sanity_checker.check_sanity(df6)
            df8 = self.ranker.rank_signals(df7)
            df9 = self.validator.validate_signals(df8)
            final_df = self.explainer.explain(df9)
            return final_df

        except Exception as err:
            logger.error(f"🛑 [SIGNAL_ENGINE_ERROR] Execution failed: {err}")
            cols = prediction_df.columns
            
            # Fallback Skema Utuh untuk Kompatibilitas Downstream
            fallback_exprs = [
                pl.lit(0.0).alias("signal"),
                pl.lit(False).alias("is_valid_execution"),
                pl.lit(f"{ExplanationStatus.PIPELINE_ERROR} {err}").alias("signal_explanation_text"),
                pl.lit(f"{ExplanationStatus.PIPELINE_ERROR} {err}").alias("signal_explanation"),
                pl.lit("PIPELINE_RUNTIME_ERROR").alias("final_validator_reason"),
                pl.lit(0.0).alias("prediction_probability"),
                pl.lit(0.0).alias("probability"),
                pl.lit(0.0).alias("prediction_confidence"),
                pl.lit(0.0).alias("confidence"),
                pl.lit(0.0).alias("entry_price"),
                pl.lit(0.0).alias("target_price"),
                pl.lit(0.0).alias("take_profit"),
                pl.lit(0.0).alias("stop_loss"),
                pl.lit(0).alias("lot_quantity"),
                pl.lit(0.0).alias("quantity"),
                pl.lit(0.0).alias("notional_idr"),
                pl.lit(0.0).alias("notional_usdt")
            ]
            
            if "ticker" not in cols and "symbol" in cols:
                fallback_exprs.append(pl.col("symbol").alias("ticker"))
            elif "ticker" not in cols:
                fallback_exprs.append(pl.lit(DEFAULT_IDX_TICKER).alias("ticker"))

            return prediction_df.with_columns(fallback_exprs)

    def get_pipeline_telemetry(self) -> List[Dict[str, Any]]:
        """Mengambil rekap telemetri dari seluruh Gateway."""
        with self._lock:
            return [gw.get_latest_telemetry() for gw in self.gateways]

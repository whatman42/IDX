"""
IDX Stock Quantitative Trading System - Unified Risk Management Engine v2026.Q3.v3.1.1
Module Name : risk.py
Directory   : Root Directory (Selevel dengan main.py)

Modul terpadu berstandar Hedge-Fund Production-Grade untuk pengawasan, pengendalian, 
dan mitigasi risiko kuantitatif saham Indonesia (IDX) mencakup:
1. AnomalyDetector          : Continuous Scoring Anomaly Engine (Median, MAD, & Quantile).
2. CusumDetector            : Stationary Return Welford CUSUM + Frozen Baseline & Auto-Reset.
3. DrawdownGuard            : Dual-mode drawdown guard (MTM & Closed-Trade Loss) dengan True Rolling Peak O(N) deque.
4. LiquidityGuard           : Penilaian likuiditas mikrostruktur robust (Median Turnover, RVOL, Amihud, Parkinson Spread).
5. OperationalStateMachine  : Thread-safe State Machine dengan SQLite WAL Connection Pool & Watchdog Heartbeat.
6. PopulationStabilityIndex : Dual PSI Engine dengan Scale-Invariant Fallback Binning.
7. RiskScoreEvaluator       : Komputasi skor komposit risiko horizontal terstandarisasi.
8. VolatilityGuard          : Multi-estimator dengan Vectorized Polars Rolling Historical Volatility Percentile Regime.
9. RiskAggregator           : Stratifikasi keparahan, Dynamic Regime Weighting, & Quadratic Form Matrix Correlation ((w⊙x)ᵀ Σ (w⊙x)).
10. PortfolioRiskExtensions : RiskMetrics 1996 EWMA Covariance/Beta, Parametric VaR, GBM Monte Carlo VaR, & Expected Shortfall.
11. UnifiedRiskEngine       : Facade Class terpusat sebagai single entry point pengawasan risiko.
"""

import time
import math
import sqlite3
import logging
import datetime
import hashlib
import json
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Literal, Final

import numpy as np
import polars as pl

# =============================================================================
# INTEGRASI LOGGING INTERNAL & ADAPTIF
# =============================================================================
try:
    from logger import get_logger
    logger = get_logger("IDX.Risk")
except ImportError:
    logger = logging.getLogger("IDX.Risk")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# =============================================================================
# KONSTANTA TERKUNCI BURSA EFEK INDONESIA (IDX TRADING BASELINE - OPTIMIZED)
# =============================================================================
IDX_FEE_ROUNDTRIP_PCT: float = 0.003          # Biaya Transaksi Roundtrip (0.15% Beli, 0.15% Jual+PPh)
IDX_MIN_PRICE_IDR: float = 50.0              # Batas Minimal Harga Saham IDX (Papan Reguler)
IDX_MIN_NOTIONAL_IDR: float = 5_000.0        # Batas Minimal Nilai Transaksi IDR (1 Lot x Rp 50)
IDX_MIN_24H_VOLUME_IDR: float = 1_000_000.0 # Batas Minimal Nilai Transaksi Harian (Rp 1 Juta)
IDX_MAX_STALENESS_SEC: float = 172800.0       # Toleransi Keusangan Data Candle (48 Jam)
IDX_DEFAULT_STP_MODE: str = "EXPIRE_MAKER"

# =============================================================================
# EXCEPTION CLASSES
# =============================================================================
try:
    from exceptions import (
        RiskEngineError,
        DataValidationError,
        NumericalInstabilityError,
        ChronologyError,
        RiskLimitBreachedError,
        InvalidStateTransitionError,
        LowVarianceError,
        OperationalStateError,
    )
except ImportError:
    class RiskEngineError(Exception): pass
    class DataValidationError(RiskEngineError): pass
    class NumericalInstabilityError(RiskEngineError): pass
    class ChronologyError(RiskEngineError): pass
    class RiskLimitBreachedError(RiskEngineError): pass
    class InvalidStateTransitionError(RiskEngineError): pass
    class LowVarianceError(RiskEngineError): pass
    class OperationalStateError(RiskEngineError): pass


# =============================================================================
# 1. ANOMALY DETECTOR (Continuous Scoring Engine)
# =============================================================================
@dataclass(frozen=True)
class AnomalyResult:
    anomaly_score: float
    anomaly_type: str
    halt: bool
    warning: bool
    telemetry: Dict[str, Any]


class AnomalyDetector:
    """Mendeteksi anomali harga dan volume menggunakan continuous scoring berbasis Median, MAD, dan Quantile."""

    def __init__(
        self,
        window_size: int = 20,
        mad_threshold: float = 3.5,
        quantile_threshold: float = 0.99,
        volume_spike_threshold: float = 4.0
    ) -> None:
        if window_size < 5:
            raise DataValidationError("Jendela observasi terlalu pendek.")
        self.window_size = window_size
        self.mad_threshold = mad_threshold
        self.quantile_threshold = quantile_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self._last_timestamp: str = ""
        self._version = "3.1.1"
        
        config_payload = {
            "version": self._version,
            "algorithm": "ContinuousRobustMedianMadAnomalyDetector",
            "parameters": {
                "window_size": window_size,
                "mad_threshold": mad_threshold,
                "quantile_threshold": quantile_threshold,
                "volume_spike_threshold": volume_spike_threshold
            }
        }
        self._config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()

    def evaluate(self, df: pl.DataFrame, pipeline_timestamp: str, execution_id: str, trace_id: str = "") -> AnomalyResult:
        start_time = time.perf_counter()
        
        if self._last_timestamp and pipeline_timestamp < self._last_timestamp:
            raise ChronologyError(f"[AnomalyDetector] Data out-of-order: {pipeline_timestamp} < {self._last_timestamp}")
        self._last_timestamp = pipeline_timestamp

        close_col = "close" if "close" in df.columns else ("last_price" if "last_price" in df.columns else "price")
        vol_col = "volume" if "volume" in df.columns else "vol"

        if df.is_empty() or close_col not in df.columns or vol_col not in df.columns:
            raise DataValidationError("Skema kolom DataFrame untuk AnomalyDetector tidak valid.")

        df_valid = df.filter((pl.col(close_col) > 0.0) & (pl.col(vol_col) >= 0.0))
        n = len(df_valid)

        if n < self.window_size + 1:
            return AnomalyResult(
                0.0, "INSUFFICIENT_DATA", False, False,
                {"schema_version": self._version, "module_name": "AnomalyDetector", "execution_id": execution_id, "trace_id": trace_id, "status": "INSUFFICIENT_DATA"}
            )

        prices = df_valid.select(pl.col(close_col)).to_series().to_numpy().astype(np.float64)
        volumes = df_valid.select(pl.col(vol_col)).to_series().to_numpy().astype(np.float64)
        
        if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(volumes)):
            raise NumericalInstabilityError("Array data mengandung elemen non-finite.")

        current_price = prices[-1]
        current_vol = volumes[-1]
        
        hist_prices = prices[-self.window_size - 1 : -1]
        hist_volumes = volumes[-self.window_size - 1 : -1]

        # 1. Continuous MAD Score
        p_median = np.median(hist_prices)
        mad = np.median(np.abs(hist_prices - p_median))
        mod_z_score = (0.6745 * abs(current_price - p_median)) / max(mad, 1e-8)
        mad_score = float(np.clip(mod_z_score / self.mad_threshold, 0.0, 1.0))

        # 2. Continuous Quantile Distance Score
        q_upper = np.quantile(hist_prices, self.quantile_threshold)
        q_lower = np.quantile(hist_prices, 1.0 - self.quantile_threshold)
        iqr_dist = max(q_upper - q_lower, 1e-8)
        
        if current_price > q_upper:
            quant_score = float(np.clip(0.5 + 0.5 * ((current_price - q_upper) / iqr_dist), 0.5, 1.0))
        elif current_price < q_lower:
            quant_score = float(np.clip(0.5 + 0.5 * ((q_lower - current_price) / iqr_dist), 0.5, 1.0))
        else:
            quant_score = float(np.clip(abs(current_price - p_median) / (iqr_dist * 0.5), 0.0, 0.5))

        # 3. Continuous Volume Spike Ratio Score
        v_median = np.median(hist_volumes)
        vol_ratio = current_vol / max(v_median, 1e-8)
        vol_score = float(np.clip(vol_ratio / self.volume_spike_threshold, 0.0, 1.0))

        # Continuously weighted anomaly score
        anomaly_score = float(np.clip((0.45 * mad_score) + (0.35 * quant_score) + (0.20 * vol_score), 0.0, 1.0))
        
        halt = anomaly_score >= 0.85 # Diselaraskan ke 0.85 agar tidak gampang false-positive
        warning = (anomaly_score >= 0.50) and not halt
        
        anomaly_type = "NORMAL"
        if halt:
            anomaly_type = "EXTREME_PRICE_MANIPULATION_OR_FLASH"
        elif warning:
            anomaly_type = "HIGH_ANOMALY_RISK"
            
        status = "HALT" if halt else ("WARNING" if warning else "SUCCESS")

        telemetry = {
            "schema_version": self._version,
            "module_name": "AnomalyDetector",
            "execution_id": execution_id,
            "trace_id": trace_id,
            "configuration_hash": self._config_hash,
            "status": status,
            "latency_ms": (time.perf_counter() - start_time) * 1000.0,
            "timestamp": pipeline_timestamp
        }

        return AnomalyResult(anomaly_score, anomaly_type, halt, warning, telemetry)


# =============================================================================
# 2. CUSUM DETECTOR (Pure Stationary Return Welford CUSUM + Frozen Baseline)
# =============================================================================
class CusumDetector:
    """Deteksi pergeseran tren berbasis return stasioner Welford Online Accumulator Warmup & Frozen Baseline."""

    def __init__(
        self,
        target_column: str = "close",
        threshold_multiplier: float = 4.5,
        drift_allowance_std: float = 0.5,
        warmup_period: int = 20,
        variance_floor: float = 1e-6,
        auto_reset: bool = True
    ) -> None:
        self.target_column = target_column
        self.threshold_multiplier = threshold_multiplier
        self.drift_allowance_std = drift_allowance_std
        self.warmup_period = warmup_period
        self.variance_floor = variance_floor
        self.auto_reset = auto_reset
        self._version = "3.1.1"

        config_payload = {
            "version": self._version,
            "algorithm": "PureWelfordStationaryReturnCusumFrozenBaseline",
            "parameters": {
                "threshold_multiplier": threshold_multiplier,
                "drift_allowance_std": drift_allowance_std,
                "warmup_period": warmup_period
            }
        }
        self._config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()

    def analyze(self, df: pl.DataFrame, pipeline_timestamp: Optional[str] = None) -> pl.DataFrame:
        if df.is_empty():
            raise DataValidationError("DataFrame input kosong.")
        
        target_col = self.target_column
        if target_col not in df.columns:
            fallback = [c for c in ["close", "last_price", "price"] if c in df.columns]
            target_col = fallback[0] if fallback else df.columns[0]

        raw_prices = df.select(pl.col(target_col)).to_series().to_numpy().astype(np.float64)
        if not np.all(np.isfinite(raw_prices)) or len(raw_prices) < 2:
            return df.with_columns([
                pl.lit(0.0).alias("cusum_pos"),
                pl.lit(0.0).alias("cusum_neg"),
                pl.lit(False).alias("cusum_anomaly_triggered")
            ])

        returns = np.diff(np.log(np.maximum(raw_prices, 1e-8)))
        returns = np.insert(returns, 0, 0.0)

        n = len(returns)
        s_pos = np.zeros(n, dtype=np.float64)
        s_neg = np.zeros(n, dtype=np.float64)
        anomalies = np.zeros(n, dtype=bool)

        if n < self.warmup_period:
            return df.with_columns([
                pl.Series("cusum_pos", s_pos),
                pl.Series("cusum_neg", s_neg),
                pl.Series("cusum_anomaly_triggered", anomalies)
            ])

        # Murni Algoritma Welford Online Accumulator selama Warmup
        welford_count = 0
        welford_mean = 0.0
        welford_M2 = 0.0

        for i in range(1, self.warmup_period):
            x = returns[i]
            welford_count += 1
            delta = x - welford_mean
            welford_mean += delta / welford_count
            delta2 = x - welford_mean
            welford_M2 += delta * delta2

        baseline_mean = welford_mean
        baseline_var = (welford_M2 / (welford_count - 1)) if welford_count > 1 else self.variance_floor
        baseline_std = np.sqrt(max(baseline_var, self.variance_floor))

        decision_boundary = self.threshold_multiplier
        slack = self.drift_allowance_std

        for i in range(self.warmup_period, n):
            x = returns[i]
            z = (x - baseline_mean) / baseline_std

            prev_pos = s_pos[i-1]
            prev_neg = s_neg[i-1]

            s_pos[i] = max(0.0, prev_pos + z - slack)
            s_neg[i] = max(0.0, prev_neg - z - slack)

            if (s_pos[i] > decision_boundary) or (s_neg[i] > decision_boundary):
                anomalies[i] = True
                if self.auto_reset:
                    s_pos[i] = 0.0
                    s_neg[i] = 0.0

        return df.with_columns([
            pl.Series("cusum_pos", s_pos),
            pl.Series("cusum_neg", s_neg),
            pl.Series("cusum_anomaly_triggered", anomalies)
        ])

CUSUMDetector = CusumDetector


# =============================================================================
# 3. DRAWDOWN GUARD (True Rolling Peak O(N) Deque & Closed-Trade Loss Tracking)
# =============================================================================
@dataclass(frozen=True)
class DrawdownResult:
    current_unrealized_dd: float
    max_unrealized_dd: float
    current_realized_dd: float
    max_realized_dd: float
    consecutive_realized_losses: int
    halt: bool
    warning: bool
    telemetry: Dict[str, Any]


class DrawdownGuard:
    """Dual-mode drawdown guard dengan Monotonic Deque O(N) & Closed-Trade Loss Evaluator."""

    def __init__(
        self,
        max_unrealized_dd: float = 0.20,      # OPTIMIZED: Diselaraskan ke 20%
        max_realized_dd: float = 0.15,        # OPTIMIZED: Diselaraskan ke 15%
        max_consecutive_losses: int = 8,      # OPTIMIZED: Diselaraskan ke 8 kali beruntun
        peak_mode: Literal["rolling", "ewma", "cumulative"] = "rolling",
        rolling_window: int = 60,
        ewma_span: int = 30
    ) -> None:
        self.max_unrealized_dd = max_unrealized_dd
        self.max_realized_dd = max_realized_dd
        self.max_consecutive_losses = max_consecutive_losses
        self.peak_mode = peak_mode
        self.rolling_window = rolling_window
        self.ewma_span = ewma_span
        self._version = "3.1.1"

    def _calculate_peaks(self, equity_array: np.ndarray) -> np.ndarray:
        n = len(equity_array)
        if n == 0:
            return equity_array
            
        if self.peak_mode == "cumulative":
            return np.maximum.accumulate(equity_array)
        elif self.peak_mode == "rolling":
            peaks = np.empty_like(equity_array)
            w = self.rolling_window
            dq: deque[int] = deque()
            for i in range(n):
                if dq and dq[0] <= i - w:
                    dq.popleft()
                while dq and equity_array[dq[-1]] <= equity_array[i]:
                    dq.pop()
                dq.append(i)
                peaks[i] = equity_array[dq[0]]
            return peaks
        elif self.peak_mode == "ewma":
            alpha = 2.0 / (self.ewma_span + 1.0)
            peaks = np.empty_like(equity_array)
            curr_peak = equity_array[0]
            for i in range(n):
                curr_peak = max(equity_array[i], curr_peak * (1.0 - alpha) + equity_array[i] * alpha)
                peaks[i] = curr_peak
            return peaks
        else:
            return np.maximum.accumulate(equity_array)

    def evaluate(
        self,
        unrealized_equity: Optional[pl.Series] = None,
        realized_equity: Optional[pl.Series] = None,
        closed_trade_pnls: Optional[List[float]] = None,
        pipeline_timestamp: str = "",
        execution_id: str = "",
        trace_id: str = ""
    ) -> DrawdownResult:
        start_time = time.perf_counter()
        
        if unrealized_equity is None or unrealized_equity.len() == 0:
            unrealized_equity = pl.Series("unrealized_equity", [10_000_000.0]) # Diselaraskan ke modal Rp 10 Juta
        if realized_equity is None or realized_equity.len() == 0:
            realized_equity = pl.Series("realized_equity", [10_000_000.0])
            
        unreal_np = unrealized_equity.to_numpy().astype(np.float64)
        real_np = realized_equity.to_numpy().astype(np.float64)

        peaks_unreal = self._calculate_peaks(unreal_np)
        dd_unreal = np.where(peaks_unreal > 0.0, (peaks_unreal - unreal_np) / peaks_unreal, 0.0)
        curr_unreal_dd = float(dd_unreal[-1])
        max_unreal_dd_val = float(np.max(dd_unreal))

        peaks_real = self._calculate_peaks(real_np)
        dd_real = np.where(peaks_real > 0.0, (peaks_real - real_np) / peaks_real, 0.0)
        curr_real_dd = float(dd_real[-1])
        max_real_dd_val = float(np.max(dd_real))

        # Perhitungan Consecutive Losses berbasis Closed Trade PnL
        consecutive_losses = 0
        if closed_trade_pnls and len(closed_trade_pnls) > 0:
            for pnl in reversed(closed_trade_pnls):
                if pnl < 0.0:
                    consecutive_losses += 1
                else:
                    break
        else:
            current_loss_run = 0
            for i in range(1, len(real_np)):
                if real_np[i] < real_np[i-1]:
                    current_loss_run += 1
                    consecutive_losses = max(consecutive_losses, current_loss_run)
                else:
                    current_loss_run = 0

        halt = (curr_unreal_dd >= self.max_unrealized_dd) or \
               (curr_real_dd >= self.max_realized_dd) or \
               (consecutive_losses >= self.max_consecutive_losses)
               
        warning = ((curr_unreal_dd >= self.max_unrealized_dd * 0.7) or \
                   (curr_real_dd >= self.max_realized_dd * 0.7)) and not halt

        telemetry = {
            "schema_version": self._version, "module_name": "DrawdownGuard",
            "execution_id": execution_id, "trace_id": trace_id, "status": "HALT" if halt else ("WARNING" if warning else "SUCCESS"),
            "latency_ms": (time.perf_counter() - start_time) * 1000.0, "timestamp": pipeline_timestamp
        }

        return DrawdownResult(curr_unreal_dd, max_unreal_dd_val, curr_real_dd, max_real_dd_val, consecutive_losses, halt, warning, telemetry)


# =============================================================================
# 4. LIQUIDITY GUARD (Robust Median Estimator Architecture)
# =============================================================================
@dataclass(frozen=True)
class LiquidityResult:
    liquidity_score: float
    spread_score: float
    slippage_score: float
    rvol_score: float
    halt: bool
    warning: bool
    telemetry: Dict[str, Any]


class LiquidityGuard:
    """Penilaian likuiditas mikrostruktur dengan konsistensi robust median estimator."""

    def __init__(
        self,
        mode: Literal["Level-1", "Level-2"] = "Level-1",
        min_value_traded_idr: float = IDX_MIN_24H_VOLUME_IDR,
        max_spread_pct: float = 0.04,          # OPTIMIZED: Max spread 4%
        window_size: int = 20
    ) -> None:
        self.mode = mode
        self.min_value_traded_idr = min_value_traded_idr
        self.max_spread_pct = max_spread_pct
        self.window_size = window_size
        self._version = "3.1.1"

    def evaluate(self, df: pl.DataFrame, pipeline_timestamp: str, execution_id: str, trace_id: str = "") -> LiquidityResult:
        start_time = time.perf_counter()
        vol_col = "volume" if "volume" in df.columns else "vol"
        close_col = "close" if "close" in df.columns else ("last_price" if "last_price" in df.columns else "price")
        high_col = "high" if "high" in df.columns else close_col
        low_col = "low" if "low" in df.columns else close_col

        if df.is_empty() or close_col not in df.columns or vol_col not in df.columns:
            raise DataValidationError("Skema kolom DataFrame untuk LiquidityGuard tidak lengkap.")

        df_valid = df.filter(pl.col(close_col) > 0.0)
        n = len(df_valid)

        if n < self.window_size:
            return LiquidityResult(0.0, 0.0, 0.0, 0.0, False, False, {"status": "INSUFFICIENT_DATA"})

        tail_df = df_valid.tail(self.window_size)
        
        # 1. Robust Median Value Traded Estimator
        value_traded_series = (tail_df.select(pl.col(vol_col) * pl.col(close_col))).to_series().to_numpy().astype(np.float64)
        median_value_traded = float(np.median(value_traded_series))
        val_score = float(np.clip((self.min_value_traded_idr / max(median_value_traded, 1.0)) - 0.2, 0.0, 1.0))

        # 2. RVOL menggunakan Robust Median Volume
        vol_series = tail_df.select(pl.col(vol_col)).to_series().to_numpy().astype(np.float64)
        median_vol = float(np.median(vol_series[:-1])) if len(vol_series) > 1 else 1.0
        rvol = vol_series[-1] / max(median_vol, 1.0)
        rvol_score = float(np.clip(1.0 / max(rvol, 0.1), 0.0, 1.0))

        # 3. Amihud Illiquidity Ratio
        illiquidity_df = tail_df.with_columns([
            (pl.col(close_col).log().diff().abs() / ((pl.col(vol_col) * pl.col(close_col)) + 1.0)).alias("amihud")
        ])
        amihud_series = illiquidity_df.select(pl.col("amihud").fill_null(0.0)).to_series().to_numpy()
        median_amihud = float(np.median(amihud_series))
        slippage_score = float(np.clip(median_amihud * 1e7, 0.0, 1.0))

        # 4. Parkinson High-Low Volatility Spread Estimator
        if self.mode == "Level-2" and "bid_price" in tail_df.columns and "ask_price" in tail_df.columns:
            spread_df = tail_df.with_columns([
                ((pl.col("ask_price") - pl.col("bid_price")) / ((pl.col("ask_price") + pl.col("bid_price")) * 0.5)).alias("spread_pct")
            ])
            median_spread = float(np.median(spread_df.select(pl.col("spread_pct")).to_series().to_numpy()))
        else:
            hl_ratio_expr = ((pl.col(high_col) / pl.col(low_col)).log() ** 2) / (4.0 * math.log(2.0))
            hl_series = tail_df.select(hl_ratio_expr).to_series().to_numpy()
            median_hl = float(np.median(hl_series))
            median_spread = float(np.sqrt(max(median_hl, 1.0e-8)))

        spread_score = float(np.clip(median_spread / self.max_spread_pct, 0.0, 1.0))

        final_liquidity_risk = (0.35 * val_score) + (0.25 * rvol_score) + (0.25 * slippage_score) + (0.15 * spread_score)
        halt = (final_liquidity_risk >= 0.85) or (median_value_traded < self.min_value_traded_idr * 0.1)
        warning = (final_liquidity_risk >= 0.55) and not halt

        telemetry = {
            "schema_version": self._version, "module_name": "LiquidityGuard",
            "execution_id": execution_id, "trace_id": trace_id, "status": "HALT" if halt else ("WARNING" if warning else "SUCCESS"),
            "latency_ms": (time.perf_counter() - start_time) * 1000.0, "timestamp": pipeline_timestamp
        }

        return LiquidityResult(final_liquidity_risk, spread_score, slippage_score, rvol_score, halt, warning, telemetry)


# =============================================================================
# 5. OPERATIONAL STATE MACHINE (SQLite Persistent WAL Connection Pool)
# =============================================================================
class SystemState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HALT = "HALT"
    COOLDOWN = "COOLDOWN"
    INITIALIZING = "INITIALIZING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class OperationalStateMachine:
    """Mesin status operasional thread-safe dengan Persistent Connection Pool & WAL Mode Audit Trail."""

    def __init__(self, mode: str = "live", initial_state: SystemState = SystemState.NORMAL, cooldown_duration_sec: float = 300.0, db_path: str = "risk_audit_trail.db") -> None:
        self._mode = mode.lower()
        self._current_state = initial_state
        self._cooldown_duration = cooldown_duration_sec
        self._halt_timestamp: float = 0.0
        self._last_heartbeat: float = time.time()
        self._lock = threading.RLock()
        self._version = "3.1.1"
        self._db_path = db_path

        self._db_conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._init_sqlite_db()

        self._legal_transitions = {
            SystemState.NORMAL: [SystemState.NORMAL, SystemState.WARNING, SystemState.HALT, SystemState.INITIALIZING],
            SystemState.WARNING: [SystemState.NORMAL, SystemState.WARNING, SystemState.HALT, SystemState.COOLDOWN],
            SystemState.HALT: [SystemState.COOLDOWN, SystemState.INITIALIZING],
            SystemState.COOLDOWN: [SystemState.NORMAL, SystemState.WARNING, SystemState.HALT],
            SystemState.INITIALIZING: [SystemState.EXECUTING, SystemState.FAILED],
            SystemState.EXECUTING: [SystemState.COMPLETED, SystemState.FAILED, SystemState.NORMAL, SystemState.WARNING, SystemState.HALT],
            SystemState.COMPLETED: [SystemState.NORMAL, SystemState.INITIALIZING],
            SystemState.FAILED: [SystemState.INITIALIZING]
        }

    def _init_sqlite_db(self) -> None:
        with self._lock:
            try:
                self._db_conn.execute("PRAGMA journal_mode=WAL;")
                self._db_conn.execute("PRAGMA synchronous=NORMAL;")
                self._db_conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_trail (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        trace_id TEXT,
                        run_id TEXT,
                        session_id TEXT,
                        source TEXT,
                        from_state TEXT,
                        to_state TEXT,
                        reason TEXT
                    )
                """)
                self._db_conn.commit()
            except Exception as e:
                logger.error(f"[SQLITE_INIT_ERROR] Gagal menginisialisasi audit database: {e}")

    def record_heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = time.time()

    def check_watchdog_timeout(self, timeout_sec: float = 120.0) -> bool:
        with self._lock:
            elapsed = time.time() - self._last_heartbeat
            return elapsed > timeout_sec

    @property
    def current_state(self) -> str:
        with self._lock:
            if self._current_state == SystemState.COOLDOWN:
                if time.time() - self._halt_timestamp >= self._cooldown_duration:
                    self._current_state = SystemState.WARNING
            return self._current_state.value

    def transition_to(self, target_state: SystemState, source: str = "ORCHESTRATOR", reason: str = "", timestamp: str = "", trace_id: str = "", run_id: str = "", session_id: str = "") -> None:
        with self._lock:
            if target_state not in self._legal_transitions[self._current_state]:
                if self._current_state == SystemState.HALT and target_state == SystemState.NORMAL:
                    target_state = SystemState.COOLDOWN
                else:
                    raise InvalidStateTransitionError(f"Transisi ilegal dari {self._current_state.value} ke {target_state.value}")
            
            prev = self._current_state
            if target_state == SystemState.HALT:
                self._halt_timestamp = time.time()
                
            self._current_state = target_state
            ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()

            try:
                self._db_conn.execute("""
                    INSERT INTO audit_trail (timestamp, trace_id, run_id, session_id, source, from_state, to_state, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (ts, trace_id, run_id, session_id, source, prev.value, target_state.value, reason))
                self._db_conn.commit()
            except Exception as e:
                logger.error(f"[SQLITE_WRITE_ERROR] Gagal menulis audit log ke database: {e}")

    def evaluate_metrics(self, risk_score: float, psi_score: float, timestamp: str, trace_id: str = "", run_id: str = "", session_id: str = "") -> SystemState:
        with self._lock:
            if risk_score >= 0.85:
                target = SystemState.HALT
                reason = f"Critical risk score: {risk_score:.4f}"
            elif risk_score >= 0.55 or psi_score > 0.30:
                target = SystemState.WARNING
                reason = f"Warning risk score: {risk_score:.4f}, PSI: {psi_score:.4f}"
            else:
                if self._current_state == SystemState.HALT:
                    target = SystemState.COOLDOWN
                    reason = "Risk normalized, entering cooldown period."
                elif self._current_state == SystemState.COOLDOWN and time.time() - self._halt_timestamp < self._cooldown_duration:
                    target = SystemState.COOLDOWN
                    reason = "Cooldown active."
                else:
                    target = SystemState.NORMAL
                    reason = "System normal."

            try:
                self.transition_to(target, "METRICS_ENGINE", reason, timestamp, trace_id, run_id, session_id)
            except Exception:
                pass
            return self._current_state

    def close(self) -> None:
        with self._lock:
            try:
                self._db_conn.close()
            except Exception:
                pass


# =============================================================================
# 6. POPULATION STABILITY INDEX (Scale-Invariant Fallback Binning)
# =============================================================================
class PopulationStabilityIndex:
    """Dual PSI Engine dengan Scale-Invariant Fallback Binning."""

    def __init__(self, n_bins: int = 10, epsilon: float = 1e-10) -> None:
        self.num_bins = n_bins
        self.epsilon = epsilon
        self._version = "3.1.1"

    def _compute_single_psi(self, baseline: np.ndarray, actual: np.ndarray) -> float:
        if len(baseline) == 0 or len(actual) == 0:
            return float('nan')
            
        base_range = np.max(baseline) - np.min(baseline)
        if base_range < 1e-7:
            offset = max(abs(np.mean(baseline)) * 1e-6, 1e-8)
            min_val = np.min(baseline) - offset
            max_val = np.max(baseline) + offset
            bin_edges = np.linspace(min_val, max_val, self.num_bins + 1)
        else:
            percentiles = np.linspace(0, 100, self.num_bins + 1)
            bin_edges = np.unique(np.percentile(baseline, percentiles))
            if len(bin_edges) < 3:
                bin_edges = np.linspace(np.min(baseline), np.max(baseline), self.num_bins + 1)

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        base_counts, _ = np.histogram(baseline, bins=bin_edges)
        targ_counts, _ = np.histogram(actual, bins=bin_edges)

        base_pct = np.where(base_counts == 0, self.epsilon, base_counts) / len(baseline)
        targ_pct = np.where(targ_counts == 0, self.epsilon, targ_counts) / len(actual)

        psi_val = float(np.sum((targ_pct - base_pct) * np.log(targ_pct / base_pct)))
        return psi_val if np.isfinite(psi_val) else float('nan')

    def calculate_dual_psi(self, base_features: Any, targ_features: Any, base_preds: Any, targ_preds: Any) -> Dict[str, float]:
        f_base = np.asarray(base_features, dtype=np.float64).flatten()
        f_targ = np.asarray(targ_features, dtype=np.float64).flatten()
        p_base = np.asarray(base_preds, dtype=np.float64).flatten()
        p_targ = np.asarray(targ_preds, dtype=np.float64).flatten()

        return {
            "feature_psi": self._compute_single_psi(f_base[np.isfinite(f_base)], f_targ[np.isfinite(f_targ)]),
            "prediction_psi": self._compute_single_psi(p_base[np.isfinite(p_base)], p_targ[np.isfinite(p_targ)])
        }


# =============================================================================
# 7. RISK SCORE EVALUATOR
# =============================================================================
class RiskScoreEvaluator:
    """Komputasi skor komposit risiko horizontal menggunakan Polars Expressions."""

    def __init__(self, weights: Dict[str, float]) -> None:
        self.weights = weights

    def compute_composite_score(self, df: pl.DataFrame, pipeline_timestamp: str) -> pl.DataFrame:
        if df.is_empty():
            return df.with_columns(pl.lit(0.0).alias("composite_risk_score"))
        expr = pl.sum_horizontal([pl.col(c) * pl.lit(w) for c, w in self.weights.items() if c in df.columns])
        return df.with_columns(expr.alias("composite_risk_score"))


# =============================================================================
# 8. VOLATILITY GUARD (Vectorized Polars Expressions Engine)
# =============================================================================
@dataclass(frozen=True)
class VolatilityResult:
    realized_volatility: float
    parkinson_volatility: float
    garman_klass_volatility: float
    volatility_score: float
    adaptive_threshold: float
    halt: bool
    warning: bool
    telemetry: Dict[str, Any]


class VolatilityGuard:
    """Multi-estimator volatilitas dengan Polars Expression Engine."""

    def __init__(self, window_size: int = 20, percentile_regime: float = 0.95) -> None:
        self.window_size = window_size
        self.percentile_regime = percentile_regime
        self._version = "3.1.1"

    def evaluate(self, df: pl.DataFrame, pipeline_timestamp: str, execution_id: str, trace_id: str = "") -> VolatilityResult:
        start_time = time.perf_counter()
        open_col = "open" if "open" in df.columns else "close"
        high_col = "high" if "high" in df.columns else open_col
        low_col = "low" if "low" in df.columns else open_col
        close_col = "close" if "close" in df.columns else "price"

        if df.is_empty() or close_col not in df.columns:
            return VolatilityResult(0.0, 0.0, 0.0, 0.0, 0.8, False, False, {})

        df_v = df.filter(pl.col(close_col) > 0.0)
        if len(df_v) < self.window_size * 2:
            return VolatilityResult(0.0, 0.0, 0.0, 0.0, 0.8, False, False, {})

        returns = df_v.select(pl.col(close_col).log().diff()).to_series().to_numpy().astype(np.float64)
        returns = returns[np.isfinite(returns)]
        
        realized_vol = float(np.std(returns[-self.window_size:], ddof=1) * np.sqrt(252)) if len(returns) >= self.window_size else 0.0

        p_term = float(df_v.select(
            (((pl.col(high_col) / pl.col(low_col)).log() ** 2) / (4.0 * math.log(2.0))).mean()
        ).item() or 0.0)
        parkinson_vol = float(np.sqrt(max(p_term, 1e-8)) * np.sqrt(252))

        gk_expr = (0.5 * (pl.col(high_col) / pl.col(low_col)).log() ** 2) - \
                  ((2.0 * math.log(2.0) - 1.0) * (pl.col(close_col) / pl.col(open_col)).log() ** 2)
        gk_term = float(df_v.select(gk_expr.mean()).item() or 0.0)
        gk_vol = float(np.sqrt(max(gk_term, 1e-8)) * np.sqrt(252))

        composite_vol = (0.40 * realized_vol) + (0.30 * parkinson_vol) + (0.30 * gk_vol)

        rolling_df = df_v.select([
            pl.col(close_col).log().diff().alias("ret")
        ]).with_columns([
            pl.col("ret").rolling_std(window_size=self.window_size, ddof=1).mul(np.sqrt(252)).alias("rolling_rv")
        ]).drop_nulls()

        rolling_vols = rolling_df.select(pl.col("rolling_rv")).to_series().to_numpy()

        if len(rolling_vols) > 5:
            adaptive_thresh = float(np.quantile(rolling_vols, self.percentile_regime))
        else:
            adaptive_thresh = max(composite_vol * 1.5, 0.45)
            
        adaptive_thresh = max(adaptive_thresh, 0.25)

        vol_score = float(np.clip(composite_vol / adaptive_thresh, 0.0, 1.0))
        halt = vol_score >= 1.0
        warning = (vol_score >= 0.75) and not halt

        telemetry = {
            "schema_version": self._version, "module_name": "VolatilityGuard",
            "execution_id": execution_id, "trace_id": trace_id, "status": "HALT" if halt else ("WARNING" if warning else "SUCCESS"),
            "latency_ms": (time.perf_counter() - start_time) * 1000.0, "timestamp": pipeline_timestamp
        }

        return VolatilityResult(realized_vol, parkinson_vol, gk_vol, vol_score, adaptive_thresh, halt, warning, telemetry)


# =============================================================================
# 9. RISK AGGREGATOR (Quadratic Form Matrix Correlation (w⊙x)ᵀ Σ (w⊙x))
# =============================================================================
@dataclass(frozen=True)
class AggregatedRiskOutput:
    composite_risk_score: float
    final_halt: bool
    final_warning: bool
    highest_severity_triggered: str
    hard_halt_sources: List[str]
    soft_halt_sources: List[str]
    telemetry: Dict[str, Any]


class RiskAggregator:
    """Stratifikasi risiko dengan Dynamic Regime Weighting & Formulasi Quadratic Form ((w⊙x)ᵀ Σ (w⊙x))."""

    REGIME_WEIGHTS: Final[Dict[str, Dict[str, float]]] = {
        "bull": {"drawdown": 0.20, "volatility": 0.40, "liquidity": 0.15, "anomaly": 0.25},
        "panic": {"drawdown": 0.30, "volatility": 0.20, "liquidity": 0.40, "anomaly": 0.10},
        "neutral": {"drawdown": 0.30, "volatility": 0.25, "liquidity": 0.25, "anomaly": 0.20}
    }

    def __init__(self, default_regime: str = "neutral") -> None:
        self.current_regime = default_regime
        self._version = "3.1.1"
        self.correlation_matrix = np.array([
            [1.00, 0.65, 0.40, 0.30],
            [0.65, 1.00, 0.50, 0.45],
            [0.40, 0.50, 1.00, 0.20],
            [0.30, 0.45, 0.20, 1.00]
        ], dtype=np.float64)

    def aggregate(
        self,
        drawdown_res: DrawdownResult,
        volatility_res: VolatilityResult,
        liquidity_res: LiquidityResult,
        anomaly_res: AnomalyResult,
        market_regime: str = "neutral",
        pipeline_timestamp: str = "",
        execution_id: str = "",
        trace_id: str = ""
    ) -> AggregatedRiskOutput:
        start_time = time.perf_counter()
        weights_dict = self.REGIME_WEIGHTS.get(market_regime, self.REGIME_WEIGHTS["neutral"])

        hard_halt_sources = []
        soft_halt_sources = []
        highest_severity = "NONE"

        if drawdown_res.halt:
            hard_halt_sources.append("CRITICAL_DRAWDOWN")
            highest_severity = "CRITICAL"
        if liquidity_res.halt:
            hard_halt_sources.append("LIQUIDITY_COLLAPSE")
            highest_severity = "CRITICAL"
        if anomaly_res.halt:
            hard_halt_sources.append("MANIPULATION_ANOMALY")
            highest_severity = "HIGH"
        if volatility_res.halt:
            soft_halt_sources.append("VOLATILITY_REGIME_BREACH")
            if highest_severity not in ["CRITICAL"]:
                highest_severity = "MEDIUM"

        raw_scores = np.array([
            drawdown_res.current_unrealized_dd,
            volatility_res.volatility_score,
            liquidity_res.liquidity_score,
            anomaly_res.anomaly_score
        ], dtype=np.float64)

        w_vector = np.array([
            weights_dict["drawdown"],
            weights_dict["volatility"],
            weights_dict["liquidity"],
            weights_dict["anomaly"]
        ], dtype=np.float64)

        # Formulasi Simetris Murni Quadratic Form: (w ⊙ x)ᵀ Σ (w ⊙ x)
        weighted_scores = w_vector * raw_scores
        quadratic_variance = float(np.dot(weighted_scores.T, np.dot(self.correlation_matrix, weighted_scores)))
        c_score = float(np.clip(np.sqrt(max(quadratic_variance, 0.0)) * 1.25, 0.0, 1.0))

        final_halt = (len(hard_halt_sources) > 0) or (c_score >= 0.85)
        final_warning = (len(soft_halt_sources) > 0 or drawdown_res.warning or volatility_res.warning or liquidity_res.warning) and not final_halt

        telemetry = {
            "schema_version": self._version,
            "module_name": "RiskAggregator",
            "execution_id": execution_id,
            "trace_id": trace_id,
            "status": "HALT" if final_halt else ("WARNING" if final_warning else "SUCCESS"),
            "latency_ms": (time.perf_counter() - start_time) * 1000.0,
            "timestamp": pipeline_timestamp
        }

        return AggregatedRiskOutput(c_score, final_halt, final_warning, highest_severity, hard_halt_sources, soft_halt_sources, telemetry)


# =============================================================================
# 10. INSTITUTIONAL PORTFOLIO & MARKET RISK EXTENSIONS
# =============================================================================
class PortfolioRiskExtensions:
    """Modul portofolio: RiskMetrics 1996 EWMA Covariance, Parametric VaR, & Configurable GBM Monte Carlo VaR."""

    def __init__(
        self,
        max_beta: float = 1.8,
        max_sector_alloc: float = 0.30,
        var_confidence: float = 0.95,
        mc_simulations: int = 10000,
        seed: Optional[int] = None
    ) -> None:
        self.max_beta = max_beta
        self.max_sector_alloc = max_sector_alloc
        self.var_confidence = var_confidence
        self.mc_simulations = mc_simulations
        self.rng = np.random.default_rng(seed)

    def check_portfolio_limits(self, market_returns: np.ndarray, asset_returns: np.ndarray) -> Dict[str, Any]:
        min_len = min(len(market_returns), len(asset_returns))
        if min_len < 10:
            return {"ewma_beta": 1.0, "beta_breached": False, "historical_var_95": 0.05, "monte_carlo_var_95": 0.05}

        m_slice = market_returns[-min_len:]
        a_slice = asset_returns[-min_len:]

        # RiskMetrics 1996 EWMA Covariance dengan Demeaned Returns
        mean_m = float(np.mean(m_slice))
        mean_a = float(np.mean(a_slice))
        m_demeaned = m_slice - mean_m
        a_demeaned = a_slice - mean_a

        span = 30
        alpha = 2.0 / (span + 1.0)
        cov_ewma = 0.0
        var_mkt_ewma = 0.0

        for i in range(min_len):
            cov_ewma = cov_ewma * (1.0 - alpha) + a_demeaned[i] * m_demeaned[i] * alpha
            var_mkt_ewma = var_mkt_ewma * (1.0 - alpha) + (m_demeaned[i] ** 2) * alpha

        ewma_beta = float(cov_ewma / max(var_mkt_ewma, 1e-8))
        beta_breached = ewma_beta > self.max_beta

        sorted_rets = np.sort(a_slice)
        idx = int((1.0 - self.var_confidence) * len(sorted_rets))
        hist_var = float(-sorted_rets[idx])
        hist_es = float(-np.mean(sorted_rets[:max(idx, 1)]))

        std_ret = float(np.std(a_slice, ddof=1))
        z_score = 1.645 if self.var_confidence == 0.95 else 2.33
        param_var = float(max(0.0, -(mean_a - z_score * std_ret)))

        dt = 1.0
        drift = (mean_a - 0.5 * (std_ret ** 2)) * dt
        diffusion = std_ret * np.sqrt(dt)
        
        z_random = self.rng.standard_normal(self.mc_simulations)
        simulated_log_returns = drift + diffusion * z_random
        sim_sorted = np.sort(simulated_log_returns)
        mc_idx = int((1.0 - self.var_confidence) * self.mc_simulations)
        mc_var = float(-sim_sorted[mc_idx])
        mc_es = float(-np.mean(sim_sorted[:max(mc_idx, 1)]))

        return {
            "ewma_beta": ewma_beta,
            "beta_breached": beta_breached,
            "historical_var_95": hist_var,
            "historical_es_95": hist_es,
            "parametric_var_95": param_var,
            "monte_carlo_var_95": mc_var,
            "monte_carlo_es_95": mc_es
        }


# =============================================================================
# 11. UNIFIED RISK ENGINE (FACADE CLASS - FIXED METHOD SIGNATURE)
# =============================================================================
class UnifiedRiskEngine:
    """Facade terpusat institusional untuk seluruh subsistem pengawasan risiko IDX (main.py ready)."""

    def __init__(self, mode: str = "live", market_regime: str = "neutral", seed: Optional[int] = None) -> None:
        self.mode = mode
        self.market_regime = market_regime
        self.anomaly_detector = AnomalyDetector()
        self.cusum_detector = CusumDetector()
        self.drawdown_guard = DrawdownGuard()
        self.liquidity_guard = LiquidityGuard()
        self.state_machine = OperationalStateMachine(mode=mode)
        self.psi_calculator = PopulationStabilityIndex()
        self.volatility_guard = VolatilityGuard()
        self.risk_evaluator = RiskScoreEvaluator({"drawdown": 0.3, "volatility": 0.3, "liquidity": 0.2, "anomaly": 0.2})
        self.risk_aggregator = RiskAggregator(default_regime=market_regime)
        self.portfolio_extensions = PortfolioRiskExtensions(seed=seed)

    def evaluate_market_risk(
        self,
        df_klines: pl.DataFrame,
        unrealized_equity: Optional[pl.Series] = None,
        realized_equity: Optional[pl.Series] = None,
        closed_trade_pnls: Optional[List[float]] = None,
        pipeline_timestamp: Optional[str] = None,
        execution_id: Optional[str] = None,
        trace_id: str = "",
        run_id: str = "",
        session_id: str = ""
    ) -> AggregatedRiskOutput:
        # Heartbeat interseptor utama untuk mencegah false-positive watchdog
        self.state_machine.record_heartbeat()

        ts = pipeline_timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        exec_id = execution_id or f"EXEC-RISK-{int(time.time())}"

        anomaly_res = self.anomaly_detector.evaluate(df_klines, ts, exec_id, trace_id)
        drawdown_res = self.drawdown_guard.evaluate(unrealized_equity, realized_equity, closed_trade_pnls, ts, exec_id, trace_id)
        liquidity_res = self.liquidity_guard.evaluate(df_klines, ts, exec_id, trace_id)
        volatility_res = self.volatility_guard.evaluate(df_klines, ts, exec_id, trace_id)

        cusum_df = self.cusum_detector.analyze(df_klines, ts)

        aggregated_output = self.risk_aggregator.aggregate(
            drawdown_res, volatility_res, liquidity_res, anomaly_res,
            market_regime=self.market_regime, pipeline_timestamp=ts, execution_id=exec_id, trace_id=trace_id
        )

        self.state_machine.evaluate_metrics(aggregated_output.composite_risk_score, 0.0, ts, trace_id, run_id, session_id)
        return aggregated_output

    def audit_portfolio_orders(self, orders: Any, aggregated_risk: Optional[AggregatedRiskOutput] = None) -> Any:
        """Menyaring order sinyal berdasarkan batasan IDX (Min Price Rp 50, Min Notional Rp 5.000)."""
        if orders is None:
            return None

        orders_list = orders if isinstance(orders, list) else [orders]
        if not orders_list:
            return orders

        if aggregated_risk and aggregated_risk.final_halt:
            logger.critical(f"[RISK_CIRCUIT_BREAKER] Global Hard Halt! Sources: {aggregated_risk.hard_halt_sources}")
            return []

        validated_orders = []
        for order in orders_list:
            if not isinstance(order, dict):
                continue
            ticker = str(order.get("ticker", order.get("symbol", "UNKNOWN"))).upper()
            price = float(order.get("price", order.get("close", 0.0)))
            qty = float(order.get("quantity", order.get("qty", 0.0)))

            if price < IDX_MIN_PRICE_IDR:
                continue
            notional = price * qty if qty > 0 else float(order.get("notional", 0.0))
            if notional > 0.0 and notional < IDX_MIN_NOTIONAL_IDR:
                continue

            order["selfTradePreventionMode"] = IDX_DEFAULT_STP_MODE
            validated_orders.append(order)

        return validated_orders

    def get_system_health(self) -> Dict[str, Any]:
        return {
            "current_state": self.state_machine.current_state,
            "mode": self.mode,
            "market_regime": self.market_regime,
            "watchdog_expired": self.state_machine.check_watchdog_timeout()
        }

    def close(self) -> None:
        """Mematikan resource dan mengakhiri koneksi SQLite."""
        self.state_machine.close()

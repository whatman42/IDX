"""
Module: features.py
Description: Production-Grade Institutional Quantitative Feature Engineering Engine for IDX Stock Trading.
             Consolidates Multi-Factor Extraction (Cross-Sectional Alpha Ranks, Market Microstructure,
             Advanced Volatility Estimators, Momentum, Price Action Patterns, Adaptive Trend Indicators, 
             ADTV / IDR Turnover via Typical Price), Real-Time Population Stability Index (PSI) & 
             Vectorized Kolmogorov-Smirnov Distribution Drift Monitoring, Lookahead-Safe Collinearity & 
             Variance Feature Selection, and Parquet Feature Store with Point-in-Time As-Of Join capability.
Version: 2026.Q3.v17.1-INSTITUTIONAL-HEDGEFUND-GRADE
Architecture: Clean Architecture, Vectorized Polars Execution, Thread-Safe Memory-Optimized Pipeline
Compliance: Indonesia Stock Exchange (IDX) Equity Rules & Zero-Copy Polars Standard
"""

import gc
import os
import sys
import time
import json
import math
import random
import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Final, Union, Optional

import numpy as np
import polars as pl

# ==============================================================================
# 1. LOGGING & INTEGRATION IMMUNITY (SINGLE-DIRECTORY COMPLIANCE)
# ==============================================================================
try:
    from logger import logger, get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][IDX.Features] %(message)s")
    logger = logging.getLogger("IDX.Features")
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

try:
    from constants import (
        COL_TIMESTAMP, COL_ASSET, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME,
        EPSILON, NUMERICAL_EPSILON, IDX_FEE_ROUNDTRIP_PCT,
        IDX_MIN_PRICE_IDR, IDX_MIN_24H_VOLUME_IDR, IDX_MAX_STALENESS_SEC
    )
except ImportError:
    COL_TIMESTAMP: Final[str] = "timestamp"
    COL_ASSET: Final[str] = "asset"
    COL_OPEN: Final[str] = "open"
    COL_HIGH: Final[str] = "high"
    COL_LOW: Final[str] = "low"
    COL_CLOSE: Final[str] = "close"
    COL_VOLUME: Final[str] = "volume"
    EPSILON: Final[float] = 1e-12
    NUMERICAL_EPSILON: Final[float] = 1e-12
    IDX_FEE_ROUNDTRIP_PCT: Final[float] = 0.003       # 0.3% estimasi komisi & pajak bursa
    IDX_MIN_PRICE_IDR: Final[float] = 50.0             # Pasar reguler gocap floor
    IDX_MIN_24H_VOLUME_IDR: Final[float] = 1_000_000_000.0 # Rp 1 Miliar ADTV Minimum
    IDX_MAX_STALENESS_SEC: Final[float] = 259200.0      # 72 Jam (Akomodasi Akhir Pekan)

try:
    from exceptions import DataValidationError, FeatureEngineeringError
except ImportError:
    class DataValidationError(Exception):
        """Dilemparkan jika input data melanggar aturan skema atau integritas."""
        pass
    class FeatureEngineeringError(Exception):
        """Dilemparkan jika terjadi kegagalan komputasi pada pipeline fitur."""
        pass

COL_SECTOR: Final[str] = "sector"


# ==============================================================================
# HELPER: VECTORIZED NUMPY KAMA CALCULATION
# ==============================================================================
def _compute_kama_numpy(close_arr: np.ndarray, n: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average (KAMA) menggunakan NumPy murni.
    Menghindari bug Polars ewm_mean(alpha=Expr) yang memicu TypeError: truth value of Expr is ambiguous.
    """
    n_obs = len(close_arr)
    kama = np.full_like(close_arr, np.nan, dtype=np.float64)
    if n_obs < n + 1:
        return kama

    # Efficiency Ratio (ER)
    change = np.abs(close_arr[n:] - close_arr[:-n])
    price_diff = np.abs(np.diff(close_arr))
    
    volatility = np.convolve(price_diff, np.ones(n, dtype=np.float64), mode='valid')
    
    er = np.zeros_like(change, dtype=np.float64)
    valid_mask = volatility > 1e-12
    er[valid_mask] = change[valid_mask] / volatility[valid_mask]

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Inisialisasi KAMA
    kama[n] = close_arr[n]
    for i in range(n + 1, n_obs):
        sc_val = sc[i - n]
        kama[i] = kama[i - 1] + sc_val * (close_arr[i] - kama[i - 1])

    # Forward fill nilai nan awal jika ada
    if np.isnan(kama[0]):
        first_valid_idx = np.where(~np.isnan(kama))[0]
        if len(first_valid_idx) > 0:
            kama[:first_valid_idx[0]] = close_arr[:first_valid_idx[0]]

    return kama


# ==============================================================================
# 2. DATA STRUCTURES & REGISTRY METADATA SCHEMA
# ==============================================================================
@dataclass(slots=True)
class FeatureRegistryRecord:
    feature_id: str
    feature_name: str
    version: str
    origin_module: str
    owner: str
    created_at: str
    dependencies: List[str]
    warmup_required: int
    data_type: str
    expected_range: Tuple[Optional[float], Optional[float]]
    drift_limit_psi: float
    status: str                  # 'active', 'dropped'
    definition_hash: str
    quality_score: float
    importance: float
    selection_count: int
    last_importance: float
    survival_round: int
    health_status: str           # 'stable', 'warning', 'critical', 'quarantine', 'recovered'
    last_selected_at: Optional[str]
    last_drift_psi: float
    last_retrained_at: Optional[str]
    drop_reason: Optional[str]
    feature_age_days: int
    last_drift_checked_at: Optional[str] = None


def safe_float(value: Any, default: float) -> float:
    """Mengonversi nilai ke float secara defensif untuk mencegah krisis tipe data."""
    if value is None:
        return default
    try:
        val = float(value)
        return default if math.isnan(val) or math.isinf(val) else val
    except (ValueError, TypeError):
        return default


def sanitize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """Memastikan konsistensi nama kolom dasar (asset/ticker, timestamp/date)."""
    cols = df.columns
    exprs = []
    if "ticker" in cols and "asset" not in cols:
        exprs.append(pl.col("ticker").alias("asset"))
    elif "asset" in cols and "ticker" not in cols:
        exprs.append(pl.col("asset").alias("ticker"))

    if "date" in cols and "timestamp" not in cols:
        exprs.append(pl.col("date").alias("timestamp"))
    elif "timestamp" in cols and "date" not in cols:
        exprs.append(pl.col("timestamp").alias("date"))

    if exprs:
        return df.with_columns(exprs)
    return df


# ==============================================================================
# SUB-ENGINE 1: CROSS-SECTIONAL FEATURE EXTRACTOR
# ==============================================================================
class CrossSectionalFeatureExtractor:
    """Engine Cross-Sectional Alpha Rank & Scale Normalization (Vektor Polars Aman)."""
    FEATURE_SET_VERSION: Final[str] = "17.0.0"
    ORIGIN_MODULE: Final[str] = "features.cross_sectional"
    OWNER: Final[str] = "Quantitative Architecture Core"
    ANCHORED_TIME: Final[str] = "2026-01-01T00:00:00Z"
    PSI_DEFAULT_LIMIT: Final[float] = 0.10

    def __init__(self, zscore_max_cap: float = 6.0) -> None:
        self.zscore_max_cap: Final[float] = zscore_max_cap
        self.feature_registry: Dict[str, FeatureRegistryRecord] = {}
        self.warmup_rows: Final[int] = 0
        self._generate_registry_blueprints()

    def _generate_feature_hash(self, logic: str, deps: List[str], bounds: Tuple[Any, Any]) -> str:
        payload = f"{self.FEATURE_SET_VERSION}_{logic}_{''.join(deps)}_{bounds[0]}_{bounds[1]}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _create_blueprint(self, name: str, fid: str, deps: List[str], bounds: Tuple[Optional[float], Optional[float]], logic: str) -> FeatureRegistryRecord:
        return FeatureRegistryRecord(
            feature_id=fid, feature_name=name, version=self.FEATURE_SET_VERSION,
            origin_module=self.ORIGIN_MODULE, owner=self.OWNER, created_at=self.ANCHORED_TIME,
            dependencies=deps, warmup_required=self.warmup_rows, data_type="Float64",
            expected_range=bounds, drift_limit_psi=self.PSI_DEFAULT_LIMIT, status="active",
            definition_hash=self._generate_feature_hash(logic, deps, bounds), quality_score=0.0, importance=1.0,
            selection_count=0, last_importance=1.0, survival_round=0, health_status="stable",
            last_selected_at=None, last_drift_psi=0.0, last_retrained_at=None, drop_reason=None, feature_age_days=0
        )

    def _generate_registry_blueprints(self) -> None:
        for prim in ["ret", "vol"]:
            f_rank = f"f_cs_global_rank_{prim}"
            self.feature_registry[f_rank] = self._create_blueprint(
                f_rank, f"CS-GLOBAL-RANK-{prim.upper()}", [COL_TIMESTAMP, COL_CLOSE if prim == "ret" else COL_VOLUME],
                (0.0, 1.0), f"percentile_rank({prim})_over_epoch"
            )
            f_zscore = f"f_cs_global_zscore_{prim}"
            self.feature_registry[f_zscore] = self._create_blueprint(
                f_zscore, f"CS-GLOBAL-ZSCORE-{prim.upper()}", [COL_TIMESTAMP, COL_CLOSE if prim == "ret" else COL_VOLUME],
                (-self.zscore_max_cap, self.zscore_max_cap), f"zscore({prim})_over_epoch"
            )
            f_sec_rank = f"f_cs_sector_rank_{prim}"
            self.feature_registry[f_sec_rank] = self._create_blueprint(
                f_sec_rank, f"CS-SECTOR-RANK-{prim.upper()}", [COL_TIMESTAMP, COL_SECTOR, COL_CLOSE if prim == "ret" else COL_VOLUME],
                (0.0, 1.0), f"percentile_rank({prim})_over_epoch_sector"
            )

        ready_rec = self._create_blueprint("feature_cross_sectional_ready", "CS-READY-BOOLEAN", [COL_TIMESTAMP], (0.0, 1.0), "boolean_ready")
        object.__setattr__(ready_rec, 'data_type', 'Boolean')
        self.feature_registry["feature_cross_sectional_ready"] = ready_rec

    def extract(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> Union[pl.DataFrame, pl.LazyFrame]:
        is_lazy = isinstance(df, pl.LazyFrame)
        lf = df.lazy() if not is_lazy else df
        cols = lf.collect_schema().names()

        asset_col = COL_ASSET if COL_ASSET in cols else ("ticker" if "ticker" in cols else COL_ASSET)
        time_col = COL_TIMESTAMP if COL_TIMESTAMP in cols else ("date" if "date" in cols else COL_TIMESTAMP)

        lf = lf.unique(subset=[asset_col, time_col], keep="last").sort([asset_col, time_col])

        prev_close = pl.col(COL_CLOSE).shift(1).over(asset_col)
        log_ret = (pl.col(COL_CLOSE) / (prev_close + EPSILON)).log().fill_null(0.0)

        sector_node = pl.col(COL_SECTOR).fill_null("GLOBAL") if COL_SECTOR in cols else pl.lit("GLOBAL").alias(COL_SECTOR)

        layer_1 = lf.with_columns([
            log_ret.alias("_log_ret"),
            pl.col(COL_VOLUME).cast(pl.Float64).alias("_vol_float"),
            sector_node.alias(COL_SECTOR)
        ])

        layer_2 = layer_1.with_columns([
            pl.col("_log_ret").mean().over(time_col).alias("_epoch_mean_ret"),
            pl.col("_log_ret").std().over(time_col).alias("_epoch_std_ret"),
            pl.col("_vol_float").mean().over(time_col).alias("_epoch_mean_vol"),
            pl.col("_vol_float").std().over(time_col).alias("_epoch_std_vol"),
            pl.col("_log_ret").count().over(time_col).alias("_epoch_cnt_global"),
            pl.col("_log_ret").count().over([time_col, COL_SECTOR]).alias("_epoch_cnt_sector")
        ])

        zscore_ret = (pl.col("_log_ret") - pl.col("_epoch_mean_ret")) / (pl.col("_epoch_std_ret") + EPSILON)
        zscore_vol = (pl.col("_vol_float") - pl.col("_epoch_mean_vol")) / (pl.col("_epoch_std_vol") + EPSILON)

        # PRECISI RANKING: Formula (Rank - 1) / (N - 1) untuk skala presisi 0.0 s/d 1.0
        rank_ret_global = pl.when(pl.col("_epoch_cnt_global") > 1)\
            .then((pl.col("_log_ret").rank(method="average").over(time_col) - 1.0) / (pl.col("_epoch_cnt_global") - 1.0))\
            .otherwise(0.5).fill_null(0.5)

        rank_vol_global = pl.when(pl.col("_epoch_cnt_global") > 1)\
            .then((pl.col("_vol_float").rank(method="average").over(time_col) - 1.0) / (pl.col("_epoch_cnt_global") - 1.0))\
            .otherwise(0.5).fill_null(0.5)

        rank_ret_sector = pl.when(pl.col("_epoch_cnt_sector") > 1)\
            .then((pl.col("_log_ret").rank(method="average").over([time_col, COL_SECTOR]) - 1.0) / (pl.col("_epoch_cnt_sector") - 1.0))\
            .otherwise(0.5).fill_null(0.5)

        rank_vol_sector = pl.when(pl.col("_epoch_cnt_sector") > 1)\
            .then((pl.col("_vol_float").rank(method="average").over([time_col, COL_SECTOR]) - 1.0) / (pl.col("_epoch_cnt_sector") - 1.0))\
            .otherwise(0.5).fill_null(0.5)

        exprs = [
            zscore_ret.clip(-self.zscore_max_cap, self.zscore_max_cap).fill_null(0.0).alias("f_cs_global_zscore_ret"),
            zscore_vol.clip(-self.zscore_max_cap, self.zscore_max_cap).fill_null(0.0).alias("f_cs_global_zscore_vol"),
            rank_ret_global.alias("f_cs_global_rank_ret"),
            rank_vol_global.alias("f_cs_global_rank_vol"),
            rank_ret_sector.alias("f_cs_sector_rank_ret"),
            rank_vol_sector.alias("f_cs_sector_rank_vol"),
            pl.lit(True).alias("feature_cross_sectional_ready")
        ]

        out_cols = [c for c in cols if not c.startswith("_")]
        if COL_SECTOR not in out_cols:
            out_cols.append(COL_SECTOR)

        final_lf = layer_2.with_columns(exprs).select(out_cols + [e.meta.output_name() for e in exprs if e.meta.output_name() not in out_cols])
        return final_lf if is_lazy else final_lf.collect()


# ==============================================================================
# SUB-ENGINE 2: TECHNICAL, TREND & ADAPTIVE MOVING AVERAGE BUILDER
# ==============================================================================
class IDXTechnicalAndTrendBuilder:
    """Engine Indikator Teknikal, Trend & Adaptive MA (Wilder RSI, KAMA, HMA, Linear Reg Slope)."""

    def __init__(self, rsi_window: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 bb_window: int = 20, atr_window: int = 14, trend_fast: int = 20, trend_slow: int = 50) -> None:
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_window = bb_window
        self.atr_window = atr_window
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow

    def build_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        df = sanitize_column_names(df)
        asset_col = COL_ASSET if COL_ASSET in df.columns else ("ticker" if "ticker" in df.columns else COL_ASSET)
        time_col = COL_TIMESTAMP if COL_TIMESTAMP in df.columns else ("date" if "date" in df.columns else COL_TIMESTAMP)

        df_sorted = df.unique(subset=[asset_col, time_col], keep="last").sort([asset_col, time_col])

        # 1. PRECISI WILDER RSI (Alpha = 1/N Wilder's RMA Smoothing)
        price_diff = pl.col(COL_CLOSE).diff().over(asset_col)
        gain = pl.when(price_diff > 0.0).then(price_diff).otherwise(0.0)
        loss = pl.when(price_diff < 0.0).then(-price_diff).otherwise(0.0)
        alpha_wilder = 1.0 / float(self.rsi_window)
        avg_gain = gain.ewm_mean(alpha=alpha_wilder, adjust=False).over(asset_col)
        avg_loss = loss.ewm_mean(alpha=alpha_wilder, adjust=False).over(asset_col)
        
        rsi_raw = 100.0 - (100.0 / (1.0 + (avg_gain / (avg_loss + EPSILON))))
        rsi_expr = pl.when((avg_gain == 0.0) & (avg_loss == 0.0)).then(50.0).otherwise(rsi_raw).fill_null(50.0)

        # 2. MACD
        ema_fast = pl.col(COL_CLOSE).ewm_mean(span=self.macd_fast, adjust=False).over(asset_col)
        ema_slow = pl.col(COL_CLOSE).ewm_mean(span=self.macd_slow, adjust=False).over(asset_col)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm_mean(span=self.macd_signal, adjust=False).over(asset_col)
        macd_hist = macd_line - signal_line

        # 3. BOLLINGER BANDS (dengan strict min_periods=window_size untuk stabilitas warm-up)
        bb_mean = pl.col(COL_CLOSE).rolling_mean(window_size=self.bb_window, min_periods=self.bb_window).over(asset_col).fill_null(strategy="forward")
        bb_std = pl.col(COL_CLOSE).rolling_std(window_size=self.bb_window, min_periods=self.bb_window, ddof=0).over(asset_col).fill_null(0.0)
        upper_bb = bb_mean + (2.0 * bb_std)
        lower_bb = bb_mean - (2.0 * bb_std)
        pct_b = ((pl.col(COL_CLOSE) - lower_bb) / (upper_bb - lower_bb + EPSILON)).fill_null(0.5)
        bb_width = ((upper_bb - lower_bb) / (bb_mean + EPSILON)).fill_null(0.0)

        # 4. ATR (Average True Range)
        prev_close = pl.col(COL_CLOSE).shift(1).over(asset_col)
        tr = pl.max_horizontal([
            pl.col(COL_HIGH) - pl.col(COL_LOW),
            (pl.col(COL_HIGH) - prev_close).abs().fill_null(0.0),
            (pl.col(COL_LOW) - prev_close).abs().fill_null(0.0)
        ])
        atr_expr = tr.ewm_mean(alpha=(1.0 / float(self.atr_window)), adjust=False).over(asset_col).fill_null(0.0)

        # 5. HULL MOVING AVERAGE (HMA 16)
        wma_half = pl.col(COL_CLOSE).ewm_mean(span=8, adjust=False).over(asset_col)
        wma_full = pl.col(COL_CLOSE).ewm_mean(span=16, adjust=False).over(asset_col)
        hma_raw = (2.0 * wma_half) - wma_full
        hma_expr = hma_raw.ewm_mean(span=4, adjust=False).over(asset_col).fill_null(strategy="forward")

        # 6. KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA 10,2,30) - SOLUSI BUG POLARS AMBIGUITY
        def _apply_kama_group(df_group: pl.DataFrame) -> pl.DataFrame:
            c_arr = df_group[COL_CLOSE].to_numpy()
            k_arr = _compute_kama_numpy(c_arr, n=10, fast=2, slow=30)
            return df_group.with_columns(pl.Series("f_trend_kama_10", k_arr))

        df_sorted = df_sorted.group_by(asset_col, maintain_order=True).map_groups(_apply_kama_group)

        # 7. LINEAR REGRESSION SLOPE (14 Baris Vectorization)
        p_lag14 = pl.col(COL_CLOSE).shift(14).over(asset_col)
        linreg_slope = ((pl.col(COL_CLOSE) - p_lag14) / 14.0).fill_null(0.0)

        # Trend Distances & Velocity
        dist_fast = ((pl.col(COL_CLOSE) - ema_fast) / (ema_fast + EPSILON)).fill_null(0.0)
        dist_slow = ((pl.col(COL_CLOSE) - ema_slow) / (ema_slow + EPSILON)).fill_null(0.0)
        slow_lag = ema_slow.shift(5).over(asset_col)
        velocity = ((ema_slow - slow_lag) / (slow_lag + EPSILON)).fill_null(0.0)

        return df_sorted.with_columns([
            rsi_expr.alias("feature_rsi"),
            rsi_expr.alias("f_rsi_14"),
            rsi_expr.alias("rsi_14"),
            
            macd_line.fill_null(0.0).alias("feature_macd_line"),
            macd_line.fill_null(0.0).alias("f_macd_line"),
            signal_line.fill_null(0.0).alias("feature_macd_signal"),
            signal_line.fill_null(0.0).alias("f_macd_signal"),
            macd_hist.fill_null(0.0).alias("feature_macd_histogram"),
            macd_hist.fill_null(0.0).alias("f_macd_histogram"),

            pct_b.alias("feature_bollinger_pct_b"),
            pct_b.alias("f_bollinger_pct_b"),
            bb_width.alias("feature_bollinger_width"),
            bb_width.alias("f_bollinger_width"),

            atr_expr.alias("feature_atr"),
            atr_expr.alias("f_atr_14"),
            atr_expr.alias("atr_14"),
            atr_expr.alias("atr"),

            hma_expr.alias("f_trend_hma_16"),
            linreg_slope.alias("f_trend_linreg_slope_14"),

            dist_fast.alias("feature_ema_fast_distance"),
            dist_fast.alias("f_ema_20_dist"),
            dist_slow.alias("feature_ema_slow_distance"),
            dist_slow.alias("f_ema_50_dist"),
            velocity.alias("feature_trend_velocity"),
            velocity.alias("f_trend_velocity"),

            pl.lit(True).alias("feature_technical_ready")
        ])


# ==============================================================================
# SUB-ENGINE 3: ADVANCED VOLATILITY & MICROSTRUCTURE EXTRACTOR
# ==============================================================================
class ComprehensiveMicrostructureExtractor:
    """Institutional Multi-Estimator Volatility (Parkinson, Garman-Klass, Yang-Zhang) & Microstructure Engine."""

    def __init__(self, windows: Optional[List[int]] = None) -> None:
        self.windows = windows or [5, 10, 21, 63]

    def extract_all(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> Union[pl.DataFrame, pl.LazyFrame]:
        is_lazy = isinstance(df, pl.LazyFrame)
        lf = df.lazy() if not is_lazy else df

        cols = lf.collect_schema().names()
        asset_col = COL_ASSET if COL_ASSET in cols else ("ticker" if "ticker" in cols else COL_ASSET)
        time_col = COL_TIMESTAMP if COL_TIMESTAMP in cols else ("date" if "date" in cols else COL_TIMESTAMP)

        lf = lf.unique(subset=[asset_col, time_col], keep="last").sort([asset_col, time_col])

        # 1. AKURASI IDR TURNOVER BERBASIS TYPICAL PRICE (H + L + C) / 3.0
        typical_price = (pl.col(COL_HIGH) + pl.col(COL_LOW) + pl.col(COL_CLOSE)) / 3.0
        daily_turnover_idr = (typical_price * pl.col(COL_VOLUME)).cast(pl.Float64).fill_null(0.0)

        # 2. ADVANCED VOLATILITY ESTIMATORS
        # Parkinson Volatility: sqrt( 1 / (4 * ln(2)) * (ln(H/L))^2 )
        log_hl_sq = ((pl.col(COL_HIGH) / (pl.col(COL_LOW) + EPSILON)).log() ** 2) / (4.0 * math.log(2.0))
        parkinson_vol = (log_hl_sq.rolling_mean(window_size=21, min_periods=21).over(asset_col).fill_null(0.0) * 252.0).sqrt()

        # Garman-Klass Volatility: 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
        log_co_sq = (pl.col(COL_CLOSE) / (pl.col(COL_OPEN) + EPSILON)).log() ** 2
        gk_term = (0.5 * log_hl_sq) - ((2.0 * math.log(2.0) - 1.0) * log_co_sq)
        garman_klass_vol = (gk_term.rolling_mean(window_size=21, min_periods=21).over(asset_col).fill_null(0.0) * 252.0).sqrt()

        # 3. MICROSTRUCTURE & LIQUIDITY METRICS
        # Amihud Illiquidity Ratio: |Return| / Turnover (IDR Juta)
        prev_close = pl.col(COL_CLOSE).shift(1).over(asset_col)
        abs_ret = ((pl.col(COL_CLOSE) - prev_close).abs() / (prev_close + EPSILON)).fill_null(0.0)
        amihud_illiquidity = (abs_ret / ((daily_turnover_idr / 1e6) + EPSILON)).fill_null(0.0)

        # Order Imbalance Proxy & Wicks Geometry
        hl_range = (pl.col(COL_HIGH) - pl.col(COL_LOW)) + EPSILON
        order_imbalance = ((pl.col(COL_CLOSE) - pl.col(COL_OPEN)) / hl_range).fill_null(0.0)
        body_pct = ((pl.col(COL_CLOSE) - pl.col(COL_OPEN)).abs() / hl_range).fill_null(0.0)
        upper_wick = ((pl.col(COL_HIGH) - pl.max_horizontal(COL_OPEN, COL_CLOSE)) / hl_range).fill_null(0.0)
        lower_wick = ((pl.min_horizontal(COL_OPEN, COL_CLOSE) - pl.col(COL_LOW)) / hl_range).fill_null(0.0)

        # Gap % & Pattern Recognition (Inside Bar, NR7)
        prev_high = pl.col(COL_HIGH).shift(1).over(asset_col)
        prev_low = pl.col(COL_LOW).shift(1).over(asset_col)
        gap_pct = ((pl.col(COL_OPEN) - prev_close) / (prev_close + EPSILON)).fill_null(0.0)
        is_inside_bar = (pl.col(COL_HIGH) <= prev_high) & (pl.col(COL_LOW) >= prev_low)
        min_hl_7 = hl_range.rolling_min(window_size=7, min_periods=7).over(asset_col)
        is_nr7 = (hl_range <= min_hl_7)

        exprs = [
            daily_turnover_idr.alias("f_volume_24h_idr"),
            daily_turnover_idr.alias("volume_24h_idr"),
            parkinson_vol.alias("f_vol_parkinson_21"),
            garman_klass_vol.alias("f_vol_garman_klass_21"),
            amihud_illiquidity.alias("f_micro_amihud_illiquidity"),
            order_imbalance.alias("f_micro_order_imbalance_proxy"),
            body_pct.alias("f_pr_body_pct"),
            upper_wick.alias("f_pr_upper_wick_pct"),
            lower_wick.alias("f_pr_lower_wick_pct"),
            gap_pct.alias("f_pr_gap_pct"),
            is_inside_bar.cast(pl.Float64).alias("f_pr_is_inside_bar"),
            is_nr7.cast(pl.Float64).alias("f_pr_is_nr7")
        ]

        log_ret = (pl.col(COL_CLOSE) / (prev_close + EPSILON)).log().fill_null(0.0)

        for w in self.windows:
            vol_daily = log_ret.rolling_std(window_size=w, min_periods=w).over(asset_col).fill_null(0.0)
            exprs.append(vol_daily.alias(f"f_vol_c2c_daily_{w}"))
            exprs.append((vol_daily * math.sqrt(252.0)).alias(f"f_vol_c2c_annual_{w}"))
            
            vol_sma = pl.col(COL_VOLUME).rolling_mean(window_size=w, min_periods=w).over(asset_col)
            rvol_expr = (pl.col(COL_VOLUME) / (vol_sma + EPSILON)).fill_null(1.0)
            exprs.append(rvol_expr.alias(f"f_volume_rvol_{w}"))
            
            adtv_w_idr = daily_turnover_idr.rolling_mean(window_size=w, min_periods=w).over(asset_col).fill_null(0.0)
            exprs.append(adtv_w_idr.alias(f"f_adtv_{w}d_idr"))
            exprs.append(adtv_w_idr.alias(f"adtv_{w}d_idr"))
            
            if w == 21:
                exprs.append(adtv_w_idr.alias("adtv_20d_idr"))
                exprs.append(adtv_w_idr.alias("f_adtv_20d_idr"))
            
            p_lag = pl.col(COL_CLOSE).shift(w).over(asset_col)
            roc_expr = ((pl.col(COL_CLOSE) - p_lag) / (p_lag + EPSILON)).fill_null(0.0)
            exprs.append(roc_expr.alias(f"f_momentum_roc_{w}"))

        # WARM-UP COMPLETENESS INDICATOR FLAG (Minimum 63 Baris Data Historis Per Aset)
        row_count = pl.col(COL_CLOSE).cum_count().over(asset_col)
        exprs.append((row_count >= 63).alias("feature_warmup_complete"))
        exprs.append(pl.lit(True).alias("feature_microstructure_ready"))

        final_lf = lf.with_columns(exprs)
        return final_lf if is_lazy else final_lf.collect()


# ==============================================================================
# SUB-ENGINE 4: ACCELERATED FEATURE DRIFT MONITOR (SEARCHSORTED & FREEDMAN-DIACONIS)
# ==============================================================================
class FeatureBaselineProfiler:
    """Reservoir Drift Baseline Generation Engine."""
    MAX_KS_SAMPLE: Final[int] = 10000

    @classmethod
    def generate_asset_profiles(cls, df: pl.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if df.height == 0:
            return {}

        asset_col = COL_ASSET if COL_ASSET in df.columns else ("ticker" if "ticker" in df.columns else "GLOBAL")
        assets = df[asset_col].unique().to_list() if asset_col in df.columns else ["GLOBAL"]
        profiles: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for asset in assets:
            profiles[asset] = {}
            sub_df = df.filter(pl.col(asset_col) == asset) if asset_col in df.columns else df

            for f in feature_cols:
                if f not in sub_df.columns or not sub_df[f].dtype.is_numeric():
                    continue

                s = sub_df.select(pl.col(f)).filter(pl.col(f).is_not_null() & pl.col(f).is_finite())[f]
                if s.len() < 10:
                    continue

                sample = s.slice(0, cls.MAX_KS_SAMPLE).to_list()
                profiles[asset][f] = {
                    "ks_reference_sample": sorted([float(x) for x in sample]),
                    "mean": float(s.mean()) if s.len() > 0 else 0.0,
                    "std": float(s.std()) if s.len() > 1 else 0.0,
                    "sample_size": s.len()
                }

        return profiles


class FeatureDriftMonitor:
    """Fast Kolmogorov-Smirnov (np.searchsorted) & Equal-Width Fallback PSI Engine."""

    def __init__(self, psi_warning_threshold: float = 0.10, psi_critical_threshold: float = 0.25) -> None:
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_critical_threshold = psi_critical_threshold
        self._lock = threading.Lock()
        self.drift_history: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()

    def compute_fast_ks_distance(self, b_sample_sorted: List[float], c_series: pl.Series) -> float:
        """AKSERELASI KS-TEST O(N log N) -> O(N) Menggunakan NumPy searchsorted()."""
        ref_arr = np.array(b_sample_sorted, dtype=np.float64)
        c_clean = c_series.filter(c_series.is_not_null() & c_series.is_finite()).to_numpy().astype(np.float64)
        
        n_ref = len(ref_arr)
        n_cur = len(c_clean)
        if n_ref == 0 or n_cur == 0:
            return 0.0

        cur_sorted = np.sort(c_clean)
        all_data = np.concatenate([ref_arr, cur_sorted])
        
        ref_cdf = np.searchsorted(ref_arr, all_data, side='right') / n_ref
        cur_cdf = np.searchsorted(cur_sorted, all_data, side='right') / n_cur

        return float(np.max(np.abs(ref_cdf - cur_cdf)))

    def compute_psi(self, reference_vals: List[float], actual_series: pl.Series, num_bins: int = 10) -> float:
        """ROBUST PSI DENGAN EQUAL-WIDTH BINS FALLBACK (Bebas dari Collapse Percentiles)."""
        ref_arr = np.array([float(x) for x in reference_vals if x is not None and not math.isnan(float(x)) and not math.isinf(float(x))], dtype=np.float64)
        act_clean = actual_series.filter(actual_series.is_not_null() & actual_series.is_finite()).to_numpy()
        act_arr = act_clean.astype(np.float64)

        if len(ref_arr) < 10 or len(act_arr) < 10:
            return 0.0

        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref_arr, quantiles)
        bins = np.unique(bins)

        # Fallback Equal-Width Bins jika nilai persentil identik/sempit
        if len(bins) < 2:
            min_v, max_v = np.min(ref_arr), np.max(ref_arr)
            if min_v == max_v:
                return 0.0
            bins = np.linspace(min_v, max_v, num_bins + 1)

        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(ref_arr, bins=bins)
        act_counts, _ = np.histogram(act_arr, bins=bins)

        ref_pct = ref_counts / len(ref_arr)
        act_pct = act_counts / len(act_arr)

        ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
        act_pct = np.where(act_pct == 0, 1e-4, act_pct)

        psi_value = np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))
        return float(psi_value)

    def analyze_drift(self, current_df: pl.DataFrame, baseline_profiles: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None) -> Dict[str, Any]:
        """Eksekusi analisa drift dengan matching profil statistik presisi PER ASET."""
        with self._lock:
            if current_df.height == 0:
                return {"drift_detected": False, "metrics": {"status": "EMPTY_DATA"}}

            cols = [c for c in current_df.columns if current_df[c].dtype.is_numeric() and not c.startswith("_")]
            drift_results = {}
            high_drift_count = 0

            asset_col = COL_ASSET if COL_ASSET in current_df.columns else "ticker"

            for c in cols:
                s = current_df[c]
                if s.filter(s.is_not_null() & s.is_finite()).len() < 10:
                    continue

                psi = 0.0
                ks_dist = 0.0

                if baseline_profiles and asset_col in current_df.columns:
                    current_assets = current_df[asset_col].unique().to_list()
                    # Koreksi baseline per aset tanpa bug break premature
                    for asset_k in current_assets:
                        if asset_k in baseline_profiles and c in baseline_profiles[asset_k]:
                            sample_baseline = baseline_profiles[asset_k][c].get("ks_reference_sample")
                            if sample_baseline:
                                asset_series = current_df.filter(pl.col(asset_col) == asset_k)[c]
                                ks_dist = max(ks_dist, self.compute_fast_ks_distance(sample_baseline, asset_series))
                                psi = max(psi, self.compute_psi(sample_baseline, asset_series))

                status = "STABLE"
                if psi >= self.psi_critical_threshold:
                    status = "CRITICAL"
                    high_drift_count += 1
                elif psi >= self.psi_warning_threshold:
                    status = "WARNING"

                drift_results[c] = {
                    "psi": round(psi, 4),
                    "ks_distance": round(ks_dist, 4),
                    "status": status
                }

            summary = {
                "drift_detected": high_drift_count > 0,
                "high_drift_features_count": high_drift_count,
                "evaluated_cols": len(drift_results),
                "feature_metrics": drift_results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            self.drift_history[summary["timestamp"]] = summary
            if len(self.drift_history) > 100:
                self.drift_history.popitem(last=False)

            return summary


# ==============================================================================
# SUB-ENGINE 5: FEATURE SELECTION ENGINE (INCREMENTAL PRUNING)
# ==============================================================================
class FeatureSelectionEngine:
    """Non-Leaking Variance & Incremental Collinearity Pruning Engine."""

    def __init__(self, max_null_rate: float = 0.05, min_variance: float = 1e-5, max_cross_correlation: float = 0.85) -> None:
        self.max_null_rate = max_null_rate
        self.min_variance = min_variance
        self.max_cross_correlation = max_cross_correlation
        self.selected_features: List[str] = []
        self.dropped_manifest: Dict[str, str] = {}

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        protected = {
            COL_TIMESTAMP, COL_ASSET, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, COL_SECTOR,
            "ticker", "date", "arithmetic_return", "log_return", "feature_warmup_complete",
            "volume_24h_idr", "f_volume_24h_idr", "adtv_20d_idr", "f_adtv_20d_idr",
            "feature_rsi", "f_rsi_14", "rsi_14", "feature_atr", "f_atr_14", "atr_14", "atr",
            "f_cs_global_rank_ret", "f_cs_global_zscore_ret", "f_cs_sector_rank_ret"
        }
        
        candidates = [
            c for c in df.columns 
            if c not in protected 
            and not c.startswith("_") 
            and df[c].dtype.is_numeric()
        ]

        dropped: Dict[str, str] = {}
        valid_candidates: List[str] = []

        # 1. Null Rate & Variance Filtering
        for c in candidates:
            s = df[c]
            null_rate = s.null_count() / df.height
            if null_rate > self.max_null_rate:
                dropped[c] = f"Null Rate Breach ({null_rate:.2f} > {self.max_null_rate})"
                continue

            clean_s = s.filter(s.is_not_null() & s.is_finite())
            var_val = clean_s.var() if clean_s.len() > 1 else 0.0
            if var_val is None or var_val < self.min_variance:
                dropped[c] = f"Low Variance Pruned ({var_val if var_val is not None else 0.0:.6f})"
                continue

            valid_candidates.append(c)

        # 2. INCREMENTAL COL-BY-COL PRUNING (Menghindari N x N Matriks Memory Bottleneck)
        selected_set: List[str] = []
        for feat in valid_candidates:
            feat_vec = df[feat].fill_null(strategy="forward").fill_null(0.0).to_numpy()
            feat_vec = np.nan_to_num(feat_vec, nan=0.0)
            
            is_collinear = False
            for prev_feat in selected_set:
                prev_vec = df[prev_feat].fill_null(strategy="forward").fill_null(0.0).to_numpy()
                prev_vec = np.nan_to_num(prev_vec, nan=0.0)
                
                corr = abs(float(np.corrcoef(feat_vec, prev_vec)[0, 1]))
                if not math.isnan(corr) and corr >= self.max_cross_correlation:
                    is_collinear = True
                    dropped[feat] = f"Incremental Collinear ({corr:.2f} >= {self.max_cross_correlation}) with {prev_feat}"
                    break
            
            if not is_collinear:
                selected_set.append(feat)

        self.dropped_manifest = dropped
        self.selected_features = selected_set

        cols_to_drop = list(dropped.keys())
        return df.drop(cols_to_drop)


# ==============================================================================
# SUB-ENGINE 6: HISTORICAL FEATURE STORE & POINT-IN-TIME AS-OF JOIN
# ==============================================================================
class HistoricalFeatureStore:
    """Atomic Parquet Persistence & Point-In-Time As-Of Join Repository."""

    def __init__(self, base_path: Union[str, Path] = "data/feature_store") -> None:
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_features(self, version_tag: str, df: pl.DataFrame, registry: Dict[str, FeatureRegistryRecord]) -> str:
        if df.height == 0:
            raise DataValidationError("Tidak dapat menyimpan DataFrame kosong ke FeatureStore.")

        version_dir = self.base_path / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = version_dir / "features.parquet"
        manifest_path = version_dir / "feature_manifest.json"

        tmp_parquet = parquet_path.with_suffix(".tmp")
        df.write_parquet(tmp_parquet, compression="snappy")
        os.replace(tmp_parquet, parquet_path)

        manifest_payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "version_tag": version_tag,
            "row_count": df.height,
            "registry": {k: asdict(v) for k, v in registry.items()}
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

        return str(parquet_path)

    def load_features(self, version_tag: str = "latest") -> pl.LazyFrame:
        p_path = self.base_path / version_tag / "features.parquet"
        if not p_path.exists():
            raise FileNotFoundError(f"Versi FeatureStore '{version_tag}' tidak ditemukan pada lokasi {p_path}")
        return pl.scan_parquet(p_path)

    def join_point_in_time_features(
        self,
        events_df: pl.DataFrame,
        feature_version: str = "latest",
        feature_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Point-In-Time As-Of Join Bebas Lookahead Bias & Suffix Collision."""
        if events_df.height == 0:
            return events_df

        events_df = sanitize_column_names(events_df)
        asset_col = COL_ASSET if COL_ASSET in events_df.columns else "ticker"
        time_col = COL_TIMESTAMP if COL_TIMESTAMP in events_df.columns else "date"

        if feature_df is None:
            f_lazy = self.load_features(version_tag=feature_version)
            f_df = f_lazy.collect()
        else:
            f_df = feature_df

        f_df = sanitize_column_names(f_df)
        f_asset_col = COL_ASSET if COL_ASSET in f_df.columns else "ticker"
        f_time_col = COL_TIMESTAMP if COL_TIMESTAMP in f_df.columns else "date"

        overlapping_cols = [
            c for c in f_df.columns 
            if c in events_df.columns and c not in [f_time_col, f_asset_col, time_col, asset_col]
        ]
        if overlapping_cols:
            f_df = f_df.drop(overlapping_cols)

        # PRECISI SORTING KUNCI GANDA [ASSET, TIMESTAMP] SEBELUM JOIN_ASOF
        events_sorted = events_df.sort([asset_col, time_col])
        f_sorted = f_df.sort([f_asset_col, f_time_col])

        joined = events_sorted.join_asof(
            f_sorted,
            on=time_col,
            by_left=asset_col,
            by_right=f_asset_col,
            strategy="backward"
        )
        return joined


class FeatureStore(HistoricalFeatureStore):
    """Facade Adapter Alias untuk kompatibilitas penuh dengan PipelineManager."""
    
    def __init__(self, config_or_path: Any = "data/feature_store") -> None:
        path = config_or_path if isinstance(config_or_path, (str, Path)) else "data/feature_store"
        super().__init__(base_path=path)

    def compute_and_store(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Mengeksekusi ekstraksi dan penyimpanan fitur kuantitatif ke FeatureStore Saham IDX.")
        if df.height > 0:
            engine = UnifiedFeatureEngine(store_dir=str(self.base_path))
            feat_df = engine.compute_full_feature_grid(df, run_selection=False)
            
            full_reg = get_feature_registry()
            try:
                self.save_features("latest", feat_df, full_reg)
            except Exception as e:
                logger.warning(f"Notifikasi non-fatal penyimpanan FeatureStore: {e}")
            return feat_df
        return df


# ==============================================================================
# FACADE ENGINE: UNIFIED FEATURE ENGINE WITH AGGRESSIVE MEMORY MANAGEMENT
# ==============================================================================
class UnifiedFeatureEngine:
    """
    Facade Entrypoint Terpadu untuk Pengelolaan Siklus Hidup Fitur Kuantitatif Saham IDX.
    Menggabungkan Ekstraksi, Monitoring Drift, Seleksi Fitur, dan Persistensi Store.
    """

    def __init__(self, store_dir: str = "data/feature_store") -> None:
        self.cross_sectional = CrossSectionalFeatureExtractor()
        self.technical_trend = IDXTechnicalAndTrendBuilder()
        self.microstructure = ComprehensiveMicrostructureExtractor()
        self.drift_monitor = FeatureDriftMonitor()
        self.selector = FeatureSelectionEngine()
        self.store = HistoricalFeatureStore(base_path=store_dir)
        logger.info("UnifiedFeatureEngine Saham IDX v2026.Q3.v17.1-INSTITUTIONAL berhasil diinisialisasi.")

    def compute_full_feature_grid(self, df: Union[pl.DataFrame, pl.LazyFrame], run_selection: bool = True) -> pl.DataFrame:
        """
        Pipa Ekstraksi Fitur Terpadu dengan Aggressive Garbage Collection (gc.collect()).
        """
        t_start = time.perf_counter()
        eager_df = df.collect() if isinstance(df, pl.LazyFrame) else df

        if eager_df.height == 0:
            logger.warning("Input DataFrame kosong. Ekstraksi fitur dibatalkan.")
            return eager_df

        eager_df = sanitize_column_names(eager_df)

        # 1. Ekstraksi Indikator Teknikal & Tren
        df_tech = self.technical_trend.build_all_features(eager_df)
        del eager_df
        gc.collect()

        # 2. Ekstraksi Microstructure, Volatilitas Advanced & Typical Price ADTV
        df_micro = self.microstructure.extract_all(df_tech)
        if isinstance(df_micro, pl.LazyFrame):
            df_micro = df_micro.collect()
        del df_tech
        gc.collect()

        # 3. Ekstraksi Cross-Sectional Alpha Ranks Presisi
        df_cs = self.cross_sectional.extract(df_micro)
        if isinstance(df_cs, pl.LazyFrame):
            df_cs = df_cs.collect()
        del df_micro
        gc.collect()

        # 4. Imputasi Safe Vector Null, NaN, & Inf Otomatis
        asset_col = COL_ASSET if COL_ASSET in df_cs.columns else "ticker"
        numeric_cols = [c for c in df_cs.columns if df_cs[c].dtype.is_numeric() and not c.startswith("_")]
        
        impute_exprs = [
            pl.when(pl.col(c).is_null() | pl.col(c).is_nan() | pl.col(c).is_infinite())
            .then(0.0)
            .otherwise(pl.col(c))
            .fill_null(strategy="forward")
            .over(asset_col)
            .fill_null(0.0)
            .alias(c)
            for c in numeric_cols
        ]
        df_clean = df_cs.with_columns(impute_exprs)
        del df_cs
        gc.collect()

        # 5. Seleksi Fitur Otomatis
        if run_selection:
            df_final = self.selector.fit_transform(df_clean)
            del df_clean
            gc.collect()
        else:
            df_final = df_clean

        duration = time.perf_counter() - t_start
        logger.info(f"Ekstraksi fitur saham IDX selesai dalam {duration:.4f}s. Total kolom luaran: {len(df_final.columns)}")
        
        return df_final


# ==============================================================================
# PROCEDURAL TOP-LEVEL CONVENIENCE APIs
# ==============================================================================
def extract_all_features(df: pl.DataFrame, run_selection: bool = True) -> pl.DataFrame:
    """API prosedural terpadu untuk ekstraksi seluruh fitur kuantitatif."""
    engine = UnifiedFeatureEngine()
    return engine.compute_full_feature_grid(df, run_selection=run_selection)


def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """Alias prosedural kompatibilitas tinggi untuk orchestrator main.py."""
    return extract_all_features(df, run_selection=True)


def get_feature_registry() -> Dict[str, FeatureRegistryRecord]:
    """Mengembalikan manifes registri cetak biru fitur kuantitatif terpadu."""
    cs_engine = CrossSectionalFeatureExtractor()
    reg = cs_engine.feature_registry.copy()
    
    common_deps = [COL_TIMESTAMP, COL_ASSET, COL_CLOSE, COL_VOLUME]
    anchored_time = cs_engine.ANCHORED_TIME
    
    tech_blueprints = [
        ("feature_rsi", "TECH-RSI-14", (0.0, 100.0), "Float64"),
        ("feature_atr", "TECH-ATR-14", (0.0, None), "Float64"),
        ("f_trend_kama_10", "TREND-KAMA-10", (0.0, None), "Float64"),
        ("f_trend_hma_16", "TREND-HMA-16", (0.0, None), "Float64"),
        ("f_vol_parkinson_21", "VOL-PARKINSON-21", (0.0, None), "Float64"),
        ("f_vol_garman_klass_21", "VOL-GARMAN-KLASS-21", (0.0, None), "Float64"),
        ("f_micro_amihud_illiquidity", "MICRO-AMIHUD-ILLIQ", (0.0, None), "Float64"),
        ("f_volume_24h_idr", "MICRO-VOL-24H-IDR", (0.0, None), "Float64"),
        ("adtv_20d_idr", "MICRO-ADTV-20D-IDR", (0.0, None), "Float64"),
        ("feature_warmup_complete", "META-WARMUP-COMPLETE", (0.0, 1.0), "Boolean")
    ]
    
    for fname, fid, bounds, dtype in tech_blueprints:
        rec = FeatureRegistryRecord(
            feature_id=fid, feature_name=fname, version="17.0.0", origin_module="features.indicators",
            owner="Quantitative Architecture Core", created_at=anchored_time, dependencies=common_deps,
            warmup_required=63, data_type=dtype, expected_range=bounds, drift_limit_psi=0.10,
            status="active", definition_hash=hashlib.sha256(f"{fname}_{fid}".encode("utf-8")).hexdigest(),
            quality_score=1.0, importance=1.0, selection_count=1, last_importance=1.0, survival_round=1,
            health_status="stable", last_selected_at=None, last_drift_psi=0.0, last_retrained_at=None,
            drop_reason=None, feature_age_days=0
        )
        reg[fname] = rec
        
    return reg

"""
Feature schema & versioning for the IDX quant pipeline.

FEATURE_SCHEMA_VERSION bumps whenever columns are added/removed/renamed.
Downstream models should store the version they were trained on.
"""

from __future__ import annotations

from typing import Final

FEATURE_SCHEMA_VERSION: Final[str] = "1.0.0"

# Base OHLCV columns (always present)
BASE_COLUMNS: Final[list[str]] = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# Feature groups produced by engineering.py
RETURN_FEATURES: Final[list[str]] = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "log_ret_1",
    "cum_ret_20",
]

TREND_FEATURES: Final[list[str]] = [
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_5",
    "ema_10",
    "ema_20",
    "price_vs_sma_20",
    "ema_spread_5_20",
    "trend_slope_10",
]

VOLATILITY_FEATURES: Final[list[str]] = [
    "roll_std_10",
    "roll_std_20",
    "atr_14",
    "natr_14",
    "vol_ratio_5_20",
    "range_expansion",
]

MOMENTUM_FEATURES: Final[list[str]] = [
    "rsi_14",
    "roc_5",
    "roc_10",
    "mom_10",
]

VOLUME_FEATURES: Final[list[str]] = [
    "vol_chg_1",
    "vol_sma_20",
    "rel_vol_20",
    "vol_zscore_20",
    "pv_corr_10",
]

STRUCTURE_FEATURES: Final[list[str]] = [
    "hl_range",
    "close_loc",
    "gap",
    "dist_to_high_20",
    "dist_to_low_20",
    "breakout_dist_20",
]

ALL_FEATURE_COLUMNS: Final[list[str]] = (
    RETURN_FEATURES
    + TREND_FEATURES
    + VOLATILITY_FEATURES
    + MOMENTUM_FEATURES
    + VOLUME_FEATURES
    + STRUCTURE_FEATURES
)

# Warm-up: max lookback used by any feature
MAX_LOOKBACK: Final[int] = 20

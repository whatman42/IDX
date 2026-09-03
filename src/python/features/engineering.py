"""
Causal Feature Engineering Pipeline (Polars-first).

Strict causality
----------------
Every feature at timestamp *t* uses only information available up to and
including *t*.  All rolling windows are trailing (not centered).  No future
prices, highs, lows, or volumes ever enter a feature calculation.

Warm-up policy
--------------
Rolling features produce nulls for the first MAX_LOOKBACK rows per symbol.
``build_ml_dataset`` drops those rows so the ML-ready frame contains no
unexpected nulls or infs.

Multi-symbol
------------
All rolling / ewm operations are partitioned by ``symbol`` so that BBCA
features are never contaminated by TLKM data.
"""

from __future__ import annotations

from typing import Optional, Union

import polars as pl

from src.python.features.schema import (
    ALL_FEATURE_COLUMNS,
    BASE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    MAX_LOOKBACK,
)

FrameLike = Union[pl.DataFrame, "pd.DataFrame"]  # noqa: F821


def _to_polars(df: FrameLike) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    import pandas as pd
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Expected polars or pandas DataFrame, got {type(df)}")


def _validate_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    colmap = {c.lower(): c for c in df.columns}
    missing = required - set(colmap)
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")
    df = df.rename({colmap[k]: k for k in required})
    df = df.sort(["symbol", "timestamp"])
    df = df.unique(subset=["symbol", "timestamp"], keep="first", maintain_order=True)
    df = df.sort(["symbol", "timestamp"])
    for col in ("open", "high", "low", "close"):
        if df.filter(pl.col(col) <= 0).height > 0:
            raise ValueError(f"Non-positive prices in '{col}'")
    if df.filter(pl.col("volume") < 0).height > 0:
        raise ValueError("Negative volume detected")
    return df


def compute_features(ohlcv: FrameLike) -> pl.DataFrame:
    """Compute the full causal feature set (trailing windows only, per symbol)."""
    df = _validate_ohlcv(_to_polars(ohlcv))

    # Price / Return
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias("ret_1"),
        (pl.col("close") / pl.col("close").shift(3).over("symbol") - 1.0).alias("ret_3"),
        (pl.col("close") / pl.col("close").shift(5).over("symbol") - 1.0).alias("ret_5"),
        (pl.col("close") / pl.col("close").shift(10).over("symbol") - 1.0).alias("ret_10"),
        (pl.col("close") / pl.col("close").shift(20).over("symbol") - 1.0).alias("ret_20"),
        (pl.col("close") / pl.col("close").shift(1).over("symbol")).log().alias("log_ret_1"),
        (pl.col("close") / pl.col("close").shift(20).over("symbol") - 1.0).alias("cum_ret_20"),
    ])

    # Trend
    df = df.with_columns([
        pl.col("close").rolling_mean(5).over("symbol").alias("sma_5"),
        pl.col("close").rolling_mean(10).over("symbol").alias("sma_10"),
        pl.col("close").rolling_mean(20).over("symbol").alias("sma_20"),
        pl.col("close").ewm_mean(span=5, adjust=False).over("symbol").alias("ema_5"),
        pl.col("close").ewm_mean(span=10, adjust=False).over("symbol").alias("ema_10"),
        pl.col("close").ewm_mean(span=20, adjust=False).over("symbol").alias("ema_20"),
    ])
    df = df.with_columns([
        (pl.col("close") / pl.col("sma_20") - 1.0).alias("price_vs_sma_20"),
        (pl.col("ema_5") - pl.col("ema_20")).alias("ema_spread_5_20"),
        ((pl.col("close") - pl.col("close").shift(10).over("symbol")) / 10.0).alias("trend_slope_10"),
    ])

    # Volatility
    prev_close = pl.col("close").shift(1).over("symbol")
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    df = df.with_columns([
        pl.col("close").rolling_std(10).over("symbol").alias("roll_std_10"),
        pl.col("close").rolling_std(20).over("symbol").alias("roll_std_20"),
        tr.alias("_tr"),
        pl.col("close").rolling_std(5).over("symbol").alias("roll_std_5"),
    ])
    df = df.with_columns(pl.col("_tr").rolling_mean(14).over("symbol").alias("atr_14"))
    df = df.with_columns([
        (pl.col("atr_14") / pl.col("close")).alias("natr_14"),
        (pl.col("roll_std_5") / pl.col("roll_std_20")).alias("vol_ratio_5_20"),
        ((pl.col("high") - pl.col("low")) / (pl.col("high") - pl.col("low")).rolling_mean(20).over("symbol")).alias("range_expansion"),
    ])

    # Momentum (RSI via ewm of gains/losses)
    delta = pl.col("close") - pl.col("close").shift(1).over("symbol")
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(span=14, adjust=False).over("symbol")
    avg_loss = loss.ewm_mean(span=14, adjust=False).over("symbol")
    rs = avg_gain / avg_loss
    df = df.with_columns([
        (100.0 - (100.0 / (1.0 + rs))).alias("rsi_14"),
        (pl.col("close") / pl.col("close").shift(5).over("symbol") - 1.0).alias("roc_5"),
        (pl.col("close") / pl.col("close").shift(10).over("symbol") - 1.0).alias("roc_10"),
        (pl.col("close") - pl.col("close").shift(10).over("symbol")).alias("mom_10"),
    ])

    # Volume
    df = df.with_columns([
        (pl.col("volume") / pl.col("volume").shift(1).over("symbol") - 1.0).alias("vol_chg_1"),
        pl.col("volume").rolling_mean(20).over("symbol").alias("vol_sma_20"),
    ])
    df = df.with_columns([
        (pl.col("volume") / pl.col("vol_sma_20")).alias("rel_vol_20"),
        ((pl.col("volume") - pl.col("vol_sma_20")) / pl.col("volume").rolling_std(20).over("symbol")).alias("vol_zscore_20"),
        (pl.col("ret_1") * pl.col("vol_chg_1")).alias("pv_corr_10"),
    ])

    # Market structure
    df = df.with_columns([
        (pl.col("high") - pl.col("low")).alias("hl_range"),
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-12)).alias("close_loc"),
        (pl.col("open") / pl.col("close").shift(1).over("symbol") - 1.0).alias("gap"),
        (pl.col("close") / pl.col("high").rolling_max(20).over("symbol") - 1.0).alias("dist_to_high_20"),
        (pl.col("close") / pl.col("low").rolling_min(20).over("symbol") - 1.0).alias("dist_to_low_20"),
        (pl.col("close") / pl.col("high").rolling_max(20).over("symbol") - 1.0).alias("breakout_dist_20"),
    ])

    drop_cols = [c for c in ("_tr", "roll_std_5") if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    df = df.with_columns(pl.lit(FEATURE_SCHEMA_VERSION).alias("feature_schema_version"))
    return df


def build_ml_dataset(
    ohlcv: FrameLike,
    labels: Optional[FrameLike] = None,
    drop_warmup: bool = True,
    min_history: int = MAX_LOOKBACK,
) -> dict:
    """Produce ML-ready dataset: X, y (optional), metadata."""
    feats = compute_features(ohlcv)

    if drop_warmup:
        feats = feats.with_columns(
            pl.col("timestamp").rank("ordinal").over("symbol").alias("_rn")
        )
        feats = feats.filter(pl.col("_rn") > min_history).drop("_rn")

    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in feats.columns]
    for c in feature_cols:
        feats = feats.filter(pl.col(c).is_not_null() & pl.col(c).is_finite())

    y = None
    if labels is not None:
        lab = _to_polars(labels)
        lab_cols = {c.lower(): c for c in lab.columns}
        rename = {}
        for k in ("timestamp", "symbol", "label"):
            if k in lab_cols:
                rename[lab_cols[k]] = k
        lab = lab.rename(rename)
        feats = feats.join(
            lab.select(["timestamp", "symbol", "label"]),
            on=["timestamp", "symbol"],
            how="inner",
        )
        y = feats["label"]

    meta = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "n_rows": feats.height,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "symbols": feats["symbol"].unique().to_list() if feats.height else [],
        "max_lookback": MAX_LOOKBACK,
        "warmup_dropped": drop_warmup,
    }

    id_cols = ["timestamp", "symbol"]
    X = feats.select(
        id_cols + feature_cols + (
            ["feature_schema_version"] if "feature_schema_version" in feats.columns else []
        )
    )
    return {"X": X, "y": y, "metadata": meta}

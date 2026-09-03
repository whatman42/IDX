"""
Comprehensive tests for the causal Feature Engineering pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.python.features import (
    ALL_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    MAX_LOOKBACK,
    build_ml_dataset,
    compute_features,
    validate_features,
)
from src.python.labeling.triple_barrier import apply_triple_barrier


def _make_panel(
    symbols: list[str] = ("BBCA",),
    n: int = 60,
    start: str = "2024-01-01",
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i, sym in enumerate(symbols):
        idx = pd.date_range(start, periods=n, freq="B")
        close = 100 + i * 50 + np.cumsum(rng.normal(0, 0.8, n))
        high = close + rng.uniform(0.2, 1.5, n)
        low = close - rng.uniform(0.2, 1.5, n)
        open_ = close + rng.normal(0, 0.3, n)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        vol = rng.integers(100_000, 5_000_000, n).astype(float)
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": idx,
                    "symbol": sym,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": vol,
                }
            )
        )
    return pl.concat(frames).sort(["symbol", "timestamp"])


class TestBasic:
    def test_ohlcv_input_produces_features(self):
        df = _make_panel(n=40)
        out = compute_features(df)
        assert out.height == df.height
        for c in ALL_FEATURE_COLUMNS:
            assert c in out.columns, f"missing {c}"

    def test_schema_version_attached(self):
        out = compute_features(_make_panel(n=30))
        assert "feature_schema_version" in out.columns
        assert out["feature_schema_version"][0] == FEATURE_SCHEMA_VERSION

    def test_base_columns_preserved(self):
        df = _make_panel(n=25)
        out = compute_features(df)
        for c in ("timestamp", "symbol", "open", "high", "low", "close", "volume"):
            assert c in out.columns


class TestCausality:
    def test_future_mutation_does_not_change_history(self):
        base = _make_panel(n=50, seed=7)
        feats_a = compute_features(base)

        mutated = base.with_columns(
            [
                pl.when(pl.col("timestamp") >= base["timestamp"][-5])
                .then(pl.lit(999_999_999.0))
                .otherwise(pl.col("close"))
                .alias("close"),
                pl.when(pl.col("timestamp") >= base["timestamp"][-5])
                .then(pl.lit(999_999_999.0))
                .otherwise(pl.col("high"))
                .alias("high"),
                pl.when(pl.col("timestamp") >= base["timestamp"][-5])
                .then(pl.lit(1.0))
                .otherwise(pl.col("low"))
                .alias("low"),
                pl.when(pl.col("timestamp") >= base["timestamp"][-5])
                .then(pl.lit(999_999_999.0))
                .otherwise(pl.col("volume"))
                .alias("volume"),
            ]
        )
        feats_b = compute_features(mutated)

        cutoff = base["timestamp"][-6]
        a = feats_a.filter(pl.col("timestamp") <= cutoff).select(ALL_FEATURE_COLUMNS)
        b = feats_b.filter(pl.col("timestamp") <= cutoff).select(ALL_FEATURE_COLUMNS)

        assert a.shape == b.shape
        for c in ALL_FEATURE_COLUMNS:
            diff = (a[c] - b[c]).abs().max()
            assert a[c].null_count() == b[c].null_count()
            if diff is not None and not (isinstance(diff, float) and np.isnan(diff)):
                assert diff < 1e-9 or (diff != diff), f"Leakage in {c}: max|Δ|={diff}"

    def test_no_centered_rolling(self):
        df = _make_panel(n=30)
        out = compute_features(df)
        nulls = out.filter(pl.col("symbol") == "BBCA")["sma_20"].null_count()
        assert nulls >= 19


class TestMultiSymbol:
    def test_symbols_independent(self):
        panel = _make_panel(symbols=["BBCA", "TLKM", "AAPL"], n=40, seed=11)
        out = compute_features(panel)
        bbca = out.filter(pl.col("symbol") == "BBCA").select(["timestamp"] + ALL_FEATURE_COLUMNS)
        solo = compute_features(panel.filter(pl.col("symbol") == "BBCA")).select(
            ["timestamp"] + ALL_FEATURE_COLUMNS
        )
        assert bbca.shape == solo.shape
        for c in ALL_FEATURE_COLUMNS:
            d = (bbca[c] - solo[c]).abs().max()
            if d is not None and d == d:
                assert d < 1e-9, f"Cross-symbol leakage in {c}"


class TestTimestamps:
    def test_unsorted_input(self):
        df = _make_panel(n=20)
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=1)
        out = compute_features(shuffled)
        ts = out.filter(pl.col("symbol") == "BBCA")["timestamp"].to_list()
        assert ts == sorted(ts)

    def test_duplicate_timestamps(self):
        df = _make_panel(n=15)
        dup = pl.concat([df, df.head(2)])
        out = compute_features(dup)
        n_unique = out.select(["symbol", "timestamp"]).unique().height
        assert n_unique == out.height


class TestNumerical:
    def test_constant_price(self):
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pl.DataFrame(
            {
                "timestamp": idx,
                "symbol": "FLAT",
                "open": [100.0] * n,
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
                "volume": [1e6] * n,
            }
        )
        out = compute_features(df)
        assert out.height == n
        rets = out["ret_1"].drop_nulls()
        assert (rets.abs() < 1e-12).all()

    def test_zero_volume_ok(self):
        df = _make_panel(n=30)
        df = df.with_columns(
            pl.when(pl.int_range(0, pl.len()) == 5)
            .then(pl.lit(0.0))
            .otherwise(pl.col("volume"))
            .alias("volume")
        )
        out = compute_features(df)
        assert out.height == 30

    def test_non_positive_price_raises(self):
        df = _make_panel(n=10)
        df = df.with_columns(pl.lit(-1.0).alias("close"))
        with pytest.raises(ValueError, match="Non-positive"):
            compute_features(df)

    def test_missing_column_raises(self):
        df = _make_panel(n=10).drop("volume")
        with pytest.raises(ValueError, match="missing columns"):
            compute_features(df)


class TestDeterminism:
    def test_same_input_same_output(self):
        df = _make_panel(n=35, seed=99)
        a = compute_features(df)
        b = compute_features(df)
        for c in ALL_FEATURE_COLUMNS:
            assert a[c].equals(b[c]), f"Non-deterministic column {c}"


class TestMLDataset:
    def test_build_drops_warmup(self):
        df = _make_panel(n=50)
        result = build_ml_dataset(df, drop_warmup=True)
        X = result["X"]
        assert X.height < 50
        assert result["metadata"]["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        for c in result["metadata"]["feature_columns"]:
            assert X[c].null_count() == 0

    def test_with_labels_join(self):
        df = _make_panel(n=50)
        labels = df.select(
            [
                "timestamp",
                "symbol",
                (pl.col("close") > pl.col("close").shift(1)).cast(pl.Int8).alias("label"),
            ]
        )
        result = build_ml_dataset(df, labels=labels, drop_warmup=True)
        assert result["y"] is not None
        assert result["X"].height == result["y"].len()


class TestTBMIntegration:
    def test_features_then_tbm(self):
        raw = _make_panel(n=40, seed=3)
        pdf = raw.filter(pl.col("symbol") == "BBCA")
        ohlc = pd.DataFrame(
            {
                "open": pdf["open"].to_list(),
                "high": pdf["high"].to_list(),
                "low": pdf["low"].to_list(),
                "close": pdf["close"].to_list(),
            },
            index=pd.DatetimeIndex(pdf["timestamp"].to_list()),
        )
        events = ohlc.index[MAX_LOOKBACK : MAX_LOOKBACK + 5]
        labels = apply_triple_barrier(
            ohlc, events, pt_sl=(0.02, 0.01), molecule=5, side=1
        )
        assert len(labels) > 0
        assert "label" in labels.columns
        feats = compute_features(raw)
        assert feats.height == 40


class TestValidation:
    def test_validate_clean_frame(self):
        result = build_ml_dataset(_make_panel(n=50), drop_warmup=True)
        issues = validate_features(result["X"], allow_null=False)
        assert issues == []

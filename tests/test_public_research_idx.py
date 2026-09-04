"""Public research IDX mapping — no embedded bulk data."""
from __future__ import annotations
import pandas as pd
import pytest
from src.python.data.public_research_idx import (
    LENSETEK_OHLCV_MAPPING, map_lensetek_ohlcv, drop_duplicate_keys, filter_ohlc_envelope,
)

def test_mapping_keys():
    assert set(LENSETEK_OHLCV_MAPPING) == {"date", "ticker", "open", "high", "low", "close", "volume"}

def test_map_and_dedupe():
    raw = pd.DataFrame({
        "date": ["2020-01-02", "2020-01-02", "2020-01-03"],
        "ticker": ["BBCA", "BBCA", "BBCA"],
        "open": [1.0, 1.1, 1.2], "high": [1.2, 1.2, 1.3],
        "low": [0.9, 0.9, 1.0], "close": [1.1, 1.15, 1.25], "volume": [100, 200, 150],
    })
    m = map_lensetek_ohlcv(raw)
    assert list(m.columns) == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    d, n = drop_duplicate_keys(m)
    assert n == 1 and len(d) == 2

def test_filter_envelope():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-02", "2020-01-03"]),
        "symbol": ["X", "X"],
        "open": [10.0, 10.0], "high": [9.0, 11.0], "low": [8.0, 9.0], "close": [9.5, 10.5],
        "volume": [1.0, 1.0],
    })
    clean, n = filter_ohlc_envelope(df)
    assert n == 1 and len(clean) == 1

def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing_raw_columns"):
        map_lensetek_ohlcv(pd.DataFrame({"date": ["2020-01-01"], "open": [1.0]}))

"""Public IDX research subset helpers (lensetek/idx-panel-data-descriptor).

NOT an official BEI production feed. yfinance auto_adjust=True → ADJUSTED basis,
corporate actions UNVERIFIED beyond vendor auto-adjust.

Never commit raw proprietary data; this module only maps/validates external paths.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

LENSETEK_OHLCV_MAPPING = {
    "date": "timestamp", "ticker": "symbol", "open": "open",
    "high": "high", "low": "low", "close": "close", "volume": "volume",
}
SOURCE_REPO = "https://github.com/lensetek/idx-panel-data-descriptor"
SOURCE_DOI = "10.5281/zenodo.21110404"
LICENSE_CLAIM = "CC-BY 4.0"

def map_lensetek_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    need = set(LENSETEK_OHLCV_MAPPING)
    missing = need - set(cols)
    if missing:
        raise ValueError(f"missing_raw_columns:{sorted(missing)}")
    return pd.DataFrame({
        "timestamp": pd.to_datetime(df[cols["date"]]),
        "symbol": df[cols["ticker"]].astype(str).str.upper(),
        "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
        "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
        "low": pd.to_numeric(df[cols["low"]], errors="coerce"),
        "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
        "volume": pd.to_numeric(df[cols["volume"]], errors="coerce"),
    })

def drop_duplicate_keys(df: pd.DataFrame):
    before = len(df)
    out = df.drop_duplicates(["timestamp", "symbol"], keep="last")
    return out, before - len(out)

def filter_ohlc_envelope(df: pd.DataFrame):
    bad = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    return df.loc[~bad].copy(), int(bad.sum())

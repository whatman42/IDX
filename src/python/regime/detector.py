"""Lightweight regime detection from realized volatility & trend."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class Regime(str, Enum):
    LOW_VOL_TREND = "LOW_VOL_TREND"
    HIGH_VOL = "HIGH_VOL"
    MEAN_REVERT = "MEAN_REVERT"
    UNKNOWN = "UNKNOWN"


def detect_regime(close: pd.Series, lookback: int = 20, high_vol_z: float = 1.25) -> Regime:
    if close is None or len(close) < lookback + 1:
        return Regime.UNKNOWN
    rets = close.pct_change().dropna()
    if len(rets) < lookback:
        return Regime.UNKNOWN
    window = rets.iloc[-lookback:]
    vol = float(window.std())
    hist = rets.iloc[max(0, len(rets) - 5 * lookback) : -lookback]
    base = float(hist.std()) if len(hist) > 5 else vol
    if base < 1e-12:
        return Regime.UNKNOWN
    z = vol / base
    trend = float(close.iloc[-1] / close.iloc[-lookback] - 1.0)
    if z >= high_vol_z:
        return Regime.HIGH_VOL
    if abs(trend) > 0.03:
        return Regime.LOW_VOL_TREND
    return Regime.MEAN_REVERT

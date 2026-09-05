"""Capital constraint instrumentation — behavior-preserving."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.python.data.costs import CostModel
from src.python.validation.economic_sim import simulate_long_only
from src.python.validation.capital_constraint_audit import summarize_fills, equity_identity_ok

def _bars(n=40, sym="BBCA"):
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = np.linspace(100, 120, n)
    return pd.DataFrame({"timestamp": idx, "symbol": sym, "open": px, "high": px+1, "low": px-1, "close": px, "volume": 1e6})

def test_fill_events_present_and_full_fill():
    bars = _bars(30)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    sim = simulate_long_only(bars, sig, cost=CostModel(0, 0), hold_bars=3)
    assert "fill_events" in sim
    s = summarize_fills(sim["fill_events"])
    assert s["FULL_FILL"] >= 1
    assert s["PARTIAL_FILL"] == 0

def test_existing_position_skip():
    bars = _bars(40)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    sim = simulate_long_only(bars, sig, cost=CostModel(0, 0), hold_bars=10)
    s = summarize_fills(sim["fill_events"])
    assert s["SKIPPED_EXISTING_POSITION"] >= 1

def test_cash_skip_with_tiny_capital():
    bars = _bars(20)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    sim = simulate_long_only(bars, sig, cost=CostModel(0, 0), hold_bars=2, initial_cash=500.0, lot_size=100.0)
    s = summarize_fills(sim["fill_events"])
    assert s["SKIPPED_CASH"] >= 1 or s["FULL_FILL"] == 0

def test_equity_identity():
    bars = _bars(25)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    sim = simulate_long_only(bars, sig, cost=CostModel(5, 5), hold_bars=3)
    r = equity_identity_ok(sim["equity_curve"])
    assert r["status"] == "PASS"

def test_window_reset_vs_combined_carry():
    bars = _bars(50)
    mid = bars["timestamp"].iloc[25]
    a = bars[bars.timestamp <= mid][["timestamp","symbol"]].assign(side=1)
    b = bars[bars.timestamp > mid][["timestamp","symbol"]].assign(side=1)
    wa = simulate_long_only(bars, a, cost=CostModel(10, 0), hold_bars=3)
    comb = simulate_long_only(bars, pd.concat([a,b], ignore_index=True), cost=CostModel(10, 0), hold_bars=3)
    assert wa["metrics"]["initial_cash"] == comb["metrics"]["initial_cash"]
    assert "signal_attrition" in comb["metrics"]

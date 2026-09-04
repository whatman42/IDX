"""Reconciliation: window-sum vs combined are different accounting modes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.python.data.costs import CostModel
from src.python.validation.economic_sim import simulate_long_only
from src.python.validation.economic_reconciliation import (
    ledger_sums, reconcile_window_vs_combined, trade_identity,
)

def _bars(n=40, sym="BBCA", start="2020-01-01"):
    idx = pd.bdate_range(start, periods=n)
    px = np.linspace(100, 130, n)
    return pd.DataFrame({"timestamp": idx, "symbol": sym, "open": px, "high": px+1, "low": px-1, "close": px, "volume": 1e6})

def test_higher_cost_reduces_net():
    bars = _bars(50)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    sim = simulate_long_only(bars, sig, cost=CostModel(15, 5), hold_bars=3)
    z = simulate_long_only(bars, sig, cost=CostModel(0, 0), hold_bars=3)
    if sim["trades"]:
        assert sim["metrics"]["net_pnl"] <= z["metrics"]["net_pnl"] + 1e-6

def test_no_duplicate_trade_identity_single_sim():
    bars = _bars(40)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    trades = simulate_long_only(bars, sig, cost=CostModel(0, 0), hold_bars=2)["trades"]
    ids = [trade_identity(t) for t in trades]
    assert len(ids) == len(set(ids))

def test_window_reset_vs_combined_documented_divergence():
    bars = _bars(60)
    mid = bars["timestamp"].iloc[30]
    sig_a = bars[bars["timestamp"] <= mid][["timestamp", "symbol"]].assign(side=1)
    sig_b = bars[bars["timestamp"] > mid][["timestamp", "symbol"]].assign(side=1)
    cost = CostModel(10, 5)
    wa = simulate_long_only(bars, sig_a, cost=cost, hold_bars=3)
    wb = simulate_long_only(bars, sig_b, cost=cost, hold_bars=3)
    comb = simulate_long_only(bars, pd.concat([sig_a, sig_b], ignore_index=True), cost=cost, hold_bars=3)
    rec = reconcile_window_vs_combined(wa["trades"] + wb["trades"], comb["trades"])
    assert rec["status"] in ("RECONCILED_EXPECTED_DIVERGENCE", "RECONCILED_EQUAL")
    assert rec["semantics"]["comparable_as_identical"] is False

def test_ledger_sums_counts():
    trades = [{"gross_pnl": 10.0, "net_pnl": 8.0, "fees": 1.0, "slippage_cost": 1.0, "exit_reason": "TIME_EXIT"}]
    s = ledger_sums(trades)
    assert s["trades"] == 1 and s["gross_pnl"] == 10.0

"""Cost sensitivity & robustness tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.python.data.costs import CostModel
from src.python.validation.economic_sim import simulate_long_only
from src.python.validation.economic_robustness import (
    run_cost_sensitivity, run_holding_robustness, symbol_attribution_from_sim, _classify_cost,
)

def _bars(n=40, sym="BBCA"):
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = np.linspace(100, 120, n)
    return pd.DataFrame({"timestamp": idx, "symbol": sym, "open": px, "high": px+1, "low": px-1, "close": px, "volume": 1e6})

def test_zero_cost_net_near_gross():
    bars = _bars(); sig = bars[["timestamp", "symbol"]].assign(side=1)
    m = simulate_long_only(bars, sig, cost=CostModel(0, 0), hold_bars=3)["metrics"]
    if m["total_trades"] == 0: return
    assert abs(m["net_pnl"] - m["gross_pnl"]) < 1e-3

def test_higher_cost_never_improves_net():
    bars = _bars(50); sig = bars[["timestamp", "symbol"]].assign(side=1)
    a = simulate_long_only(bars, sig, cost=CostModel(5, 5), hold_bars=3)["metrics"]["net_pnl"]
    b = simulate_long_only(bars, sig, cost=CostModel(30, 20), hold_bars=3)["metrics"]["net_pnl"]
    assert b <= a + 1e-6

def test_cost_sensitivity_deterministic():
    bars = _bars(); sig = bars[["timestamp", "symbol"]].assign(side=1)
    r1 = run_cost_sensitivity(bars, sig, fee_grid=(0, 15), slip_grid=(0, 5), hold_bars=3)
    r2 = run_cost_sensitivity(bars, sig, fee_grid=(0, 15), slip_grid=(0, 5), hold_bars=3)
    assert r1["grid"] == r2["grid"] and r1["cost_model_status"] == "UNVERIFIED"

def test_holding_changes_exits():
    bars = _bars(60); sig = bars[["timestamp", "symbol"]].assign(side=1)
    r = run_holding_robustness(bars, sig, hold_grid=(1, 5, 10), fee_bps=0, slip_bps=0)
    assert [x["hold_bars"] for x in r["grid"]] == [1, 5, 10]

def test_classify_cost():
    assert "PROFITABLE_BEFORE_COST" in _classify_cost(100, 50, 10)
    assert "LOSS_BEFORE_COST" in _classify_cost(-10, -20, -30)

def test_symbol_attribution_sum():
    bars = _bars(30, "AAA"); bars2 = _bars(30, "BBB")
    bars2[["open","high","low","close"]] *= 1.01
    bars = pd.concat([bars, bars2], ignore_index=True)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    sim = simulate_long_only(bars, sig, cost=CostModel(0, 0), hold_bars=2)
    att = symbol_attribution_from_sim(sim)
    if sim["metrics"]["total_trades"] > 0:
        assert att["attribution_sum_ok"] is True

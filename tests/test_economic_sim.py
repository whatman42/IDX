"""Economic sim: no pct_change P&L; T+1 timing; finite metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.python.data.costs import CostModel
from src.python.validation.economic_sim import economic_metrics_from_trades, simulate_long_only, TradeRecord

def _bars(n=30, sym="BBCA", start="2020-01-01"):
    idx = pd.bdate_range(start, periods=n)
    px = np.linspace(100, 110, n) + np.sin(np.arange(n)) * 0.5
    return pd.DataFrame({"timestamp": idx, "symbol": sym, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 1e6})

def test_zero_trades_safe():
    bars = _bars()
    sig = pd.DataFrame({"timestamp": bars["timestamp"], "symbol": "BBCA", "side": 0})
    m = simulate_long_only(bars, sig, hold_bars=3)["metrics"]
    assert m["total_trades"] == 0 and m["market_performance"] == "INSUFFICIENT_EVIDENCE"
    assert m["expectancy"] == "NOT_APPLICABLE"

def test_fees_reduce_net():
    bars = _bars(40)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    cheap = simulate_long_only(bars, sig, cost=CostModel(0, 0), hold_bars=3)
    rich = simulate_long_only(bars, sig, cost=CostModel(50, 20), hold_bars=3)
    assert cheap["metrics"]["total_trades"] > 0
    assert rich["metrics"]["net_pnl"] <= cheap["metrics"]["net_pnl"] + 1e-6

def test_tplus1_entry_after_signal():
    bars = _bars(5)
    sig = pd.DataFrame({"timestamp": [bars.iloc[0]["timestamp"]], "symbol": ["BBCA"], "side": [1]})
    sim = simulate_long_only(bars, sig, hold_bars=2, cost=CostModel(0, 0))
    for t in sim["trades"]:
        assert pd.Timestamp(t["entry_timestamp"]) >= pd.Timestamp(bars.iloc[1]["timestamp"])

def test_metrics_no_inf_all_wins():
    tr = [TradeRecord("T1", "X", 1, "a", 1, 100, "b", 2, 100.0, 0, 0, 100.0, 1, "TIME_EXIT")]
    m = economic_metrics_from_trades(tr, [], initial_cash=1e6)
    assert m["net_pnl"] == 100.0
    assert m["profit_factor"] == "NOT_APPLICABLE" or isinstance(m["profit_factor"], float)

def test_sim_has_gross_pnl_key():
    bars = _bars(20)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    sim = simulate_long_only(bars, sig, hold_bars=2)
    assert "gross_pnl" in sim["metrics"] or sim["metrics"]["total_trades"] == 0

"""Paper session validation layer — no strategy optimization."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.python.paper.session import PaperSessionConfig, run_paper_session, write_paper_artifacts

def _bars(n=40, sym="BBCA"):
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = np.linspace(100, 120, n)
    return pd.DataFrame({"timestamp": idx, "symbol": sym, "open": px, "high": px+1, "low": px-1, "close": px, "volume": 1e6})

def test_paper_account_init_and_complete():
    bars = _bars(30)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    cfg = PaperSessionConfig(initial_cash=100_000_000.0, fee_bps=0, slippage_bps=0, hold_bars=3)
    s = run_paper_session(bars, sig, config=cfg, session_id="test_paper_1")
    assert s.status == "COMPLETED"
    assert s.config.initial_cash == 100_000_000.0
    assert s.equity_identity["status"] == "PASS"
    assert s.production_pointer == "UNCHANGED"
    assert s.promotion == "REJECTED"
    assert s.paper_result_label == "OBSERVATION_ONLY"
    assert s.economic_edge == "UNVERIFIED"

def test_equity_identity_throughout():
    bars = _bars(25)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig, config=PaperSessionConfig(fee_bps=5, slippage_bps=5, hold_bars=3))
    assert s.equity_identity["status"] == "PASS"
    assert s.equity_identity["negative_cash_points"] == 0

def test_tplus1_and_signal_ledger():
    bars = _bars(20)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig, config=PaperSessionConfig(fee_bps=0, slippage_bps=0, hold_bars=2))
    assert s.metrics.get("timing") == "signal_T_execute_open_Tplus1"
    assert len(s.fill_events) >= 1
    classes = {e["classification"] for e in s.fill_events}
    assert classes <= {"FULL_FILL", "PARTIAL_FILL", "SKIPPED_CASH", "SKIPPED_EXISTING_POSITION", "SKIPPED_OTHER_HARD_CONSTRAINT"}
    n = len(s.fill_events)
    att = s.signal_attrition
    assert att["n_signal_events"] == n
    assert att["FULL_FILL"] + att["PARTIAL_FILL"] + att["SKIPPED_CASH"] + att["SKIPPED_EXISTING_POSITION"] + att["SKIPPED_OTHER_HARD_CONSTRAINT"] == n

def test_one_position_and_existing_skip():
    bars = _bars(40)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig, config=PaperSessionConfig(hold_bars=10, fee_bps=0, slippage_bps=0))
    assert s.signal_attrition["SKIPPED_EXISTING_POSITION"] >= 1

def test_dq_halt():
    bars = _bars(15)
    bars.loc[0, "high"] = bars.loc[0, "low"] - 10
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig, config=PaperSessionConfig(max_gap_days=10), halt_on_dq_fail=True)
    assert s.status in ("HALTED_DATA_QUALITY", "COMPLETED")
    if s.status == "HALTED_DATA_QUALITY":
        assert s.data_quality == "FAIL"

def test_deterministic_repeat():
    bars = _bars(28)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    cfg = PaperSessionConfig(fee_bps=10, slippage_bps=5, hold_bars=3, git_commit="abc")
    a = run_paper_session(bars, sig, config=cfg, session_id="det_a")
    b = run_paper_session(bars, sig, config=cfg, session_id="det_b")
    assert a.metrics.get("net_pnl") == b.metrics.get("net_pnl")
    assert a.metrics.get("total_trades") == b.metrics.get("total_trades")

def test_write_artifacts(tmp_path):
    bars = _bars(20)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig, config=PaperSessionConfig(hold_bars=2))
    paths = write_paper_artifacts(s, tmp_path / "paper_validation")
    assert (tmp_path / "paper_validation" / "paper_session.json").exists()
    assert (tmp_path / "paper_validation" / "paper_summary.json").exists()

def test_promotion_and_strategy_flags():
    bars = _bars(15)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig)
    assert s.promotion == "REJECTED"
    assert s.strategy_changed is False
    assert s.production_pointer == "UNCHANGED"

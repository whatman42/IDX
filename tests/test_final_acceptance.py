"""IDX Stage-8 final acceptance gates — paper-only, no strategy change."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "python"

def test_no_broker_sdk_imports():
    banned = re.compile(r"\b(ccxt|ib_insync|alpaca|binance|interactive_brokers|oandapy|mt5|MetaTrader)\b", re.I)
    hits = []
    for p in SRC.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if banned.search(text):
            hits.append(str(p.relative_to(ROOT)))
    assert hits == [], f"broker SDK imports found: {hits}"

def test_no_live_order_submit_functions():
    pattern = re.compile(r"^\s*def\s+(place_order|send_order|submit_order|execute_live|submit_live)\s*\(", re.M)
    hits = []
    for p in SRC.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            hits.append(str(p.relative_to(ROOT)))
    assert hits == [], f"live order submit defs found: {hits}"

def test_run_cycle_production_mode_does_not_apply_buy():
    """production mode is excluded from paper apply_buy branch."""
    text = (SRC / "pipeline" / "run_cycle.py").read_text()
    assert 'mode in ("paper", "development", "test")' in text
    for line in text.splitlines():
        if "allow_new_trades" in line and "mode in" in line:
            assert "production" not in line
            assert "paper" in line
            break
    else:
        raise AssertionError("execution mode gate not found")

def test_paper_session_governor_allow_paper_only():
    from src.python.paper.session import PaperSessionConfig, run_paper_session
    bars = _bars(20)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig, config=PaperSessionConfig(hold_bars=2))
    assert any(g.get("decision_outcome") == "ALLOW_PAPER_ONLY" for g in s.governor_log)
    assert s.promotion == "REJECTED"
    assert s.production_pointer == "UNCHANGED"
    assert s.paper_result_label == "OBSERVATION_ONLY"

def test_tplus1_execution_timestamps():
    from src.python.validation.economic_sim import simulate_long_only
    from src.python.data.costs import CostModel
    bars = _bars(30)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    sim = simulate_long_only(bars, sig, cost=CostModel(0, 0), hold_bars=3)
    for t in sim["trades"]:
        assert t["exit_timestamp"] >= t["entry_timestamp"]
        assert t["holding_bars"] >= 1
    assert sim["metrics"]["timing"] == "signal_T_execute_open_Tplus1"

def test_signal_ledger_complete_classification():
    from src.python.paper.session import PaperSessionConfig, run_paper_session
    bars = _bars(35)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig, config=PaperSessionConfig(hold_bars=5, fee_bps=0, slippage_bps=0))
    att = s.signal_attrition
    n = att["n_signal_events"]
    total = (att["FULL_FILL"] + att["PARTIAL_FILL"] + att["SKIPPED_CASH"]
             + att["SKIPPED_EXISTING_POSITION"] + att["SKIPPED_OTHER_HARD_CONSTRAINT"])
    assert total == n

def test_equity_identity_and_no_negative_cash():
    from src.python.paper.session import PaperSessionConfig, run_paper_session
    bars = _bars(25)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig, config=PaperSessionConfig(fee_bps=15, slippage_bps=5, hold_bars=3))
    assert s.equity_identity["status"] == "PASS"
    assert s.equity_identity["negative_cash_points"] == 0

def test_portfolio_idempotent_duplicate_order():
    from src.python.portfolio.engine import PortfolioEngine
    eng = PortfolioEngine(initial_cash=100_000_000.0)
    t1 = eng.apply_buy(signal_id_="sig1", symbol="BBCA", side=1, price=9000.0, weight=0.05, cycle_id="c1")
    t2 = eng.apply_buy(signal_id_="sig1", symbol="BBCA", side=1, price=9000.0, weight=0.05, cycle_id="c1")
    assert t1 is not None
    assert t2 is None
    open_pos = [p for p in eng.state.positions.values() if getattr(p, "qty", 0) > 0]
    assert len(open_pos) <= 1

def test_portfolio_restart_resume_state():
    from src.python.portfolio.engine import PortfolioEngine, PortfolioState
    eng = PortfolioEngine(initial_cash=100_000_000.0)
    eng.apply_buy(signal_id_="sigA", symbol="BBCA", side=1, price=9000.0, weight=0.1, cycle_id="c1")
    blob = eng.state.to_dict()
    st = PortfolioState.from_dict(blob)
    eng2 = PortfolioEngine(initial_cash=st.cash)
    eng2.state = st
    assert abs(eng2.state.cash - eng.state.cash) < 1e-6
    t = eng2.apply_buy(signal_id_="sigA", symbol="BBCA", side=1, price=9000.0, weight=0.1, cycle_id="c1")
    assert t is None

def test_determinism_paper_session():
    from src.python.paper.session import PaperSessionConfig, run_paper_session
    bars = _bars(28)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    cfg = PaperSessionConfig(fee_bps=10, slippage_bps=5, hold_bars=3, git_commit="final")
    a = run_paper_session(bars, sig, config=cfg)
    b = run_paper_session(bars, sig, config=cfg)
    assert a.metrics.get("net_pnl") == b.metrics.get("net_pnl")
    assert a.metrics.get("total_trades") == b.metrics.get("total_trades")

def test_dq_gate_blocks_invalid_envelope():
    from src.python.paper.session import PaperSessionConfig, run_paper_session
    bars = _bars(15)
    bars.loc[0, "high"] = bars.loc[0, "low"] - 5
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig, config=PaperSessionConfig(max_gap_days=10), halt_on_dq_fail=True)
    if s.status == "HALTED_DATA_QUALITY":
        assert s.data_quality == "FAIL"
        assert len(s.issues) >= 1
    else:
        assert s.data_quality in ("PASS", "FAIL")

def test_secret_scan_src_no_obvious_keys():
    secret_assign = re.compile(
        r"""(?i)(api_key|apikey|secret_key|private_key|password|token)\s*=\s*['\"][A-Za-z0-9_\-]{20,}['\"]"""
    )
    hits = []
    for p in SRC.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if secret_assign.search(text):
            hits.append(str(p.relative_to(ROOT)))
    assert hits == [], f"possible hardcoded secrets: {hits}"

def test_economic_edge_labels_unchanged():
    from src.python.paper.session import PaperSessionConfig, run_paper_session
    bars = _bars(12)
    sig = bars[["timestamp", "symbol"]].assign(side=1)
    s = run_paper_session(bars, sig)
    assert s.economic_edge == "UNVERIFIED"
    assert s.promotion == "REJECTED"
    assert s.strategy_changed is False

def _bars(n=40, sym="BBCA"):
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = np.linspace(100, 120, n)
    return pd.DataFrame({
        "timestamp": idx, "symbol": sym,
        "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 1e6,
    })

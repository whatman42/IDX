"""Deterministic paper session runner.

SIGNAL-ONLY + PAPER/SIMULATION. No live broker. No production pointer mutation.
Uses existing economic_sim (T+1 open, long-only, one position per symbol).
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from src.python.data.costs import CostModel
from src.python.data.quality import validate_ohlcv
from src.python.validation.economic_sim import simulate_long_only
from src.python.validation.capital_constraint_audit import summarize_fills, equity_identity_ok

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _config_hash(cfg: dict) -> str:
    return _sha256_bytes(json.dumps(cfg, sort_keys=True, default=str).encode())

@dataclass
class PaperSessionConfig:
    initial_cash: float = 100_000_000.0
    currency: str = "IDR"
    fee_bps: float = 15.0
    slippage_bps: float = 5.0
    hold_bars: int = 5
    lot_size: float = 100.0
    max_position_pct: float = 0.10
    max_gap_days: float = 45.0
    cost_model_status: str = "UNVERIFIED_ASSUMPTION"
    git_commit: str = ""
    model_version: str = "paper_primary"
    dataset_hash: str = ""
    universe: list = field(default_factory=list)
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class PaperSession:
    session_id: str
    config: PaperSessionConfig
    start_timestamp: str = ""
    end_timestamp: str = ""
    status: str = "INIT"
    data_quality: str = "UNVERIFIED"
    issues: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    signal_attrition: dict = field(default_factory=dict)
    capital_utilization: dict = field(default_factory=dict)
    equity_identity: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    fill_events: list = field(default_factory=list)
    daily_snapshots: list = field(default_factory=list)
    governor_log: list = field(default_factory=list)
    paper_result_label: str = "OBSERVATION_ONLY"
    economic_edge: str = "UNVERIFIED"
    production_pointer: str = "UNCHANGED"
    promotion: str = "REJECTED"
    strategy_changed: bool = False
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id, "config": self.config.to_dict(),
            "start_timestamp": self.start_timestamp, "end_timestamp": self.end_timestamp,
            "status": self.status, "data_quality": self.data_quality, "issues": self.issues,
            "metrics": self.metrics, "signal_attrition": self.signal_attrition,
            "capital_utilization": self.capital_utilization, "equity_identity": self.equity_identity,
            "paper_result_label": self.paper_result_label, "economic_edge": self.economic_edge,
            "production_pointer": self.production_pointer, "promotion": self.promotion,
            "strategy_changed": self.strategy_changed, "trade_count": len(self.trades),
            "fill_event_count": len(self.fill_events), "daily_snapshot_count": len(self.daily_snapshots),
            "governor_log": self.governor_log,
        }

def run_paper_session(bars: pd.DataFrame, signals: pd.DataFrame, *, config: Optional[PaperSessionConfig] = None,
    session_id: Optional[str] = None, halt_on_dq_fail: bool = True) -> PaperSession:
    """Run deterministic paper session. Signal T → execute open T+1. No live execution."""
    cfg = config or PaperSessionConfig()
    cfg_hash = _config_hash(cfg.to_dict())
    sid = session_id or f"paper_{_sha256_bytes(cfg_hash.encode())[:12]}"
    session = PaperSession(session_id=sid, config=cfg)
    session.governor_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_mode": "PAPER_SIMULATION", "reason": "paper_validation_layer",
        "policy_version": "v1", "decision_outcome": "ALLOW_PAPER_ONLY",
    })
    q = validate_ohlcv(bars, max_gap_days=cfg.max_gap_days)
    session.data_quality = "PASS" if q.ok else "FAIL"
    if not q.ok:
        session.issues.extend(q.issues)
        session.status = "HALTED_DATA_QUALITY"
        if halt_on_dq_fail:
            return session
    cost = CostModel(fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps)
    sim = simulate_long_only(bars, signals, cost=cost, initial_cash=cfg.initial_cash,
        hold_bars=cfg.hold_bars, lot_size=cfg.lot_size, max_position_pct=cfg.max_position_pct,
        cost_model_status=cfg.cost_model_status)
    session.trades = sim.get("trades") or []
    session.fill_events = sim.get("fill_events") or []
    session.metrics = sim.get("metrics") or {}
    session.signal_attrition = summarize_fills(session.fill_events)
    session.capital_utilization = session.metrics.get("capital_utilization") or {}
    eq_pts = sim.get("equity_curve") or []
    session.equity_identity = equity_identity_ok(eq_pts)
    session.daily_snapshots = [{
        "date": e.get("timestamp"), "cash": e.get("cash"), "equity": e.get("equity"),
        "market_value": e.get("market_value"), "realized_pnl": e.get("realized_pnl"),
        "fees_cum": e.get("fees_cum"),
    } for e in eq_pts]
    if session.daily_snapshots:
        session.start_timestamp = str(session.daily_snapshots[0]["date"])
        session.end_timestamp = str(session.daily_snapshots[-1]["date"])
    session.metrics["config_hash"] = cfg_hash
    session.metrics["dataset_hash"] = cfg.dataset_hash
    session.metrics["git_commit"] = cfg.git_commit
    session.metrics["model_version"] = cfg.model_version
    session.metrics["cost_model_status"] = cfg.cost_model_status
    session.metrics["timing"] = "signal_T_execute_open_Tplus1"
    session.metrics["paper_result_label"] = "OBSERVATION_ONLY"
    session.metrics["economic_edge"] = "UNVERIFIED"
    if session.equity_identity.get("status") != "PASS":
        session.status = "HALTED_RECONCILIATION"
        session.issues.append("equity_identity_fail")
    else:
        session.status = "COMPLETED"
    return session

def write_paper_artifacts(session: PaperSession, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    p_session = out / "paper_session.json"
    p_session.write_text(json.dumps(session.to_dict(), indent=2, default=str))
    paths["paper_session"] = str(p_session)
    p_sig = out / "signal_ledger.jsonl"
    with p_sig.open("w") as f:
        for e in session.fill_events:
            f.write(json.dumps(e, default=str) + "\n")
    paths["signal_ledger"] = str(p_sig)
    p_tr = out / "trade_ledger.jsonl"
    with p_tr.open("w") as f:
        for t in session.trades:
            f.write(json.dumps(t, default=str) + "\n")
    paths["trade_ledger"] = str(p_tr)
    p_eq = out / "daily_equity.jsonl"
    with p_eq.open("w") as f:
        for d in session.daily_snapshots:
            f.write(json.dumps(d, default=str) + "\n")
    paths["daily_equity"] = str(p_eq)
    summary = {
        "session_id": session.session_id, "status": session.status, "data_quality": session.data_quality,
        "initial_cash": session.config.initial_cash, "final_equity": session.metrics.get("final_equity"),
        "total_return": session.metrics.get("total_return"), "gross_pnl": session.metrics.get("gross_pnl"),
        "net_pnl": session.metrics.get("net_pnl"), "total_fees": session.metrics.get("total_fees"),
        "total_slippage_cost": session.metrics.get("total_slippage_cost"),
        "trade_count": session.metrics.get("total_trades"), "win_rate": session.metrics.get("win_rate"),
        "expectancy": session.metrics.get("expectancy"), "profit_factor": session.metrics.get("profit_factor"),
        "max_drawdown": session.metrics.get("max_drawdown"), "signal_attrition": session.signal_attrition,
        "capital_utilization": session.capital_utilization, "equity_identity": session.equity_identity,
        "paper_result_label": "OBSERVATION_ONLY", "economic_edge": "UNVERIFIED",
        "production_pointer": "UNCHANGED", "promotion": "REJECTED", "strategy_changed": False,
        "cost_model_status": session.config.cost_model_status, "timing": "signal_T_execute_open_Tplus1",
        "git_commit": session.config.git_commit, "dataset_hash": session.config.dataset_hash,
        "config_hash": session.metrics.get("config_hash"),
    }
    p_sum = out / "paper_summary.json"
    p_sum.write_text(json.dumps(summary, indent=2, default=str))
    paths["paper_summary"] = str(p_sum)
    return paths

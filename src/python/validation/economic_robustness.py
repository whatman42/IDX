"""Cost sensitivity + holding-period robustness for economic OOS.

Does not claim verified IDX broker costs. All cost runs: cost_model_status=UNVERIFIED.
Uses existing simulate_long_only (T+1, trade ledger). No feature pct_change P&L.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd
from src.python.data.costs import CostModel
from src.python.validation.economic_sim import _finite, simulate_long_only

FEE_GRID = (0, 5, 10, 15, 20, 30, 50)
SLIP_GRID = (0, 5, 10, 15, 20, 30)
HOLD_GRID = (1, 3, 5, 10)
STRATEGY_SCOPE = "LONG_ONLY"

def _classify_cost(gross: float, net_base: float, net_high: float) -> str:
    labels = []
    if gross > 1e-6: labels.append("PROFITABLE_BEFORE_COST")
    elif gross < -1e-6: labels.append("LOSS_BEFORE_COST")
    else: labels.append("BREAK_EVEN_BEFORE_COST")
    if gross > 1e-6 and net_high < 0: labels.append("COST_SENSITIVE")
    elif gross > 1e-6 and net_base > 0 and net_high > 0: labels.append("ROBUST_TO_COST")
    elif gross <= 0: labels.append("COST_SENSITIVE")
    return "+".join(labels)

def run_cost_sensitivity(bars, signals, *, fee_grid=FEE_GRID, slip_grid=SLIP_GRID, hold_bars=5,
    initial_cash=100_000_000.0, provenance=None) -> dict[str, Any]:
    rows = []; zero = None
    for fee in fee_grid:
        for slip in slip_grid:
            if fee + slip > 80: continue
            sim = simulate_long_only(bars, signals, cost=CostModel(float(fee), float(slip)),
                hold_bars=hold_bars, initial_cash=initial_cash, cost_model_status="UNVERIFIED")
            m = sim["metrics"]
            row = {"fee_bps": fee, "slippage_bps": slip, "total_cost_bps": fee + slip,
                "trades": m.get("total_trades"), "gross_pnl": m.get("gross_pnl"), "net_pnl": m.get("net_pnl"),
                "expectancy": m.get("expectancy") if isinstance(m.get("expectancy"), (int, float)) else None,
                "profit_factor": m.get("profit_factor") if isinstance(m.get("profit_factor"), (int, float)) else None,
                "win_rate": m.get("win_rate") if isinstance(m.get("win_rate"), (int, float)) else None,
                "max_drawdown": m.get("max_drawdown") if isinstance(m.get("max_drawdown"), (int, float)) else None,
                "total_return": m.get("total_return") if isinstance(m.get("total_return"), (int, float)) else None,
                "cost_model_status": "UNVERIFIED"}
            rows.append(row)
            if fee == 0 and slip == 0: zero = row
    base = next((r for r in rows if r["fee_bps"] == 15 and r["slippage_bps"] == 5), rows[0] if rows else {})
    high = next((r for r in rows if r["fee_bps"] == 30 and r["slippage_bps"] == 20), rows[-1] if rows else {})
    gross0 = float(zero["gross_pnl"]) if zero and zero.get("gross_pnl") is not None else 0.0
    net_b = float(base.get("net_pnl") or 0); net_h = float(high.get("net_pnl") or 0)
    cost_abs = gross0 - net_b if zero else None
    cost_pct = _finite(cost_abs / abs(gross0)) if zero and abs(gross0) > 1e-6 and cost_abs is not None else None
    return {"methodology": "simulate_long_only T+1 open; grid fee\u00d7slip; UNVERIFIED costs",
        "strategy_scope": STRATEGY_SCOPE, "hold_bars": hold_bars, "provenance": provenance or {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cost_model_status": "UNVERIFIED",
        "grid": rows, "diagnostic": {"gross_without_cost": gross0, "net_base_15_5": net_b, "net_high_30_20": net_h,
            "cost_impact_absolute": cost_abs, "cost_impact_pct_of_gross": cost_pct,
            "classification": _classify_cost(gross0, net_b, net_h)}}

def run_holding_robustness(bars, signals, *, hold_grid=HOLD_GRID, fee_bps=15.0, slip_bps=5.0,
    initial_cash=100_000_000.0, provenance=None) -> dict[str, Any]:
    rows = []
    for h in hold_grid:
        sim = simulate_long_only(bars, signals, cost=CostModel(fee_bps, slip_bps), hold_bars=int(h), initial_cash=initial_cash)
        m = sim["metrics"]
        rows.append({"hold_bars": int(h), "trades": m.get("total_trades"), "gross_pnl": m.get("gross_pnl"),
            "net_pnl": m.get("net_pnl"),
            "expectancy": m.get("expectancy") if isinstance(m.get("expectancy"), (int, float)) else None,
            "max_drawdown": m.get("max_drawdown") if isinstance(m.get("max_drawdown"), (int, float)) else None,
            "total_return": m.get("total_return") if isinstance(m.get("total_return"), (int, float)) else None,
            "average_holding_bars": m.get("average_holding_bars"), "cost_model_status": "UNVERIFIED"})
    return {"methodology": "same signals; vary hold_bars only; T+1 preserved", "strategy_scope": STRATEGY_SCOPE,
        "fee_bps": fee_bps, "slippage_bps": slip_bps, "provenance": provenance or {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "grid": rows}

def symbol_attribution_from_sim(sim: dict[str, Any]) -> dict[str, Any]:
    m = sim.get("metrics") or {}; attr = m.get("symbol_attribution") or {}
    total_net = m.get("net_pnl") if isinstance(m.get("net_pnl"), (int, float)) else 0.0
    rows = []; sum_net = 0.0
    for sym, v in sorted(attr.items()):
        npnl = float(v.get("net_pnl") or 0); sum_net += npnl
        rows.append({"symbol": sym, "trades": v.get("trades"), "net_pnl": npnl, "expectancy": v.get("expectancy"),
            "contribution_pct": _finite(npnl / abs(total_net)) if abs(total_net) > 1e-6 else None})
    return {"symbols": rows, "sum_symbol_net_pnl": _finite(sum_net), "reported_total_net_pnl": total_net,
        "attribution_sum_ok": abs(sum_net - float(total_net or 0)) < 1.0, "strategy_scope": STRATEGY_SCOPE}

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))

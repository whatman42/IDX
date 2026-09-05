"""Economic OOS simulation — trades/P&L from prices, not feature pct_change.

Timing (leakage-safe):
  signal decision uses information available at bar T close
  execution at next bar open (T+1) with configurable slippage
  exit after hold_bars at open (or last bar END_OF_TEST)

cost_model_status is always ASSUMED/UNVERIFIED unless caller proves otherwise.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Optional
import numpy as np
import pandas as pd
from src.python.data.costs import CostModel

@dataclass
class TradeRecord:
    trade_id: str; symbol: str; side: int
    entry_timestamp: str; entry_price: float; quantity: float
    exit_timestamp: str; exit_price: float
    gross_pnl: float; fees: float; slippage_cost: float; net_pnl: float
    holding_bars: int; exit_reason: str
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class EquityPoint:
    timestamp: str; cash: float; market_value: float; equity: float
    realized_pnl: float; fees_cum: float
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _finite(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v or abs(v) == float("inf"):
            return default
        return v
    except Exception:
        return default

def economic_metrics_from_trades(trades, equity, *, initial_cash: float) -> dict[str, Any]:
    out: dict[str, Any] = {"total_trades": len(trades), "cost_model_status": "UNVERIFIED", "market_performance": "UNVERIFIED"}
    if not trades:
        out.update({"winning_trades": 0, "losing_trades": 0, "win_rate": "NOT_APPLICABLE", "expectancy": "NOT_APPLICABLE",
                    "profit_factor": "NOT_APPLICABLE", "gross_pnl": 0.0, "net_pnl": 0.0, "max_drawdown": "NOT_APPLICABLE",
                    "market_performance": "INSUFFICIENT_EVIDENCE"})
        return out
    nets = np.array([t.net_pnl for t in trades], dtype=float)
    grosses = np.array([t.gross_pnl for t in trades], dtype=float)
    wins, losses = nets[nets > 0], nets[nets < 0]
    out["winning_trades"] = int(len(wins)); out["losing_trades"] = int(len(losses))
    out["win_rate"] = _finite(len(wins) / len(nets))
    out["average_win"] = _finite(wins.mean()) if len(wins) else 0.0
    out["average_loss"] = _finite(losses.mean()) if len(losses) else 0.0
    out["expectancy"] = _finite(nets.mean())
    sum_w, sum_l = float(wins.sum()) if len(wins) else 0.0, float(-losses.sum()) if len(losses) else 0.0
    if sum_l > 1e-12:
        out["profit_factor"] = _finite(sum_w / sum_l)
    elif sum_w > 0:
        out["profit_factor"] = "NOT_APPLICABLE"
    else:
        out["profit_factor"] = 0.0
    out["gross_pnl"] = _finite(grosses.sum()); out["net_pnl"] = _finite(nets.sum())
    out["total_fees"] = _finite(sum(t.fees for t in trades))
    out["total_slippage_cost"] = _finite(sum(t.slippage_cost for t in trades))
    out["average_holding_bars"] = _finite(np.mean([t.holding_bars for t in trades]))
    out["worst_trade"] = _finite(nets.min()); out["best_trade"] = _finite(nets.max())
    max_cl = cl = 0
    for n in nets:
        if n < 0: cl += 1; max_cl = max(max_cl, cl)
        else: cl = 0
    out["max_consecutive_losses"] = int(max_cl)
    if equity and len(equity) >= 2:
        eq = np.array([e.equity for e in equity], dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-12)
        out["max_drawdown"] = _finite(float(dd.min()))
        out["final_equity"] = _finite(eq[-1])
        out["total_return"] = _finite((eq[-1] - initial_cash) / initial_cash) if initial_cash else 0.0
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        rets = rets[np.isfinite(rets)]
        if len(rets) >= 5 and float(np.std(rets)) > 1e-12:
            out["volatility"] = _finite(float(np.std(rets)))
            out["sharpe_like"] = _finite(float(np.mean(rets) / np.std(rets) * np.sqrt(252)))
        else:
            out["volatility"] = "INSUFFICIENT_DATA"; out["sharpe_like"] = "INSUFFICIENT_DATA"
    else:
        out["max_drawdown"] = "INSUFFICIENT_DATA"; out["final_equity"] = initial_cash
    out["statistical_evidence"] = "INSUFFICIENT" if len(trades) < 30 else "PRELIMINARY"
    return out

def simulate_long_only(bars: pd.DataFrame, signals: pd.DataFrame, *, cost: Optional[CostModel] = None,
    initial_cash: float = 100_000_000.0, hold_bars: int = 5, lot_size: float = 100.0,
    max_position_pct: float = 0.10, cost_model_status: str = "UNVERIFIED") -> dict[str, Any]:
    """Long-only sim with fill instrumentation (sizing behavior unchanged).

    Fill classes when long signal at T for T+1 open execution:
      FULL_FILL | SKIPPED_EXISTING_POSITION | SKIPPED_CASH | SKIPPED_OTHER_HARD_CONSTRAINT
    Partial fills not produced (all-or-nothing lot grid).
    """
    cost = cost or CostModel()
    fee_frac, slip_frac = cost.fee_bps / 10_000.0, cost.slippage_bps / 10_000.0
    bars = bars.copy(); bars["timestamp"] = pd.to_datetime(bars["timestamp"]); bars["symbol"] = bars["symbol"].astype(str)
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    sig = signals.copy(); sig["timestamp"] = pd.to_datetime(sig["timestamp"]); sig["symbol"] = sig["symbol"].astype(str)
    sig["side"] = sig["side"].astype(int)
    by_sym = {s: g.reset_index(drop=True) for s, g in bars.groupby("symbol")}
    sig_map = {(r.symbol, pd.Timestamp(r.timestamp)): int(r.side) for r in sig.itertuples(index=False)}
    cash, fees_cum, realized = float(initial_cash), 0.0, 0.0
    trades: list = []; equity_curve: list = []; open_pos: dict = {}; trade_i = 0
    fill_events: list = []
    all_dates = sorted(bars["timestamp"].unique())
    for dt in all_dates:
        for sym in list(open_pos.keys()):
            pos = open_pos[sym]; g = by_sym.get(sym)
            if g is None: continue
            rows = g[g["timestamp"] == dt]
            if rows.empty: continue
            open_px = float(rows.iloc[0]["open"]); bars_held = pos["bars_held"] + 1; pos["bars_held"] = bars_held
            if bars_held >= hold_bars or dt == all_dates[-1]:
                exit_px = open_px * (1.0 - slip_frac); fee = exit_px * pos["qty"] * fee_frac
                gross = (exit_px - pos["entry_price"]) * pos["qty"]
                net = gross - fee - pos["entry_fee"] - pos["entry_slip_cost"]
                cash += exit_px * pos["qty"] - fee; fees_cum += fee; realized += net
                reason = "TIME_EXIT" if bars_held >= hold_bars else "END_OF_TEST"
                trade_i += 1
                trades.append(TradeRecord(f"T{trade_i:05d}", sym, 1, str(pos["entry_ts"]), _finite(pos["entry_price"]),
                    _finite(pos["qty"]), str(dt), _finite(exit_px), _finite(gross),
                    _finite(pos["entry_fee"] + fee), _finite(pos["entry_slip_cost"] + open_px * pos["qty"] * slip_frac),
                    _finite(net), int(bars_held), reason))
                del open_pos[sym]
        for sym, g in by_sym.items():
            rows = g[g["timestamp"] == dt]
            if rows.empty: continue
            idx = g.index[g["timestamp"] == dt]
            if len(idx) == 0: continue
            i = int(idx[0])
            if i == 0: continue
            prev = g.iloc[i - 1]
            if sig_map.get((sym, pd.Timestamp(prev["timestamp"])), 0) != 1: continue
            cash_before = cash
            open_px = float(rows.iloc[0]["open"]); entry_px = open_px * (1.0 + slip_frac)
            budget = cash * max_position_pct
            requested_qty = float(int((budget / entry_px if entry_px > 0 else 0) / lot_size) * lot_size)
            requested_notional = requested_qty * entry_px if requested_qty > 0 else 0.0
            if sym in open_pos:
                fill_events.append({"signal_timestamp": str(prev["timestamp"]), "exec_timestamp": str(dt), "symbol": sym,
                    "classification": "SKIPPED_EXISTING_POSITION", "cash_before": _finite(cash_before), "cash_after": _finite(cash),
                    "requested_qty": _finite(requested_qty), "actual_qty": 0.0, "requested_notional": _finite(requested_notional),
                    "actual_notional": 0.0, "existing_position": True, "reason": "symbol_already_open"})
                continue
            qty = requested_qty
            if qty < lot_size:
                fill_events.append({"signal_timestamp": str(prev["timestamp"]), "exec_timestamp": str(dt), "symbol": sym,
                    "classification": "SKIPPED_CASH", "cash_before": _finite(cash_before), "cash_after": _finite(cash),
                    "requested_qty": _finite(requested_qty), "actual_qty": 0.0, "requested_notional": _finite(requested_notional),
                    "actual_notional": 0.0, "existing_position": False, "reason": "qty_below_lot_after_budget"})
                continue
            notional = qty * entry_px; fee = notional * fee_frac
            if notional + fee > cash:
                fill_events.append({"signal_timestamp": str(prev["timestamp"]), "exec_timestamp": str(dt), "symbol": sym,
                    "classification": "SKIPPED_CASH", "cash_before": _finite(cash_before), "cash_after": _finite(cash),
                    "requested_qty": _finite(requested_qty), "actual_qty": 0.0, "requested_notional": _finite(requested_notional),
                    "actual_notional": 0.0, "existing_position": False, "reason": "notional_plus_fee_exceeds_cash"})
                continue
            cash -= notional + fee; fees_cum += fee
            open_pos[sym] = {"entry_ts": dt, "entry_price": entry_px, "qty": qty, "bars_held": 0,
                             "entry_fee": fee, "entry_slip_cost": open_px * qty * slip_frac}
            fill_events.append({"signal_timestamp": str(prev["timestamp"]), "exec_timestamp": str(dt), "symbol": sym,
                "classification": "FULL_FILL", "cash_before": _finite(cash_before), "cash_after": _finite(cash),
                "requested_qty": _finite(requested_qty), "actual_qty": _finite(qty), "requested_notional": _finite(requested_notional),
                "actual_notional": _finite(notional), "existing_position": False, "reason": "filled_at_lot_grid"})
        mv = 0.0
        for sym, pos in open_pos.items():
            rows = by_sym[sym][by_sym[sym]["timestamp"] == dt]
            if not rows.empty: mv += float(rows.iloc[0]["close"]) * pos["qty"]
        equity_curve.append(EquityPoint(str(dt), _finite(cash), _finite(mv), _finite(cash + mv), _finite(realized), _finite(fees_cum)))
    metrics = economic_metrics_from_trades(trades, equity_curve, initial_cash=initial_cash)
    metrics["cost_model_status"] = cost_model_status
    metrics["timing"] = "signal_T_execute_open_Tplus1"
    metrics["hold_bars"] = hold_bars; metrics["initial_cash"] = initial_cash
    metrics["fee_bps"] = cost.fee_bps; metrics["slippage_bps"] = cost.slippage_bps
    by_s: dict = {}
    for t in trades: by_s.setdefault(t.symbol, []).append(t.net_pnl)
    metrics["symbol_attribution"] = {s: {"trades": len(v), "net_pnl": _finite(sum(v)), "expectancy": _finite(float(np.mean(v)))} for s, v in by_s.items()}
    if equity_curve:
        cashs = np.array([e.cash for e in equity_curve], dtype=float)
        mvs = np.array([e.market_value for e in equity_curve], dtype=float)
        eqs = np.array([e.equity for e in equity_curve], dtype=float)
        deployed = mvs
        metrics["capital_utilization"] = {
            "initial_cash": _finite(initial_cash),
            "peak_deployed": _finite(float(deployed.max()) if len(deployed) else 0),
            "average_deployed": _finite(float(deployed.mean()) if len(deployed) else 0),
            "average_cash": _finite(float(cashs.mean())),
            "minimum_cash": _finite(float(cashs.min())),
            "max_exposure_pct": _finite(float((deployed / np.maximum(eqs, 1e-12)).max())) if len(eqs) else 0.0,
            "average_exposure_pct": _finite(float((deployed / np.maximum(eqs, 1e-12)).mean())) if len(eqs) else 0.0,
            "final_equity": _finite(float(eqs[-1])),
            "negative_cash_points": int((cashs < -1e-6).sum()),
        }
    n_sig = len(fill_events)
    def _cnt(c): return sum(1 for e in fill_events if e["classification"] == c)
    metrics["signal_attrition"] = {
        "raw_long_signals_seen": n_sig,
        "FULL_FILL": _cnt("FULL_FILL"), "PARTIAL_FILL": _cnt("PARTIAL_FILL"),
        "SKIPPED_CASH": _cnt("SKIPPED_CASH"), "SKIPPED_EXISTING_POSITION": _cnt("SKIPPED_EXISTING_POSITION"),
        "SKIPPED_OTHER_HARD_CONSTRAINT": _cnt("SKIPPED_OTHER_HARD_CONSTRAINT"),
        "pct_full_fill": _finite(_cnt("FULL_FILL") / n_sig) if n_sig else None,
        "pct_skipped_cash": _finite(_cnt("SKIPPED_CASH") / n_sig) if n_sig else None,
        "pct_skipped_existing": _finite(_cnt("SKIPPED_EXISTING_POSITION") / n_sig) if n_sig else None,
    }
    return {"trades": [t.to_dict() for t in trades], "equity_curve": [e.to_dict() for e in equity_curve[-500:]],
            "equity_curve_len": len(equity_curve), "metrics": metrics, "fill_events": fill_events}

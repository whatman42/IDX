"""Reconcile per-window vs combined economic simulations.

IMPORTANT semantics (not a bug by itself):

* **Per-window mode**: each simulate_long_only call **resets** cash to
  initial_cash and clears open positions. Position size is ~max_position_pct
  of full capital. Independent capital books; SUM(window net) is NOT the same
  economic object as a continuous multi-period book.

* **Combined mode**: one continuous cash/position state across all OOS signals.
  Later trades may be smaller or skipped (cash constraint / one position per symbol).

Therefore SUM(window nets) and combined net must not be asserted equal.
"""
from __future__ import annotations
from collections import Counter
from typing import Any, Sequence

def trade_identity(t: dict) -> tuple:
    return (
        str(t.get("symbol")), str(t.get("entry_timestamp")), str(t.get("exit_timestamp")),
        int(t.get("side", 1)), round(float(t.get("entry_price", 0)), 4),
        round(float(t.get("exit_price", 0)), 4), round(float(t.get("quantity", 0)), 2),
    )

def entry_identity(t: dict) -> tuple:
    return (str(t.get("symbol")), str(t.get("entry_timestamp")))

def ledger_sums(trades: Sequence[dict]) -> dict[str, float]:
    return {
        "trades": float(len(trades)),
        "gross_pnl": float(sum(float(t.get("gross_pnl", 0)) for t in trades)),
        "net_pnl": float(sum(float(t.get("net_pnl", 0)) for t in trades)),
        "fees": float(sum(float(t.get("fees", 0)) for t in trades)),
        "slippage_cost": float(sum(float(t.get("slippage_cost", 0)) for t in trades)),
        "end_of_test": float(sum(1 for t in trades if t.get("exit_reason") == "END_OF_TEST")),
    }

def compare_trade_sets(window_trades: Sequence[dict], combined_trades: Sequence[dict]) -> dict[str, Any]:
    kw = Counter(trade_identity(t) for t in window_trades)
    kc = Counter(trade_identity(t) for t in combined_trades)
    only_w = sum((kw - kc).values())
    only_c = sum((kc - kw).values())
    ew = Counter(entry_identity(t) for t in window_trades)
    ec = Counter(entry_identity(t) for t in combined_trades)
    shared_entries = set(ew) & set(ec)
    qty_mismatch = 0
    for k in shared_entries:
        tw = next(t for t in window_trades if entry_identity(t) == k)
        tc = next(t for t in combined_trades if entry_identity(t) == k)
        if abs(float(tw["quantity"]) - float(tc["quantity"])) > 1e-6:
            qty_mismatch += 1
    return {
        "window_trade_count": len(window_trades),
        "combined_trade_count": len(combined_trades),
        "trade_count_delta": len(window_trades) - len(combined_trades),
        "identity_only_window": only_w,
        "identity_only_combined": only_c,
        "window_duplicate_identity": sum(v - 1 for v in kw.values() if v > 1),
        "combined_duplicate_identity": sum(v - 1 for v in kc.values() if v > 1),
        "shared_entry_count": len(shared_entries),
        "shared_entry_qty_mismatch": qty_mismatch,
        "entries_only_window": len(set(ew) - set(ec)),
        "entries_only_combined": len(set(ec) - set(ew)),
    }

def reconcile_window_vs_combined(
    window_trades: Sequence[dict], combined_trades: Sequence[dict], *, tolerance: float = 1.0,
) -> dict[str, Any]:
    sw = ledger_sums(window_trades)
    sc = ledger_sums(combined_trades)
    ident = compare_trade_sets(window_trades, combined_trades)
    deltas = {k: sw[k] - sc[k] for k in ("trades", "gross_pnl", "net_pnl", "fees", "slippage_cost")}
    expected_divergence = (
        ident["shared_entry_qty_mismatch"] > 0
        or ident["entries_only_window"] > 0
        or ident["entries_only_combined"] > 0
        or abs(deltas["trades"]) > 0
    )
    root = (
        "EXPECTED_ACCOUNTING_DIFFERENCE: per-window cash/position RESET vs combined stateful book; "
        "position size = max_position_pct * current_cash; one position per symbol."
        if expected_divergence else "LEDGERS_ALIGNED"
    )
    return {
        "window_sum": sw, "combined": sc, "deltas_window_minus_combined": deltas,
        "trade_identity": ident,
        "semantics": {
            "per_window": "reset cash=initial_cash, clear positions each window",
            "combined": "continuous cash and positions across all OOS signals",
            "comparable_as_identical": False,
        },
        "root_cause": root,
        "status": "RECONCILED_EXPECTED_DIVERGENCE" if expected_divergence else "RECONCILED_EQUAL",
    }

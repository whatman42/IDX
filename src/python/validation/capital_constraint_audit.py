"""Capital-constraint & paper-readiness audit helpers (no strategy change)."""
from __future__ import annotations
from typing import Any, Sequence
from src.python.validation.economic_sim import _finite

def summarize_fills(fill_events: Sequence[dict]) -> dict[str, Any]:
    n = len(fill_events)
    def c(name: str) -> int:
        return sum(1 for e in fill_events if e.get("classification") == name)
    return {
        "n_signal_events": n,
        "FULL_FILL": c("FULL_FILL"),
        "PARTIAL_FILL": c("PARTIAL_FILL"),
        "SKIPPED_CASH": c("SKIPPED_CASH"),
        "SKIPPED_EXISTING_POSITION": c("SKIPPED_EXISTING_POSITION"),
        "SKIPPED_OTHER_HARD_CONSTRAINT": c("SKIPPED_OTHER_HARD_CONSTRAINT"),
        "pct_full_fill": _finite(c("FULL_FILL") / n) if n else None,
        "pct_skipped_cash": _finite(c("SKIPPED_CASH") / n) if n else None,
        "pct_skipped_existing": _finite(c("SKIPPED_EXISTING_POSITION") / n) if n else None,
    }

def equity_identity_ok(equity_curve: Sequence[dict], *, tol: float = 1e-4) -> dict[str, Any]:
    bad = 0
    neg_cash = 0
    for e in equity_curve:
        cash = float(e["cash"]); mv = float(e["market_value"]); eq = float(e["equity"])
        if abs((cash + mv) - eq) > tol:
            bad += 1
        if cash < -1e-6:
            neg_cash += 1
    return {
        "points_checked": len(equity_curve),
        "equity_mismatch_points": bad,
        "negative_cash_points": neg_cash,
        "status": "PASS" if bad == 0 and neg_cash == 0 else "FAIL",
    }

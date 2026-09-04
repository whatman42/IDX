"""Portfolio reconciliation — detect inconsistencies; never silent-repair."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.python.portfolio.engine import PortfolioEngine, PositionState


@dataclass
class ReconcileReport:
    ok: bool
    issues: list[str] = field(default_factory=list)


def reconcile(engine: PortfolioEngine) -> ReconcileReport:
    issues: list[str] = []
    st = engine.state
    if st.cash != st.cash or st.equity != st.equity:
        issues.append("NaN cash/equity")
    if st.cash < -1e-6:
        issues.append(f"negative cash: {st.cash}")
    ids = [t.tx_id for t in st.transactions]
    if len(ids) != len(set(ids)):
        issues.append("duplicate transaction ids in ledger")
    for k, p in st.positions.items():
        if p.state != PositionState.CLOSED and p.qty < 0:
            issues.append(f"negative qty on {k}")
        if p.state != PositionState.CLOSED and p.avg_price <= 0:
            issues.append(f"invalid avg_price on {k}")
    for t in st.transactions:
        if t.order_id not in st.applied_order_ids:
            issues.append(f"tx {t.tx_id} order not in applied set")
    return ReconcileReport(ok=len(issues) == 0, issues=issues)

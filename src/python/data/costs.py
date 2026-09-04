"""Transaction cost model hooks — report gross vs net."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class CostModel:
    fee_bps: float = 15.0
    slippage_bps: float = 5.0

    def cost_fraction(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000.0

    def apply(self, gross_returns: np.ndarray, traded: Optional[np.ndarray] = None) -> dict:
        r = np.asarray(gross_returns, dtype=float)
        traded = np.ones(len(r), dtype=bool) if traded is None else np.asarray(traded, dtype=bool)
        n_trades = int(traded.sum())
        cost_each = self.cost_fraction()
        fees = float(n_trades * (self.fee_bps / 10_000.0))
        slip = float(n_trades * (self.slippage_bps / 10_000.0))
        net = r.copy()
        net[traded] = net[traded] - cost_each
        return {
            "gross_sum": float(r[traded].sum()) if n_trades else 0.0,
            "fees": fees, "slippage": slip,
            "net_sum": float(net[traded].sum()) if n_trades else 0.0,
            "n_trades": float(n_trades), "cost_fraction_per_trade": cost_each,
        }

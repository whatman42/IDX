"""Paper Portfolio Engine — cash, positions, ledger, MTM, idempotent orders.

Accounting identities:
  equity = cash + market_value(positions)
  market_value(long)  = +qty * mark
  market_value(short) = -qty * mark
Open long:  cash -= notional + fee
Open short: cash += notional - fee
Close long: cash += notional - fee;  pnl = (exit - avg) * qty - fee
Close short: cash -= notional + fee; pnl = (avg - exit) * qty - fee
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.python.core.ids import order_id, tx_id


class PositionState(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    CLOSED = "CLOSED"


class ExitReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TIME_EXIT = "TIME_EXIT"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    RISK_EXIT = "RISK_EXIT"
    SYSTEM_EXIT = "SYSTEM_EXIT"
    MANUAL = "MANUAL"


@dataclass
class Position:
    symbol: str
    side: int
    qty: float
    avg_price: float
    state: PositionState = PositionState.OPEN
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: str = ""
    closed_at: Optional[str] = None
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    signal_id: str = ""
    exit_reason: Optional[str] = None


@dataclass
class Transaction:
    tx_id: str
    order_id: str
    signal_id: str
    symbol: str
    side: int
    action: str
    qty: float
    price: float
    fee: float
    timestamp: str
    cycle_id: str = ""


@dataclass
class PortfolioState:
    cash: float
    equity: float
    positions: dict[str, Position] = field(default_factory=dict)
    transactions: list[Transaction] = field(default_factory=list)
    realized_pnl: float = 0.0
    fees_total: float = 0.0
    peak_equity: float = 0.0
    applied_order_ids: set[str] = field(default_factory=set)
    cycle_id: str = ""
    updated_at: str = ""
    initial_cash: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash, "equity": self.equity,
            "realized_pnl": self.realized_pnl, "fees_total": self.fees_total,
            "peak_equity": self.peak_equity, "cycle_id": self.cycle_id,
            "updated_at": self.updated_at, "initial_cash": self.initial_cash,
            "applied_order_ids": sorted(self.applied_order_ids),
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "transactions": [asdict(t) for t in self.transactions],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PortfolioState":
        positions = {}
        for k, v in d.get("positions", {}).items():
            st = v.get("state", "OPEN")
            positions[k] = Position(
                symbol=v["symbol"], side=int(v["side"]), qty=float(v["qty"]),
                avg_price=float(v["avg_price"]),
                state=PositionState(st) if not isinstance(st, PositionState) else st,
                stop_loss=v.get("stop_loss"), take_profit=v.get("take_profit"),
                opened_at=v.get("opened_at", ""), closed_at=v.get("closed_at"),
                realized_pnl=float(v.get("realized_pnl", 0)), fees_paid=float(v.get("fees_paid", 0)),
                signal_id=v.get("signal_id", ""), exit_reason=v.get("exit_reason"),
            )
        return cls(
            cash=float(d["cash"]), equity=float(d["equity"]), positions=positions,
            transactions=[Transaction(**{**t}) for t in d.get("transactions", [])],
            realized_pnl=float(d.get("realized_pnl", 0)), fees_total=float(d.get("fees_total", 0)),
            peak_equity=float(d.get("peak_equity", d.get("equity", d["cash"]))),
            applied_order_ids=set(d.get("applied_order_ids", [])),
            cycle_id=d.get("cycle_id", ""), updated_at=d.get("updated_at", ""),
            initial_cash=float(d.get("initial_cash", d.get("cash", 0))),
        )


class PortfolioEngine:
    def __init__(self, initial_cash: float = 100_000_000.0, fee_bps: float = 10.0,
                 slippage_bps: float = 5.0, max_position_pct: float = 0.20, lot_size: float = 100.0):
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.fee_bps, self.slippage_bps = fee_bps, slippage_bps
        self.max_position_pct, self.lot_size = max_position_pct, lot_size
        self.state = PortfolioState(
            cash=initial_cash, equity=initial_cash, peak_equity=initial_cash,
            initial_cash=initial_cash, updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.fee_bps / 10_000.0

    def _slip_price(self, price: float, side: int) -> float:
        return price + side * (price * self.slippage_bps / 10_000.0)

    def market_value(self, marks: dict[str, float]) -> float:
        mv = 0.0
        for pos in self.state.positions.values():
            if pos.state == PositionState.CLOSED or pos.qty <= 0:
                continue
            px = marks.get(pos.symbol, pos.avg_price)
            if px <= 0 or px != px:
                px = pos.avg_price
            mv += pos.qty * px if pos.side == 1 else -pos.qty * px
        return mv

    def apply_buy(self, *, signal_id_: str, symbol: str, side: int, price: float, weight: float,
                  stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                  cycle_id: str = "", timestamp: Optional[str] = None) -> Optional[Transaction]:
        if side not in (1, -1):
            raise ValueError("side must be +1 or -1")
        if price <= 0 or price != price:
            raise ValueError(f"invalid price: {price}")
        if weight <= 0:
            return None
        oid = order_id(signal_id_, "BUY")
        if oid in self.state.applied_order_ids:
            return None
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        exec_px = self._slip_price(price, side)
        target_notional = min(weight, self.max_position_pct) * self.state.equity
        raw_shares = target_notional / exec_px
        qty = float(int(raw_shares / self.lot_size) * self.lot_size)
        if qty <= 0 and raw_shares >= self.lot_size * 0.9:
            qty = float(self.lot_size)
        if qty <= 0:
            return None
        notional, fee = qty * exec_px, self._fee(qty * exec_px)
        if side == 1:
            cost = notional + fee
            if cost > self.state.cash:
                affordable = (self.state.cash / (1 + self.fee_bps / 10_000.0)) / exec_px
                qty = float(int(affordable / self.lot_size) * self.lot_size)
                if qty <= 0:
                    return None
                notional, fee = qty * exec_px, self._fee(qty * exec_px)
                cost = notional + fee
            self.state.cash -= cost
        else:
            self.state.cash += notional - fee
        self.state.fees_total += fee
        key = f"{symbol}:{side}"
        if key in self.state.positions and self.state.positions[key].state != PositionState.CLOSED:
            pos = self.state.positions[key]
            new_qty = pos.qty + qty
            pos.avg_price = (pos.avg_price * pos.qty + exec_px * qty) / new_qty
            pos.qty, pos.fees_paid = new_qty, pos.fees_paid + fee
        else:
            self.state.positions[key] = Position(
                symbol=symbol, side=side, qty=qty, avg_price=exec_px, state=PositionState.OPEN,
                stop_loss=stop_loss, take_profit=take_profit, opened_at=ts, fees_paid=fee, signal_id=signal_id_,
            )
        txn = Transaction(
            tx_id=tx_id(oid, 0), order_id=oid, signal_id=signal_id_, symbol=symbol,
            side=side, action="BUY", qty=qty, price=exec_px, fee=fee, timestamp=ts, cycle_id=cycle_id,
        )
        self.state.transactions.append(txn)
        self.state.applied_order_ids.add(oid)
        self.state.cycle_id, self.state.updated_at = cycle_id, ts
        self.mark_to_market({symbol: exec_px})
        return txn

    def apply_close(self, *, symbol: str, side: int, price: float, reason: ExitReason,
                    signal_id_: str = "", cycle_id: str = "", timestamp: Optional[str] = None,
                    qty: Optional[float] = None) -> Optional[Transaction]:
        key = f"{symbol}:{side}"
        if key not in self.state.positions:
            return None
        pos = self.state.positions[key]
        if pos.state == PositionState.CLOSED or pos.qty <= 0:
            return None
        close_qty = pos.qty if qty is None else min(qty, pos.qty)
        close_qty = float(int(close_qty / self.lot_size) * self.lot_size)
        if close_qty <= 0:
            close_qty = pos.qty
        oid = order_id(signal_id_ or pos.signal_id or key, f"CLOSE:{reason.value}:{close_qty}")
        if oid in self.state.applied_order_ids:
            return None
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        exec_px = self._slip_price(price, -side)
        notional, fee = close_qty * exec_px, self._fee(close_qty * exec_px)
        if side == 1:
            pnl = (exec_px - pos.avg_price) * close_qty - fee
            self.state.cash += notional - fee
        else:
            pnl = (pos.avg_price - exec_px) * close_qty - fee
            self.state.cash -= notional + fee
        self.state.realized_pnl += pnl
        self.state.fees_total += fee
        pos.realized_pnl += pnl
        pos.fees_paid += fee
        pos.qty -= close_qty
        if pos.qty <= 1e-12:
            pos.qty, pos.state, pos.closed_at, pos.exit_reason = 0.0, PositionState.CLOSED, ts, reason.value
        else:
            pos.state = PositionState.PARTIAL_EXIT
        txn = Transaction(
            tx_id=tx_id(oid, 0), order_id=oid, signal_id=signal_id_ or pos.signal_id,
            symbol=symbol, side=side, action=f"CLOSE:{reason.value}", qty=close_qty,
            price=exec_px, fee=fee, timestamp=ts, cycle_id=cycle_id,
        )
        self.state.transactions.append(txn)
        self.state.applied_order_ids.add(oid)
        self.state.updated_at = ts
        self.mark_to_market({symbol: exec_px})
        return txn

    def mark_to_market(self, marks: dict[str, float]) -> float:
        mv = self.market_value(marks)
        self.state.equity = self.state.cash + mv
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        return self.state.equity

    def check_exits(self, marks: dict[str, float], cycle_id: str = "") -> list[Transaction]:
        closed: list[Transaction] = []
        for pos in list(self.state.positions.values()):
            if pos.state == PositionState.CLOSED or pos.qty <= 0:
                continue
            px = marks.get(pos.symbol)
            if px is None or px <= 0 or px != px:
                continue
            reason = None
            if pos.side == 1:
                if pos.stop_loss is not None and px <= pos.stop_loss:
                    reason = ExitReason.STOP_LOSS
                elif pos.take_profit is not None and px >= pos.take_profit:
                    reason = ExitReason.TAKE_PROFIT
            else:
                if pos.stop_loss is not None and px >= pos.stop_loss:
                    reason = ExitReason.STOP_LOSS
                elif pos.take_profit is not None and px <= pos.take_profit:
                    reason = ExitReason.TAKE_PROFIT
            if reason is not None:
                txn = self.apply_close(symbol=pos.symbol, side=pos.side, price=px, reason=reason,
                                       signal_id_=pos.signal_id, cycle_id=cycle_id)
                if txn:
                    closed.append(txn)
        return closed

    def drawdown(self) -> float:
        if self.state.peak_equity <= 0:
            return 0.0
        return (self.state.equity - self.state.peak_equity) / self.state.peak_equity

    def accounting_identity(self, marks: Optional[dict[str, float]] = None) -> dict[str, float]:
        if marks is None:
            marks = {p.symbol: p.avg_price for p in self.state.positions.values()}
        mv = self.market_value(marks)
        equity = self.state.cash + mv
        return {
            "cash": self.state.cash, "market_value": mv, "equity": equity,
            "stated_equity": self.state.equity, "realized_pnl": self.state.realized_pnl,
            "fees_total": self.state.fees_total, "initial_cash": self.state.initial_cash,
            "identity_gap": equity - self.state.equity,
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state.to_dict(), indent=2, default=str))

    def load(self, path: Path | str) -> None:
        self.state = PortfolioState.from_dict(json.loads(Path(path).read_text()))

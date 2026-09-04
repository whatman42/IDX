"""Corporate action contract + policy — no fabricated live feeds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CorporateActionType(str, Enum):
    DIVIDEND = "DIVIDEND"
    STOCK_SPLIT = "STOCK_SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    STOCK_DIVIDEND = "STOCK_DIVIDEND"
    RIGHTS_ISSUE = "RIGHTS_ISSUE"
    TICKER_CHANGE = "TICKER_CHANGE"
    DELISTING = "DELISTING"
    SUSPENSION = "SUSPENSION"
    RESUMPTION = "RESUMPTION"
    OTHER = "OTHER"


@dataclass
class CorporateActionRecord:
    symbol: str
    effective_date: str
    action_type: CorporateActionType
    ratio: Optional[float] = None
    cash_component: Optional[float] = None
    new_symbol: Optional[str] = None
    source: str = ""
    id: str = ""


class CorporateActionProvider(ABC):
    @abstractmethod
    def get_actions(self, symbol: str, start: str, end: str) -> list[CorporateActionRecord]:
        ...


class EmptyCorporateActionProvider(CorporateActionProvider):
    def get_actions(self, symbol: str, start: str, end: str) -> list[CorporateActionRecord]:
        return []


class FileCorporateActionProvider(CorporateActionProvider):
    def __init__(self, records: list[CorporateActionRecord] | None = None):
        self._records = records or []

    def get_actions(self, symbol: str, start: str, end: str) -> list[CorporateActionRecord]:
        return [r for r in self._records if r.symbol == symbol and start <= r.effective_date <= end]


@dataclass
class PositionAdjustment:
    new_qty: float
    new_avg_price: float
    cash_delta: float


class CorporateActionPolicy:
    @staticmethod
    def apply_to_position(qty: float, avg_price: float, action: CorporateActionRecord) -> PositionAdjustment:
        if action.action_type == CorporateActionType.STOCK_SPLIT:
            ratio = float(action.ratio or 1.0)
            if ratio <= 0:
                raise ValueError("split ratio must be positive")
            return PositionAdjustment(new_qty=qty * ratio, new_avg_price=avg_price / ratio, cash_delta=0.0)
        if action.action_type == CorporateActionType.REVERSE_SPLIT:
            ratio = float(action.ratio or 1.0)
            if ratio <= 0:
                raise ValueError("reverse split ratio must be positive")
            return PositionAdjustment(new_qty=qty / ratio, new_avg_price=avg_price * ratio, cash_delta=0.0)
        if action.action_type in (CorporateActionType.DIVIDEND, CorporateActionType.STOCK_DIVIDEND):
            cash = float(action.cash_component or 0.0) * qty
            return PositionAdjustment(new_qty=qty, new_avg_price=avg_price, cash_delta=cash)
        return PositionAdjustment(new_qty=qty, new_avg_price=avg_price, cash_delta=0.0)

    @staticmethod
    def material_unresolved(open_symbols: list[str], actions: list[CorporateActionRecord], price_basis_raw: bool) -> list[str]:
        if not price_basis_raw:
            return []
        blocked = []
        for a in actions:
            if a.symbol in open_symbols and a.action_type in (
                CorporateActionType.STOCK_SPLIT, CorporateActionType.REVERSE_SPLIT,
                CorporateActionType.DELISTING, CorporateActionType.SUSPENSION,
            ):
                blocked.append(a.symbol)
        return sorted(set(blocked))

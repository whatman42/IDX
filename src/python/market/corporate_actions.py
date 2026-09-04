"""Corporate action architecture — no fabricated live CA data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CorporateActionType(str, Enum):
    DIVIDEND = "DIVIDEND"
    STOCK_SPLIT = "STOCK_SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    RIGHTS_ISSUE = "RIGHTS_ISSUE"
    TICKER_CHANGE = "TICKER_CHANGE"
    OTHER = "OTHER"


@dataclass
class CorporateAction:
    symbol: str
    effective_date: str
    action_type: CorporateActionType
    ratio: Optional[float] = None
    cash_component: Optional[float] = None
    new_symbol: Optional[str] = None
    source: str = ""


class CorporateActionProvider(ABC):
    @abstractmethod
    def get_actions(self, symbol: str, start: str, end: str) -> list[CorporateAction]:
        ...


class EmptyCorporateActionProvider(CorporateActionProvider):
    def get_actions(self, symbol: str, start: str, end: str) -> list[CorporateAction]:
        return []

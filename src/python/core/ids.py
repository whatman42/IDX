"""Deterministic ID generation for idempotent cycles/signals/orders."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone


def new_cycle_id(ts: datetime | None = None) -> str:
    ts = ts or datetime.now(timezone.utc)
    return f"cyc_{ts.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"


def signal_id(cycle_id: str, symbol: str, side: int, ts: str) -> str:
    raw = f"{cycle_id}|{symbol}|{side}|{ts}"
    return "sig_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def order_id(signal_id_: str, action: str) -> str:
    raw = f"{signal_id_}|{action}"
    return "ord_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def tx_id(order_id_: str, seq: int = 0) -> str:
    raw = f"{order_id_}|{seq}"
    return "tx_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

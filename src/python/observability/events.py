"""Structured observability events."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    DATA_LOADED = "DATA_LOADED"
    DATA_VALIDATED = "DATA_VALIDATED"
    FEATURES_BUILT = "FEATURES_BUILT"
    MODEL_LOADED = "MODEL_LOADED"
    MODEL_INFERRED = "MODEL_INFERRED"
    REGIME_DETECTED = "REGIME_DETECTED"
    GOVERNOR_DECIDED = "GOVERNOR_DECIDED"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_DENIED = "RISK_DENIED"
    ORDER_CREATED = "ORDER_CREATED"
    TRANSACTION_COMMITTED = "TRANSACTION_COMMITTED"
    POSITION_UPDATED = "POSITION_UPDATED"
    PORTFOLIO_UPDATED = "PORTFOLIO_UPDATED"
    NOTIFICATION_QUEUED = "NOTIFICATION_QUEUED"
    NOTIFICATION_SENT = "NOTIFICATION_SENT"
    ERROR = "ERROR"
    DEGRADED = "DEGRADED"


class EventLog:
    def __init__(self, cycle_id: str, git_sha: str = ""):
        self.cycle_id = cycle_id
        self.git_sha = git_sha
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: EventType | str, payload: Optional[dict] = None, severity: str = "INFO") -> dict:
        ev = {
            "event_id": f"ev_{uuid.uuid4().hex[:12]}",
            "cycle_id": self.cycle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type.value if isinstance(event_type, EventType) else event_type,
            "severity": severity,
            "git_sha": self.git_sha,
            "payload": payload or {},
        }
        self.events.append(ev)
        return ev

    def to_list(self) -> list[dict]:
        return list(self.events)

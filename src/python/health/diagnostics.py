"""System health diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass
class ComponentHealth:
    name: str
    status: str
    detail: str = ""


@dataclass
class SystemHealth:
    overall: str
    components: list[ComponentHealth] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"overall": self.overall, "components": [asdict(c) for c in self.components]}


def probe_health(
    *,
    data_ok: bool = True,
    model_ok: bool = True,
    rust_ok: bool = True,
    db_ok: bool = True,
    telegram_configured: bool = False,
    portfolio_ok: bool = True,
) -> SystemHealth:
    comps = [
        ComponentHealth("data", HealthStatus.HEALTHY.value if data_ok else HealthStatus.FAILED.value),
        ComponentHealth("model", HealthStatus.HEALTHY.value if model_ok else HealthStatus.DEGRADED.value),
        ComponentHealth("rust", HealthStatus.HEALTHY.value if rust_ok else HealthStatus.DEGRADED.value,
                        detail="" if rust_ok else "binary missing or fail-closed"),
        ComponentHealth("database", HealthStatus.HEALTHY.value if db_ok else HealthStatus.FAILED.value),
        ComponentHealth("telegram", HealthStatus.HEALTHY.value if telegram_configured else HealthStatus.DEGRADED.value,
                        detail="not configured" if not telegram_configured else ""),
        ComponentHealth("portfolio", HealthStatus.HEALTHY.value if portfolio_ok else HealthStatus.FAILED.value),
    ]
    statuses = [c.status for c in comps]
    if HealthStatus.FAILED.value in statuses:
        overall = HealthStatus.FAILED.value
    elif HealthStatus.DEGRADED.value in statuses:
        overall = HealthStatus.DEGRADED.value
    else:
        overall = HealthStatus.HEALTHY.value
    return SystemHealth(overall=overall, components=comps)

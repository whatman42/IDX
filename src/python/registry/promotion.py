"""Model promotion gates — accuracy alone is insufficient."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateResult:
    passed: bool
    name: str
    detail: str = ""


@dataclass
class PromotionReport:
    approved: bool
    gates: list[GateResult] = field(default_factory=list)
    reason: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in self.gates],
        }


def evaluate_promotion(
    metrics: dict[str, float],
    *,
    min_samples: int = 30,
    min_accuracy: float = 0.52,
    max_drawdown: float = 0.25,
    min_expectancy: float = -0.01,
    min_calibration_ok: bool = True,
) -> PromotionReport:
    gates: list[GateResult] = []
    n = int(metrics.get("n_samples", metrics.get("support", 0)))
    gates.append(GateResult(n >= min_samples, "sample_size", f"n={n} min={min_samples}"))
    acc = float(metrics.get("accuracy", 0))
    gates.append(GateResult(acc >= min_accuracy, "oos_accuracy", f"acc={acc:.4f}"))
    dd = float(metrics.get("max_drawdown", metrics.get("drawdown", 0)))
    gates.append(GateResult(abs(dd) <= max_drawdown, "max_drawdown", f"dd={dd}"))
    exp = float(metrics.get("expectancy", metrics.get("avg_return", 0)))
    if "expectancy" in metrics or "avg_return" in metrics:
        gates.append(GateResult(exp >= min_expectancy, "expectancy", f"exp={exp}"))
    if "brier" in metrics:
        brier = float(metrics["brier"])
        gates.append(GateResult(brier <= 0.30, "calibration_brier", f"brier={brier}"))
    if not min_calibration_ok:
        gates.append(GateResult(False, "calibration_flag", "calibration marked bad"))
    if acc >= 0.99 and "expectancy" not in metrics and "max_drawdown" not in metrics:
        gates.append(GateResult(False, "suspicious_perfect_accuracy", "require trading metrics"))
    failed = [g for g in gates if not g.passed]
    if failed:
        return PromotionReport(approved=False, gates=gates, reason="gates_failed: " + ",".join(g.name for g in failed))
    return PromotionReport(approved=True, gates=gates, reason="all_gates_passed")

"""Training runtime estimator with actual feedback for Governor budget allocation."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

ACTION_COMPLEXITY = {
    "SKIP": 0.0, "DEFER": 0.0, "LIGHTWEIGHT": 0.4, "PRIMARY": 1.0, "PRIMARY_META": 1.5,
    "ENSEMBLE": 2.5, "FULL_RETRAIN": 3.0, "MODEL_PRIMARY": 1.0, "MODEL_PRIMARY_META": 1.5,
    "LIGHTWEIGHT_ENSEMBLE": 2.0, "FULL_ENSEMBLE": 2.8, "DEFENSIVE": 0.0,
}

@dataclass
class RuntimeEstimate:
    estimated_seconds: float
    confidence: float
    action: str
    n_rows: int = 0
    n_features: int = 0
    hardware_factor: float = 1.0
    basis: str = "heuristic"
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class RuntimeMemory:
    path: Optional[Path] = None
    records: list = field(default_factory=list)
    valid: bool = True

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, list):
                self.valid = False
                return
            self.records = data[-200:]
        except Exception:
            self.valid = False
            self.records = []

    def save(self) -> None:
        if not self.path or not self.valid:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records[-200:], indent=2))

    def record(self, *, action: str, n_rows: int, n_features: int,
               estimated_seconds: float, actual_seconds: float, hardware: str = "cpu") -> None:
        if not self.valid:
            return
        self.records.append({
            "action": action, "n_rows": n_rows, "n_features": n_features,
            "estimated_seconds": estimated_seconds, "actual_seconds": actual_seconds, "hardware": hardware,
        })
        self.save()

    def mean_ratio(self, action: str) -> Optional[float]:
        xs = [r["actual_seconds"] / max(r["estimated_seconds"], 1e-6)
              for r in self.records if r.get("action") == action and r.get("estimated_seconds", 0) > 0]
        if len(xs) < 2:
            return None
        return sum(xs) / len(xs)

def estimate_runtime(*, action: str, n_rows: int, n_features: int = 40, n_symbols: int = 1,
                     gpu_available: bool = False, vram_gb: float = 0.0,
                     memory: Optional[RuntimeMemory] = None) -> RuntimeEstimate:
    if action in ("SKIP", "DEFER", "DEFENSIVE"):
        return RuntimeEstimate(0.0, 1.0, action, n_rows, n_features, 1.0, "noop")
    if n_rows < 0 or n_features < 0:
        return RuntimeEstimate(1e9, 0.0, action, n_rows, n_features, 1.0, "invalid_input")
    complexity = ACTION_COMPLEXITY.get(action, 1.0)
    base = max(1.0, n_rows * max(n_features, 1) * 0.00015 * complexity * max(n_symbols, 1) ** 0.5)
    hw = 1.0
    if gpu_available and vram_gb >= 8:
        hw = 0.55
    elif gpu_available:
        hw = 0.75
    est = base * hw
    conf, basis = 0.35, "heuristic"
    if memory and memory.valid:
        ratio = memory.mean_ratio(action)
        if ratio is not None:
            est = est * ratio
            conf = min(0.85, 0.35 + 0.1 * len([r for r in memory.records if r.get("action") == action]))
            basis = "calibrated"
    return RuntimeEstimate(float(max(0.5, est)), float(conf), action, n_rows, n_features, hw, basis)

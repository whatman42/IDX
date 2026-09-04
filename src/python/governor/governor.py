"""ML Governor — adaptive control plane (regime × performance × resources)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.python.regime.detector import Regime


@dataclass
class ResourceBudget:
    n_cpu: int = 4
    ram_gb: float = 12.0
    has_gpu: bool = False
    max_parallel_models: int = 2
    cycle_timeout_sec: int = 900


@dataclass
class GovernorConfig:
    version: str = "gov_v001"
    meta_threshold: float = 0.55
    side_threshold: float = 0.50
    max_position_pct: float = 0.15
    kelly_fraction: float = 0.35
    sizing_method: str = "sigmoid"
    active_models: list[str] = field(default_factory=lambda: ["primary_lgbm", "meta_rf"])
    pt: float = 0.02
    sl: float = 0.01
    embargo_days: int = 5
    allow_new_trades: bool = True
    regime: str = Regime.UNKNOWN.value
    reason: str = ""
    updated_at: str = ""


def detect_resources() -> ResourceBudget:
    import os
    n_cpu = max(1, min(os.cpu_count() or 2, 4))
    ram_gb = 12.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    ram_gb = min(12.0, int(line.split()[1]) / 1e6)
                    break
    except Exception:
        pass
    return ResourceBudget(n_cpu=n_cpu, ram_gb=ram_gb, max_parallel_models=2 if n_cpu >= 2 else 1)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class MLGovernor:
    def __init__(self, version: str = "gov_v001"):
        self.version = version
        self.config = GovernorConfig(version=version)
        self.resources = detect_resources()
        self._perf_history: list[dict[str, float]] = []

    def observe_performance(self, metrics: dict[str, float]) -> None:
        self._perf_history.append(metrics)
        self._perf_history = self._perf_history[-60:]

    def decide(
        self,
        regime: Regime,
        recent_drawdown: float = 0.0,
        data_ok: bool = True,
        model_health: Optional[dict[str, bool]] = None,
    ) -> GovernorConfig:
        model_health = model_health or {"primary_lgbm": True, "meta_rf": True}
        cfg = GovernorConfig(version=self.version, regime=regime.value)
        cfg.updated_at = datetime.now(timezone.utc).isoformat()

        if not data_ok:
            cfg.allow_new_trades = False
            cfg.reason = "data_quality_fail"
            self.config = cfg
            return cfg

        if regime == Regime.HIGH_VOL:
            cfg.meta_threshold, cfg.max_position_pct, cfg.kelly_fraction = 0.65, 0.08, 0.20
            cfg.pt, cfg.sl, cfg.reason = 0.025, 0.015, "high_vol_defensive"
        elif regime == Regime.LOW_VOL_TREND:
            cfg.meta_threshold, cfg.max_position_pct, cfg.kelly_fraction = 0.52, 0.18, 0.40
            cfg.reason = "trend_permissive"
        elif regime == Regime.MEAN_REVERT:
            cfg.meta_threshold, cfg.max_position_pct, cfg.kelly_fraction = 0.58, 0.12, 0.30
            cfg.reason = "mean_revert_moderate"
        else:
            cfg.reason = "default"

        if recent_drawdown < -0.05:
            cfg.meta_threshold = min(0.75, cfg.meta_threshold + 0.10)
            cfg.max_position_pct *= 0.5
            cfg.allow_new_trades = recent_drawdown > -0.10
            cfg.reason += "+dd_brake"

        if self._perf_history:
            avg_hit = _mean([m.get("hit_rate", 0.5) for m in self._perf_history[-10:]])
            if avg_hit < 0.40:
                cfg.meta_threshold = min(0.70, cfg.meta_threshold + 0.05)
                cfg.reason += "+low_hit"
            elif avg_hit > 0.55:
                cfg.meta_threshold = max(0.50, cfg.meta_threshold - 0.03)
                cfg.reason += "+good_hit"

        active = []
        if model_health.get("primary_lgbm", True):
            active.append("primary_lgbm")
        if model_health.get("meta_rf", True) and self.resources.max_parallel_models >= 2:
            active.append("meta_rf")
        if not active:
            cfg.allow_new_trades = False
            cfg.reason += "+no_healthy_model"
        cfg.active_models = active
        if self.resources.ram_gb < 4:
            cfg.active_models = cfg.active_models[:1]
            cfg.reason += "+low_ram"

        self.config = cfg
        return cfg

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self.config), indent=2))

    def load(self, path: Path | str) -> GovernorConfig:
        d = json.loads(Path(path).read_text())
        fields = set(GovernorConfig.__dataclass_fields__)
        self.config = GovernorConfig(**{k: v for k, v in d.items() if k in fields})
        return self.config

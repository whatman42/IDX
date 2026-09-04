"""ML Governor — multi-objective utility + backward-compatible decide API."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _regime_str(regime: Any) -> str:
    if regime is None:
        return "neutral"
    if hasattr(regime, "value"):
        s = str(regime.value).lower()
    else:
        s = str(regime).lower()
    mapping = {"high_vol": "high_vol", "low_vol_trend": "low_vol", "low_vol": "low_vol",
               "crisis": "crisis", "unknown": "unknown", "neutral": "neutral", "trend": "low_vol"}
    for k, v in mapping.items():
        if k in s:
            return v
    return s or "neutral"

@dataclass
class ResourceProfile:
    cpu_cores: int = 2
    ram_gb: float = 4.0
    max_parallel_models: int = 2
    training_budget_sec: int = 1200
    is_high_end: bool = False
    @classmethod
    def detect(cls) -> "ResourceProfile":
        import os
        try:
            cores = os.cpu_count() or 2
        except Exception:
            cores = 2
        ram = 4.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram = int(line.split()[1]) / 1e6
                        break
        except Exception:
            pass
        high = cores >= 4 and ram >= 8.0
        return cls(cpu_cores=cores, ram_gb=ram,
                   max_parallel_models=min(4, max(1, cores // 2)) if high else 1,
                   training_budget_sec=1200 if high else 600, is_high_end=high)

@dataclass
class UtilityWeights:
    alpha_edge: float = 1.0
    beta_drawdown: float = 1.5
    gamma_volatility: float = 0.5
    delta_turnover: float = 0.3
    epsilon_cost: float = 0.4
    zeta_complexity: float = 0.2
    eta_runtime: float = 0.3

@dataclass
class GovernorConfig:
    governor_version: str = "gov_util_v2"
    version: str = "gov_util_v2"
    decision_id: str = ""
    strategy: str = "primary_meta"
    active_models: list = field(default_factory=lambda: ["primary_lgbm", "meta_rf"])
    model_weights: dict = field(default_factory=lambda: {"primary_lgbm": 1.0, "meta_rf": 1.0})
    meta_threshold: float = 0.55
    primary_threshold: float = 0.50
    risk_budget: float = 0.20
    max_position_pct: float = 0.20
    compute_budget: float = 1.0
    max_positions: int = 5
    allow_new_trades: bool = True
    regime: str = "neutral"
    utility_score: float = 0.0
    utility_components: dict = field(default_factory=dict)
    training_candidate_limit: int = 2
    training_search_budget: int = 20
    reason: str = "default"
    timestamp: str = ""

@dataclass
class CandidateScore:
    name: str
    expected_edge: float = 0.0
    drawdown: float = 0.0
    volatility: float = 0.0
    turnover: float = 0.0
    cost: float = 0.0
    complexity: float = 0.5
    runtime_cost: float = 0.3
    reliability: float = 0.5
    regime_fit: float = 1.0

class MLGovernor:
    def __init__(self, resources: Optional[ResourceProfile] = None, weights: Optional[UtilityWeights] = None, version: str = "gov_util_v2"):
        self.resources = resources or ResourceProfile.detect()
        self.weights = weights or UtilityWeights()
        self.version = version
        self.config = GovernorConfig(version=version, governor_version=version)
        self._perf_history: list = []

    def record_performance(self, metrics: dict) -> None:
        self._perf_history.append(dict(metrics))
        self._perf_history = self._perf_history[-50:]

    def observe_performance(self, metrics: dict) -> None:
        self.record_performance(metrics)

    def score_utility(self, c: CandidateScore):
        w = self.weights
        components = {
            "edge": w.alpha_edge * c.expected_edge * max(c.reliability, 0.05) * max(c.regime_fit, 0.05),
            "drawdown": -w.beta_drawdown * abs(c.drawdown),
            "volatility": -w.gamma_volatility * abs(c.volatility),
            "turnover": -w.delta_turnover * abs(c.turnover),
            "cost": -w.epsilon_cost * abs(c.cost),
            "complexity": -w.zeta_complexity * abs(c.complexity),
            "runtime": -w.eta_runtime * abs(c.runtime_cost),
        }
        return sum(components.values()), components

    def decide(self, regime: Any = "neutral", data_ok: bool = True, recent_drawdown: float = 0.0, *,
               drawdown: Optional[float] = None, model_health: Optional[dict] = None,
               candidates: Optional[list] = None, turnover: float = 0.0,
               decision_id: str = "", timestamp: str = "") -> GovernorConfig:
        reg = _regime_str(regime)
        dd = float(drawdown if drawdown is not None else recent_drawdown)
        model_health = model_health or {"primary_lgbm": True, "meta_rf": True}
        cfg = GovernorConfig(decision_id=decision_id or f"gov_{timestamp or 'now'}", regime=reg, timestamp=timestamp)
        if not data_ok or reg == "unknown":
            cfg.allow_new_trades = False
            cfg.risk_budget = 0.0
            cfg.max_position_pct = 0.0
            cfg.reason = "data_fail" if not data_ok else "unknown_regime"
            self.config = cfg
            return cfg
        w = UtilityWeights()
        if reg in ("high_vol", "crisis"):
            w.beta_drawdown, w.gamma_volatility, w.alpha_edge = 2.5, 1.2, 0.7
            cfg.meta_threshold, cfg.risk_budget, cfg.max_position_pct = 0.62, 0.10, 0.10
            cfg.reason = "high_vol_defensive"
        elif reg == "low_vol":
            w.alpha_edge, w.beta_drawdown = 1.2, 1.0
            cfg.meta_threshold, cfg.risk_budget, cfg.max_position_pct = 0.52, 0.25, 0.25
            cfg.reason = "low_vol_expand"
        else:
            cfg.reason = "neutral_baseline"
            cfg.max_position_pct = 0.20
        self.weights = w
        if dd < -0.10:
            cfg.allow_new_trades = False
            cfg.risk_budget = cfg.max_position_pct = 0.05
            cfg.reason += "+dd_halt"
        elif dd <= -0.05:
            cfg.risk_budget = min(cfg.risk_budget, 0.10)
            cfg.max_position_pct = min(cfg.max_position_pct, 0.10)
            cfg.meta_threshold = min(0.70, cfg.meta_threshold + 0.05)
            cfg.reason += "+dd_brake"
        if self._perf_history:
            avg_hit = _mean([m.get("hit_rate", 0.5) for m in self._perf_history[-10:]])
            if avg_hit < 0.40:
                cfg.meta_threshold = min(0.70, cfg.meta_threshold + 0.05)
                cfg.reason += "+low_hit"
            elif avg_hit > 0.55:
                cfg.meta_threshold = max(0.50, cfg.meta_threshold - 0.03)
                cfg.reason += "+good_hit"
        if candidates:
            scored = sorted(((self.score_utility(c)[0], c, self.score_utility(c)[1]) for c in candidates),
                            key=lambda x: x[0], reverse=True)
            best_u, best_c, best_comps = scored[0]
            cfg.utility_score, cfg.utility_components, cfg.strategy = best_u, best_comps, best_c.name
            if best_u <= 0:
                cfg.allow_new_trades = False
                cfg.reason += "+neg_utility"
        else:
            default = CandidateScore(name="primary_meta", expected_edge=0.03, drawdown=abs(dd), turnover=turnover,
                                     reliability=0.6 if model_health.get("primary_lgbm") else 0.2,
                                     regime_fit=0.3 if reg == "crisis" else 1.0, complexity=0.3, runtime_cost=0.2)
            u, comps = self.score_utility(default)
            cfg.utility_score, cfg.utility_components = u, comps
        active = []
        if model_health.get("primary_lgbm", True):
            active.append("primary_lgbm")
        if model_health.get("meta_rf", True) and self.resources.max_parallel_models >= 2 and cfg.utility_score > -0.1:
            active.append("meta_rf")
        if not active:
            cfg.allow_new_trades = False
            cfg.reason += "+no_healthy_model"
        if self.resources.ram_gb < 4 or self.resources.cpu_cores <= 1:
            active = active[:1]
            cfg.compute_budget = 0.4
            cfg.reason += "+low_resource"
        elif self.resources.is_high_end:
            cfg.compute_budget = 1.0
            cfg.reason += "+high_end_selective"
        else:
            cfg.compute_budget = 0.7
        cfg.active_models = active
        cfg.model_weights = {m: 1.0 for m in active}
        if self.resources.is_high_end:
            cfg.training_candidate_limit, cfg.training_search_budget = 4, 40
        else:
            cfg.training_candidate_limit, cfg.training_search_budget = 1, 10
        if turnover > 0.5:
            cfg.meta_threshold = min(0.72, cfg.meta_threshold + 0.04)
            cfg.risk_budget = min(cfg.risk_budget, 0.12)
            cfg.max_position_pct = min(cfg.max_position_pct, 0.12)
            cfg.reason += "+high_turnover"
        self.config = cfg
        return cfg

    def training_plan(self, remaining_sec: float) -> dict:
        limit, budget = self.config.training_candidate_limit, self.config.training_search_budget
        if remaining_sec < 300:
            limit, budget = 1, 3
        elif remaining_sec < 600:
            limit, budget = min(limit, 2), min(budget, 10)
        return {"candidate_limit": limit, "search_budget": budget,
                "n_jobs": 1 if self.resources.cpu_cores <= 2 else min(2, self.resources.cpu_cores // 2),
                "remaining_sec": remaining_sec, "allow_train": remaining_sec > 60}

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self.config), indent=2))

    def load(self, path) -> GovernorConfig:
        d = json.loads(Path(path).read_text())
        fields = set(GovernorConfig.__dataclass_fields__)
        self.config = GovernorConfig(**{k: v for k, v in d.items() if k in fields})
        return self.config

"""Adaptive training allocation Governor layer — additive; does not replace MLGovernor."""
from __future__ import annotations
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from src.python.governor.hardware import HardwareProfile, detect_hardware
from src.python.governor.learning import GovernorMemory, OutcomeRecord, compute_reward
from src.python.governor.runtime_estimator import RuntimeMemory, estimate_runtime

TRAIN_ACTIONS = ("SKIP", "DEFER", "LIGHTWEIGHT", "PRIMARY", "PRIMARY_META", "ENSEMBLE", "FULL_RETRAIN")
ACTION_TO_POLICY = {
    "LIGHTWEIGHT": "MODEL_PRIMARY", "PRIMARY": "MODEL_PRIMARY", "PRIMARY_META": "MODEL_PRIMARY_META",
    "ENSEMBLE": "LIGHTWEIGHT_ENSEMBLE", "FULL_RETRAIN": "FULL_ENSEMBLE", "SKIP": "DEFENSIVE", "DEFER": "DEFENSIVE",
}

@dataclass
class SymbolContext:
    symbol: str
    n_rows: int = 0
    n_features: int = 40
    data_ok: bool = True
    data_quality_score: float = 1.0
    liquidity_score: float = 0.5
    model_age_days: float = 999.0
    recent_oos_accuracy: float = 0.5
    recent_expectancy: float = 0.0
    max_drawdown: float = 0.0
    feature_drift: float = 0.0
    prediction_drift: float = 0.0
    regime: str = "neutral"
    regime_changed: bool = False
    uncertainty: float = 0.5
    last_train_benefit: float = 0.0
    sample_count: int = 0

@dataclass
class CandidateDecision:
    symbol: str
    priority: float
    action: str
    reason_codes: list = field(default_factory=list)
    estimated_seconds: float = 0.0
    estimate_confidence: float = 0.0
    training_utility: float = 0.0
    policy_action: str = "DEFENSIVE"
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class AllocationPlan:
    cycle_id: str
    timestamp: str
    hardware: dict
    budget_seconds: float
    budget_minutes: float
    candidates: list
    selected_count: int = 0
    skipped_count: int = 0
    deferred_count: int = 0
    policy_version: str = "gov_learn_v1"
    governor_version: str = "gov_train_alloc_v1"
    notes: list = field(default_factory=list)
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def score_training_utility(ctx: SymbolContext, hardware: HardwareProfile) -> tuple:
    reasons: list = []
    if not ctx.data_ok or ctx.data_quality_score < 0.4:
        return -1.0, ["BAD_DATA_QUALITY"]
    if ctx.n_rows < 40:
        return -1.0, ["INSUFFICIENT_DATA"]
    benefit = 0.0
    if ctx.model_age_days > 30:
        benefit += 0.15; reasons.append("MODEL_AGE")
    if ctx.recent_oos_accuracy < 0.52:
        benefit += 0.2; reasons.append("MODEL_DEGRADATION")
    if ctx.feature_drift > 0.3:
        benefit += 0.15; reasons.append("FEATURE_DRIFT")
    if ctx.prediction_drift > 0.3:
        benefit += 0.15; reasons.append("PREDICTION_DRIFT")
    if ctx.regime_changed:
        benefit += 0.2; reasons.append("REGIME_CHANGE")
    if ctx.uncertainty > 0.6:
        benefit += 0.1; reasons.append("HIGH_UNCERTAINTY")
    if ctx.last_train_benefit > 0:
        benefit += min(0.15, ctx.last_train_benefit); reasons.append("HISTORICAL_BENEFIT")
    if ctx.liquidity_score > 0.6:
        benefit += 0.05; reasons.append("LIQUIDITY_OK")
    benefit += _clamp(ctx.n_rows / 500.0) * 0.1
    cost = 0.05
    if ctx.max_drawdown > 0.2:
        cost += 0.2; reasons.append("HIGH_DRAWDOWN")
    if ctx.liquidity_score < 0.2:
        cost += 0.15; reasons.append("LOW_LIQUIDITY")
    cost += (1.0 - hardware.training_capacity_score) * 0.05
    utility = _clamp(benefit - cost, -1.0, 1.0)
    if utility < 0.05 and not reasons:
        reasons.append("LOW_UTILITY")
    return utility, reasons

def select_action_for_utility(utility: float, ctx: SymbolContext, memory: Optional[GovernorMemory],
                              context_key: str, *, explore: bool = False) -> str:
    if utility <= -0.5 or not ctx.data_ok or ctx.n_rows < 40:
        return "SKIP"
    if utility < 0.1:
        return "SKIP"
    if utility < 0.25:
        return "LIGHTWEIGHT"
    if utility < 0.45:
        return "PRIMARY"
    if utility < 0.65:
        return "PRIMARY_META"
    if utility < 0.8:
        return "ENSEMBLE"
    action = "FULL_RETRAIN"
    if explore and memory and memory.policy.valid:
        alt = memory.select_action(context_key)
        mapped = {"MODEL_PRIMARY": "PRIMARY", "MODEL_PRIMARY_META": "PRIMARY_META",
                  "LIGHTWEIGHT_ENSEMBLE": "ENSEMBLE", "FULL_ENSEMBLE": "FULL_RETRAIN", "DEFENSIVE": "SKIP"}.get(alt, action)
        return mapped
    return action

class TrainingAllocator:
    def __init__(self, *, hardware: Optional[HardwareProfile] = None, memory: Optional[GovernorMemory] = None,
                 runtime_memory: Optional[RuntimeMemory] = None, budget_minutes: float = 20.0,
                 max_candidates: int = 20, max_exploration_frac: float = 0.1,
                 governor_version: str = "gov_train_alloc_v1"):
        self.hardware = hardware or detect_hardware()
        self.memory = memory
        self.runtime_memory = runtime_memory or RuntimeMemory()
        self.budget_minutes = float(budget_minutes)
        self.max_candidates = int(max_candidates)
        self.max_exploration_frac = float(max_exploration_frac)
        self.governor_version = governor_version

    def allocate(self, symbols: list, *, cycle_id: Optional[str] = None, regime: str = "neutral") -> AllocationPlan:
        cycle_id = cycle_id or f"train_{uuid.uuid4().hex[:12]}"
        ts = datetime.now(timezone.utc).isoformat()
        budget_sec = max(0.0, self.budget_minutes * 60.0)
        if budget_sec <= 0:
            return AllocationPlan(cycle_id=cycle_id, timestamp=ts, hardware=self.hardware.to_dict(),
                                 budget_seconds=0.0, budget_minutes=0.0, candidates=[],
                                 notes=["INVALID_BUDGET_FAIL_CLOSED"])
        scored = []
        for ctx in symbols:
            u, reasons = score_training_utility(ctx, self.hardware)
            scored.append((u, ctx, reasons))
        scored.sort(key=lambda x: x[0], reverse=True)
        remaining = budget_sec
        selected = []
        skipped = deferred = 0
        explore_budget = max(0, int(len(scored) * self.max_exploration_frac))
        for i, (utility, ctx, reasons) in enumerate(scored):
            if len(selected) >= self.max_candidates:
                deferred += 1
                selected.append(CandidateDecision(symbol=ctx.symbol, priority=utility, action="DEFER",
                    reason_codes=reasons + ["MAX_CANDIDATES"], estimated_seconds=0.0))
                continue
            context_key = f"{regime}:{ctx.regime}"
            explore = i < explore_budget and utility > 0.1
            action = select_action_for_utility(utility, ctx, self.memory, context_key, explore=explore)
            if action == "SKIP":
                skipped += 1
                selected.append(CandidateDecision(symbol=ctx.symbol, priority=max(0.0, utility), action="SKIP",
                    reason_codes=reasons or ["LOW_UTILITY"], estimated_seconds=0.0,
                    training_utility=utility, policy_action="DEFENSIVE"))
                continue
            est = estimate_runtime(action=action, n_rows=ctx.n_rows, n_features=ctx.n_features, n_symbols=1,
                gpu_available=self.hardware.gpu_available, vram_gb=self.hardware.vram_gb,
                memory=self.runtime_memory if self.runtime_memory.valid else None)
            if est.confidence <= 0.0 and est.estimated_seconds >= 1e8:
                skipped += 1
                selected.append(CandidateDecision(symbol=ctx.symbol, priority=0.0, action="SKIP",
                    reason_codes=reasons + ["INVALID_ESTIMATE"], estimated_seconds=0.0,
                    estimate_confidence=0.0, training_utility=utility, policy_action="DEFENSIVE"))
                continue
            if est.estimated_seconds > remaining:
                deferred += 1
                selected.append(CandidateDecision(symbol=ctx.symbol, priority=max(0.0, utility), action="DEFER",
                    reason_codes=reasons + ["BUDGET_EXHAUSTED"], estimated_seconds=est.estimated_seconds,
                    estimate_confidence=est.confidence, training_utility=utility, policy_action="DEFENSIVE"))
                continue
            remaining -= est.estimated_seconds
            selected.append(CandidateDecision(
                symbol=ctx.symbol, priority=max(0.0, min(1.0, (utility + 1.0) / 2.0)), action=action,
                reason_codes=reasons, estimated_seconds=est.estimated_seconds,
                estimate_confidence=est.confidence, training_utility=utility,
                policy_action=ACTION_TO_POLICY.get(action, "DEFENSIVE"),
            ))
        sel_n = sum(1 for c in selected if c.action not in ("SKIP", "DEFER"))
        return AllocationPlan(
            cycle_id=cycle_id, timestamp=ts, hardware=self.hardware.to_dict(),
            budget_seconds=budget_sec, budget_minutes=self.budget_minutes,
            candidates=[c.to_dict() for c in selected], selected_count=sel_n,
            skipped_count=skipped, deferred_count=deferred,
            policy_version=getattr(getattr(self.memory, "policy", None), "version", "gov_learn_v1") if self.memory else "gov_learn_v1",
            governor_version=self.governor_version,
        )

    def record_training_outcome(self, *, decision_id: str, action: str, estimated_seconds: float,
                                actual_seconds: float, oos_accuracy: float = 0.5, expectancy: float = 0.0,
                                max_drawdown: float = 0.0, n_rows: int = 0, n_features: int = 40,
                                context_key: str = "neutral") -> float:
        if self.runtime_memory and self.runtime_memory.valid:
            hw = "gpu" if self.hardware.gpu_available else "cpu"
            self.runtime_memory.record(action=action, n_rows=n_rows, n_features=n_features,
                estimated_seconds=estimated_seconds, actual_seconds=actual_seconds, hardware=hw)
        reward = compute_reward(actual_return=expectancy, drawdown_impact=max_drawdown,
            runtime_actual=actual_seconds, runtime_budget=max(estimated_seconds, 1.0),
            prediction_error=max(0.0, 0.55 - oos_accuracy))
        if self.memory is not None:
            from src.python.governor.learning import DecisionRecord
            if decision_id not in self.memory.decisions:
                self.memory.record_decision(DecisionRecord(
                    decision_id=decision_id, timestamp=datetime.now(timezone.utc).isoformat(),
                    policy_action=ACTION_TO_POLICY.get(action, action), context_key=context_key,
                ))
            self.memory.record_outcome(OutcomeRecord(
                decision_id=decision_id, actual_return=expectancy, drawdown_impact=max_drawdown,
                runtime_actual=actual_seconds, actual_utility=reward,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
        return reward

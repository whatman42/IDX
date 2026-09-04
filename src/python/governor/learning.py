"""Governor online learning — contextual bandit style, production-safe.

Stores decision → outcome → reward; updates policy with bounded steps.
Corrupt state → SAFE DEFAULT POLICY (fail-closed, not fail-open).
"""
from __future__ import annotations
import json, math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

POLICY_ACTIONS = (
    "MODEL_PRIMARY", "MODEL_PRIMARY_META", "LIGHTWEIGHT_ENSEMBLE", "FULL_ENSEMBLE", "DEFENSIVE",
)
MIN_SAMPLES_TO_LEARN = 5
MAX_UPDATE = 0.05
EXPLORATION_RATE = 0.08
DECAY = 0.98

@dataclass
class DecisionRecord:
    decision_id: str
    timestamp: str
    market_regime: str = "neutral"
    volatility: float = 0.0
    drawdown: float = 0.0
    available_compute: float = 1.0
    runtime_budget: float = 1.0
    selected_models: list = field(default_factory=list)
    ensemble_weights: dict = field(default_factory=dict)
    confidence_threshold: float = 0.55
    risk_multiplier: float = 1.0
    expected_utility: float = 0.0
    policy_action: str = "MODEL_PRIMARY_META"
    context_key: str = "neutral"

@dataclass
class OutcomeRecord:
    decision_id: str
    actual_return: float = 0.0
    actual_pnl: float = 0.0
    drawdown_impact: float = 0.0
    turnover: float = 0.0
    transaction_cost: float = 0.0
    prediction_error: float = 0.0
    calibration_error: float = 0.0
    runtime_actual: float = 0.0
    actual_utility: float = 0.0
    timestamp: str = ""

def compute_reward(*, actual_return: float = 0.0, drawdown_impact: float = 0.0,
                   turnover: float = 0.0, transaction_cost: float = 0.0,
                   runtime_actual: float = 0.0, runtime_budget: float = 1.0,
                   prediction_error: float = 0.0) -> float:
    r = actual_return
    r -= 1.5 * abs(drawdown_impact)
    r -= 0.3 * abs(turnover)
    r -= 0.4 * abs(transaction_cost)
    if runtime_budget > 0:
        r -= 0.2 * max(0.0, runtime_actual / runtime_budget - 1.0)
    r -= 0.2 * abs(prediction_error)
    return max(-2.0, min(2.0, r))

@dataclass
class PolicyState:
    version: str = "gov_learn_v1"
    values: dict = field(default_factory=dict)
    decision_count: int = 0
    outcome_count: int = 0
    exploration_rate: float = EXPLORATION_RATE
    valid: bool = True
    last_update: str = ""

    def ensure_action(self, context: str, action: str) -> None:
        self.values.setdefault(context, {})
        self.values[context].setdefault(action, {"n": 0.0, "mean": 0.0})

    def best_action(self, context: str, candidates: Optional[list] = None) -> str:
        candidates = candidates or list(POLICY_ACTIONS)
        if not self.valid or self.outcome_count < MIN_SAMPLES_TO_LEARN:
            return "MODEL_PRIMARY_META"
        bucket = self.values.get(context, {})
        best, best_v = "MODEL_PRIMARY_META", -1e9
        for a in candidates:
            stats = bucket.get(a, {"n": 0.0, "mean": 0.0})
            bonus = math.sqrt(2 * math.log(max(self.outcome_count, 1)) / max(stats["n"], 1))
            v = stats["mean"] + 0.1 * bonus
            if v > best_v:
                best_v, best = v, a
        return best

    def update(self, context: str, action: str, reward: float) -> None:
        if not self.valid:
            return
        self.ensure_action(context, action)
        st = self.values[context][action]
        n = st["n"]
        n_new = n * DECAY + 1.0
        mean = st["mean"]
        delta = max(-MAX_UPDATE, min(MAX_UPDATE, reward - mean))
        st["mean"] = mean + delta * (1.0 / n_new)
        st["n"] = n_new
        self.outcome_count += 1
        self.last_update = datetime.now(timezone.utc).isoformat()

class GovernorMemory:
    def __init__(self, path: Path | str = "state/governor_memory.json"):
        self.path = Path(path)
        self.decisions: dict = {}
        self.outcomes: dict = {}
        self.policy = PolicyState()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text())
            self.decisions = d.get("decisions", {})
            self.outcomes = d.get("outcomes", {})
            ps = d.get("policy", {})
            self.policy = PolicyState(
                version=ps.get("version", "gov_learn_v1"),
                values=ps.get("values", {}),
                decision_count=int(ps.get("decision_count", 0)),
                outcome_count=int(ps.get("outcome_count", 0)),
                exploration_rate=float(ps.get("exploration_rate", EXPLORATION_RATE)),
                valid=bool(ps.get("valid", True)),
                last_update=ps.get("last_update", ""),
            )
            if self.policy.outcome_count < 0 or self.policy.exploration_rate < 0:
                self.policy.valid = False
        except Exception:
            self.policy = PolicyState(valid=False)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "decisions": self.decisions, "outcomes": self.outcomes, "policy": asdict(self.policy),
        }, indent=2, default=str))

    def record_decision(self, rec: DecisionRecord) -> None:
        self.decisions[rec.decision_id] = asdict(rec)
        self.policy.decision_count += 1
        if len(self.decisions) > 500:
            for k in sorted(self.decisions.keys())[:-500]:
                self.decisions.pop(k, None)
        self.save()

    def record_outcome(self, out: OutcomeRecord) -> float:
        self.outcomes[out.decision_id] = asdict(out)
        dec = self.decisions.get(out.decision_id, {})
        reward = out.actual_utility
        if reward == 0.0 and (out.actual_return or out.drawdown_impact):
            reward = compute_reward(
                actual_return=out.actual_return, drawdown_impact=out.drawdown_impact,
                turnover=out.turnover, transaction_cost=out.transaction_cost,
                runtime_actual=out.runtime_actual, prediction_error=out.prediction_error,
            )
            out.actual_utility = reward
            self.outcomes[out.decision_id] = asdict(out)
        ctx = dec.get("context_key", "neutral")
        action = dec.get("policy_action", "MODEL_PRIMARY_META")
        if self.policy.valid:
            self.policy.update(ctx, action, reward)
        if len(self.outcomes) > 500:
            for k in sorted(self.outcomes.keys())[:-500]:
                self.outcomes.pop(k, None)
        self.save()
        return reward

    def select_action(self, context_key: str, rng_seed: Optional[int] = None) -> str:
        import random
        if not self.policy.valid:
            return "DEFENSIVE"
        if self.policy.outcome_count < MIN_SAMPLES_TO_LEARN:
            return "MODEL_PRIMARY_META"
        rng = random.Random(rng_seed)
        if rng.random() < self.policy.exploration_rate:
            return rng.choice(list(POLICY_ACTIONS))
        return self.policy.best_action(context_key)

    def rollback_policy(self) -> None:
        self.policy = PolicyState(valid=True, version="gov_learn_v1_rollback")
        self.save()

    def mark_corrupt(self) -> None:
        self.policy.valid = False
        self.save()

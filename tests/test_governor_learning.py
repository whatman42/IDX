from __future__ import annotations
from src.python.governor.learning import (
    DecisionRecord, OutcomeRecord, GovernorMemory, compute_reward, MIN_SAMPLES_TO_LEARN,
)
from src.python.governor.governor import MLGovernor, apply_policy_action, decide_with_memory

def test_reward_bounded():
    assert -2 <= compute_reward(actual_return=10.0) <= 2
    assert -2 <= compute_reward(actual_return=-10.0) <= 2

def test_memory_decision_outcome_updates_policy(tmp_path):
    mem = GovernorMemory(tmp_path / "m.json")
    for i in range(MIN_SAMPLES_TO_LEARN + 2):
        did = f"d{i}"
        mem.record_decision(DecisionRecord(
            decision_id=did, timestamp="t", market_regime="neutral",
            policy_action="MODEL_PRIMARY_META", context_key="neutral|dd=0.0",
        ))
        mem.record_outcome(OutcomeRecord(decision_id=did, actual_return=0.02, actual_utility=0.02, timestamp="t"))
    assert mem.policy.outcome_count >= MIN_SAMPLES_TO_LEARN and mem.policy.valid
    assert mem.select_action("neutral|dd=0.0", rng_seed=0) in (
        "MODEL_PRIMARY", "MODEL_PRIMARY_META", "LIGHTWEIGHT_ENSEMBLE", "FULL_ENSEMBLE", "DEFENSIVE",
    )

def test_corrupt_policy_safe_default(tmp_path):
    mem = GovernorMemory(tmp_path / "c.json")
    mem.mark_corrupt()
    assert mem.select_action("neutral") == "DEFENSIVE"

def test_rollback(tmp_path):
    mem = GovernorMemory(tmp_path / "r.json")
    mem.mark_corrupt()
    mem.rollback_policy()
    assert mem.policy.valid

def test_decide_with_memory_records(tmp_path):
    mem = GovernorMemory(tmp_path / "g.json")
    cfg = decide_with_memory(MLGovernor(), mem, regime="neutral", data_ok=True, decision_id="x1")
    assert "x1" in mem.decisions or cfg.decision_id in mem.decisions
    assert "policy:" in cfg.reason

def test_apply_policy_defensive():
    from src.python.governor.governor import GovernorConfig
    cfg = apply_policy_action(GovernorConfig(), "DEFENSIVE")
    assert cfg.strategy == "defensive" and cfg.meta_threshold >= 0.65

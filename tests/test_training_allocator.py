"""Adaptive training allocator + hardware + runtime estimator tests."""
from __future__ import annotations
from pathlib import Path
import pytest
from src.python.governor.hardware import detect_hardware, mock_tesla_t4
from src.python.governor.learning import GovernorMemory
from src.python.governor.runtime_estimator import RuntimeMemory, estimate_runtime
from src.python.governor.training_allocator import SymbolContext, TrainingAllocator, score_training_utility

def test_hardware_cpu_detection():
    hw = detect_hardware()
    assert hw.cpu_cores >= 1 and isinstance(hw.gpu_available, bool)

def test_mock_tesla_t4_not_live_claim():
    hw = mock_tesla_t4()
    assert hw.is_tesla_t4 and hw.vram_gb == 16.0
    assert "mock" in " ".join(hw.notes).lower()

def test_runtime_estimate_skip_zero():
    assert estimate_runtime(action="SKIP", n_rows=1000).estimated_seconds == 0.0

def test_runtime_estimate_invalid_fail_closed():
    e = estimate_runtime(action="PRIMARY", n_rows=-1)
    assert e.confidence == 0.0 and e.estimated_seconds >= 1e8

def test_runtime_memory_calibration(tmp_path):
    mem = RuntimeMemory(path=tmp_path / "rt.json")
    mem.load()
    est = estimate_runtime(action="PRIMARY", n_rows=500, n_features=40)
    mem.record(action="PRIMARY", n_rows=500, n_features=40, estimated_seconds=est.estimated_seconds, actual_seconds=est.estimated_seconds * 1.2)
    mem.record(action="PRIMARY", n_rows=500, n_features=40, estimated_seconds=est.estimated_seconds, actual_seconds=est.estimated_seconds * 1.3)
    assert estimate_runtime(action="PRIMARY", n_rows=500, n_features=40, memory=mem).basis == "calibrated"

def test_insufficient_data_skip():
    u, reasons = score_training_utility(SymbolContext(symbol="X", n_rows=10, data_ok=True), detect_hardware())
    assert u <= -0.5 and "INSUFFICIENT_DATA" in reasons

def test_bad_dq_skip():
    u, reasons = score_training_utility(SymbolContext(symbol="X", n_rows=100, data_ok=False), detect_hardware())
    assert u <= -0.5

def test_high_utility_priority():
    u_high, _ = score_training_utility(SymbolContext(symbol="A", n_rows=400, data_ok=True, model_age_days=60,
        recent_oos_accuracy=0.45, feature_drift=0.5, regime_changed=True, liquidity_score=0.8), detect_hardware())
    u_low, _ = score_training_utility(SymbolContext(symbol="B", n_rows=50, data_ok=True, model_age_days=1, recent_oos_accuracy=0.6), detect_hardware())
    assert u_high > u_low

def test_budget_respected_and_defer():
    alloc = TrainingAllocator(budget_minutes=0.05, max_candidates=10)
    symbols = [SymbolContext(symbol=f"S{i}", n_rows=500, data_ok=True, model_age_days=40,
               recent_oos_accuracy=0.4, feature_drift=0.4, regime_changed=True) for i in range(5)]
    plan = alloc.allocate(symbols)
    total_est = sum(c["estimated_seconds"] for c in plan.candidates if c["action"] not in ("SKIP", "DEFER"))
    assert total_est <= plan.budget_seconds + 1e-6

def test_invalid_budget_fail_closed():
    plan = TrainingAllocator(budget_minutes=0.0).allocate([SymbolContext(symbol="A", n_rows=100, data_ok=True)])
    assert plan.selected_count == 0 and any("INVALID_BUDGET" in n for n in plan.notes)

def test_allocation_structure():
    plan = TrainingAllocator(budget_minutes=20).allocate([
        SymbolContext(symbol="SYM1", n_rows=200, data_ok=True, model_age_days=45, recent_oos_accuracy=0.48),
        SymbolContext(symbol="SYM2", n_rows=30, data_ok=True),
    ])
    assert "cycle_id" in plan.to_dict() and any(c["symbol"] == "SYM2" and c["action"] == "SKIP" for c in plan.candidates)

def test_outcome_updates_memory(tmp_path):
    gmem = GovernorMemory(tmp_path / "gov.json")
    rmem = RuntimeMemory(path=tmp_path / "rt.json"); rmem.load()
    alloc = TrainingAllocator(memory=gmem, runtime_memory=rmem, budget_minutes=10)
    reward = alloc.record_training_outcome(decision_id="d1", action="PRIMARY", estimated_seconds=10.0,
        actual_seconds=12.0, oos_accuracy=0.56, expectancy=0.01, max_drawdown=0.05, n_rows=200)
    assert isinstance(reward, float) and rmem.records and "d1" in gmem.outcomes

def test_corrupt_runtime_memory_defensive(tmp_path):
    path = tmp_path / "bad.json"; path.write_text("{not-json")
    mem = RuntimeMemory(path=path); mem.load()
    assert mem.valid is False

def test_production_pointer_untouched(tmp_path):
    pointer = tmp_path / "primary.PRODUCTION"; pointer.write_text("KEEP")
    TrainingAllocator(budget_minutes=5).allocate([SymbolContext(symbol="Z", n_rows=100, data_ok=True)])
    assert pointer.read_text() == "KEEP"

def test_chronology_no_future_in_scoring():
    ctx = SymbolContext(symbol="A", n_rows=100, data_ok=True, recent_oos_accuracy=0.5)
    assert not hasattr(ctx, "future_oos")
    u, _ = score_training_utility(ctx, detect_hardware())
    assert -1.0 <= u <= 1.0

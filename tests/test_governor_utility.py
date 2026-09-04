from src.python.governor.governor import CandidateScore, MLGovernor, ResourceProfile

def test_high_drawdown_halts():
    g = MLGovernor(resources=ResourceProfile(cpu_cores=8, ram_gb=16, is_high_end=True, max_parallel_models=4))
    cfg = g.decide(regime="neutral", drawdown=-0.12)
    assert cfg.allow_new_trades is False and "dd_halt" in cfg.reason

def test_high_vol_defensive():
    g = MLGovernor(resources=ResourceProfile(cpu_cores=4, ram_gb=8, is_high_end=True, max_parallel_models=2))
    cfg = g.decide(regime="high_vol", drawdown=-0.01)
    assert cfg.risk_budget <= 0.12 and cfg.meta_threshold >= 0.60

def test_low_resource_limits_models():
    g = MLGovernor(resources=ResourceProfile(cpu_cores=1, ram_gb=2.0, is_high_end=False, max_parallel_models=1))
    cfg = g.decide(regime="neutral", model_health={"primary_lgbm": True, "meta_rf": True})
    assert len(cfg.active_models) <= 1

def test_utility_prefers_better_candidate():
    g = MLGovernor(resources=ResourceProfile(cpu_cores=4, ram_gb=8, is_high_end=True, max_parallel_models=2))
    good = CandidateScore(name="good", expected_edge=0.05, reliability=0.8, drawdown=0.02, turnover=0.1)
    bad = CandidateScore(name="bad", expected_edge=-0.02, reliability=0.3, drawdown=0.2, turnover=0.8)
    cfg = g.decide(regime="neutral", candidates=[bad, good])
    assert cfg.strategy == "good"
    assert g.score_utility(good)[0] > g.score_utility(bad)[0]

def test_training_plan_tightens_near_deadline():
    g = MLGovernor(resources=ResourceProfile(is_high_end=True, training_budget_sec=1200, max_parallel_models=4, cpu_cores=8, ram_gb=16))
    g.decide()
    assert g.training_plan(120)["candidate_limit"] == 1
    assert g.training_plan(30)["allow_train"] is False

def test_high_turnover_tightens_threshold():
    cfg = MLGovernor().decide(regime="neutral", turnover=0.8)
    assert "high_turnover" in cfg.reason

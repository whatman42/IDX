"""Final hardening tests."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.python.market.calendar import in_session, is_stale
from src.python.market.corporate_actions import CorporateActionPolicy, CorporateActionRecord, CorporateActionType
from src.python.portfolio.idx_rules import round_lot, round_to_tick, tick_size
from src.python.registry.promotion import evaluate_promotion
from src.python.risk_bridge.rust_client import invoke_rust_risk


class TestCorporateActions:
    def test_stock_split_adjusts_qty_and_cost(self):
        adj = CorporateActionPolicy.apply_to_position(
            100, 1000.0, CorporateActionRecord("BBCA", "2024-06-01", CorporateActionType.STOCK_SPLIT, ratio=2.0))
        assert adj.new_qty == 200 and adj.new_avg_price == 500.0

    def test_dividend_cash(self):
        adj = CorporateActionPolicy.apply_to_position(
            100, 1000.0, CorporateActionRecord("BBCA", "2024-06-01", CorporateActionType.DIVIDEND, cash_component=50.0))
        assert adj.cash_delta == 5000.0

    def test_material_blocks_raw(self):
        blocked = CorporateActionPolicy.material_unresolved(
            ["BBCA"], [CorporateActionRecord("BBCA", "2024-01-01", CorporateActionType.STOCK_SPLIT, ratio=2)], True)
        assert "BBCA" in blocked


class TestPromotionGates:
    def test_reject_perfect_accuracy_without_trading_metrics(self):
        r = evaluate_promotion({"accuracy": 1.0, "n_samples": 100})
        assert r.approved is False

    def test_approve_with_trading_metrics(self):
        r = evaluate_promotion({"accuracy": 0.58, "n_samples": 80, "max_drawdown": -0.08, "expectancy": 0.002})
        assert r.approved is True


class TestIdxRules:
    def test_tick_and_lot(self):
        assert tick_size(150) == 1.0
        assert round_lot(250) == 200


class TestCalendar:
    def test_weekend_not_session(self):
        sun = datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)
        assert in_session(sun) is False

    def test_stale(self):
        assert is_stale(datetime(2020, 1, 1, tzinfo=timezone.utc), max_age_days=5)


class TestRustIntegration:
    @pytest.fixture
    def binary(self):
        for c in [Path("risk_engine/target/release/risk_engine"), Path("risk_engine/target/debug/risk_engine")]:
            if c.exists():
                return str(c)
        pytest.skip("Rust binary not built")

    def test_valid_allow(self, binary, tmp_path):
        d = invoke_rust_risk(
            {"symbol": "BBCA", "side": 1, "primary_probability": 0.6, "meta_probability": 0.70,
             "suggested_size": 0.12, "timestamp": "t", "mode": "paper"},
            {"equity": 1e8, "cash": 1e8, "positions": {}, "daily_pnl_pct": 0, "max_drawdown_pct": 0.01},
            binary=binary, work_dir=tmp_path)
        assert d.allow is True and abs(d.final_weight - 0.12) < 1e-6

    def test_deny_low_meta(self, binary, tmp_path):
        d = invoke_rust_risk(
            {"symbol": "BBCA", "side": 1, "primary_probability": 0.6, "meta_probability": 0.40,
             "suggested_size": 0.12, "timestamp": "t", "mode": "paper"},
            {"equity": 1e8, "cash": 1e8, "positions": {}, "daily_pnl_pct": 0, "max_drawdown_pct": 0.01},
            binary=binary, work_dir=tmp_path)
        assert d.allow is False


class TestTursoAuthority:
    def test_missing_credentials_sqlite_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
        monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
        from src.python.persistence import open_repository, SQLiteStateRepository
        repo = open_repository({"sqlite_path": str(tmp_path / "x.db"), "authority": "auto"})
        assert isinstance(repo, SQLiteStateRepository)

    def test_turso_authority_without_creds_fails(self, monkeypatch):
        monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
        monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
        from src.python.persistence import open_repository
        with pytest.raises(RuntimeError, match="credentials"):
            open_repository({"authority": "turso"})

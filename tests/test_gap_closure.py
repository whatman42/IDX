"""Gap-closure tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.python.market.providers import CSVProvider, RuntimeMode, SyntheticProvider, resolve_provider
from src.python.persistence.repository import SQLiteStateRepository
from src.python.portfolio.engine import PortfolioEngine
from src.python.portfolio.reconcile import reconcile
from src.python.registry.artifacts import load_primary_verified, promote_to_production, save_primary_artifact
from src.python.ml.primary_side import PrimarySideModel
from src.python.features.schema import ALL_FEATURE_COLUMNS
from src.python.risk_bridge.rust_client import invoke_rust_risk
from src.python.notify.telegram import format_message, NotifyEventType
from src.python.observability.events import EventLog, EventType
from src.python.health.diagnostics import probe_health, HealthStatus


class TestProviders:
    def test_synthetic_dev_ok(self):
        p = resolve_provider(RuntimeMode.DEVELOPMENT, {})
        assert p.is_synthetic
        assert p.fetch(["BBCA"]).validate().ok

    def test_production_forbids_silent_synthetic(self):
        with pytest.raises(RuntimeError, match="Synthetic is forbidden"):
            resolve_provider(RuntimeMode.PRODUCTION, {})

    def test_csv_provider(self, tmp_path):
        df = SyntheticProvider(n=30).fetch(["BBCA"]).df
        path = tmp_path / "mkt.csv"
        df.to_csv(path, index=False)
        assert CSVProvider(path).fetch(["BBCA"]).validate().ok


class TestArtifacts:
    def test_checksum_roundtrip(self, tmp_path):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(60, len(ALL_FEATURE_COLUMNS))), columns=ALL_FEATURE_COLUMNS)
        y = (X.iloc[:, 0] > 0).astype(int)
        m = PrimarySideModel(model_version="primary_lgbm_vtest").fit(X, y)
        save_primary_artifact(m, tmp_path)
        loaded = load_primary_verified(tmp_path, m.model_version)
        np.testing.assert_allclose(m.predict_proba(X), loaded.predict_proba(X), atol=1e-8)
        p = tmp_path / f"{m.model_version}.txt"
        p.write_bytes(p.read_bytes() + b"x")
        with pytest.raises(ValueError, match="Checksum"):
            load_primary_verified(tmp_path, m.model_version)

    def test_promote_pointer(self, tmp_path):
        promote_to_production(tmp_path, "primary_lgbm", "v9")
        assert (tmp_path / "primary_lgbm.PRODUCTION").read_text().strip() == "v9"


class TestPersistence:
    def test_idempotent_tx(self, tmp_path):
        repo = SQLiteStateRepository(tmp_path / "t.db")
        tx = {"tx_id": "tx_1", "order_id": "ord_1", "signal_id": "s", "symbol": "BBCA",
              "side": 1, "action": "BUY", "qty": 100, "price": 10, "fee": 1, "cycle_id": "c", "timestamp": "t"}
        assert repo.put_transaction(tx) is True
        assert repo.put_transaction(tx) is False

    def test_outbox(self, tmp_path):
        repo = SQLiteStateRepository(tmp_path / "t.db")
        repo.enqueue_notification("e1", "BUY", {"symbol": "BBCA"})
        assert len(repo.pending_notifications()) == 1
        repo.mark_notification("e1", "SENT")
        assert repo.pending_notifications() == []


class TestRustBridge:
    def test_missing_binary_fail_closed(self):
        d = invoke_rust_risk(
            {"symbol": "BBCA", "side": 1, "primary_probability": 0.6, "meta_probability": 0.7,
             "suggested_size": 0.1, "timestamp": "t", "mode": "paper"},
            {"equity": 1e8, "cash": 1e8, "positions": {}, "daily_pnl_pct": 0, "max_drawdown_pct": 0},
            binary="/nonexistent/risk_engine",
        )
        assert d.allow is False and d.final_weight == 0.0


class TestReconcile:
    def test_clean(self):
        eng = PortfolioEngine(initial_cash=1e7, lot_size=100)
        eng.apply_buy(signal_id_="s1", symbol="BBCA", side=1, price=9000, weight=0.1)
        assert reconcile(eng).ok


class TestNotifyFormat:
    def test_buy_message(self):
        msg = format_message(NotifyEventType.BUY.value, {
            "symbol": "BBCA", "entry_price": 9000, "qty": 100, "side": 1,
            "primary_probability": 0.6, "meta_probability": 0.7, "confidence": 0.7,
            "stop_loss": 8800, "take_profit": 9200, "model_version": "v1",
            "governor_version": "g1", "regime": "HIGH_VOL", "equity": 1e8, "cash": 9e7,
            "signal_id": "sig", "tx_id": "tx", "timestamp": "t",
        })
        assert "IDX BUY" in msg and "BBCA" in msg


class TestObservability:
    def test_event_log(self):
        log = EventLog("cyc1", "abc")
        log.emit(EventType.DATA_LOADED, {"n": 1})
        assert log.events[0]["cycle_id"] == "cyc1"


class TestHealth:
    def test_degraded(self):
        h = probe_health(rust_ok=False, telegram_configured=False)
        assert h.overall == HealthStatus.DEGRADED.value

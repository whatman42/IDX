"""Governor, regime, data quality tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.python.data.quality import validate_ohlcv
from src.python.governor.governor import MLGovernor
from src.python.regime.detector import Regime, detect_regime
from src.python.registry.models import ModelRecord, ModelRegistry, ModelStatus


class TestDataQuality:
    def test_clean_ok(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        df = pd.DataFrame({"timestamp": idx, "symbol": "BBCA", "open": 100.0, "high": 101.0,
                           "low": 99.0, "close": 100.5, "volume": 1e6})
        assert validate_ohlcv(df).ok

    def test_negative_price_fails(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="B"),
                           "symbol": "X", "open": 1, "high": 1, "low": 1, "close": -1, "volume": 1})
        assert not validate_ohlcv(df).ok

    def test_high_lt_low_fails(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="B"),
                           "symbol": "X", "open": 10, "high": 9, "low": 11, "close": 10, "volume": 1})
        assert not validate_ohlcv(df).ok


class TestRegime:
    def test_detect_returns_enum(self):
        close = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1, 50)) + 100)
        assert isinstance(detect_regime(close), Regime)


class TestGovernor:
    def test_high_vol_tightens_threshold(self):
        cfg = MLGovernor().decide(Regime.HIGH_VOL)
        assert cfg.meta_threshold >= 0.60 and cfg.max_position_pct <= 0.10

    def test_data_fail_blocks_trades(self):
        assert MLGovernor().decide(Regime.UNKNOWN, data_ok=False).allow_new_trades is False

    def test_drawdown_brake(self):
        cfg = MLGovernor().decide(Regime.LOW_VOL_TREND, recent_drawdown=-0.08)
        assert "dd_brake" in cfg.reason


class TestRegistry:
    def test_promote_retires_old(self, tmp_path):
        reg = ModelRegistry(tmp_path)
        reg.register(ModelRecord(model_id="primary_lgbm", model_version="v1", model_type="lgbm",
                                 feature_schema_version="1.0.0", status=ModelStatus.PRODUCTION.value))
        reg.register(ModelRecord(model_id="primary_lgbm", model_version="v2", model_type="lgbm",
                                 feature_schema_version="1.0.0", status=ModelStatus.APPROVED.value))
        reg.promote("primary_lgbm", "v2", ModelStatus.PRODUCTION)
        assert reg.get_production("primary_lgbm").model_version == "v2"

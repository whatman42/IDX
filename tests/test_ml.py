"""Tests for Primary Side + Meta-Labeling pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.python.features.schema import ALL_FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from src.python.ml.evaluation import classification_metrics, trading_metrics
from src.python.ml.meta_labeling import MetaLabelGovernor
from src.python.ml.primary_side import PrimarySideModel
from src.python.ml.signal import generate_signals
from src.python.ml.temporal import (
    apply_embargo,
    chronological_split,
    make_purged_split,
    purge_overlapping,
)


def _synthetic_xy(n: int = 120, n_features: int = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols = ALL_FEATURE_COLUMNS if n_features is None else ALL_FEATURE_COLUMNS[:n_features]
    X = pd.DataFrame(rng.normal(0, 1, (n, len(cols))), columns=cols)
    score = X.iloc[:, 0] * 0.8 + X.iloc[:, 1] * 0.3 + rng.normal(0, 0.5, n)
    y = (score > 0).astype(int)
    ts = pd.date_range("2023-01-01", periods=n, freq="B")
    return X, y, ts


class TestTemporal:
    def test_chronological_order(self):
        ts = pd.date_range("2024-01-01", periods=100, freq="D")
        tr, va, te = chronological_split(ts, 0.6, 0.2, 0.2)
        assert tr.max() < va.min()
        assert va.max() < te.min()

    def test_purge_removes_overlap(self):
        n = 50
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
        vb = ts + pd.Timedelta(days=10)
        train_idx = np.arange(0, 30)
        val_idx = np.arange(30, 40)
        purged = purge_overlapping(train_idx, val_idx, ts, vb)
        val_start = ts.iloc[val_idx].min()
        assert (vb.iloc[purged] < val_start).all()

    def test_embargo(self):
        n = 40
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
        train_idx = np.arange(0, 30)
        val_idx = np.arange(30, 40)
        out = apply_embargo(train_idx, val_idx, ts, embargo=pd.Timedelta(days=5))
        val_start = ts.iloc[val_idx].min()
        assert (ts.iloc[out] < val_start - pd.Timedelta(days=5)).all()

    def test_purged_split_structure(self):
        n = 80
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
        vb = ts + pd.Timedelta(days=3)
        split = make_purged_split(ts, vb, embargo=pd.Timedelta(days=2))
        assert len(split.train_idx) > 0
        assert len(split.val_idx) > 0
        assert len(split.test_idx) > 0


class TestPrimarySide:
    def test_fit_predict(self):
        X, y, _ = _synthetic_xy(100)
        m = PrimarySideModel()
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (100,)
        assert (proba >= 0).all() and (proba <= 1).all()
        side = m.predict_side(X)
        assert set(np.unique(side)).issubset({-1, 1})

    def test_deterministic(self):
        X, y, _ = _synthetic_xy(80, seed=7)
        a = PrimarySideModel().fit(X, y)
        b = PrimarySideModel().fit(X, y)
        np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-8)

    def test_save_load(self, tmp_path):
        X, y, _ = _synthetic_xy(60)
        m = PrimarySideModel(model_version="primary_lgbm_vtest")
        m.fit(X, y)
        m.save(tmp_path)
        loaded = PrimarySideModel.load(tmp_path, "primary_lgbm_vtest")
        np.testing.assert_allclose(m.predict_proba(X), loaded.predict_proba(X), atol=1e-8)
        assert loaded.meta.feature_schema_version == FEATURE_SCHEMA_VERSION

    def test_missing_feature_raises(self):
        X, y, _ = _synthetic_xy(40)
        m = PrimarySideModel().fit(X, y)
        bad = X.drop(columns=[ALL_FEATURE_COLUMNS[0]])
        with pytest.raises(ValueError, match="Missing"):
            m.predict_proba(bad)

    def test_empty_raises(self):
        m = PrimarySideModel()
        with pytest.raises(ValueError, match="empty"):
            m.fit(pd.DataFrame(), pd.Series(dtype=float))

    def test_feature_importance(self):
        X, y, _ = _synthetic_xy(80)
        m = PrimarySideModel().fit(X, y)
        imp = m.feature_importance(top_n=5)
        assert len(imp) <= 5

    def test_nan_target_raises(self):
        X, y, _ = _synthetic_xy(30)
        y.iloc[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            PrimarySideModel().fit(X, y)


class TestMetaLabel:
    def test_fit_predict_platt(self):
        X, y, _ = _synthetic_xy(100, seed=1)
        g = MetaLabelGovernor(calibration="platt", threshold=0.55)
        g.fit(X, y)
        proba = g.predict_proba(X)
        assert proba.shape == (100,)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_fit_predict_isotonic(self):
        X, y, _ = _synthetic_xy(100, seed=2)
        g = MetaLabelGovernor(calibration="isotonic", threshold=0.5)
        g.fit(X, y)
        assert g.predict_proba(X).shape == (100,)

    def test_threshold_gate(self):
        X, y, _ = _synthetic_xy(80, seed=3)
        g = MetaLabelGovernor(calibration="platt", threshold=0.9)
        g.fit(X, y)
        accepted = g.accept(X)
        assert accepted.dtype == bool
        assert accepted.sum() <= len(X)

    def test_sizing_clamped(self):
        g = MetaLabelGovernor()
        proba = np.array([0.1, 0.5, 0.9, 1.0])
        w = g.size_from_proba(proba, method="sigmoid", max_weight=0.15)
        assert (w >= 0).all() and (w <= 0.15).all()
        w2 = g.size_from_proba(proba, method="kelly", max_weight=0.10, kelly_fraction=0.5)
        assert (w2 >= 0).all() and (w2 <= 0.10).all()

    def test_save_load(self, tmp_path):
        X, y, _ = _synthetic_xy(70, seed=4)
        g = MetaLabelGovernor(calibration="platt", model_version="meta_rf_vtest")
        g.fit(X, y)
        g.save(tmp_path)
        loaded = MetaLabelGovernor.load(tmp_path, "meta_rf_vtest")
        np.testing.assert_allclose(g.predict_proba(X), loaded.predict_proba(X), atol=1e-7)

    def test_missing_feature_raises(self):
        X, y, _ = _synthetic_xy(40, seed=5)
        g = MetaLabelGovernor().fit(X, y)
        with pytest.raises(ValueError, match="Missing"):
            g.predict_proba(X.drop(columns=[ALL_FEATURE_COLUMNS[0]]))


class TestEvaluation:
    def test_classification_metrics(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.4, 0.3, 0.9])
        m = classification_metrics(y_true, y_pred, y_proba)
        assert "accuracy" in m and "f1" in m and "roc_auc" in m

    def test_trading_metrics(self):
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        accepted = np.array([True, True, False, True, True])
        m = trading_metrics(returns, accepted)
        assert m["n_accepted"] == 4
        assert "hit_rate" in m and "max_drawdown" in m


class TestSignalIntegration:
    def test_end_to_end_signal(self):
        X, y, ts = _synthetic_xy(100, seed=9)
        split = make_purged_split(
            pd.Series(ts),
            vertical_barrier_times=pd.Series(ts) + pd.Timedelta(days=5),
            embargo=pd.Timedelta(days=2),
        )
        Xtr, ytr = X.iloc[split.train_idx], y.iloc[split.train_idx]
        Xva, yva = X.iloc[split.val_idx], y.iloc[split.val_idx]
        Xte = X.iloc[split.test_idx]

        primary = PrimarySideModel().fit(Xtr, ytr, eval_set=(Xva, yva))
        primary_side = primary.predict_side(Xtr)
        y_meta = ((primary_side == 1) == (ytr == 1)).astype(int)
        meta = MetaLabelGovernor(calibration="platt", threshold=0.5)
        meta.fit(Xtr, y_meta)

        signals = generate_signals(
            Xte, primary, meta,
            timestamps=pd.Series(ts).iloc[split.test_idx].reset_index(drop=True),
            symbols=pd.Series(["BBCA"] * len(Xte)),
        )
        for col in ("side", "primary_probability", "meta_probability", "accepted",
                    "suggested_size", "model_version", "feature_schema_version"):
            assert col in signals.columns
        assert signals["feature_schema_version"].iloc[0] == FEATURE_SCHEMA_VERSION
        assert (signals.loc[~signals["accepted"], "suggested_size"] == 0).all()

    def test_no_future_in_train(self):
        X, y, ts = _synthetic_xy(90, seed=11)
        split = make_purged_split(pd.Series(ts), embargo=pd.Timedelta(days=1))
        assert pd.Series(ts).iloc[split.train_idx].max() <= pd.Series(ts).iloc[split.val_idx].min() + pd.Timedelta(days=20)

"""Benchmark adapter tests — no fake GPU metrics."""
from __future__ import annotations
import numpy as np
import pytest
from colab.benchmark_gpu import (
    benchmark_suite, inspect_backends, make_xy, nvidia_smi_query,
    train_lgbm, train_rf_meta, _stats,
)

def test_inspect_backends_primary_default_cpu():
    backends = {b.model_name: b for b in inspect_backends()}
    assert "PRIMARY" in backends
    assert "cpu" in backends["PRIMARY"].cpu_backend.lower() or "CPU" in backends["PRIMARY"].default_configuration

def test_meta_not_gpu():
    backends = {b.model_name: b for b in inspect_backends()}
    meta = backends.get("META / PRIMARY_META")
    assert meta is not None and meta.gpu_support_available is False

def test_nvidia_smi_no_fabricate():
    q = nvidia_smi_query()
    assert q["status"] in ("OK", "UNAVAILABLE")
    if q["status"] == "UNAVAILABLE":
        assert "reason" in q

def test_make_xy_reproducible():
    X1, y1 = make_xy(100, 10, 42)
    X2, y2 = make_xy(100, 10, 42)
    assert np.allclose(X1, X2) and np.array_equal(y1, y2)

def test_stats_median():
    assert _stats([1.0, 2.0, 3.0])["median"] == 2.0

def test_train_lgbm_cpu():
    X, y = make_xy(200, 15, 0)
    r = train_lgbm(X, y, device="cpu", n_estimators=20)
    assert r["device_used"] == "cpu" and r["training_time_sec"] > 0

def test_train_rf_meta_not_applicable():
    X, y = make_xy(150, 10, 1)
    r = train_rf_meta(X, y)
    assert r["status"] == "NOT_APPLICABLE" and r["device_used"] == "cpu"

def test_benchmark_suite_cpu_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts/colab_candidates").mkdir(parents=True)
    report = benchmark_suite(n_rows=300, n_features=20, seed=7, warmup=0, runs=2, n_estimators=30, force_device="cpu")
    assert report["dataset_type"] == "SYNTHETIC"
    assert report["market_performance"] == "UNVERIFIED"
    assert report["production_unchanged"] is True and report["promotion"] is False
    gpu_rows = [b for b in report["benchmarks"] if b["model_name"] == "PRIMARY" and b["device_requested"] == "gpu"]
    assert gpu_rows and gpu_rows[0]["status"] in ("GPU_BLOCKED", "SKIPPED")

def test_gpu_force_without_gpu_graceful(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts/colab_candidates").mkdir(parents=True)
    report = benchmark_suite(n_rows=200, n_features=15, seed=1, warmup=0, runs=1, n_estimators=15, force_device="gpu")
    assert report["production_unchanged"] is True

def test_governor_note_no_force_all_gpu():
    report = benchmark_suite(n_rows=150, n_features=12, seed=2, warmup=0, runs=1, n_estimators=15, force_device="cpu")
    assert report["governor"]["gpu_does_not_force_all_models"] is True

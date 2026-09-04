"""Colab adapter tests — no weakening of frozen core."""
from __future__ import annotations
from pathlib import Path
import pytest
from colab.artifact_export import build_manifest, export_bundle, reject_if_corrupted, verify_bundle
from colab.colab_config import ColabConfig, DatasetType, classify_dataset_type
from colab.environment_check import check_environment, detect_gpu
from colab.colab_train import run_colab_training

def test_env_detection():
    rep = check_environment()
    assert rep.python_version and rep.status in ("PASS", "DEGRADED")

def test_gpu_detect_no_hardcode():
    ok, name, vram, cuda = detect_gpu()
    assert isinstance(ok, bool)

def test_dataset_classification():
    assert classify_dataset_type(None) == DatasetType.SYNTHETIC_DATA
    assert classify_dataset_type("/data/mock_x.csv") == DatasetType.MOCK_DATA
    assert classify_dataset_type("/data/idx_bars.parquet") == DatasetType.REAL_MARKET_DATA

def test_artifact_sha_and_corrupt(tmp_path):
    files = {"model.bin": b"abc", "validation.json": b"{}"}
    man = build_manifest(
        model_version="vtest", repository_commit="abc", dataset_version="SYNTHETIC:1",
        dataset_type="SYNTHETIC_DATA", model_type="lgbm", feature_schema_version="1.0.0",
        validation={"accuracy": 0.5}, risk_metrics={}, governor_policy_version="gov_util_v2",
        hardware={}, dependencies={}, artifact_sha256="",
    )
    bundle = export_bundle(tmp_path, "vtest", files, man)
    ok, reason = verify_bundle(Path(bundle["dir"]))
    assert ok, reason
    (Path(bundle["dir"]) / "model.bin").write_bytes(b"CORRUPTED")
    assert not verify_bundle(Path(bundle["dir"]))[0]
    with pytest.raises(ValueError, match="CORRUPTED"):
        reject_if_corrupted(Path(bundle["dir"]))

def test_manifest_rejects_secret_keys(tmp_path):
    man = build_manifest(
        model_version="v", repository_commit="c", dataset_version="d",
        dataset_type="SYNTHETIC_DATA", model_type="t", feature_schema_version="1",
        validation={}, risk_metrics={}, governor_policy_version="g",
        hardware={}, dependencies={}, artifact_sha256="x",
    )
    man["api_key"] = "SHOULD_NOT"
    with pytest.raises(ValueError, match="secret"):
        export_bundle(tmp_path, "v", {"a.txt": b"1"}, man)

def test_smoke_training_synthetic(tmp_path):
    cfg = ColabConfig(out_dir=str(tmp_path / "out"), n_bars=80,
                      dataset_type=DatasetType.SYNTHETIC_DATA, promote=False,
                      gemini_advisory=False, training_timeout_sec=600)
    report = run_colab_training(cfg)
    assert report.production_unchanged and report.market_performance == "UNVERIFIED"
    assert report.real_idx_data == "BLOCKED"
    statuses = {s.name: s.status for s in report.steps}
    assert statuses.get("Training") == "PASS" and statuses.get("SHA256") == "PASS"
    assert report.artifact_dir and report.approved is False

def test_production_pointer_unchanged(tmp_path):
    prod = tmp_path / "models" / "production"
    prod.mkdir(parents=True)
    pointer = prod / "primary_lgbm.PRODUCTION"
    pointer.write_text("KEEP_ME")
    cfg = ColabConfig(out_dir=str(tmp_path / "c"), n_bars=60, promote=False, gemini_advisory=False)
    run_colab_training(cfg)
    assert pointer.read_text() == "KEEP_ME"

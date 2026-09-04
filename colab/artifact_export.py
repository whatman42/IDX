"""Artifact export + SHA256 + manifest for Colab training outputs."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def build_manifest(
    *,
    model_version: str,
    repository_commit: str,
    dataset_version: str,
    dataset_type: str,
    model_type: str,
    feature_schema_version: str,
    validation: dict,
    risk_metrics: dict,
    governor_policy_version: str,
    hardware: dict,
    dependencies: dict,
    artifact_sha256: str,
    training_timestamp: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "model_version": model_version,
        "repository_commit": repository_commit,
        "dataset_version": dataset_version,
        "dataset_type": dataset_type,
        "model_type": model_type,
        "feature_schema_version": feature_schema_version,
        "training_timestamp": training_timestamp or datetime.now(timezone.utc).isoformat(),
        "validation": validation,
        "risk_metrics": risk_metrics,
        "governor_policy_version": governor_policy_version,
        "hardware": hardware,
        "dependencies": dependencies,
        "artifact_sha256": artifact_sha256,
    }

def export_bundle(out_dir: Path, model_version: str, files: dict, manifest: dict) -> dict:
    banned = ("api_key", "token", "password", "secret", "private_key")
    for k in list(manifest.keys()):
        if any(b in k.lower() for b in banned):
            raise ValueError(f"secret_key_in_manifest:{k}")
    dest = Path(out_dir) / model_version
    dest.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, data in files.items():
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        checksums[name] = sha256_bytes(data)
    agg = hashlib.sha256()
    for name in sorted(checksums):
        agg.update(checksums[name].encode())
    manifest = dict(manifest)
    manifest["artifact_sha256"] = agg.hexdigest()
    manifest["file_checksums"] = checksums
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return {"dir": str(dest), "manifest": manifest, "checksums": checksums}

def verify_bundle(dir_path: Path) -> tuple:
    dir_path = Path(dir_path)
    man_path = dir_path / "manifest.json"
    if not man_path.exists():
        return False, "missing_manifest"
    manifest = json.loads(man_path.read_text())
    checksums = manifest.get("file_checksums", {})
    if not checksums:
        return False, "missing_file_checksums"
    for name, expected in checksums.items():
        p = dir_path / name
        if not p.exists():
            return False, f"missing_file:{name}"
        if sha256_file(p) != expected:
            return False, f"checksum_mismatch:{name}"
    agg = hashlib.sha256()
    for name in sorted(checksums):
        agg.update(checksums[name].encode())
    if agg.hexdigest() != manifest.get("artifact_sha256"):
        return False, "aggregate_sha_mismatch"
    return True, "ok"

def reject_if_corrupted(dir_path: Path) -> None:
    ok, reason = verify_bundle(dir_path)
    if not ok:
        raise ValueError(f"CORRUPTED_ARTIFACT:{reason}")

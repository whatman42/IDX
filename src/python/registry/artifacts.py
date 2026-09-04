"""Production model artifact integrity — checksum, schema, version gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.python.features.schema import FEATURE_SCHEMA_VERSION
from src.python.ml.meta_labeling import MetaLabelGovernor
from src.python.ml.primary_side import PrimarySideModel


@dataclass
class ArtifactManifest:
    model_id: str
    model_version: str
    model_type: str
    feature_schema_version: str
    sha256: str
    git_sha: str = ""
    created_at: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    status: str = "CANDIDATE"
    training_rows: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_primary_artifact(model: PrimarySideModel, directory: Path | str, git_sha: str = "",
                          metrics: Optional[dict] = None) -> ArtifactManifest:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory)
    model_path = directory / f"{model.model_version}.txt"
    manifest = ArtifactManifest(
        model_id="primary_lgbm", model_version=model.model_version, model_type="primary_lgbm",
        feature_schema_version=FEATURE_SCHEMA_VERSION, sha256=file_sha256(model_path),
        git_sha=git_sha, created_at=datetime.now(timezone.utc).isoformat(),
        metrics=metrics or {}, hyperparameters=model.meta.hyperparameters,
        status="CANDIDATE", training_rows=model.meta.training_rows,
    )
    (directory / f"{model.model_version}.manifest.json").write_text(manifest.to_json())
    return manifest


def save_meta_artifact(model: MetaLabelGovernor, directory: Path | str, git_sha: str = "",
                       metrics: Optional[dict] = None) -> ArtifactManifest:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory)
    model_path = directory / f"{model.model_version}.joblib"
    manifest = ArtifactManifest(
        model_id="meta_rf", model_version=model.model_version, model_type="meta_rf",
        feature_schema_version=FEATURE_SCHEMA_VERSION, sha256=file_sha256(model_path),
        git_sha=git_sha, created_at=datetime.now(timezone.utc).isoformat(),
        metrics=metrics or {}, status="CANDIDATE", training_rows=model.meta.training_rows,
    )
    (directory / f"{model.model_version}.manifest.json").write_text(manifest.to_json())
    return manifest


def load_primary_verified(directory: Path | str, model_version: str) -> PrimarySideModel:
    directory = Path(directory)
    model_path = directory / f"{model_version}.txt"
    manifest_path = directory / f"{model_version}.manifest.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Primary artifact missing: {model_path}")
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        actual = file_sha256(model_path)
        if man.get("sha256") and man["sha256"] != actual:
            raise ValueError(f"Checksum mismatch for {model_version}")
        if man.get("feature_schema_version") and man["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("Feature schema mismatch")
    return PrimarySideModel.load(directory, model_version)


def load_meta_verified(directory: Path | str, model_version: str) -> MetaLabelGovernor:
    directory = Path(directory)
    model_path = directory / f"{model_version}.joblib"
    manifest_path = directory / f"{model_version}.manifest.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Meta artifact missing: {model_path}")
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        if man.get("sha256") and man["sha256"] != file_sha256(model_path):
            raise ValueError(f"Checksum mismatch for {model_version}")
        if man.get("feature_schema_version") and man["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("Feature schema mismatch for meta model")
    return MetaLabelGovernor.load(directory, model_version)


def find_production_version(directory: Path | str, model_id: str) -> Optional[str]:
    ptr = Path(directory) / f"{model_id}.PRODUCTION"
    return ptr.read_text().strip() if ptr.exists() else None


def promote_to_production(directory: Path | str, model_id: str, model_version: str) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ptr = directory / f"{model_id}.PRODUCTION"
    tmp = directory / f"{model_id}.PRODUCTION.tmp"
    tmp.write_text(model_version)
    tmp.replace(ptr)

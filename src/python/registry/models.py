"""File-based model registry with promotion lifecycle."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ModelStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


@dataclass
class ModelRecord:
    model_id: str
    model_version: str
    model_type: str
    feature_schema_version: str
    status: str = ModelStatus.CANDIDATE.value
    created_at: str = ""
    git_sha: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_path: str = ""


class ModelRegistry:
    def __init__(self, root: Path | str = "models"):
        self.root = Path(root)
        for sub in ("candidates", "production", "archive"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self._records: dict[str, ModelRecord] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self.index_path.exists():
            for k, v in json.loads(self.index_path.read_text()).items():
                self._records[k] = ModelRecord(**v)

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps({k: asdict(v) for k, v in self._records.items()}, indent=2))

    def register(self, record: ModelRecord) -> ModelRecord:
        if not record.created_at:
            record.created_at = datetime.now(timezone.utc).isoformat()
        self._records[f"{record.model_id}:{record.model_version}"] = record
        self._save_index()
        return record

    def promote(self, model_id: str, model_version: str, to_status: ModelStatus) -> ModelRecord:
        key = f"{model_id}:{model_version}"
        if key not in self._records:
            raise KeyError(key)
        rec = self._records[key]
        if to_status == ModelStatus.PRODUCTION:
            for r in self._records.values():
                if r.model_id == model_id and r.status == ModelStatus.PRODUCTION.value:
                    r.status = ModelStatus.RETIRED.value
                    if r.artifact_path and Path(r.artifact_path).exists():
                        dest = self.root / "archive" / Path(r.artifact_path).name
                        try:
                            shutil.move(r.artifact_path, str(dest))
                            r.artifact_path = str(dest)
                        except Exception:
                            pass
        rec.status = to_status.value
        self._save_index()
        return rec

    def get_production(self, model_id: str) -> Optional[ModelRecord]:
        for r in self._records.values():
            if r.model_id == model_id and r.status == ModelStatus.PRODUCTION.value:
                return r
        return None

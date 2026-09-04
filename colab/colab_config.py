"""Colab training configuration — external research only, not production runtime."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

class DatasetType(str, Enum):
    REAL_MARKET_DATA = "REAL_MARKET_DATA"
    SYNTHETIC_DATA = "SYNTHETIC_DATA"
    MOCK_DATA = "MOCK_DATA"
    DEMO_DATA = "DEMO_DATA"

@dataclass
class ColabConfig:
    repo_url: str = "https://github.com/whatman42/idx.git"
    repo_branch: str = "main"
    expected_freeze_sha: str = "91192e9ac64ce488f60cc9f89bedd2bb5c83d1ce"
    out_dir: str = "artifacts/colab_candidates"
    drive_root: str = "/content/drive/MyDrive/IDX"
    dataset_type: DatasetType = DatasetType.SYNTHETIC_DATA
    dataset_path: Optional[str] = None
    symbols: list = field(default_factory=lambda: ["BBCA"])
    n_bars: int = 120
    seed: int = 7
    min_oos_accuracy: float = 0.52
    promote: bool = False
    training_timeout_sec: int = 1800
    validation_timeout_sec: int = 300
    use_gpu: bool = True
    gemini_advisory: bool = True
    require_real_data_for_promotion: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dataset_type"] = self.dataset_type.value
        return d

def classify_dataset_type(path: Optional[str], explicit: Optional[str] = None) -> DatasetType:
    if explicit:
        return DatasetType(explicit)
    if not path:
        return DatasetType.SYNTHETIC_DATA
    low = path.lower()
    if "mock" in low:
        return DatasetType.MOCK_DATA
    if "demo" in low:
        return DatasetType.DEMO_DATA
    if "synthetic" in low:
        return DatasetType.SYNTHETIC_DATA
    return DatasetType.REAL_MARKET_DATA

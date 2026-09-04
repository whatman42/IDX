"""Dataset provenance — required for REAL_MARKET_DATA claims."""
from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import pandas as pd

class DatasetType(str, Enum):
    REAL_MARKET_DATA = "REAL_MARKET_DATA"
    SYNTHETIC_DATA = "SYNTHETIC_DATA"
    MOCK_DATA = "MOCK_DATA"
    DEMO_DATA = "DEMO_DATA"
    FIXTURE = "FIXTURE"

@dataclass
class DatasetProvenance:
    dataset_type: DatasetType
    source: str
    acquired_at: str
    coverage_start: str
    coverage_end: str
    symbols: list
    row_count: int
    schema_version: str = "ohlcv_v1"
    data_version: str = ""
    data_hash: str = ""
    price_basis: str = "RAW"
    adjustment_status: str = "UNKNOWN"
    corporate_action_available: bool = False
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dataset_type"] = self.dataset_type.value
        return d

    def assert_no_secrets(self) -> None:
        blob = json.dumps(self.to_dict()).lower()
        for bad in ("api_key", "auth_token", "password", "private_key", "bearer "):
            if bad in blob:
                raise ValueError(f"secret_pattern_in_provenance:{bad}")

def hash_dataframe(df: pd.DataFrame) -> str:
    cols = [c for c in ("timestamp", "symbol", "open", "high", "low", "close", "volume") if c in df.columns]
    if not cols:
        cols = list(df.columns)
    payload = df[cols].astype(str).to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def build_provenance_from_df(
    df: pd.DataFrame, *, dataset_type: DatasetType, source: str,
    price_basis: str = "RAW", adjustment_status: str = "UNKNOWN",
    corporate_action_available: bool = False, data_version: str = "",
    notes: Optional[list] = None,
) -> DatasetProvenance:
    work = df.copy()
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"])
        tmin, tmax = str(work["timestamp"].min()), str(work["timestamp"].max())
    else:
        tmin = tmax = "UNKNOWN"
    symbols = sorted(work["symbol"].astype(str).unique().tolist()) if "symbol" in work.columns else []
    h = hash_dataframe(work)
    prov = DatasetProvenance(
        dataset_type=dataset_type, source=source,
        acquired_at=datetime.now(timezone.utc).isoformat(),
        coverage_start=tmin, coverage_end=tmax, symbols=symbols, row_count=len(work),
        data_version=data_version or h[:16], data_hash=h, price_basis=price_basis,
        adjustment_status=adjustment_status, corporate_action_available=corporate_action_available,
        notes=list(notes or []),
    )
    prov.assert_no_secrets()
    return prov

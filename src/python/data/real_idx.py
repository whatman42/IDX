"""REAL IDX data resolution — fail-closed, never silent synthetic fallback.

Environment:
  REAL_IDX_DATA_PATH  — absolute path to licensed CSV or Parquet (not committed to git)

dataset_type REAL_MARKET_DATA is only valid when the caller supplies a real file
and labels it explicitly. Missing path → BLOCKED.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from src.python.data.provenance import DatasetType
from src.python.market.providers import CSVProvider, MarketDataContract, ParquetProvider, PriceBasis

@dataclass
class RealIdxStatus:
    status: str  # PASS | BLOCKED
    path: Optional[str] = None
    reason: str = ""
    format: str = ""  # csv | parquet | ""
    notes: list = field(default_factory=list)
    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "path": self.path, "reason": self.reason,
                "format": self.format, "notes": list(self.notes)}

def resolve_real_idx_path(explicit: Optional[str] = None, env_var: str = "REAL_IDX_DATA_PATH") -> RealIdxStatus:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get(env_var, "").strip()
    if env:
        candidates.append(env)
    if not candidates:
        return RealIdxStatus(status="BLOCKED", reason=f"{env_var}_unset_and_no_explicit_path",
            notes=["Set REAL_IDX_DATA_PATH to a licensed CSV/Parquet outside the git repo"])
    for c in candidates:
        p = Path(c).expanduser()
        if not p.exists():
            continue
        if not p.is_file():
            return RealIdxStatus(status="BLOCKED", path=str(p), reason="path_is_not_file")
        suf = p.suffix.lower()
        if suf not in (".csv", ".parquet", ".pq"):
            return RealIdxStatus(status="BLOCKED", path=str(p), reason=f"unsupported_suffix:{suf}",
                notes=["expected .csv or .parquet"])
        fmt = "parquet" if suf in (".parquet", ".pq") else "csv"
        parts = {x.lower() for x in p.parts}
        if "fixtures" in parts or p.name.startswith("idx_ohlcv_fixture"):
            return RealIdxStatus(status="BLOCKED", path=str(p), reason="fixture_path_cannot_be_real_idx",
                notes=["Fixture files must use dataset_type=FIXTURE, not REAL_MARKET_DATA"])
        return RealIdxStatus(status="PASS", path=str(p.resolve()), format=fmt, reason="file_found")
    return RealIdxStatus(status="BLOCKED", path=candidates[0], reason="file_not_found",
        notes=[f"looked_for={candidates}"])

def load_real_idx_contract(path: Optional[str] = None, symbols: Optional[list[str]] = None):
    st = resolve_real_idx_path(path)
    if st.status != "PASS" or not st.path:
        return st, None
    p = Path(st.path)
    if st.format == "parquet":
        contract = ParquetProvider(p).fetch(symbols or [])
    else:
        contract = CSVProvider(p).fetch(symbols or [])
    contract.source = f"real_idx:{p.name}"
    contract.price_basis = PriceBasis.RAW
    return st, contract

def assert_not_fixture_as_real(dataset_type: DatasetType, path: str) -> None:
    if dataset_type != DatasetType.REAL_MARKET_DATA:
        return
    pl = path.lower()
    if "fixture" in pl or "/tests/" in pl.replace("\\", "/"):
        raise ValueError(
            "REAL_MARKET_DATA cannot be claimed for fixture/test paths. "
            "Use DatasetType.FIXTURE and keep market_performance=UNVERIFIED."
        )

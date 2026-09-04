"""Data Quality Engine — reject untrustworthy OHLCV before trading."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    ok: bool
    issues: list[str] = field(default_factory=list)
    n_rows: int = 0
    symbols: list[str] = field(default_factory=list)

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise ValueError("Data quality failed:\n  - " + "\n  - ".join(self.issues))


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    require_volume: bool = True,
    max_gap_days: float = 10.0,
) -> QualityReport:
    issues: list[str] = []
    if df is None or len(df) == 0:
        return QualityReport(ok=False, issues=["empty dataframe"])

    cols = {c.lower(): c for c in df.columns}
    required = {"open", "high", "low", "close"}
    if require_volume:
        required.add("volume")
    missing = required - set(cols)
    if missing:
        return QualityReport(ok=False, issues=[f"missing columns: {missing}"])

    work = df.rename(columns={cols[k]: k for k in required if k in cols}).copy()
    if "timestamp" in {c.lower() for c in df.columns}:
        ts_col = next(c for c in df.columns if c.lower() == "timestamp")
        work["timestamp"] = pd.to_datetime(df[ts_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        work["timestamp"] = df.index
    else:
        issues.append("no timestamp column or DatetimeIndex")

    if "symbol" in {c.lower() for c in df.columns}:
        sym_col = next(c for c in df.columns if c.lower() == "symbol")
        work["symbol"] = df[sym_col].astype(str)
    else:
        work["symbol"] = "UNKNOWN"

    for col in ("open", "high", "low", "close"):
        s = work[col]
        if s.isna().any():
            issues.append(f"{col} has NaN ({int(s.isna().sum())})")
        if np.isinf(s.to_numpy(dtype=float)).any():
            issues.append(f"{col} has Inf")
        if (s <= 0).any():
            issues.append(f"{col} has non-positive values")

    bad_hl = work["high"] < work["low"]
    if bad_hl.any():
        issues.append(f"high < low on {int(bad_hl.sum())} rows")
    bad_range = (work["high"] < work[["open", "close"]].max(axis=1)) | (
        work["low"] > work[["open", "close"]].min(axis=1)
    )
    if bad_range.any():
        issues.append(f"OHLC envelope broken on {int(bad_range.sum())} rows")

    if require_volume and "volume" in work.columns and (work["volume"] < 0).any():
        issues.append("negative volume")

    if "timestamp" in work.columns:
        dup = work.duplicated(subset=["symbol", "timestamp"], keep=False)
        if dup.any():
            issues.append(f"duplicate (symbol,timestamp) rows: {int(dup.sum())}")
        for sym, g in work.groupby("symbol"):
            g = g.sort_values("timestamp")
            if len(g) < 2:
                continue
            deltas = g["timestamp"].diff().dt.total_seconds().dropna() / 86400.0
            if (deltas > max_gap_days).any():
                issues.append(f"{sym}: gap > {max_gap_days}d detected")

    symbols = sorted(work["symbol"].unique().tolist())
    return QualityReport(ok=len(issues) == 0, issues=issues, n_rows=len(work), symbols=symbols)

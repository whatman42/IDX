"""Multi-symbol / preprocessing leakage guards."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class LeakageReport:
    ok: bool
    issues: list = field(default_factory=list)
    def raise_if_bad(self) -> None:
        if not self.ok:
            raise ValueError("Leakage audit failed:\n  - " + "\n  - ".join(self.issues))

def assert_train_only_fit(train_idx: np.ndarray, full_n: int, *, scaler_fit_on: str) -> LeakageReport:
    issues = []
    if scaler_fit_on.lower() not in ("train", "train_only"):
        issues.append(f"scaler_fit_on={scaler_fit_on} (must be train)")
    if len(train_idx) == 0:
        issues.append("empty train_idx")
    if len(train_idx) > full_n:
        issues.append("train_idx longer than dataset")
    return LeakageReport(ok=len(issues) == 0, issues=issues)

def assert_no_future_features(
    feature_df: pd.DataFrame, timestamps: pd.Series, *,
    symbols: Optional[pd.Series] = None, known_shift_cols: Optional[list] = None,
) -> LeakageReport:
    issues = []
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    if symbols is not None:
        sym = symbols.reset_index(drop=True).astype(str)
        for s, g in pd.DataFrame({"ts": ts, "sym": sym}).groupby("sym"):
            if not g["ts"].is_monotonic_increasing:
                issues.append(f"timestamps_not_sorted_within_symbol:{s}")
    else:
        if not ts.is_monotonic_increasing:
            issues.append("timestamps_not_sorted")
    return LeakageReport(ok=len(issues) == 0, issues=issues)

def assert_split_chronological(timestamps: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray) -> LeakageReport:
    issues = []
    if len(train_idx) == 0 or len(test_idx) == 0:
        return LeakageReport(ok=False, issues=["empty_split"])
    ts = pd.to_datetime(timestamps)
    train_max, test_min = ts.iloc[train_idx].max(), ts.iloc[test_idx].min()
    if train_max >= test_min:
        issues.append(f"train_max={train_max} >= test_min={test_min}")
    return LeakageReport(ok=len(issues) == 0, issues=issues)

def assert_no_cross_sectional_future_norm(train_mask: np.ndarray, values: np.ndarray) -> LeakageReport:
    issues = []
    if train_mask.dtype != bool:
        train_mask = train_mask.astype(bool)
    if not train_mask.any():
        issues.append("no_train_rows_for_norm")
    return LeakageReport(ok=len(issues) == 0, issues=issues)

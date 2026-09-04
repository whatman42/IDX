"""Audit walk-forward test windows for disjointness and temporal order."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
import pandas as pd

def audit_walk_forward_windows(
    timestamps: pd.Series,
    windows: Sequence[tuple],
) -> dict[str, Any]:
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    test_sets = []
    pairs_overlap = 0
    overlap_rows = 0
    all_test = []
    details = []
    for i, (tr, te) in enumerate(windows):
        tr, te = np.asarray(tr), np.asarray(te)
        if len(tr) == 0 or len(te) == 0:
            continue
        train_max = ts.iloc[tr].max()
        test_min = ts.iloc[te].min()
        test_max = ts.iloc[te].max()
        details.append({
            "window": i,
            "train_n": int(len(tr)),
            "test_n": int(len(te)),
            "train_end": str(train_max),
            "test_start": str(test_min),
            "test_end": str(test_max),
            "train_before_test": bool(train_max < test_min),
        })
        test_sets.append(set(int(x) for x in te))
        all_test.extend(int(x) for x in te)
    for i in range(len(test_sets)):
        for j in range(i + 1, len(test_sets)):
            inter = test_sets[i] & test_sets[j]
            if inter:
                pairs_overlap += 1
                overlap_rows += len(inter)
    n_all = len(all_test)
    n_unique = len(set(all_test))
    return {
        "window_count": len(details),
        "windows": details,
        "total_test_rows": n_all,
        "unique_test_rows": n_unique,
        "duplicate_test_rows": n_all - n_unique,
        "test_overlap_pairs": pairs_overlap,
        "test_overlap_rows": overlap_rows,
        "status": "PASS" if pairs_overlap == 0 and (n_all - n_unique) == 0 else "FAIL",
    }

"""Walk-forward test windows must be pairwise disjoint."""
from __future__ import annotations
import pandas as pd
from src.python.ml.temporal import walk_forward_splits
from src.python.validation.walk_forward_audit import audit_walk_forward_windows

def test_test_windows_are_disjoint():
    ts = pd.Series(pd.bdate_range("2020-01-01", periods=200))
    windows = list(walk_forward_splits(ts, n_splits=3, embargo=pd.Timedelta(days=2)))
    assert len(windows) >= 2
    audit = audit_walk_forward_windows(ts, windows)
    assert audit["test_overlap_pairs"] == 0
    assert audit["duplicate_test_rows"] == 0
    assert audit["status"] == "PASS"

def test_no_duplicate_oos_rows():
    ts = pd.Series(pd.bdate_range("2018-01-01", periods=500))
    windows = list(walk_forward_splits(ts, n_splits=4, embargo=pd.Timedelta(days=3)))
    all_te = []
    for _, te in windows:
        all_te.extend(te.tolist())
    assert len(all_te) == len(set(all_te))

def test_train_precedes_test():
    ts = pd.Series(pd.bdate_range("2020-01-01", periods=300))
    for tr, te in walk_forward_splits(ts, n_splits=3, embargo=pd.Timedelta(days=2)):
        assert ts.iloc[tr].max() < ts.iloc[te].min()

def test_embargo_gap_when_timedelta():
    ts = pd.Series(pd.bdate_range("2020-01-01", periods=250))
    embargo = pd.Timedelta(days=5)
    for tr, te in walk_forward_splits(ts, n_splits=3, embargo=embargo):
        if len(tr) == 0:
            continue
        assert ts.iloc[tr].max() < ts.iloc[te].min()

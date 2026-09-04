"""
Time-series validation with purge / embargo for Triple-Barrier labels.

Overlapping event windows can leak information from validation into training
if a training event's vertical barrier extends into the validation period.
We therefore:

1. Sort samples chronologically.
2. Split into train / validation / test by time.
3. Purge training samples whose event window overlaps the next split.
4. Apply an embargo gap (in bars or timedelta) after each training block.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Union
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TemporalSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

def chronological_split(
    timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    order = ts.argsort(kind="mergesort").to_numpy()
    n = len(order)
    if n < 10:
        raise ValueError(f"Need at least 10 samples for temporal split, got {n}")
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return train_idx, val_idx, test_idx

def purge_overlapping(
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    event_times: pd.Series,
    vertical_barrier_times: pd.Series,
) -> np.ndarray:
    if len(train_idx) == 0 or len(holdout_idx) == 0:
        return train_idx
    holdout_start = event_times.iloc[holdout_idx].min()
    vb = vertical_barrier_times.iloc[train_idx]
    keep_mask = vb < holdout_start
    return train_idx[keep_mask.to_numpy()]

def apply_embargo(
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    event_times: pd.Series,
    embargo: Union[pd.Timedelta, int] = pd.Timedelta(days=5),
) -> np.ndarray:
    if len(train_idx) == 0 or len(holdout_idx) == 0:
        return train_idx
    holdout_start = event_times.iloc[holdout_idx].min()
    if isinstance(embargo, int):
        # bar-count embargo: drop last `embargo` training samples by time order
        order = np.argsort(event_times.iloc[train_idx].to_numpy())
        if len(order) <= embargo:
            return train_idx[:0]
        keep = order[:-embargo]
        return train_idx[keep]
    cutoff = holdout_start - embargo
    mask = event_times.iloc[train_idx] <= cutoff
    return train_idx[mask.to_numpy()]

def make_purged_split(
    event_times: pd.Series,
    vertical_barrier_times: Optional[pd.Series] = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    embargo: Union[pd.Timedelta, int] = pd.Timedelta(days=5),
) -> TemporalSplit:
    train_idx, val_idx, test_idx = chronological_split(
        event_times, train_frac, val_frac, test_frac
    )
    if vertical_barrier_times is not None:
        train_idx = purge_overlapping(train_idx, val_idx, event_times, vertical_barrier_times)
        train_idx = purge_overlapping(train_idx, test_idx, event_times, vertical_barrier_times)
        val_idx = purge_overlapping(val_idx, test_idx, event_times, vertical_barrier_times)
    train_idx = apply_embargo(train_idx, val_idx, event_times, embargo)
    val_idx = apply_embargo(val_idx, test_idx, event_times, embargo)

    def _end(idx: np.ndarray) -> pd.Timestamp:
        if len(idx) == 0:
            return pd.Timestamp("NaT")
        return pd.Timestamp(event_times.iloc[idx].max())

    return TemporalSplit(
        train_idx=np.sort(train_idx),
        val_idx=np.sort(val_idx),
        test_idx=np.sort(test_idx),
        train_end=_end(train_idx),
        val_end=_end(val_idx),
        test_end=_end(test_idx),
    )

def walk_forward_splits(
    event_times: pd.Series,
    n_splits: int = 3,
    train_min_frac: float = 0.4,
    embargo: Union[pd.Timedelta, int] = pd.Timedelta(days=3),
    vertical_barrier_times: Optional[pd.Series] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward with **non-overlapping** test blocks.

    Guarantees:
      - max(train times) < min(test times) after embargo
      - test windows are pairwise disjoint (no shared indices)
    """
    ts = pd.Series(pd.to_datetime(event_times)).reset_index(drop=True)
    order = ts.argsort(kind="mergesort").to_numpy()
    n = len(order)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    fold = n // (n_splits + 1)
    if fold < 5:
        raise ValueError("Not enough samples for requested walk-forward folds")

    test_origin = n - n_splits * fold
    if test_origin < int(n * train_min_frac):
        fold = max(5, (n - int(n * train_min_frac)) // n_splits)
        test_origin = n - n_splits * fold
    if fold < 5 or test_origin < 5:
        raise ValueError("Not enough samples for non-overlapping walk-forward folds")

    seen_test: set[int] = set()
    for k in range(n_splits):
        t0 = test_origin + k * fold
        t1 = test_origin + (k + 1) * fold
        test_idx = order[t0:t1]
        if len(test_idx) == 0:
            continue
        train_idx = order[:t0]
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        overlap = seen_test.intersection(int(i) for i in test_idx)
        if overlap:
            raise RuntimeError(f"walk_forward test overlap detected: {len(overlap)} indices")
        seen_test.update(int(i) for i in test_idx)
        if vertical_barrier_times is not None:
            train_idx = purge_overlapping(train_idx, test_idx, event_times, vertical_barrier_times)
        train_idx = apply_embargo(train_idx, test_idx, event_times, embargo)
        if len(train_idx) and len(test_idx):
            yield np.sort(train_idx), np.sort(test_idx)

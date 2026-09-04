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
    """Simple chronological 3-way split. No shuffle."""
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
    """Remove training samples whose vertical barrier overlaps any holdout event."""
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
    """Drop training samples within embargo before the holdout window."""
    if len(train_idx) == 0 or len(holdout_idx) == 0:
        return train_idx

    holdout_start = event_times.iloc[holdout_idx].min()

    if isinstance(embargo, (int, np.integer)):
        ordered = np.sort(train_idx)
        before = ordered[event_times.iloc[ordered] < holdout_start]
        if len(before) <= embargo:
            return np.array([], dtype=int)
        return before[:-int(embargo)]

    cutoff = holdout_start - embargo
    mask = event_times.iloc[train_idx] < cutoff
    return train_idx[mask.to_numpy()]


def make_purged_split(
    event_times: pd.Series,
    vertical_barrier_times: Optional[pd.Series] = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    embargo: Union[pd.Timedelta, int] = pd.Timedelta(days=5),
) -> TemporalSplit:
    """Full train/val/test split with purge + embargo."""
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
    """Expanding-window walk-forward splits."""
    ts = pd.Series(pd.to_datetime(event_times)).reset_index(drop=True)
    order = ts.argsort(kind="mergesort").to_numpy()
    n = len(order)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    fold = n // (n_splits + 1)
    if fold < 5:
        raise ValueError("Not enough samples for requested walk-forward folds")

    for i in range(1, n_splits + 1):
        train_end = fold * (i + 1) if i < n_splits else n - fold
        train_end = max(train_end, int(n * train_min_frac))
        train_idx = order[:train_end]
        test_idx = order[train_end : train_end + fold] if i < n_splits else order[train_end:]
        if vertical_barrier_times is not None:
            train_idx = purge_overlapping(train_idx, test_idx, event_times, vertical_barrier_times)
        train_idx = apply_embargo(train_idx, test_idx, event_times, embargo)
        if len(train_idx) and len(test_idx):
            yield np.sort(train_idx), np.sort(test_idx)

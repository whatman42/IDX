"""
Triple-Barrier Method (TBM) — production-grade vectorized implementation.

Barriers
--------
- Upper  : Profit-Taking (PT)
- Lower  : Stop-Loss (SL)
- Vertical: Time expiration (molecule / horizon)

No static time-based labeling. Barriers are evaluated on OHLC paths
with explicit same-candle collision policy.

Causality
---------
All barrier lookups start strictly *after* the event timestamp.
Volatility / target widths may only use information available at
event time (caller responsibility for pre-computed targets).

Same-candle collision policy (configurable)
-------------------------------------------
When HIGH touches upper *and* LOW touches lower inside the same bar:

  conservative   → stop-loss wins (default, safer for simulation)
  optimistic     → take-profit wins
  stop_first     → same as conservative
  take_profit_first → same as optimistic

Labels
------
  +1  profit-taking barrier hit
  -1  stop-loss barrier hit
   0  vertical barrier (or neutral / no decisive move)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

CollisionPolicy = Literal[
    "conservative",
    "optimistic",
    "stop_first",
    "take_profit_first",
]

BarrierHit = Literal["pt", "sl", "vertical", "none"]


class Side(int, Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_prices(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    cols = {c.lower(): c for c in df.columns}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"OHLC DataFrame missing columns: {missing}")

    for col in ("open", "high", "low", "close"):
        real = cols[col]
        s = df[real]
        if s.isna().all():
            raise ValueError(f"Column '{real}' is entirely NaN")
        if (s <= 0).any():
            bad = s[s <= 0]
            raise ValueError(
                f"Non-positive prices detected in '{real}' "
                f"(count={len(bad)}, min={bad.min()})"
            )
        if np.isinf(s.to_numpy(dtype=float)).any():
            raise ValueError(f"Inf values detected in '{real}'")


def _ensure_sorted_unique_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    if not idx.is_monotonic_increasing:
        idx = idx.sort_values()
    if idx.has_duplicates:
        # keep first occurrence deterministically
        idx = idx[~idx.duplicated(keep="first")]
    return idx


def _normalize_ohlc(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with lower-case OHLC columns and clean index."""
    colmap = {c: c.lower() for c in prices.columns}
    df = prices.rename(columns=colmap)
    # Validate required columns first (before slicing)
    _validate_prices(df)
    df = df[["open", "high", "low", "close"]].copy()
    # Drop duplicate timestamps deterministically (keep first)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# Core barrier search
# ---------------------------------------------------------------------------

def _first_barrier_hit(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_price: float,
    pt: float,
    sl: float,
    side: int,
    collision: CollisionPolicy,
) -> Tuple[int, str, float]:
    """
    Scan a contiguous price path and return the first barrier hit.

    Returns
    -------
    (offset, barrier_type, exit_price)
        offset is relative index into the path arrays;
        -1 means vertical (path exhausted without PT/SL).
    """
    if side == 0 or entry_price <= 0 or pt <= 0 or sl <= 0:
        return -1, "vertical", float(close[-1]) if len(close) else entry_price

    if side > 0:  # LONG
        upper = entry_price * (1.0 + pt)
        lower = entry_price * (1.0 - sl)
    else:  # SHORT
        upper = entry_price * (1.0 + sl)   # stop for short
        lower = entry_price * (1.0 - pt)   # take-profit for short

    n = len(high)
    for i in range(n):
        hit_upper = high[i] >= upper
        hit_lower = low[i] <= lower

        if hit_upper and hit_lower:
            # same-candle collision
            if collision in ("conservative", "stop_first"):
                if side > 0:
                    return i, "sl", lower
                else:
                    return i, "sl", upper
            else:  # optimistic / take_profit_first
                if side > 0:
                    return i, "pt", upper
                else:
                    return i, "pt", lower

        if hit_upper:
            if side > 0:
                return i, "pt", upper
            else:
                return i, "sl", upper

        if hit_lower:
            if side > 0:
                return i, "sl", lower
            else:
                return i, "pt", lower

    # vertical barrier
    return -1, "vertical", float(close[-1])


def _label_from_barrier(barrier: str) -> int:
    if barrier == "pt":
        return 1
    if barrier == "sl":
        return -1
    return 0


def _empty_result() -> pd.DataFrame:
    cols = [
        "event_time",
        "vertical_barrier_time",
        "entry_price",
        "side",
        "pt",
        "sl",
        "barrier_hit",
        "label",
        "exit_time",
        "exit_price",
        "return",
    ]
    return pd.DataFrame(columns=cols).set_index(
        pd.DatetimeIndex([], name="event_time")
    )


def _vertical_only_record(ev, entry, side, pt, sl):
    return {
        "event_time": ev,
        "vertical_barrier_time": ev,
        "entry_price": entry if not (isinstance(entry, float) and np.isnan(entry)) else np.nan,
        "side": int(side),
        "pt": float(pt),
        "sl": float(sl),
        "barrier_hit": "vertical",
        "label": 0,
        "exit_time": ev,
        "exit_price": entry if not (isinstance(entry, float) and np.isnan(entry)) else np.nan,
        "return": 0.0,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_triple_barrier(
    prices: pd.DataFrame,
    events: Union[pd.DatetimeIndex, pd.Series, pd.Index],
    pt_sl: Tuple[float, float] = (0.02, 0.01),
    molecule: Optional[Union[pd.Timedelta, int]] = None,
    side: Optional[Union[pd.Series, int]] = None,
    trgt: Optional[pd.Series] = None,
    min_ret: float = 0.0,
    collision_policy: CollisionPolicy = "conservative",
    vertical_barrier_bars: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply the Triple-Barrier Method to a set of events.

    Parameters
    ----------
    prices : pd.DataFrame
        Must contain OHLC columns (case-insensitive). Index must be
        DatetimeIndex (timezone-aware or naive — consistency required).
    events : DatetimeIndex | Series | Index
        Event timestamps (entry points). Must lie within the price index.
    pt_sl : (pt, sl)
        Profit-taking and stop-loss widths expressed as fractions of
        entry price (e.g. 0.02 = 2 %).
    molecule : Timedelta | int | None
        Vertical barrier horizon.  If int → number of bars.
        If Timedelta → calendar horizon.  Default 5 bars.
    side : Series | int | None
        Position side for each event (+1 long, -1 short).
        Default = +1 (long).
    trgt : Series | None
        Per-event target width (fraction). When provided, overrides
        the pt component of pt_sl; sl is taken from pt_sl[1].
    min_ret : float
        Minimum absolute return required to accept a PT/SL label.
        Below this threshold the label is forced to 0 (vertical).
    collision_policy : str
        See module docstring. Default "conservative".
    vertical_barrier_bars : int | None
        Alternative to `molecule` when an integer bar count is preferred.

    Returns
    -------
    pd.DataFrame
        Indexed by event time with columns:

        event_time, vertical_barrier_time, entry_price, side,
        pt, sl, barrier_hit, label, exit_time, exit_price, return
    """
    if collision_policy not in (
        "conservative",
        "optimistic",
        "stop_first",
        "take_profit_first",
    ):
        raise ValueError(f"Unknown collision_policy: {collision_policy}")

    ohlc = _normalize_ohlc(prices)
    price_index = ohlc.index

    # --- normalise events -------------------------------------------------
    if isinstance(events, pd.Series):
        event_times = pd.DatetimeIndex(events.index)
    else:
        event_times = pd.DatetimeIndex(events)

    event_times = _ensure_sorted_unique_index(event_times)

    # drop events that fall outside the price series
    mask = (event_times >= price_index[0]) & (event_times <= price_index[-1])
    event_times = event_times[mask]
    if len(event_times) == 0:
        return _empty_result()

    # --- side series ------------------------------------------------------
    if side is None:
        side_s = pd.Series(1, index=event_times, dtype=int)
    elif isinstance(side, (int, np.integer)):
        side_s = pd.Series(int(side), index=event_times, dtype=int)
    else:
        side_s = pd.Series(side).reindex(event_times).fillna(0).astype(int)

    # --- target widths ----------------------------------------------------
    pt_default, sl_default = float(pt_sl[0]), float(pt_sl[1])
    if pt_default <= 0 or sl_default <= 0:
        raise ValueError(f"pt_sl must be positive, got {pt_sl}")

    if trgt is not None:
        trgt = pd.Series(trgt).reindex(event_times)
        pt_s = trgt.fillna(pt_default)
        sl_s = pd.Series(sl_default, index=event_times)
    else:
        pt_s = pd.Series(pt_default, index=event_times)
        sl_s = pd.Series(sl_default, index=event_times)

    # --- vertical barrier horizon ----------------------------------------
    if vertical_barrier_bars is not None:
        if vertical_barrier_bars <= 0:
            raise ValueError("vertical_barrier_bars must be positive")
        horizon_bars = int(vertical_barrier_bars)
        use_timedelta = False
    elif molecule is None:
        horizon_bars = 5
        use_timedelta = False
    elif isinstance(molecule, (int, np.integer)):
        if molecule <= 0:
            raise ValueError("molecule (bars) must be positive")
        horizon_bars = int(molecule)
        use_timedelta = False
    else:
        if molecule <= pd.Timedelta(0):
            raise ValueError("molecule Timedelta must be positive")
        use_timedelta = True
        horizon_td = molecule

    # --- locate event positions (causal: first bar *after* event) --------
    positions = price_index.searchsorted(event_times, side="right")

    high_arr = ohlc["high"].to_numpy(dtype=float)
    low_arr = ohlc["low"].to_numpy(dtype=float)
    close_arr = ohlc["close"].to_numpy(dtype=float)
    open_arr = ohlc["open"].to_numpy(dtype=float)
    times_arr = price_index.to_numpy()

    records = []
    n_prices = len(price_index)

    for i, ev in enumerate(event_times):
        pos = int(positions[i])
        if pos >= n_prices:
            records.append(
                _vertical_only_record(ev, np.nan, side_s.iloc[i], pt_s.iloc[i], sl_s.iloc[i])
            )
            continue

        entry_price = float(open_arr[pos])

        if use_timedelta:
            vertical_ts = ev + horizon_td
            end_pos = int(price_index.searchsorted(vertical_ts, side="right"))
        else:
            end_pos = min(pos + horizon_bars, n_prices)

        if end_pos <= pos:
            records.append(
                _vertical_only_record(ev, entry_price, side_s.iloc[i], pt_s.iloc[i], sl_s.iloc[i])
            )
            continue

        path_high = high_arr[pos:end_pos]
        path_low = low_arr[pos:end_pos]
        path_close = close_arr[pos:end_pos]
        path_times = times_arr[pos:end_pos]

        offset, barrier, exit_px = _first_barrier_hit(
            path_high,
            path_low,
            path_close,
            entry_price,
            float(pt_s.iloc[i]),
            float(sl_s.iloc[i]),
            int(side_s.iloc[i]),
            collision_policy,
        )

        if offset < 0:
            exit_time = path_times[-1]
            exit_price = float(path_close[-1])
            barrier = "vertical"
            label = 0
        else:
            exit_time = path_times[offset]
            exit_price = exit_px
            label = _label_from_barrier(barrier)

        raw_ret = (exit_price / entry_price - 1.0) * int(side_s.iloc[i])
        if abs(raw_ret) < min_ret and barrier != "vertical":
            label = 0
            barrier = "vertical"

        if use_timedelta:
            vertical_barrier_time = min(ev + horizon_td, price_index[-1])
        else:
            vb_idx = min(pos + horizon_bars - 1, n_prices - 1)
            vertical_barrier_time = times_arr[vb_idx]

        records.append(
            {
                "event_time": ev,
                "vertical_barrier_time": vertical_barrier_time,
                "entry_price": entry_price,
                "side": int(side_s.iloc[i]),
                "pt": float(pt_s.iloc[i]),
                "sl": float(sl_s.iloc[i]),
                "barrier_hit": barrier,
                "label": label,
                "exit_time": pd.Timestamp(exit_time),
                "exit_price": exit_price,
                "return": raw_ret,
            }
        )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return _empty_result()
    out = out.set_index("event_time", drop=False)
    out.index.name = "event_time"
    return out


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: Tuple[float, float],
    molecule: Union[pd.Timedelta, int],
    min_ret: float = 0.0,
    side: Optional[pd.Series] = None,
    prices: Optional[pd.DataFrame] = None,
    collision_policy: CollisionPolicy = "conservative",
) -> pd.DataFrame:
    """
    Convenience wrapper matching common quant-literature API.

    If only `close` is supplied, a synthetic OHLC is built (high=low=close)
    so the function remains usable, but full OHLC is strongly preferred.
    """
    if prices is None:
        prices = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
            },
            index=close.index,
        )
    return apply_triple_barrier(
        prices=prices,
        events=t_events,
        pt_sl=pt_sl,
        molecule=molecule,
        side=side,
        min_ret=min_ret,
        collision_policy=collision_policy,
    )


def get_bins(
    events: pd.DataFrame,
    close: pd.Series = None,
) -> pd.DataFrame:
    """
    Legacy helper — returns the label column aligned to events.
    Kept for pipeline compatibility.
    """
    if "label" in events.columns:
        return events[["label"]].copy()
    raise ValueError("events DataFrame does not contain 'label' column")

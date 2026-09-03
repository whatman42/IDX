"""
Comprehensive tests for Triple-Barrier Method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.python.labeling.triple_barrier import (
    apply_triple_barrier,
    get_events,
    get_bins,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlc(
    n: int = 30,
    start: str = "2024-01-01",
    freq: str = "B",
    base: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    close = base + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.2, n)
    # ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=idx
    )


def _deterministic_path(
    entry: float = 100.0,
    moves: list[tuple[float, float, float, float]] | None = None,
) -> pd.DataFrame:
    """
    Build a fully deterministic OHLC path.
    moves: list of (o, h, l, c) relative offsets from entry.
    """
    if moves is None:
        moves = [
            (0.0, 0.5, -0.3, 0.2),
            (0.2, 1.5, 0.0, 1.2),   # will hit +1% PT if pt=0.01
            (1.2, 1.3, 1.0, 1.1),
        ]
    idx = pd.date_range("2024-06-01", periods=len(moves) + 1, freq="D")
    rows = [{"open": entry, "high": entry, "low": entry, "close": entry}]
    for o, h, l, c in moves:
        rows.append(
            {
                "open": entry + o,
                "high": entry + h,
                "low": entry + l,
                "close": entry + c,
            }
        )
    return pd.DataFrame(rows, index=idx)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestBasicTP:
    def test_long_take_profit(self):
        # Path: after event, next bar high goes +2%
        prices = _deterministic_path(
            100.0,
            [
                (0.0, 0.5, -0.2, 0.1),
                (0.1, 2.5, 0.0, 2.0),  # high +2.5% → PT
            ],
        )
        events = pd.DatetimeIndex([prices.index[0]])
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.02, 0.01), molecule=5, side=1
        )
        assert len(out) == 1
        assert out.iloc[0]["barrier_hit"] == "pt"
        assert out.iloc[0]["label"] == 1
        assert out.iloc[0]["side"] == 1
        assert out.iloc[0]["exit_price"] == pytest.approx(102.0)

    def test_short_take_profit(self):
        # Short: PT when price drops
        prices = _deterministic_path(
            100.0,
            [
                (0.0, 0.3, -0.5, -0.2),
                (-0.2, 0.0, -2.5, -2.0),  # low -2.5% → PT for short
            ],
        )
        events = pd.DatetimeIndex([prices.index[0]])
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.02, 0.01), molecule=5, side=-1
        )
        assert out.iloc[0]["barrier_hit"] == "pt"
        assert out.iloc[0]["label"] == 1
        assert out.iloc[0]["side"] == -1


class TestBasicSL:
    def test_long_stop_loss(self):
        prices = _deterministic_path(
            100.0,
            [
                (0.0, 0.3, -0.5, -0.2),
                (-0.2, 0.0, -1.5, -1.2),  # low -1.5% → SL (sl=0.01)
            ],
        )
        events = pd.DatetimeIndex([prices.index[0]])
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.02, 0.01), molecule=5, side=1
        )
        assert out.iloc[0]["barrier_hit"] == "sl"
        assert out.iloc[0]["label"] == -1

    def test_short_stop_loss(self):
        prices = _deterministic_path(
            100.0,
            [
                (0.0, 0.5, -0.2, 0.2),
                (0.2, 1.5, 0.0, 1.2),  # high +1.5% → SL for short
            ],
        )
        events = pd.DatetimeIndex([prices.index[0]])
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.02, 0.01), molecule=5, side=-1
        )
        assert out.iloc[0]["barrier_hit"] == "sl"
        assert out.iloc[0]["label"] == -1


class TestVerticalBarrier:
    def test_vertical_when_no_hit(self):
        prices = _deterministic_path(
            100.0,
            [
                (0.0, 0.3, -0.2, 0.1),
                (0.1, 0.4, -0.1, 0.2),
                (0.2, 0.5, 0.0, 0.3),
            ],
        )
        events = pd.DatetimeIndex([prices.index[0]])
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.05, 0.05), molecule=3, side=1
        )
        assert out.iloc[0]["barrier_hit"] == "vertical"
        assert out.iloc[0]["label"] == 0

    def test_vertical_bars_param(self):
        prices = _make_ohlc(20)
        events = pd.DatetimeIndex([prices.index[2]])
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.10, 0.10), vertical_barrier_bars=3, side=1
        )
        assert len(out) == 1
        # exit should be within 3 bars
        delta = (out.iloc[0]["exit_time"] - out.iloc[0]["event_time"]).days
        assert delta <= 5  # business days tolerance


# ---------------------------------------------------------------------------
# Same-candle collision
# ---------------------------------------------------------------------------

class TestCollision:
    def _collision_path(self):
        # Single bar after entry that hits both +2% and -2%
        entry = 100.0
        idx = pd.date_range("2024-07-01", periods=2, freq="D")
        return pd.DataFrame(
            {
                "open": [entry, entry],
                "high": [entry, entry + 3.0],
                "low": [entry, entry - 3.0],
                "close": [entry, entry],
            },
            index=idx,
        )

    def test_conservative_sl_wins(self):
        prices = self._collision_path()
        events = pd.DatetimeIndex([prices.index[0]])
        out = apply_triple_barrier(
            prices,
            events,
            pt_sl=(0.02, 0.02),
            molecule=5,
            side=1,
            collision_policy="conservative",
        )
        assert out.iloc[0]["barrier_hit"] == "sl"
        assert out.iloc[0]["label"] == -1

    def test_optimistic_pt_wins(self):
        prices = self._collision_path()
        events = pd.DatetimeIndex([prices.index[0]])
        out = apply_triple_barrier(
            prices,
            events,
            pt_sl=(0.02, 0.02),
            molecule=5,
            side=1,
            collision_policy="optimistic",
        )
        assert out.iloc[0]["barrier_hit"] == "pt"
        assert out.iloc[0]["label"] == 1

    def test_stop_first_alias(self):
        prices = self._collision_path()
        events = pd.DatetimeIndex([prices.index[0]])
        out = apply_triple_barrier(
            prices,
            events,
            pt_sl=(0.02, 0.02),
            molecule=5,
            side=1,
            collision_policy="stop_first",
        )
        assert out.iloc[0]["barrier_hit"] == "sl"


# ---------------------------------------------------------------------------
# Look-ahead / causality
# ---------------------------------------------------------------------------

class TestNoLookAhead:
    def test_barrier_starts_after_event(self):
        """Entry bar itself must not be used for barrier detection."""
        entry = 100.0
        idx = pd.date_range("2024-08-01", periods=3, freq="D")
        # Event bar has a huge high that would hit PT — must be ignored
        prices = pd.DataFrame(
            {
                "open": [entry, entry, entry],
                "high": [entry + 10, entry + 0.1, entry + 0.1],
                "low": [entry - 0.1, entry - 0.1, entry - 0.1],
                "close": [entry, entry, entry],
            },
            index=idx,
        )
        events = pd.DatetimeIndex([idx[0]])
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.02, 0.02), molecule=5, side=1
        )
        # Should NOT hit PT from the event bar itself
        assert out.iloc[0]["barrier_hit"] == "vertical"
        assert out.iloc[0]["label"] == 0

    def test_exit_time_after_event(self):
        prices = _make_ohlc(15)
        events = prices.index[1:5]
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.01, 0.01), molecule=5, side=1
        )
        for _, row in out.iterrows():
            assert row["exit_time"] > row["event_time"] or row["barrier_hit"] == "vertical"


# ---------------------------------------------------------------------------
# Molecule equivalence
# ---------------------------------------------------------------------------

class TestMoleculeEquivalence:
    def test_full_vs_split_molecules(self):
        prices = _make_ohlc(40, seed=7)
        all_events = prices.index[2:30:3]
        full = apply_triple_barrier(
            prices, all_events, pt_sl=(0.015, 0.01), molecule=5, side=1
        )
        # split into two molecules
        mid = len(all_events) // 2
        part1 = apply_triple_barrier(
            prices, all_events[:mid], pt_sl=(0.015, 0.01), molecule=5, side=1
        )
        part2 = apply_triple_barrier(
            prices, all_events[mid:], pt_sl=(0.015, 0.01), molecule=5, side=1
        )
        combined = pd.concat([part1, part2]).sort_index()
        # Compare key columns
        cols = ["label", "barrier_hit", "exit_price", "return"]
        assert_frame_equal(
            full[cols].sort_index(),
            combined[cols].sort_index(),
            check_dtype=False,
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_events(self):
        prices = _make_ohlc(10)
        out = apply_triple_barrier(
            prices, pd.DatetimeIndex([]), pt_sl=(0.02, 0.01), molecule=5
        )
        assert out.empty

    def test_single_event(self):
        prices = _make_ohlc(10)
        events = pd.DatetimeIndex([prices.index[3]])
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.02, 0.01), molecule=3, side=1
        )
        assert len(out) == 1

    def test_event_near_end(self):
        prices = _make_ohlc(10)
        events = pd.DatetimeIndex([prices.index[-2]])
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.02, 0.01), molecule=10, side=1
        )
        assert len(out) == 1
        # vertical or early exit — must not crash
        assert out.iloc[0]["barrier_hit"] in ("pt", "sl", "vertical")

    def test_duplicate_timestamps_in_prices(self):
        prices = _make_ohlc(8)
        # inject duplicate
        prices = pd.concat([prices, prices.iloc[[3]]])
        prices = prices.sort_index()
        events = pd.DatetimeIndex([prices.index[1]])
        # must not raise; duplicates are deduped deterministically
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.02, 0.01), molecule=3, side=1
        )
        assert len(out) == 1

    def test_unsorted_events(self):
        prices = _make_ohlc(15)
        events = prices.index[[5, 2, 8]]
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.02, 0.01), molecule=4, side=1
        )
        assert out.index.is_monotonic_increasing

    def test_zero_pt_raises(self):
        prices = _make_ohlc(5)
        with pytest.raises(ValueError, match="positive"):
            apply_triple_barrier(
                prices, prices.index[:1], pt_sl=(0.0, 0.01), molecule=3
            )

    def test_negative_horizon_raises(self):
        prices = _make_ohlc(5)
        with pytest.raises(ValueError):
            apply_triple_barrier(
                prices, prices.index[:1], pt_sl=(0.02, 0.01), molecule=-1
            )

    def test_non_positive_price_raises(self):
        prices = _make_ohlc(5)
        prices.iloc[2, prices.columns.get_loc("close")] = -1.0
        with pytest.raises(ValueError, match="Non-positive"):
            apply_triple_barrier(
                prices, prices.index[:1], pt_sl=(0.02, 0.01), molecule=3
            )

    def test_missing_ohlc_raises(self):
        prices = _make_ohlc(5)[["open", "close"]]
        with pytest.raises(ValueError, match="missing columns"):
            apply_triple_barrier(
                prices, prices.index[:1], pt_sl=(0.02, 0.01), molecule=3
            )

    def test_timezone_aware(self):
        prices = _make_ohlc(10)
        prices.index = prices.index.tz_localize("Asia/Jakarta")
        events = prices.index[2:4]
        out = apply_triple_barrier(
            prices, events, pt_sl=(0.02, 0.01), molecule=3, side=1
        )
        assert len(out) == 2
        assert out.index.tz is not None

    def test_min_ret_demotes_label(self):
        prices = _deterministic_path(
            100.0,
            [(0.0, 0.3, -0.1, 0.05), (0.05, 0.6, 0.0, 0.5)],  # tiny move
        )
        events = pd.DatetimeIndex([prices.index[0]])
        out = apply_triple_barrier(
            prices,
            events,
            pt_sl=(0.005, 0.005),
            molecule=5,
            side=1,
            min_ret=0.02,  # 2% minimum
        )
        # even if PT hit, min_ret forces vertical
        assert out.iloc[0]["label"] == 0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestCompat:
    def test_get_events_with_close_only(self):
        prices = _make_ohlc(12)
        close = prices["close"]
        events = close.index[1:5]
        out = get_events(
            close, events, pt_sl=(0.02, 0.01), molecule=4, side=pd.Series(1, index=events)
        )
        assert "label" in out.columns
        assert len(out) == len(events)

    def test_get_bins(self):
        prices = _make_ohlc(10)
        events = apply_triple_barrier(
            prices, prices.index[1:4], pt_sl=(0.02, 0.01), molecule=3
        )
        bins = get_bins(events)
        assert "label" in bins.columns


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

class TestSchema:
    REQUIRED = [
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

    def test_columns_present(self):
        prices = _make_ohlc(10)
        out = apply_triple_barrier(
            prices, prices.index[1:3], pt_sl=(0.02, 0.01), molecule=3
        )
        for c in self.REQUIRED:
            assert c in out.columns, f"missing column {c}"

    def test_label_domain(self):
        prices = _make_ohlc(20, seed=99)
        out = apply_triple_barrier(
            prices, prices.index[1:15], pt_sl=(0.01, 0.01), molecule=4, side=1
        )
        assert set(out["label"].unique()).issubset({-1, 0, 1})

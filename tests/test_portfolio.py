"""Portfolio accounting, lifecycle, idempotency tests."""
from __future__ import annotations

import pytest
from src.python.portfolio.engine import ExitReason, PortfolioEngine, PositionState


class TestPortfolioBasics:
    def test_initial_equity(self):
        eng = PortfolioEngine(initial_cash=1_000_000)
        assert eng.state.equity == 1_000_000

    def test_buy_reduces_cash_creates_position(self):
        eng = PortfolioEngine(initial_cash=10_000_000, fee_bps=10, lot_size=100)
        txn = eng.apply_buy(signal_id_="sig_test1", symbol="BBCA", side=1, price=10000, weight=0.10)
        assert txn is not None
        assert eng.state.cash < 10_000_000
        assert eng.state.positions["BBCA:1"].state == PositionState.OPEN
        eng.mark_to_market({"BBCA": 10000})
        pos = eng.state.positions["BBCA:1"]
        mv = pos.qty * 10000
        assert abs(eng.state.cash + mv - eng.state.equity) < eng.state.fees_total + 1

    def test_idempotent_buy(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100)
        t1 = eng.apply_buy(signal_id_="sig_same", symbol="BBCA", side=1, price=9000, weight=0.1)
        t2 = eng.apply_buy(signal_id_="sig_same", symbol="BBCA", side=1, price=9000, weight=0.1)
        assert t1 is not None and t2 is None and len(eng.state.transactions) == 1

    def test_stop_loss_exit(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100, fee_bps=0, slippage_bps=0)
        eng.apply_buy(signal_id_="sig_sl", symbol="BBCA", side=1, price=10000, weight=0.2,
                      stop_loss=9500, take_profit=11000)
        closed = eng.check_exits({"BBCA": 9400})
        assert len(closed) == 1
        assert eng.state.positions["BBCA:1"].exit_reason == ExitReason.STOP_LOSS.value

    def test_take_profit_exit(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100, fee_bps=0, slippage_bps=0)
        eng.apply_buy(signal_id_="sig_tp", symbol="BBCA", side=1, price=10000, weight=0.15,
                      stop_loss=9000, take_profit=10500)
        closed = eng.check_exits({"BBCA": 10600})
        assert eng.state.positions["BBCA:1"].exit_reason == ExitReason.TAKE_PROFIT.value

    def test_save_load_roundtrip(self, tmp_path):
        eng = PortfolioEngine(initial_cash=5_000_000, lot_size=100)
        eng.apply_buy(signal_id_="sig_rt", symbol="TLKM", side=1, price=4000, weight=0.1)
        path = tmp_path / "pf.json"
        eng.save(path)
        eng2 = PortfolioEngine(initial_cash=1)
        eng2.load(path)
        assert eng2.state.cash == eng.state.cash
        assert eng2.apply_buy(signal_id_="sig_rt", symbol="TLKM", side=1, price=4000, weight=0.1) is None

    def test_invalid_price_raises(self):
        eng = PortfolioEngine(initial_cash=1_000_000)
        with pytest.raises(ValueError):
            eng.apply_buy(signal_id_="x", symbol="A", side=1, price=-1, weight=0.1)

    def test_drawdown(self):
        eng = PortfolioEngine(initial_cash=1_000_000, lot_size=1, fee_bps=0, slippage_bps=0)
        eng.apply_buy(signal_id_="d1", symbol="X", side=1, price=100, weight=0.5)
        eng.mark_to_market({"X": 50})
        assert eng.drawdown() < 0

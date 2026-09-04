"""Forensic accounting, idempotency, CA, reconciliation tests."""
from __future__ import annotations

import pytest

from src.python.market.corporate_actions import CorporateActionPolicy, CorporateActionRecord, CorporateActionType
from src.python.portfolio.engine import ExitReason, PortfolioEngine, PositionState
from src.python.portfolio.reconcile import reconcile
from src.python.persistence.repository import SQLiteStateRepository

TOL = 1e-4


class TestLongAccounting:
    def test_buy_fee_equity_identity(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100, fee_bps=10, slippage_bps=0)
        t = eng.apply_buy(signal_id_="b1", symbol="BBCA", side=1, price=10000, weight=0.2)
        assert t is not None
        assert abs(eng.accounting_identity({"BBCA": t.price})["identity_gap"]) < TOL

    def test_sell_profit(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100, fee_bps=10, slippage_bps=0)
        C0 = eng.state.cash
        eng.apply_buy(signal_id_="p1", symbol="BBCA", side=1, price=10000, weight=0.2)
        qty = eng.state.positions["BBCA:1"].qty
        eng.apply_close(symbol="BBCA", side=1, price=11000, reason=ExitReason.TAKE_PROFIT, signal_id_="p1")
        expected = C0 + (11000 - 10000) * qty - eng.state.fees_total
        assert abs(eng.state.cash - expected) < TOL
        assert eng.state.realized_pnl > 0

    def test_sell_loss(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100, fee_bps=10, slippage_bps=0)
        C0 = eng.state.cash
        eng.apply_buy(signal_id_="l1", symbol="BBCA", side=1, price=10000, weight=0.2)
        qty = eng.state.positions["BBCA:1"].qty
        eng.apply_close(symbol="BBCA", side=1, price=9000, reason=ExitReason.STOP_LOSS, signal_id_="l1")
        expected = C0 + (9000 - 10000) * qty - eng.state.fees_total
        assert abs(eng.state.cash - expected) < TOL
        assert eng.state.realized_pnl < 0

    def test_sl_tp(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100, fee_bps=0, slippage_bps=0)
        eng.apply_buy(signal_id_="sl", symbol="BBCA", side=1, price=10000, weight=0.2, stop_loss=9500, take_profit=11000)
        assert len(eng.check_exits({"BBCA": 9400})) == 1

    def test_partial_exit(self):
        eng = PortfolioEngine(initial_cash=50_000_000, lot_size=100, fee_bps=0, slippage_bps=0)
        eng.apply_buy(signal_id_="pa", symbol="BBCA", side=1, price=10000, weight=0.4)
        q = eng.state.positions["BBCA:1"].qty
        half = float(int((q / 2) / 100) * 100)
        eng.apply_close(symbol="BBCA", side=1, price=10500, reason=ExitReason.MANUAL, signal_id_="pa", qty=half)
        assert eng.state.positions["BBCA:1"].state == PositionState.PARTIAL_EXIT
        assert abs(eng.accounting_identity({"BBCA": 10500})["identity_gap"]) < TOL


class TestShortAccounting:
    def test_short_open_close_profit(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100, fee_bps=0, slippage_bps=0)
        C0 = eng.state.cash
        eng.apply_buy(signal_id_="sh", symbol="BBCA", side=-1, price=10000, weight=0.1)
        assert eng.state.cash > C0
        assert abs(eng.accounting_identity({"BBCA": 10000})["identity_gap"]) < TOL
        eng.apply_close(symbol="BBCA", side=-1, price=9000, reason=ExitReason.TAKE_PROFIT, signal_id_="sh")
        assert eng.state.realized_pnl > 0


class TestIdempotency:
    def test_same_signal_no_double_buy(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100)
        a = eng.apply_buy(signal_id_="same", symbol="BBCA", side=1, price=9000, weight=0.1)
        b = eng.apply_buy(signal_id_="same", symbol="BBCA", side=1, price=9000, weight=0.1)
        assert a is not None and b is None and len(eng.state.transactions) == 1

    def test_restart_preserves_idempotency(self, tmp_path):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100)
        eng.apply_buy(signal_id_="rs", symbol="BBCA", side=1, price=9000, weight=0.1)
        path = tmp_path / "p.json"
        eng.save(path)
        eng2 = PortfolioEngine(initial_cash=1)
        eng2.load(path)
        assert eng2.apply_buy(signal_id_="rs", symbol="BBCA", side=1, price=9000, weight=0.1) is None


class TestCorporateActionAccounting:
    def test_split_preserves_cost(self):
        adj = CorporateActionPolicy.apply_to_position(
            100.0, 1000.0, CorporateActionRecord("X", "2024-01-01", CorporateActionType.STOCK_SPLIT, ratio=2.0))
        assert adj.new_qty == 200 and abs(adj.new_qty * adj.new_avg_price - 100000) < TOL

    def test_dividend_cash(self):
        adj = CorporateActionPolicy.apply_to_position(
            100, 1000, CorporateActionRecord("X", "2024-01-01", CorporateActionType.DIVIDEND, cash_component=25))
        assert adj.cash_delta == 2500


class TestReconcileHalt:
    def test_clean_ok(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100, fee_bps=0, slippage_bps=0)
        eng.apply_buy(signal_id_="r1", symbol="BBCA", side=1, price=9000, weight=0.1)
        assert reconcile(eng, {"BBCA": 9000}).ok

    def test_corrupt_cash_fails(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100)
        eng.apply_buy(signal_id_="r2", symbol="BBCA", side=1, price=9000, weight=0.1)
        eng.state.cash = -1
        assert not reconcile(eng, {"BBCA": 9000}).ok

    def test_equity_gap_fails(self):
        eng = PortfolioEngine(initial_cash=10_000_000, lot_size=100, fee_bps=0, slippage_bps=0)
        eng.apply_buy(signal_id_="r3", symbol="BBCA", side=1, price=9000, weight=0.1)
        eng.state.equity = eng.state.cash
        assert not reconcile(eng, {"BBCA": 9000}).ok


class TestOutboxClaim:
    def test_claim_exclusive(self, tmp_path):
        repo = SQLiteStateRepository(tmp_path / "o.db")
        repo.enqueue_notification("e1", "BUY", {"x": 1})
        a = repo.claim_pending_notifications(limit=10, worker_id="a")
        b = repo.claim_pending_notifications(limit=10, worker_id="b")
        assert len(a) == 1 and len(b) == 0

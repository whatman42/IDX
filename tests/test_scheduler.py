from __future__ import annotations
from datetime import date, datetime
from zoneinfo import ZoneInfo
from src.python.scheduler.schedule import (
    FixedClock, ScheduleType, TrainingDeadline, classify_schedule,
    is_after_market_close, jakarta_to_utc_cron_docs, production_cycle_key, training_run_key,
)

def test_weekday_eod_production():
    clk = FixedClock(datetime(2026, 9, 7, 17, 0, tzinfo=ZoneInfo("Asia/Jakarta")))
    plan = classify_schedule(clk)
    assert plan.schedule_type == ScheduleType.WEEKDAY_PRODUCTION
    assert plan.allow_training is False and plan.allow_new_trades is True

def test_weekday_before_close_outside_window():
    clk = FixedClock(datetime(2026, 9, 7, 12, 0, tzinfo=ZoneInfo("Asia/Jakarta")))
    assert classify_schedule(clk).schedule_type == ScheduleType.OUTSIDE_WINDOW

def test_saturday_exploration():
    clk = FixedClock(datetime(2026, 9, 5, 10, 0, tzinfo=ZoneInfo("Asia/Jakarta")))
    plan = classify_schedule(clk)
    assert plan.schedule_type == ScheduleType.SATURDAY_EXPLORATION
    assert plan.allow_training and not plan.allow_promotion

def test_sunday_validation():
    clk = FixedClock(datetime(2026, 9, 6, 10, 0, tzinfo=ZoneInfo("Asia/Jakarta")))
    plan = classify_schedule(clk)
    assert plan.schedule_type == ScheduleType.SUNDAY_VALIDATION and plan.allow_promotion

def test_manual_production_not_training():
    clk = FixedClock(datetime(2026, 9, 5, 10, 0, tzinfo=ZoneInfo("Asia/Jakarta")))
    plan = classify_schedule(clk, manual=True, manual_mode="production")
    assert plan.schedule_type == ScheduleType.MANUAL_PRODUCTION and not plan.allow_training

def test_non_trading_day_flag():
    clk = FixedClock(datetime(2026, 9, 7, 17, 0, tzinfo=ZoneInfo("Asia/Jakarta")))
    assert not classify_schedule(clk, is_trading_day=False).allow_new_trades

def test_cycle_key_idempotent():
    assert production_cycle_key(date(2026, 9, 4)) == "IDX_PRODUCTION:2026-09-04"
    assert training_run_key(date(2026, 9, 6), "validation") == "IDX_TRAIN:2026-09-06:validation"

def test_deadline_expires():
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
    dl = TrainingDeadline(internal_budget_sec=60)
    dl.start(FixedClock(t0))
    assert not dl.expired(FixedClock(t0))
    assert dl.expired(FixedClock(datetime(2026, 1, 1, 0, 2, tzinfo=ZoneInfo("Asia/Jakarta"))))

def test_cron_docs_utc_conversion():
    docs = jakarta_to_utc_cron_docs()
    assert docs["weekday_production_cron_utc"] == "30 9 * * 1-5"

def test_after_market_close():
    assert is_after_market_close(datetime(2026, 9, 7, 16, 0, tzinfo=ZoneInfo("Asia/Jakarta")))
    assert not is_after_market_close(datetime(2026, 9, 7, 12, 0, tzinfo=ZoneInfo("Asia/Jakarta")))

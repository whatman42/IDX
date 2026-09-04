from datetime import date, datetime
from zoneinfo import ZoneInfo
from src.python.market.calendar import TradingCalendar, is_trading_day, is_weekday
from src.python.scheduler.schedule import FixedClock, classify_schedule

def test_weekend_not_trading():
    assert not is_trading_day(date(2026, 9, 5))
    assert not is_trading_day(date(2026, 9, 6))

def test_weekday_default_trading():
    assert is_weekday(date(2026, 9, 7)) and is_trading_day(date(2026, 9, 7))

def test_holiday_injection():
    cal = TradingCalendar(extra_holidays={date(2026, 9, 7)})
    assert not cal.is_trading_day(date(2026, 9, 7))

def test_provider_override():
    assert not TradingCalendar(provider=lambda d: False).is_trading_day(date(2026, 9, 8))

def test_schedule_respects_non_trading_day():
    clk = FixedClock(datetime(2026, 9, 7, 17, 0, tzinfo=ZoneInfo("Asia/Jakarta")))
    assert not classify_schedule(clk, is_trading_day=False).allow_new_trades

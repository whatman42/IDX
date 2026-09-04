"""IDX schedule — injectable clock, Asia/Jakarta, weekday EOD / weekend train.

GitHub cron is UTC-only. WIB=UTC+7 (no DST).
  Weekday EOD 16:30 WIB → 30 9 * * 1-5
  Sat explore 09:00 WIB → 0 2 * * 6
  Sun validate 09:00 WIB → 0 2 * * 0
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Optional, Protocol
from zoneinfo import ZoneInfo
from src.python.market.calendar import is_trading_day as cal_is_trading_day

IDX_TZ = ZoneInfo("Asia/Jakarta")
SESSION_CLOSE = time(15, 50)

class ScheduleType(str, Enum):
    WEEKDAY_PRODUCTION = "WEEKDAY_PRODUCTION"
    SATURDAY_EXPLORATION = "SATURDAY_EXPLORATION"
    SUNDAY_VALIDATION = "SUNDAY_VALIDATION"
    MANUAL_PRODUCTION = "MANUAL_PRODUCTION"
    MANUAL_TRAINING = "MANUAL_TRAINING"
    NON_TRADING_DAY = "NON_TRADING_DAY"
    OUTSIDE_WINDOW = "OUTSIDE_WINDOW"

class Clock(Protocol):
    def now(self) -> datetime: ...

class SystemClock:
    def now(self) -> datetime:
        return datetime.now(IDX_TZ)

@dataclass
class FixedClock:
    instant: datetime
    def now(self) -> datetime:
        if self.instant.tzinfo is None:
            return self.instant.replace(tzinfo=IDX_TZ)
        return self.instant.astimezone(IDX_TZ)

@dataclass
class RunPlan:
    schedule_type: ScheduleType
    trading_date: date
    cycle_key: str
    allow_training: bool
    allow_promotion: bool
    allow_new_trades: bool
    reason: str

def is_weekday(d: date) -> bool:
    return d.weekday() < 5

def is_after_market_close(ts: datetime) -> bool:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=IDX_TZ)
    return ts.astimezone(IDX_TZ).time() >= SESSION_CLOSE

def production_cycle_key(trading_date: date) -> str:
    return f"IDX_PRODUCTION:{trading_date.isoformat()}"

def training_run_key(run_date: date, stage: str) -> str:
    return f"IDX_TRAIN:{run_date.isoformat()}:{stage}"

def classify_schedule(clock: Clock, *, manual: bool = False, manual_mode: str = "production",
                      is_trading_day: Optional[bool] = None) -> RunPlan:
    now = clock.now()
    local = now.astimezone(IDX_TZ) if now.tzinfo else now.replace(tzinfo=IDX_TZ)
    d = local.date()
    trading = is_trading_day if is_trading_day is not None else cal_is_trading_day(d)
    if manual:
        if manual_mode == "training":
            stage = "exploration" if d.weekday() == 5 else "validation"
            return RunPlan(ScheduleType.MANUAL_TRAINING, d, training_run_key(d, stage),
                           True, True, False, "manual_training")
        return RunPlan(ScheduleType.MANUAL_PRODUCTION, d, production_cycle_key(d),
                       False, False, trading, "manual_production")
    if not trading:
        if d.weekday() == 5:
            return RunPlan(ScheduleType.SATURDAY_EXPLORATION, d, training_run_key(d, "exploration"),
                           True, False, False, "saturday_exploration")
        if d.weekday() == 6:
            return RunPlan(ScheduleType.SUNDAY_VALIDATION, d, training_run_key(d, "validation"),
                           True, True, False, "sunday_validation")
        return RunPlan(ScheduleType.NON_TRADING_DAY, d, production_cycle_key(d),
                       False, False, False, "non_trading_day")
    if not is_after_market_close(local):
        return RunPlan(ScheduleType.OUTSIDE_WINDOW, d, production_cycle_key(d),
                       False, False, False, "before_market_close")
    return RunPlan(ScheduleType.WEEKDAY_PRODUCTION, d, production_cycle_key(d),
                   False, False, True, "weekday_eod_production")

def jakarta_to_utc_cron_docs() -> dict[str, str]:
    return {
        "weekday_production_local": "Mon-Fri 16:30 Asia/Jakarta (after IDX close 15:50)",
        "weekday_production_cron_utc": "30 9 * * 1-5",
        "saturday_exploration_local": "Sat 09:00 Asia/Jakarta",
        "saturday_exploration_cron_utc": "0 2 * * 6",
        "sunday_validation_local": "Sun 09:00 Asia/Jakarta",
        "sunday_validation_cron_utc": "0 2 * * 0",
        "note": "GitHub Actions cron is UTC-only; WIB=UTC+7 (no DST)",
    }

@dataclass
class TrainingDeadline:
    workflow_timeout_min: int = 22
    internal_budget_sec: int = 20 * 60
    started_at: Optional[datetime] = None
    def start(self, clock: Clock) -> None:
        self.started_at = clock.now()
    def remaining_sec(self, clock: Clock) -> float:
        if self.started_at is None:
            return float(self.internal_budget_sec)
        return max(0.0, self.internal_budget_sec - (clock.now() - self.started_at).total_seconds())
    def expired(self, clock: Clock) -> bool:
        return self.remaining_sec(clock) <= 0

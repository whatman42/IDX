"""IDX trading session / calendar helpers."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

IDX_TZ = ZoneInfo("Asia/Jakarta")
SESSION_OPEN = time(9, 0)
SESSION_CLOSE = time(15, 50)


def is_weekday(d: date) -> bool:
    return d.weekday() < 5


def in_session(ts: datetime) -> bool:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=IDX_TZ)
    local = ts.astimezone(IDX_TZ)
    if not is_weekday(local.date()):
        return False
    t = local.time()
    return SESSION_OPEN <= t <= SESSION_CLOSE


def is_stale(last_bar_ts: datetime, now: datetime | None = None, max_age_days: float = 5.0) -> bool:
    now = now or datetime.now(IDX_TZ)
    if last_bar_ts.tzinfo is None:
        last_bar_ts = last_bar_ts.replace(tzinfo=IDX_TZ)
    age = (now - last_bar_ts.astimezone(IDX_TZ)).total_seconds() / 86400.0
    return age > max_age_days

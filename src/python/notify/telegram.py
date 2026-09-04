"""Telegram notifications via outbox — never rolls back paper transactions."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import httpx


class NotifyEventType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TIME_EXIT = "TIME_EXIT"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    DEGRADED_MODE = "DEGRADED_MODE"
    MODEL_PROMOTION = "MODEL_PROMOTION"
    MODEL_REJECTION = "MODEL_REJECTION"
    DAILY_REPORT = "DAILY_REPORT"
    WEEKLY_REPORT = "WEEKLY_REPORT"


class NotificationProvider(ABC):
    @abstractmethod
    def send(self, event_type: str, payload: dict[str, Any]) -> None:
        ...


class NullNotificationProvider(NotificationProvider):
    def send(self, event_type: str, payload: dict[str, Any]) -> None:
        return


@dataclass
class TelegramProvider(NotificationProvider):
    bot_token: str
    chat_id: str
    timeout: float = 10.0

    @classmethod
    def from_env(cls) -> Optional["TelegramProvider"]:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat:
            return None
        return cls(bot_token=token, chat_id=chat)

    def send(self, event_type: str, payload: dict[str, Any]) -> None:
        text = format_message(event_type, payload)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        resp = httpx.post(url, json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}, timeout=self.timeout)
        resp.raise_for_status()


def format_message(event_type: str, p: dict[str, Any]) -> str:
    if event_type == NotifyEventType.BUY.value:
        return (
            f"🟢 <b>IDX BUY</b>\nSymbol: <code>{p.get('symbol')}</code>\nEntry: {p.get('entry_price')}\n"
            f"Qty: {p.get('qty')}\nSide: {p.get('side')}\nPrimary P: {p.get('primary_probability')}\n"
            f"Meta P: {p.get('meta_probability')}\nConfidence: {p.get('confidence')}\n"
            f"SL: {p.get('stop_loss')}  TP: {p.get('take_profit')}\nModel: {p.get('model_version')}\n"
            f"Governor: {p.get('governor_version')}  Regime: {p.get('regime')}\n"
            f"Equity: {p.get('equity')}  Cash: {p.get('cash')}\n"
            f"Signal: <code>{p.get('signal_id')}</code>\nTx: <code>{p.get('tx_id')}</code>\nTime: {p.get('timestamp')}"
        )
    if event_type in (NotifyEventType.STOP_LOSS.value, NotifyEventType.TAKE_PROFIT.value):
        return f"🔔 <b>IDX {event_type}</b>\n{json.dumps(p, default=str)[:800]}"
    if event_type == NotifyEventType.SYSTEM_ERROR.value:
        return f"🔴 <b>IDX ERROR</b>\n{p.get('message', p)}"
    if event_type == NotifyEventType.DEGRADED_MODE.value:
        return f"🟡 <b>IDX DEGRADED</b>\n{p.get('reason', p)}"
    return f"IDX {event_type}\n{json.dumps(p, default=str)[:800]}"


def drain_outbox(repo, provider: Optional[NotificationProvider], limit: int = 10) -> int:
    if provider is None:
        provider = NullNotificationProvider()
    sent = 0
    for item in repo.pending_notifications(limit=limit):
        try:
            provider.send(item["event_type"], item["payload"])
            repo.mark_notification(item["event_id"], "SENT")
            sent += 1
        except Exception as e:
            repo.mark_notification(item["event_id"], "FAILED", str(e)[:300])
    return sent

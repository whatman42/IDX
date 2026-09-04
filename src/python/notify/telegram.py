"""Telegram notifications — Gemini advisory optional, deterministic ID fallback mandatory."""
from __future__ import annotations
import json, os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import httpx
from src.python.advisory.gemini import GeminiAdvisor, is_high_value_event
from src.python.advisory.templates_id import format_deterministic_id

class NotifyEventType(str, Enum):
    BUY = "BUY"
    NO_BUY = "NO_BUY"
    SELL = "SELL"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TIME_EXIT = "TIME_EXIT"
    PORTFOLIO = "PORTFOLIO"
    GOVERNOR = "GOVERNOR"
    TRAINING = "TRAINING"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    DEGRADED_MODE = "DEGRADED_MODE"
    HALT = "HALT"
    MODEL_PROMOTION = "MODEL_PROMOTION"
    MODEL_REJECTION = "MODEL_REJECTION"
    DAILY_REPORT = "DAILY_REPORT"
    WEEKLY_REPORT = "WEEKLY_REPORT"

class NotificationProvider(ABC):
    @abstractmethod
    def send(self, event_type: str, payload: dict[str, Any]) -> None: ...

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

def format_message(event_type: str, p: dict[str, Any], advisor: Optional[GeminiAdvisor] = None) -> str:
    use_gemini = is_high_value_event(event_type) and advisor is not None and advisor.enabled
    if use_gemini:
        result = advisor.explain(event_type, p)
        if result.ok:
            why = "\n".join(f"- {w}" for w in result.why) if result.why else "-"
            return (
                f"<b>{result.title}</b>\n\n{result.summary}\n\n"
                f"<b>Kenapa?</b>\n{why}\n\n"
                f"<b>Risiko:</b>\n{result.risk_explanation}\n\n"
                f"<b>Tindakan sistem:</b>\n{result.system_action}\n"
                f"\n<i>Sumber: advisory (angka dari sistem deterministik)</i>"
            )
    body = format_deterministic_id(event_type, p)
    return body.replace("<", "<").replace(">", ">")

def drain_outbox(repo, provider: Optional[NotificationProvider], limit: int = 10,
                 advisor: Optional[GeminiAdvisor] = None) -> int:
    if provider is None:
        provider = NullNotificationProvider()
    sent = 0
    claim = getattr(repo, "claim_pending_notifications", None)
    items = claim(limit=limit) if callable(claim) else repo.pending_notifications(limit=limit)
    for item in items:
        try:
            payload = dict(item.get("payload") or {})
            if isinstance(provider, TelegramProvider):
                text = format_message(item["event_type"], payload, advisor=advisor)
                url = f"https://api.telegram.org/bot{provider.bot_token}/sendMessage"
                resp = httpx.post(url, json={"chat_id": provider.chat_id, "text": text, "parse_mode": "HTML"},
                                  timeout=provider.timeout)
                resp.raise_for_status()
            else:
                provider.send(item["event_type"], payload)
            repo.mark_notification(item["event_id"], "SENT")
            sent += 1
        except Exception as e:
            repo.mark_notification(item["event_id"], "FAILED", str(e)[:300])
    return sent

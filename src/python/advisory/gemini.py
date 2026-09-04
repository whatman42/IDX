"""Gemini Flash-Lite advisory intelligence — NO trading authority.

Reads GEMINI_API_KEY from environment only. Failures → deterministic ID templates.
Rejects outputs that alter factual numbers from the system context.
"""
from __future__ import annotations
import json, os, re, time
from dataclasses import dataclass, field
from typing import Any, Optional
import httpx

GEMINI_MODEL = "gemini-2.0-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_TIMEOUT = 8.0
MAX_RETRIES = 1
CIRCUIT_FAILS = 3
FACT_KEYS = (
    "symbol", "entry_price", "qty", "equity", "cash", "pnl", "drawdown",
    "primary_probability", "meta_probability", "confidence", "stop_loss", "take_profit",
)

@dataclass
class AdvisoryResult:
    ok: bool
    language: str = "id-ID"
    audience: str = "beginner"
    title: str = ""
    summary: str = ""
    why: list = field(default_factory=list)
    risk_explanation: str = ""
    system_action: str = ""
    severity: str = "INFO"
    confidence: Optional[float] = None
    source: str = "gemini"
    raw_error: str = ""

def validate_advisory(data: dict[str, Any], context: dict[str, Any]) -> tuple[bool, str]:
    for k in ("title", "summary", "system_action", "severity"):
        if k not in data or not str(data.get(k, "")).strip():
            return False, f"missing_{k}"
    text_blob = " ".join([
        str(data.get("title", "")), str(data.get("summary", "")),
        str(data.get("risk_explanation", "")),
        " ".join(str(x) for x in data.get("why", [])),
        str(data.get("system_action", "")),
    ])
    ctx_str = json.dumps(context, default=str)
    for m in re.findall(r"\d{4,}", text_blob.replace(".", "").replace(",", "")):
        if m not in ctx_str and len(m) >= 5:
            return False, f"hallucinated_number:{m}"
    return True, "ok"

class GeminiAdvisor:
    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL,
                 timeout: float = DEFAULT_TIMEOUT, enabled: bool = True):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or ""
        self.model = model
        self.timeout = timeout
        self.enabled = enabled and bool(self.api_key)
        self._fail_count = 0
        self._circuit_open = False
        self._last_call = 0.0
        self.min_interval = 0.5

    def explain(self, event_type: str, context: dict[str, Any]) -> AdvisoryResult:
        if not self.enabled or self._circuit_open:
            return AdvisoryResult(ok=False, source="fallback", raw_error="disabled_or_circuit")
        now = time.monotonic()
        if now - self._last_call < self.min_interval:
            time.sleep(self.min_interval - (now - self._last_call))
        self._last_call = time.monotonic()
        prompt = self._build_prompt(event_type, context)
        url = GEMINI_URL.format(model=self.model) + f"?key={self.api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512, "responseMimeType": "application/json"},
        }
        last_err = ""
        for _ in range(MAX_RETRIES + 1):
            try:
                resp = httpx.post(url, headers={"Content-Type": "application/json"}, json=body, timeout=self.timeout)
                if resp.status_code == 429:
                    last_err = "rate_limit"
                    self._fail_count += 1
                    time.sleep(0.5)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                parsed = json.loads(text)
                ok, reason = validate_advisory(parsed, context)
                if not ok:
                    return AdvisoryResult(ok=False, source="fallback", raw_error=reason)
                self._fail_count = 0
                return AdvisoryResult(
                    ok=True, language=parsed.get("language", "id-ID"),
                    audience=parsed.get("audience", "beginner"),
                    title=str(parsed.get("title", "")), summary=str(parsed.get("summary", "")),
                    why=list(parsed.get("why", [])),
                    risk_explanation=str(parsed.get("risk_explanation", "")),
                    system_action=str(parsed.get("system_action", "")),
                    severity=str(parsed.get("severity", "INFO")),
                    confidence=parsed.get("confidence"), source="gemini",
                )
            except Exception as e:
                last_err = f"{type(e).__name__}:{e}"[:200]
                self._fail_count += 1
        if self._fail_count >= CIRCUIT_FAILS:
            self._circuit_open = True
        return AdvisoryResult(ok=False, source="fallback", raw_error=last_err)

    def _build_prompt(self, event_type: str, context: dict[str, Any]) -> str:
        safe = {k: context[k] for k in context if k not in ("api_key", "token", "password", "secret")}
        return (
            "Anda adalah asisten notifikasi trading SIMULASI (paper) untuk pemula Indonesia.\n"
            "Jelaskan dalam Bahasa Indonesia sederhana. JANGAN mengubah angka yang diberikan.\n"
            "JANGAN menganjurkan order nyata. Ini paper/simulasi.\n"
            "Output HARUS JSON dengan field: language, audience, title, summary, why (array), "
            "risk_explanation, system_action, severity, confidence.\n"
            f"Jenis peristiwa: {event_type}\n"
            f"Konteks fakta (jangan diubah angkanya): {json.dumps(safe, default=str)[:2000]}\n"
        )

def is_high_value_event(event_type: str) -> bool:
    return event_type.upper() in {
        "BUY", "NO_BUY", "STOP_LOSS", "TAKE_PROFIT", "PORTFOLIO",
        "GOVERNOR", "TRAINING", "SYSTEM_ERROR", "DEGRADED_MODE", "HALT",
    }

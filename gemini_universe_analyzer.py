"""
Module      : gemini_universe_analyzer.py
Description : Gemini Universe-Level Equity Analyzer with Native SDK & Fallback Pipeline
Version     : 2026.Q3.v18.0 (DINO IDX Master Rule Compliant)
Compliance  : IDX Trading Rules & DINO Master Engineering Baseline
"""

import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

try:
    from modules.learning_engine import get_ai_learning_context
except ImportError:
    def get_ai_learning_context() -> str:
        return "Sistem belum memiliki catatan memori evaluasi histori transaksi."

logger = logging.getLogger(__name__)

# Primary & Fallback Baseline Sesuai DINO Master Rule #3
PRIMARY_MODEL: str = "gemini-3.6-flash"
FALLBACK_MODEL: str = "gemini-3.5-flash-lite"

_ACTIVE_PRIMARY_MODEL: Optional[str] = None

# JSON Schema untuk Universe / Batch Output
UNIVERSE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "analyses": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "ticker": {"type": "STRING"},
                    "signal_candidate": {
                        "type": "STRING",
                        "enum": ["BUY", "SELL", "HOLD", "NO_SIGNAL"]
                    },
                    "confidence": {"type": "NUMBER"},
                    "reason": {"type": "STRING"},
                    "entry": {"type": "NUMBER", "nullable": True},
                    "stop_loss": {"type": "NUMBER", "nullable": True},
                    "take_profit": {"type": "NUMBER", "nullable": True},
                },
                "required": ["ticker", "signal_candidate", "confidence", "reason"]
            }
        }
    },
    "required": ["analyses"]
}


def parse_bool(val: Any) -> bool:
    """Mengonversi input ke boolean secara aman."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
    return False


def _safe_float(val: Any) -> Optional[float]:
    """Mengonversi nilai ke float atau None jika invalid."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if not (f != f) else None  # Check NaN
    except (ValueError, TypeError):
        return None


def clean_json_text(text: str) -> str:
    """Membersihkan markdown code blocks dari teks JSON."""
    if not text:
        return ""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    match = re.search(r"\{.*\}", s, re.DOTALL)
    if match:
        return match.group(0).strip()
    return s


def get_client() -> Optional[genai.Client]:
    """Menginisialisasi SDK Client Google GenAI dari Environment Variables."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.critical("[ENV] CRITICAL: GEMINI_API_KEY tidak ditemukan di environment variables!")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"[AUTH_ERROR] Gagal menginisialisasi Gemini Client: {e}")
        return None


def verify_gemini_health() -> bool:
    """
    Startup Health Check Sesuai Pipeline DINO Master Rule #4 & #26.
    Menguji Primary Model (gemini-3.6-flash), kemudian Fallback (gemini-3.5-flash-lite).
    """
    global _ACTIVE_PRIMARY_MODEL
    client = get_client()
    if not client:
        return False

    candidates = [PRIMARY_MODEL, FALLBACK_MODEL]

    for model_name in candidates:
        try:
            logger.info(f"[GEMINI] Health Check pinging candidate: '{model_name}'...")
            response = client.models.generate_content(
                model=model_name,
                contents="ping"
            )
            if response and response.text:
                _ACTIVE_PRIMARY_MODEL = model_name
                logger.info(f"[GEMINI] Health Check PASS using '{_ACTIVE_PRIMARY_MODEL}'.")
                return True
        except Exception as e:
            logger.warning(f"[GEMINI] Candidate model '{model_name}' health check failed: {e}")

    _ACTIVE_PRIMARY_MODEL = None
    logger.error("[GEMINI] ALL MODELS FAILED HEALTH CHECK. Aborting AI Analysis pipeline.")
    return False


def analyze_universe_with_gemini(universe_report: str) -> Dict[str, Dict[str, Any]]:
    """
    Menganalisis SELURUH universe saham dalam 1 PANGGILAN API dengan Bounded Retry Backoff & Jitter.
    
    Catatan Arsitektur (Rule #10):
    Gemini HANYA menghasilkan kandidat sinyal analitis (signal_candidate).
    Keputusan 'trade_allowed' sepenuhnya berada di bawah kendali Risk Engine Python.
    """
    global _ACTIVE_PRIMARY_MODEL
    client = get_client()
    if not client:
        logger.error("[CONFIG_ERROR] Gemini Client tidak tersedia. Signal -> NO_SIGNAL")
        return {}

    try:
        learning_context = get_ai_learning_context()
    except Exception:
        learning_context = "Belum ada memori pembelajaran sebelumnya."

    prompt = f"""
Anda adalah Analis Kuantitatif, Fundamental, & Portfolio Manager Senior Saham BEI (IDX).
Berikut adalah Ringkasan Ringkas Seluruh Universe Saham yang dipantau hari ini:

{universe_report}

--------------------------------------------------
🧠 MEMORI PEMBELAJARAN AI (HISTORI WIN/LOSS):
{learning_context}
--------------------------------------------------

Instruksi Analisis Universe:
1. Evaluasi seluruh ticker di atas. Lakukan *Cross-Ticker Ranking* untuk memilih emiten terbaik.
2. Gunakan Google Search jika diperlukan untuk memverifikasi isu hukum, penyidikan korupsi, atau sengketa penting pada emiten yang berpotensi BUY.
3. Berikan "signal_candidate": "BUY" jika emiten memiliki fundamental sehat (ROE, DER wajar), momentum teknikal positif, dan bebas isu risiko fatal.
4. Jika tidak layak beli, berikan signal "HOLD" atau "NO_SIGNAL".
5. Usulkan perkiraan Entry, Stop Loss, dan Take Profit awal untuk setiap ticker yang dianalisis.
"""

    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=UNIVERSE_RESPONSE_SCHEMA,
        temperature=0.1,
        tools=[{"google_search": {}}]
    )

    models_to_try = []
    if _ACTIVE_PRIMARY_MODEL:
        models_to_try.append(_ACTIVE_PRIMARY_MODEL)
    for m in [PRIMARY_MODEL, FALLBACK_MODEL]:
        if m not in models_to_try:
            models_to_try.append(m)

    for target_model in models_to_try:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[AI_CALL] Mengirim UNIVERSE PROMPT ke Gemini ({target_model}) [Attempt {attempt}/{max_retries}]...")
                
                response = client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=gen_config,
                )

                if response and response.text:
                    cleaned = clean_json_text(response.text)
                    data = json.loads(cleaned)

                    results_by_ticker: Dict[str, Dict[str, Any]] = {}
                    analyses = data.get("analyses", [])

                    for item in analyses:
                        ticker = str(item.get("ticker", "")).upper().strip()
                        if not ticker:
                            continue

                        raw_signal = str(item.get("signal_candidate", item.get("signal", "NO_SIGNAL"))).upper().strip()
                        if raw_signal not in ["BUY", "SELL", "HOLD", "NO_SIGNAL"]:
                            raw_signal = "NO_SIGNAL"

                        confidence = _safe_float(item.get("confidence"))
                        if confidence is None:
                            confidence = 0.0

                        results_by_ticker[ticker] = {
                            "status": "SUCCESS",
                            "signal": raw_signal,
                            "signal_candidate": raw_signal,
                            "confidence": confidence,
                            "reason": str(item.get("reason", "")).strip(),
                            "entry": _safe_float(item.get("entry")),
                            "stop_loss": _safe_float(item.get("stop_loss")),
                            "take_profit": _safe_float(item.get("take_profit")),
                            "model_used": target_model,
                            "data_quality": "PASS"
                        }

                    logger.info(f"[AI_SUCCESS] Berhasil memproses analisis Universe untuk {len(results_by_ticker)} ticker via '{target_model}'.")
                    return results_by_ticker

            except (ClientError, APIError, Exception) as err:
                err_str = str(err)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                    sleep_time = (attempt * 5) + random.uniform(1.0, 3.0)  # Exponential backoff + jitter
                    logger.warning(f"[API_RATE_LIMIT] Transient error pada {target_model} (Attempt {attempt}/{max_retries}). Menunggu {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"[SYSTEM_ERROR] Error pada {target_model}: {err_str}")
                    break

    logger.error("[AI_FAILURE] Seluruh model Gemini gagal/rate limited. Mengembalikan sinyal kosong (NO_SIGNAL).")
    return {}

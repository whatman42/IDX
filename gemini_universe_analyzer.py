"""
=============================================================================
IDX Quantitative Analysis System - Gemini Deep-Dive Analyzer
FileName      : gemini_universe_analyzer.py
Directory     : Flat Directory (Root Level or inside modules/)
Version       : 2026.Q3.v20.0 (Institutional Production Grade - Patched)
Compliance    : DINO IDX Master Rules & Fail-Safe Multi-Model Architecture
=============================================================================
"""

import json
import logging
import os
import random
import re
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

# Standard Zona Waktu Pasar Saham Indonesia (WIB)
WIB_TZ: ZoneInfo = ZoneInfo("Asia/Jakarta")

# Logger Setup
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s][IDX.GEMINI_ANALYZER] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Master Model Strategy Constants
PRIMARY_MODEL: str = "gemini-3.6-flash"
FALLBACK_MODEL: str = "gemini-3.5-flash-lite"

_STATE_LOCK = threading.RLock()
_ACTIVE_PRIMARY_MODEL: Optional[str] = None

# Structured Output Schema for 2-Tier Candidate Analysis
TOP_CANDIDATES_SCHEMA: Dict[str, Any] = {
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
                    "fundamental_score": {"type": "NUMBER"},
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


# ==============================================================================
# HELPER PARSERS & SANITIZERS
# ==============================================================================
def _get_wib_timestamp_str() -> str:
    """Mengembalikan timestamp ISO dengan Zona Waktu Indonesia Barat (WIB)."""
    return datetime.now(WIB_TZ).isoformat()


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Mengonversi nilai ke float secara defensif (menangani string mata uang, None, NaN)."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val) if not (isinstance(val, float) and (val != val)) else default
    try:
        clean_str = re.sub(r"[^\d.-]", "", str(val))
        if not clean_str:
            return default
        f_val = float(clean_str)
        return f_val if f_val == f_val else default
    except (ValueError, TypeError):
        return default


def clean_json_text(text: str) -> str:
    """Membersihkan markdown code blocks dan ekstrak JSON object terstruktur secara presisi."""
    if not text:
        return "{}"
    s = text.strip()
    
    # Strip Markdown Fences (```json ... ```)
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # Match JSON Object Structure ({...})
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if match:
        return match.group(0).strip()
    return s


def get_ai_learning_context() -> str:
    """Mengambil memori histori transaksi dari learning_engine secara opsional."""
    try:
        from modules.learning_engine import get_ai_learning_context as _fetch_context
        return _fetch_context()
    except Exception:
        try:
            from learning_engine import get_ai_learning_context as _fetch_context_alt
            return _fetch_context_alt()
        except Exception:
            return "Sistem belum memiliki catatan memori evaluasi histori transaksi."


# ==============================================================================
# GEMINI CLIENT & HEALTH CHECK ENGINE
# ==============================================================================
def get_client() -> Optional[genai.Client]:
    """Menginisialisasi SDK Client Google GenAI."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.critical("[AUTH] GEMINI_API_KEY tidak ditemukan di environment variables!")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"[AUTH_ERROR] Gagal menginisialisasi Gemini Client: {e}")
        return None


def verify_gemini_health() -> bool:
    """Startup Health Check & Model Active Binding."""
    global _ACTIVE_PRIMARY_MODEL
    client = get_client()
    if not client:
        return False

    with _STATE_LOCK:
        for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                logger.info(f"[GEMINI] Health Check pinging: '{model_name}'...")
                response = client.models.generate_content(
                    model=model_name,
                    contents="ping"
                )
                if response and hasattr(response, "text") and response.text:
                    _ACTIVE_PRIMARY_MODEL = model_name
                    logger.info(f"[GEMINI] Health Check PASS using '{_ACTIVE_PRIMARY_MODEL}'.")
                    return True
            except Exception as e:
                logger.warning(f"[GEMINI] Model '{model_name}' health check failed: {e}")

        _ACTIVE_PRIMARY_MODEL = None
        return False


def get_active_model() -> Optional[str]:
    """Mengembalikan nama model Gemini aktif secara thread-safe."""
    with _STATE_LOCK:
        return _ACTIVE_PRIMARY_MODEL


# ==============================================================================
# TIER-2 DEEP DIVE & ADAPTIVE LOOSENING ANALYZER
# ==============================================================================
def analyze_top_candidates_with_gemini(
    top_5_report: str, 
    relaxation_mode: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Menganalisis TOP 5 Saham Kandidat hasil filter Kuantitatif Python.
    
    Jika relaxation_mode = True, Gemini melonggarkan kriteria fundamental/teknikal
    agar menghasilkan sinyal terbaik tanpa terbentur zero-signal lock, selama tidak ada
    risiko fatal (korupsi/kebangkrutan/suspensi BEI).
    """
    client = get_client()
    if not client:
        return {}

    learning_context = get_ai_learning_context()

    if relaxation_mode:
        mode_instruction = """
        🚨 MODE ADAPTIVE LOOSENING (PELONGGARAN SYARAT):
        Sistem kuantitatif tidak mendeteksi sinyal ideal pada babak pertama.
        Misi Anda: LONGGARKAN kriteria fundamental & konfirmasi teknikal.
        - Turunkan standar confidence minimal menjadi 0.45.
        - Pilihlah minimal 1-2 emiten terbaik dari Top 5 ini untuk sinyal BUY jika kondisinya cukup wajar.
        - TOLAK HANYA JIKA emiten memiliki risiko fatal nyata (korupsi, gugatan pailit, kebangkrutan, suspensi BEI).
        """
    else:
        mode_instruction = """
        🎯 MODE EVALUASI STANDAR:
        Evaluasi secara profesional fundamental dan momentum teknikal dari Top 5 emiten ini.
        - Berikan "signal_candidate": "BUY" jika fundamental sehat dan momentum mendukung.
        - Jika dirasa belum memadai, Anda berhak memberikan "HOLD" atau "NO_SIGNAL".
        """

    prompt = f"""
Anda adalah Senior Portfolio Manager & Qualitative Stock Auditor BEI (IDX).
Berikut adalah TOP 5 SAHAM KANDIDAT TERBAIK hasil filtrasi Kuantitatif Python hari ini:

{top_5_report}

--------------------------------------------------
🧠 MEMORI PEMBELAJARAN AI (HISTORI WIN/LOSS):
{learning_context}
--------------------------------------------------

Instruksi Analisis Deep-Dive:
{mode_instruction}

1. Gunakan Google Search jika diperlukan untuk memverifikasi isu korupsi, penyelidikan hukum, suspensi, atau fraud pada emiten.
2. Tetapkan "signal_candidate": "BUY" untuk emiten yang lolos evaluasi.
3. Berikan nilai confidence (0.0 - 1.0) dan skor fundamental (0 - 100) serta alasan singkat.
4. Tentukan kisaran rasional Entry, Stop Loss, dan Take Profit.
"""

    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=TOP_CANDIDATES_SCHEMA,
        temperature=0.2 if relaxation_mode else 0.1,
        tools=[{"google_search": {}}]
    )

    models_to_try: List[str] = []
    active_m = get_active_model()
    if active_m:
        models_to_try.append(active_m)
    for m in [PRIMARY_MODEL, FALLBACK_MODEL]:
        if m not in models_to_try:
            models_to_try.append(m)

    for target_model in models_to_try:
        for attempt in range(1, 4):
            try:
                logger.info(f"[GEMINI_DEEP_DIVE] Menganalisis Top 5 Kandidat ({target_model}) [Relaxation={relaxation_mode}] [Attempt {attempt}/3]...")
                
                response = client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=gen_config,
                )

                # Inspeksi Candidate Safety & Response Validity
                if not response or not hasattr(response, "text") or not response.text:
                    logger.warning(f"⚠️ Respon kosong atau terblokir dari model '{target_model}'.")
                    continue

                cleaned = clean_json_text(response.text)
                data = json.loads(cleaned)

                results_by_ticker: Dict[str, Dict[str, Any]] = {}
                analyses = data.get("analyses", [])

                for item in analyses:
                    if not isinstance(item, dict):
                        continue

                    ticker = str(item.get("ticker", "")).upper().strip()
                    if not ticker:
                        continue

                    raw_signal = str(item.get("signal_candidate", "NO_SIGNAL")).upper().strip()
                    if raw_signal not in ["BUY", "SELL", "HOLD", "NO_SIGNAL"]:
                        raw_signal = "NO_SIGNAL"

                    confidence = _safe_float(item.get("confidence"), default=0.50) or 0.50
                    fund_score = _safe_float(item.get("fundamental_score"), default=50.0) or 50.0

                    results_by_ticker[ticker] = {
                        "status": "SUCCESS",
                        "signal": raw_signal,
                        "signal_candidate": raw_signal,
                        "confidence": float(confidence),
                        "fundamental_score": float(fund_score),
                        "reason": str(item.get("reason", "")).strip(),
                        "entry": _safe_float(item.get("entry")),
                        "stop_loss": _safe_float(item.get("stop_loss")),
                        "take_profit": _safe_float(item.get("take_profit")),
                        "model_used": target_model,
                        "relaxation_applied": relaxation_mode,
                        "timestamp_wib": _get_wib_timestamp_str()
                    }

                logger.info(f"✅ [GEMINI_DEEP_DIVE_SUCCESS] Berhasil memproses {len(results_by_ticker)} kandidat via '{target_model}'.")
                return results_by_ticker

            except (APIError, ClientError) as api_err:
                sleep_time = attempt * 3 + random.uniform(1.0, 2.0)
                logger.warning(f"⚠️ APIError/ClientError pada {target_model}: {api_err}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            except json.JSONDecodeError as json_err:
                logger.warning(f"⚠️ Gagal dekode JSON dari respon {target_model}: {json_err}")
                break
            except Exception as err:
                sleep_time = attempt * 3 + random.uniform(1.0, 2.0)
                logger.warning(f"⚠️ Error tak terduga pada {target_model}: {err}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    return {}

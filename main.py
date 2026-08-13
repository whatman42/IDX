"""
================================================================================
MODULE      : main.py
DESCRIPTION : Single-Run Production Orchestrator for Indonesia Stock Exchange (IDX)
VERSION     : v2026.Q3.v2.6.4-DINO-GEMINI-AUTOPILOT
PYTHON VER  : 3.10+ / 3.11+ / 3.12+
COMPLIANCE  : DINO IDX Master Rules & Gemini Trading Administrator Architecture
================================================================================
"""

import gc
import inspect
import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Timezone Standar BEI (Master Rule #14)
WIB_TZ = ZoneInfo("Asia/Jakarta")

# Model Baseline Sesuai DINO Master Rule #3
PRIMARY_MODEL: str = "gemini-3.6-flash"
FALLBACK_MODEL: str = "gemini-3.5-flash-lite"

# Import Google GenAI Client
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ClientError
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# =============================================================================
# 1. FAIL-FAST STATIC MODULE IMPORTS
# =============================================================================
try:
    import data
    import features
    import machine_learning
    import prediction

    try:
        import signal_idx
    except ImportError:
        import signal_crypto as signal_idx

    import validation
    import risk
    from portfolio import UnifiedPortfolioEngine, normalize_idx_symbol
    import simulation
    import evaluation
    import self_learning

    try:
        import autonomous_engine_idx
    except ImportError:
        import autonomous_engine_crypto as autonomous_engine_idx

    import research
    import reporting
    import monitoring
    import storage
except ImportError as err:
    sys.stderr.write(f"🛑 [CRITICAL_BOOTSTRAP_ERROR] Failed to load dependency module: {err}\n")
    sys.exit(1)


# =============================================================================
# HELPER: GOOGLE GEMINI AI ADMINISTRATOR & INSIGHT ENGINE
# =============================================================================
class IDXGeminiInsightEngine:
    """
    Engine integrasi Google Gemini untuk mengelola konfigurasi dinamis,
    menjalankan Health Check, serta menghasilkan analisis naratif kualitatif
    dan ringkasan eksekutif sinyal saham IDX.
    """
    def __init__(self, api_key: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("IDX.Gemini")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self.active_model: Optional[str] = None

        if not HAS_GEMINI_SDK:
            self.logger.warning("⚠️ Package 'google-genai' belum terpasang. Gemini SDK tidak tersedia.")
            return

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.logger.info("🤖 [GEMINI_INIT_SUCCESS] Google Gemini Client berhasil diinisialisasi.")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal inisialisasi Gemini Client: {e}")
        else:
            self.logger.warning("⚠️ GEMINI_API_KEY tidak ditemukan di environment. AI Engine terbatas.")

    def verify_health(self) -> Tuple[bool, Dict[str, str]]:
        """
        Menjalankan Gemini Model Validation & Health Check (Master Rule #4 & #26).
        """
        results = {
            "env_api_key": "FAIL",
            "gemini_primary": "FAIL",
            "gemini_fallback": "FAIL"
        }

        if not self.api_key or not self.client:
            return False, results

        results["env_api_key"] = "PASS"

        # Check Primary Model (gemini-3.6-flash)
        try:
            self.logger.info(f"🔍 [GEMINI_HEALTH] Testing primary model '{PRIMARY_MODEL}'...")
            res = self.client.models.generate_content(
                model=PRIMARY_MODEL,
                contents="ping"
            )
            if res and hasattr(res, "text") and res.text:
                results["gemini_primary"] = "PASS"
                self.active_model = PRIMARY_MODEL
                self.logger.info(f"✅ [GEMINI_HEALTH] Primary model '{PRIMARY_MODEL}' PASS.")
                return True, results
        except Exception as e:
            self.logger.warning(f"⚠️ [GEMINI_HEALTH] Primary model '{PRIMARY_MODEL}' FAIL: {e}")

        # Check Fallback Model (gemini-3.5-flash-lite)
        try:
            self.logger.info(f"🔍 [GEMINI_HEALTH] Testing fallback model '{FALLBACK_MODEL}'...")
            res = self.client.models.generate_content(
                model=FALLBACK_MODEL,
                contents="ping"
            )
            if res and hasattr(res, "text") and res.text:
                results["gemini_fallback"] = "PASS"
                self.active_model = FALLBACK_MODEL
                self.logger.info(f"✅ [GEMINI_HEALTH] Fallback model '{FALLBACK_MODEL}' PASS.")
                return True, results
        except Exception as e:
            self.logger.error(f"❌ [GEMINI_HEALTH] Fallback model '{FALLBACK_MODEL}' FAIL: {e}")

        self.active_model = None
        self.logger.critical("🛑 [GEMINI_HEALTH] Entire Gemini API infrastructure health check failed.")
        return False, results

    def get_dynamic_trading_parameters(self) -> Dict[str, Any]:
        """
        Gemini Administrator secara otomatis mengatur konstanta & syarat nominal fleksibel.
        Parameter immutable / hardlocked (price floor Rp 50, drawdown limit) tetap dilindungi.
        """
        defaults = {
            "min_adtv_idr": 100_000_000.0,
            "min_confidence": 0.55,
            "min_rrr": 1.5,
            "max_concurrent_positions": 5,
            "risk_scale": 1.0,
            "configured_by": "DEFAULT_FALLBACK"
        }

        if not self.client or not self.active_model:
            return defaults

        prompt = """
        Anda adalah Administrator Bot Trading Kuantitatif Saham BEI (IDX).
        Atur parameter trading dinamis yang optimal sesuai kondisi pasar saham hari ini.
        Kembalikan HANYA JSON terstruktur dengan format persis berikut:
        {
            "min_adtv_idr": 100000000.0,
            "min_confidence": 0.55,
            "min_rrr": 1.5,
            "max_concurrent_positions": 5,
            "risk_scale": 1.0
        }
        """

        try:
            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
            res = self.client.models.generate_content(
                model=self.active_model,
                contents=prompt,
                config=gen_config
            )
            if res and res.text:
                cleaned = res.text.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("
http://googleusercontent.com/immersive_entry_chip/0

---

### REGRESSION RISK
* **Sangat Rendah**: Seluruh 17 tahapan eksekusi `StepContext` dipertahankan tanpa mengubah nama fungsi maupun kontrak variabel pengembalian ke modul turunan (`data`, `risk`, `portfolio`, `reporting`).
* **Mitigasi**: Sifat Fail-Safe menjamin bahwa ketika terjadi masalah jaringan atau API Key tidak ditemukan, bot dihentikan secara aman (`exit code 1`) tanpa memunculkan transaksi palsu (`NO_TRADE`).

---

### TEST PLAN
1. **Missing GEMINI_API_KEY Test**: Unset `GEMINI_API_KEY` lalu jalankan `python main.py`. Pastikan log menampilkan `[ENV] API KEY ... FAIL` dan menghentikan proses dengan `exit 1`.
2. **Primary & Fallback Model Ping Test**: Jalankan dengan `GEMINI_API_KEY` valid. Pastikan pengujian diawali pada `gemini-3.6-flash`, lalu beralih ke `gemini-3.5-flash-lite` jika terjadi kegagalan transient.
3. **Automatic Trading Parameter Configuration Test**: Verifikasi bahwa method `get_dynamic_trading_parameters()` berhasil mengembalikan kamus terstruktur dari Gemini Administrator.
4. **Timezone Accuracy Test**: Verifikasi bahwa seluruh string timestamp dalam file JSON/log menyertakan offset `Asia/Jakarta` (`+07:00`).

---

### EXPECTED RESULT
* `main.py` berjalan penuh di bawah administrasi penuh Gemini AI, patuh 100% pada aturan `DINO IDX BOT Master Rules`.
* Pre-Flight Health Check menjamin integritas data sebelum pemrosesan ticker saham BEI.

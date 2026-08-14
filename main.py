"""
=============================================================================
IDX Quantitative Portfolio Engine - Main Production Orchestrator
FileName      : main.py
Version       : 2026.Q3.v26.3 (Full Pipeline Integration: Data + ML + Gemini)
Compliance    : Indonesia Stock Exchange (IDX) Trading Rules & Polars Engine
=============================================================================
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import polars as pl
from google import genai
from google.genai import types

# 1. Import Data Engine & Granular Exceptions (data.py v2026.8.2)
from data import (
    load_and_prepare_market_data,
    sanitize_ticker_list,
    DataSourceBlockedError,
    BEIEmptyResponseError,
    DataIngestionError
)

# 2. Import Machine Learning Engine (machine_learning.py v2026.Q3.v16.3)
try:
    from machine_learning import UnifiedModelEngine
    HAS_ML_ENGINE = True
except ImportError:
    UnifiedModelEngine = None
    HAS_ML_ENGINE = False

WIB_TZ = ZoneInfo("Asia/Jakarta")

DEFAULT_UNIVERSE_FILE = "universe.json"
DEFAULT_CACHE_DIR = ".cache"
DEFAULT_MODEL_PATH = "models/idx_ml_model.joblib"
DEFAULT_BLUECHIP_UNIVERSE = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK"]


# ==============================================================================
# HARD SAFETY BOUNDS FOR GEMINI AUTOCONFIG
# ==============================================================================
HARD_CONFIG_BOUNDS = {
    "min_adtv_idr": {"min": 50_000_000.0, "max": 10_000_000_000.0, "default": 2_500_000_000.0},
    "min_confidence": {"min": 0.50, "max": 0.85, "default": 0.65},
    "min_rrr": {"min": 1.50, "max": 3.00, "default": 1.80},
    "max_concurrent_positions": {"min": 1, "max": 10, "default": 5},
    "risk_scale": {"min": 0.10, "max": 1.00, "default": 0.80}
}


def apply_hard_safety_clamps(raw_config: Dict[str, Any], logger_instance) -> Dict[str, Any]:
    clamped_config = {}
    for key, bounds in HARD_CONFIG_BOUNDS.items():
        raw_val = raw_config.get(key, bounds["default"])
        try:
            val = float(raw_val) if key != "max_concurrent_positions" else int(raw_val)
        except (ValueError, TypeError):
            val = bounds["default"]

        clamped_val = max(bounds["min"], min(bounds["max"], val))
        if clamped_val != val:
            logger_instance.warning(
                f"🛡️ [HARD_CLAMP] Parameter '{key}' ({val}) di luar batas aman. "
                f"Di-clamp ke: {clamped_val} (Range: {bounds['min']} - {bounds['max']})"
            )
        clamped_config[key] = clamped_val

    clamped_config["configured_by"] = raw_config.get("configured_by", "SAFETY_DEFAULT")
    return clamped_config


# ==============================================================================
# LOGGER CONFIGURATION
# ==============================================================================
try:
    from logger import get_logger
    logger = get_logger("ProductionOrchestrator")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("ProductionOrchestrator")


# ==============================================================================
# GEMINI STATEFUL MANAGER WITH COOLDOWN
# ==============================================================================
class StatefulGeminiManager:
    def __init__(self, api_key: Optional[str]) -> None:
        self.api_key = api_key
        self.client = None
        self.primary_model = "gemini-3.6-flash"
        self.fallback_model = "gemini-3.5-flash-lite"
        self.active_model = None

        self.primary_in_cooldown = False
        self.primary_cooldown_until = 0.0

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("🤖 [GEMINI_INIT_SUCCESS] Google Gemini Client berhasil diinisialisasi.")
            except Exception as e:
                logger.warning(f"⚠️ [GEMINI_INIT_FAIL] Gagal inisialisasi Gemini Client: {e}")

    def check_and_select_model(self) -> Optional[str]:
        if not self.client:
            return None

        now = time.time()
        if self.primary_in_cooldown:
            if now < self.primary_cooldown_until:
                remaining_sec = int(self.primary_cooldown_until - now)
                logger.info(f"⏳ [GEMINI_COOLDOWN_ACTIVE] Primary ({self.primary_model}) dalam COOLDOWN ({remaining_sec}s tersisa). Menggunakan Fallback ({self.fallback_model}).")
                self.active_model = self.fallback_model
                return self.active_model
            else:
                logger.info(f"🔄 [GEMINI_COOLDOWN_EXPIRED] Masa cooldown untuk {self.primary_model} telah selesai.")
                self.primary_in_cooldown = False

        try:
            logger.info(f"🔍 [GEMINI_HEALTH] Testing model '{self.primary_model}'...")
            res = self.client.models.generate_content(
                model=self.primary_model,
                contents="Ping healthcheck",
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5)
            )
            if res.text:
                logger.info(f"✅ [GEMINI_HEALTH] Model '{self.primary_model}' PASS.")
                self.active_model = self.primary_model
                return self.active_model
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                logger.warning(f"🚨 [GEMINI_QUOTA_EXHAUSTED] 429 pada '{self.primary_model}'. Mengaktifkan 60s COOLDOWN.")
                self.primary_in_cooldown = True
                self.primary_cooldown_until = time.time() + 60.0
            else:
                logger.warning(f"⚠️ [GEMINI_HEALTH_FAIL] '{self.primary_model}' error: {e}")

        try:
            logger.info(f"🔍 [GEMINI_HEALTH] Testing model '{self.fallback_model}'...")
            res = self.client.models.generate_content(
                model=self.fallback_model,
                contents="Ping healthcheck",
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5)
            )
            if res.text:
                logger.info(f"✅ [GEMINI_HEALTH] Model '{self.fallback_model}' PASS.")
                self.active_model = self.fallback_model
                return self.active_model
        except Exception as e:
            logger.error(f"❌ [GEMINI_ALL_MODELS_FAIL] Fallback model '{self.fallback_model}' juga gagal: {e}")

        self.active_model = None
        return None

    def generate_adaptive_config(self, market_summary_info: str) -> Dict[str, Any]:
        if not self.client or not self.active_model:
            return apply_hard_safety_clamps({}, logger)

        prompt = f"""
        Kamu adalah AI Chief Risk Officer untuk Portofolio Quantitative Saham Indonesia (BEI).
        Berdasarkan ringkasan data pasar berikut:
        {market_summary_info}

        Berikan konfigurasi trading JSON persis dalam format ini:
        {{
            "min_adtv_idr": float,
            "min_confidence": float,
            "min_rrr": float,
            "max_concurrent_positions": int,
            "risk_scale": float
        }}
        """

        try:
            res = self.client.models.generate_content(
                model=self.active_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
            )
            raw_json = json.loads(res.text)
            raw_json["configured_by"] = f"GEMINI_{self.active_model}"
            return apply_hard_safety_clamps(raw_json, logger)
        except Exception as e:
            logger.warning(f"⚠️ [GEMINI_CONFIG_FAIL] Gagal memuat parameter adaptif Gemini: {e}.")
            return apply_hard_safety_clamps({}, logger)


# ==============================================================================
# PRODUCTION ORCHESTRATOR
# ==============================================================================
class ProductionOrchestrator:
    def __init__(self, dry_run: bool = True, self_learning: bool = False) -> None:
        self.dry_run = dry_run
        self.self_learning = self_learning
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_mgr = StatefulGeminiManager(api_key=self.api_key)
        
        self.universe_file = DEFAULT_UNIVERSE_FILE
        self.universe_tickers = []
        self.market_data = pl.DataFrame()
        self.predictions_data = pl.DataFrame()
        self.trading_config = apply_hard_safety_clamps({}, logger)
        
        # Inisialisasi ML Engine
        self.ml_engine = None
        if HAS_GEMINI_SDK and HAS_ML_ENGINE:
            os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
            loaded_engine = UnifiedModelEngine.load_model(DEFAULT_MODEL_PATH)
            if loaded_engine is not None:
                self.ml_engine = loaded_engine
                logger.info("📦 [ML_ENGINE_LOADED] Berhasil memuat model ML terkalibrasi dari disk.")
            else:
                self.ml_engine = UnifiedModelEngine(gemini_api_key=self.api_key)
                logger.info("🧠 [ML_ENGINE_INIT] Berhasil inisialisasi UnifiedModelEngine baru.")

    def _run_preflight_checks(self) -> bool:
        logger.info("==================================================")
        logger.info("🔍 STARTING PRE-FLIGHT HEALTH CHECKS")
        logger.info("==================================================")

        active_model = self.gemini_mgr.check_and_select_model()
        if active_model:
            logger.info(f"✅ [GEMINI_AVAILABLE] Gemini AI aktif menggunakan: {active_model}")
        else:
            logger.warning("⚠️ [GEMINI_UNAVAILABLE] AI Offline. Pipeline akan menggunakan parameter Deterministik Default.")

        # Load & Sanitize Universe
        if os.path.exists(self.universe_file):
            try:
                with open(self.universe_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                if isinstance(raw_data, dict):
                    raw_list = raw_data.get("symbols", list(raw_data.keys()))
                elif isinstance(raw_data, list):
                    raw_list = raw_data
                else:
                    raw_list = DEFAULT_BLUECHIP_UNIVERSE

                self.universe_tickers = sanitize_ticker_list(raw_list)
                logger.info(f"[UNIVERSE] Loaded & Sanitized {len(self.universe_tickers)} valid tickers ............. PASS")
            except Exception as e:
                logger.error(f"❌ Berkas universe.json rusak: {e}. Menggunakan default bluechip universe.")
                self.universe_tickers = DEFAULT_BLUECHIP_UNIVERSE
        else:
            logger.warning("⚠️ Berkas universe.json tidak ditemukan. Inisialisasi default bluechip universe...")
            self.universe_tickers = DEFAULT_BLUECHIP_UNIVERSE

        # Tulis ulang universe.json ter-sanitasi secara bersih
        with open(self.universe_file, "w", encoding="utf-8") as f:
            json.dump(self.universe_tickers, f, indent=2)

        logger.info("==================================================")
        return True

    def _step_1_universe_sync(self) -> None:
        logger.info("▶ [START STEP 1] Bootstrap & Dynamic Universe Synchronization")
        start = time.perf_counter()
        
        with open(self.universe_file, "w", encoding="utf-8") as f:
            json.dump(self.universe_tickers, f, indent=2)

        elapsed = time.perf_counter() - start
        logger.info(f"✔ [SUCCESS STEP 1] Universe Synchronization selesai dalam {elapsed:.4f}s. (Total: {len(self.universe_tickers)} ticker)")

    def _step_2_data_layer_isolated(self) -> None:
        logger.info("▶ [START STEP 2] Ingest Market Data via Integrated data.py Engine")
        
        max_retries = 3
        backoff_sec = 2.0

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"🔄 [STEP_2_ATTEMPT {attempt}/{max_retries}] Mengunduh data pasar BEI...")
                df, delisted = load_and_prepare_market_data(symbols=self.universe_tickers, use_cache=True)

                if df is not None and df.height > 0:
                    self.market_data = df
                    logger.info(f"✅ [SUCCESS STEP 2] Ingestion Data Pasar berhasil memuat {df.height} baris data.")

                    if delisted:
                        logger.info(f"🧹 Membersihkan {len(delisted)} ticker delisted dari universe...")
                        self.universe_tickers = [t for t in self.universe_tickers if t not in delisted]
                        with open(self.universe_file, "w", encoding="utf-8") as f:
                            json.dump(self.universe_tickers, f, indent=2)
                    return

            except DataSourceBlockedError as e:
                logger.warning(f"⚠️ [STEP_2_BLOCKED] Akses BEI Primary diblokir WAF. Failover diaktifkan: {e}")
            except BEIEmptyResponseError as e:
                logger.warning(f"⚠️ [STEP_2_EMPTY_RESPONSE] Percobaan {attempt} gagal: {e}")
            except DataIngestionError as e:
                logger.warning(f"⚠️ [STEP_2_DATA_ERROR] Ingestion error: {e}")

            if attempt < max_retries:
                logger.info(f"⏳ Menunggu {backoff_sec:.1f}s sebelum mencoba ulang Step 2...")
                time.sleep(backoff_sec)
                backoff_sec *= 1.5

        if self.market_data.is_empty():
            err_msg = f"Step 2 Gagal total setelah {max_retries} percobaan. Tidak ada DataFrame valid."
            logger.error(f"✖ [FAILED STEP 2] {err_msg}")
            raise ValueError(err_msg)

    def _step_3_adaptive_config(self) -> None:
        logger.info("▶ [START STEP 3] Gemini Adaptive Configuration (Post-Data Quality Gate)")
        
        asset_col = "ticker" if "ticker" in self.market_data.columns else "asset"
        sample_assets_count = self.market_data[asset_col].n_unique()
        latest_date = str(self.market_data["date"].max())
        total_rows = self.market_data.height

        summary_str = f"Tanggal Data: {latest_date}, Total Emiten Aktif: {sample_assets_count}, Total Row: {total_rows}"
        
        raw_config = self.gemini_mgr.generate_adaptive_config(summary_str)
        self.trading_config = apply_hard_safety_clamps(raw_config, logger)

        logger.info(f"🤖 [CONFIG_APPLIED] Parameter Trading Aktif: {self.trading_config}")

    def _step_4_machine_learning_pipeline(self) -> None:
        logger.info("▶ [START STEP 4] Machine Learning Inference, Calibration & Conformal Prediction")
        
        if not HAS_ML_ENGINE or self.ml_engine is None:
            logger.warning("⚠️ [STEP_4_SKIPPED] Modul ML tidak tersedia. Melewati inferensi ML.")
            return

        start = time.perf_counter()

        # Mode Self-Learning / Retraining jika diinstruksikan
        if self.self_learning:
            logger.info("🧠 [SELF_LEARNING] Memicu retraining dan pembaruan model ML...")
            retrain_meta = self.ml_engine.retrain_and_update_models(self.market_data)
            self.ml_engine.save_model(DEFAULT_MODEL_PATH)
            logger.info(f"✅ [SELF_LEARNING_SUCCESS] Retraining selesai. Evaluasi AI: {retrain_meta.get('ai_feature_analysis')}")

        # Inferensi & Kalibrasi Probabilitas
        self.predictions_data = self.ml_engine.predict_and_calibrate(self.market_data)
        elapsed = time.perf_counter() - start

        high_conf_signals = self.predictions_data.filter(
            pl.col("prediction_probability") >= self.trading_config["min_confidence"]
        ).height

        logger.info(
            f"✔ [SUCCESS STEP 4] ML Inference selesai dalam {elapsed:.4f}s. "
            f"Ditemukan {high_conf_signals} sinyal berkeyakinan tinggi (>= {self.trading_config['min_confidence']:.2f})."
        )

    def run(self) -> None:
        if not self._run_preflight_checks():
            logger.error("❌ Preflight healthcheck gagal. Menghentikan pipeline.")
            sys.exit(1)

        try:
            self._step_1_universe_sync()
            self._step_2_data_layer_isolated()
            self._step_3_adaptive_config()
            self._step_4_machine_learning_pipeline()

            logger.info("==================================================")
            logger.info("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("==================================================")

        except Exception as e:
            logger.error(f"💥 Pipeline terhenti karena error fatal: {e}")
            sys.exit(1)


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Quantitative Portfolio Engine")
    parser.add_argument("--dry-run", action="store_true", help="Jalankan pipeline dalam mode simulasi/dry-run")
    parser.add_argument("--self-learning", action="store_true", help="Picu retraining otomatis pada Machine Learning Engine")
    parser.add_argument("--bootstrap-universe", action="store_true", help="Bootstrap universe.json file")
    args = parser.parse_args()

    if args.bootstrap_universe:
        sanitised = sanitize_ticker_list(DEFAULT_BLUECHIP_UNIVERSE)
        with open(DEFAULT_UNIVERSE_FILE, "w", encoding="utf-8") as f:
            json.dump(sanitised, f, indent=2)
        print(f"✅ universe.json berhasil dipulihkan dengan {len(sanitised)} ticker valid.")
        sys.exit(0)

    orchestrator = ProductionOrchestrator(dry_run=args.dry_run, self_learning=args.self_learning)
    orchestrator.run()

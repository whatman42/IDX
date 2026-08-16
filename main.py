"""
=============================================================================
IDX Quantitative Portfolio Engine - Main Production Orchestrator
FileName      : main.py
Version       : 2026.Q3.v26.5 (Scalping Intraday Engine & High-ADTV Strategy)
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

# Try Google GenAI SDK Import & Define Guard
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_SDK = True
except ImportError:
    genai = None
    types = None
    HAS_GEMINI_SDK = False

# 1. Import Data Engine & Granular Exceptions (data.py v2026.8.3)
from data import (
    load_and_prepare_market_data,
    sanitize_ticker_list,
    DataSourceBlockedError,
    BEIEmptyResponseError,
    DataIngestionError
)

# 2. Import Machine Learning Engine (machine_learning.py v2026.Q3.v16.6 - Scalping)
try:
    from machine_learning import UnifiedModelEngine
    HAS_ML_ENGINE = True
except ImportError:
    UnifiedModelEngine = None
    HAS_ML_ENGINE = False

WIB_TZ = ZoneInfo("Asia/Jakarta")

DEFAULT_UNIVERSE_FILE = "universe.json"
DEFAULT_CACHE_DIR = ".cache"
DEFAULT_MODEL_PATH = "models/idx_ml_scalping_model.joblib"
DEFAULT_BLUECHIP_UNIVERSE = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK", "AMMN.JK", "BREN.JK"]


# ==============================================================================
# HARD SAFETY BOUNDS FOR SCALPING MODE (INTRADAY ULTRA-FAST)
# ==============================================================================
HARD_CONFIG_BOUNDS = {
    # Likuiditas sangat krusial untuk Scalping agar tidak terkena slippage/angkut
    "min_adtv_idr": {"min": 500_000_000.0, "max": 20_000_000_000.0, "default": 5_000_000_000.0},
    # Min Confidence dinaikkan ke 0.70 agar hanya memicu sinyal scalping ber-winrate tinggi
    "min_confidence": {"min": 0.60, "max": 0.90, "default": 0.72},
    # Risk-to-Reward Scalping rasional (1:1.0 s/d 1:2.0) dengan pergerakan cepat
    "min_rrr": {"min": 1.00, "max": 2.00, "default": 1.20},
    # Maksimal 3-5 saham bersamaan untuk kemudahan kontrol & fokus eksekusi
    "max_concurrent_positions": {"min": 1, "max": 5, "default": 3},
    # Alokasi risiko per trade lebih ketat (10% - 40% dari total ekuitas)
    "risk_scale": {"min": 0.05, "max": 0.50, "default": 0.20}
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
                f"🛡️ [HARD_CLAMP] Parameter Scalping '{key}' ({val}) di luar batas aman. "
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
# GEMINI STATEFUL MANAGER WITH COOLDOWN (SCALPING PROMPT)
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

        if HAS_GEMINI_SDK and self.api_key:
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
        Kamu adalah Chief Scalping Risk Officer untuk Portofolio Kuantitatif Saham Indonesia (BEI).
        Tujuan utama: Strategi Scalping Intraday (Hit & Run) dengan target profit kecil (1.5% - 3.5%) dan ekspektasi holding sangat singkat (intraday / T+0).

        Ringkasan Kondisi Pasar Saat Ini:
        {market_summary_info}

        Tentukan parameter strategi scalping optimal dalam format JSON persis berikut:
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
            raw_json["configured_by"] = f"GEMINI_SCALPER_{self.active_model}"
            return apply_hard_safety_clamps(raw_json, logger)
        except Exception as e:
            logger.warning(f"⚠️ [GEMINI_CONFIG_FAIL] Gagal memuat parameter adaptif Scalping Gemini: {e}.")
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
        if HAS_ML_ENGINE:
            os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
            loaded_engine = UnifiedModelEngine.load_model(DEFAULT_MODEL_PATH) if hasattr(UnifiedModelEngine, "load_model") else None
            if loaded_engine is not None:
                self.ml_engine = loaded_engine
                logger.info("📦 [ML_ENGINE_LOADED] Berhasil memuat model ML Scalping dari disk.")
            else:
                self.ml_engine = UnifiedModelEngine(gemini_api_key=self.api_key)
                logger.info("🧠 [ML_ENGINE_INIT] Berhasil inisialisasi Scalping UnifiedModelEngine baru.")

    def _run_preflight_checks(self) -> bool:
        logger.info("==================================================")
        logger.info("🔍 STARTING PRE-FLIGHT HEALTH CHECKS (SCALPING MODE)")
        logger.info("==================================================")

        active_model = self.gemini_mgr.check_and_select_model()
        if active_model:
            logger.info(f"✅ [GEMINI_AVAILABLE] Gemini AI Scalper aktif menggunakan: {active_model}")
        else:
            logger.warning("⚠️ [GEMINI_UNAVAILABLE] AI Offline. Menggunakan Parameter Scalping Default.")

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
        logger.info("▶ [START STEP 3] Gemini Scalping Adaptive Configuration")
        
        asset_col = "ticker" if "ticker" in self.market_data.columns else "asset"
        sample_assets_count = self.market_data[asset_col].n_unique()
        latest_date = str(self.market_data["date"].max())
        total_rows = self.market_data.height

        summary_str = f"Mode: Scalping Intraday, Tanggal Data: {latest_date}, Total Emiten Aktif: {sample_assets_count}, Total Row: {total_rows}"
        
        raw_config = self.gemini_mgr.generate_adaptive_config(summary_str)
        self.trading_config = apply_hard_safety_clamps(raw_config, logger)

        logger.info(f"🤖 [CONFIG_APPLIED] Parameter Scalping Aktif: {self.trading_config}")

    def _step_4_machine_learning_pipeline(self) -> None:
        logger.info("▶ [START STEP 4] Scalping ML Inference, Calibration & Signal Generation")
        
        if not HAS_ML_ENGINE or self.ml_engine is None:
            logger.warning("⚠️ [STEP_4_SKIPPED] Modul ML tidak tersedia. Melewati inferensi ML.")
            return

        start = time.perf_counter()

        # Mode Self-Learning jika diinstruksikan
        if self.self_learning:
            logger.info("🧠 [SELF_LEARNING] Memicu retraining model Scalping...")
            if hasattr(self.ml_engine, "retrain_and_update_models"):
                retrain_meta = self.ml_engine.retrain_and_update_models(self.market_data)
                if hasattr(self.ml_engine, "save_model"):
                    self.ml_engine.save_model(DEFAULT_MODEL_PATH)
                logger.info(f"✅ [SELF_LEARNING_SUCCESS] Retraining Scalping selesai.")

        # Inferensi & Kalibrasi Sinyal Scalping
        if hasattr(self.ml_engine, "predict_and_calibrate"):
            self.predictions_data = self.ml_engine.predict_and_calibrate(self.market_data)
            elapsed = time.perf_counter() - start

            high_conf_signals = self.predictions_data.filter(
                pl.col("prediction_probability") >= self.trading_config["min_confidence"]
            ).height

            logger.info(
                f"✔ [SUCCESS STEP 4] Scalping ML Inference selesai dalam {elapsed:.4f}s. "
                f"Ditemukan {high_conf_signals} sinyal scalping presisi tinggi (>= {self.trading_config['min_confidence']:.2f})."
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
            logger.info("🎉 SCALPING PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("==================================================")

        except Exception as e:
            logger.error(f"💥 Scalping Pipeline terhenti karena error fatal: {e}")
            sys.exit(1)


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Quantitative Scalping Engine")
    parser.add_argument("--dry-run", action="store_true", help="Jalankan pipeline dalam mode simulasi/dry-run (NO live order, NO production portfolio mutation)")
    parser.add_argument("--self-learning", action="store_true", help="Picu retraining otomatis pada Machine Learning Engine")
    parser.add_argument("--bootstrap-universe", action="store_true", help="Bootstrap universe.json file")
    parser.add_argument(
        "--reset-dryrun",
        action="store_true",
        help="Reset dry-run simulation state (local simulation DBs only; never touches live production portfolio)",
    )
    args = parser.parse_args()

    if args.bootstrap_universe:
        sanitised = sanitize_ticker_list(DEFAULT_BLUECHIP_UNIVERSE)
        with open(DEFAULT_UNIVERSE_FILE, "w", encoding="utf-8") as f:
            json.dump(sanitised, f, indent=2)
        print(f"✅ universe.json berhasil dipulihkan dengan {len(sanitised)} ticker valid.")
        sys.exit(0)

    if args.reset_dryrun:
        # Safe reset of dry-run / simulation artifacts only
        import pathlib
        reset_targets = [
            "data/storage_dryrun.db",
            "storage_dryrun.db",
            "autonomous_engine.db",
        ]
        removed = []
        for rel in reset_targets:
            p = pathlib.Path(rel)
            if p.exists() and p.is_file():
                try:
                    p.unlink()
                    removed.append(str(p))
                except OSError as e:
                    print(f"⚠️ Gagal menghapus {p}: {e}")
        print(f"✅ [RESET-DRYRUN] Simulasi state di-reset. Removed: {removed or 'none (already clean)'}")
        print("ℹ️ Production portfolio / live state TIDAK disentuh.")
        sys.exit(0)

    # Always prefer dry-run unless an explicit future live flag is introduced.
    # This commit does NOT enable live trading.
    orchestrator = ProductionOrchestrator(dry_run=True, self_learning=args.self_learning)
    if not args.dry_run and not args.self_learning:
        logger.info("🛡️ [SAFETY] No explicit --dry-run provided; forcing dry_run=True (no live orders).")
    orchestrator.run()

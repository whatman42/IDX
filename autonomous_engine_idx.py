"""
=============================================================================
Module      : autonomous_engine_idx.py
Description : IDX Autonomous Engine - Refactored Self-Learning & Gemini AI Feedback Loop
Version     : 2026.Q3.v18.0 (Gemini 3.6 Flash & Interactions API Integrated)
Directory   : Flat Directory (Root Level Integration with main.py)
Compliance  : Indonesia Stock Exchange (IDX) Trading Rules & Polars Memory Synergy
=============================================================================
"""

import json
import logging
import math
import os
import random
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

# Import Google GenAI Client (SDK Baru: google-genai)
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# ==============================================================================
# HARD SAFETY LIMITS & IDX CONSTANTS (Absolute Authority - Python Owned)
# ==============================================================================
HARD_LIMITS: Dict[str, float] = {
    "IDX_MIN_PRICE_IDR": 50.0,            # Batas Minimal Harga Saham IDX (Papan Efek IDR 50)
    "IDX_MIN_24H_TURNOVER_IDR": 1_000_000.0, # Nilai Transaksi Harian Minimal (Rp 1 Juta)
    "IDX_MAX_STALENESS_SEC": 172800.0,     # Toleransi Keusangan Data Candle (48 Jam)
    "MAX_DRAWDOWN_KILL_PCT": 0.20,        # Emergency Kill-Switch Drawdown (20%)
    "MAX_POSITION_PCT": 0.30,             # Alokasi Maksimal Per Ticker (30%)
    "MIN_POSITION_PCT": 0.05,             # Alokasi Minimal Per Ticker (5%)
    "MIN_CONFIDENCE": 0.45,               # Ambang Minimal Kepercayaan Model
    "MAX_CONFIDENCE": 0.75,               # Ambang Maksimal Kepercayaan Model
    "MAX_RISK_SCALE": 1.20,               # Multiplier Risiko Maksimal AI
    "MIN_RISK_SCALE": 0.50,               # Multiplier Risiko Minimal AI
    "MAX_AI_POSITION_MULTIPLIER": 1.20,   # Batas Atas Multiplier Posisi AI
    "MAX_PORTFOLIO_EXPOSURE_PCT": 1.00     # Total Maksimal Eksposure Portofolio (100%)
}

IDX_BUY_FEE_PCT: float = 0.0015            # Biaya Beli Broker + Levy (0.15%)
IDX_SELL_FEE_PCT: float = 0.0015           # Biaya Jual Broker + Levy + PPh Final (0.15%)
IDX_FEE_ROUNDTRIP_PCT: float = 0.003       # Total Roundtrip Fee Standar BEI (~0.3%)
PREDIKSI_IDX_LOG_PATH: str = "prediksi_idx_log.csv"
AUTONOMOUS_DB_PATH: str = "autonomous_engine.db"
DEFAULT_TIMEZONE: str = "Asia/Jakarta"
EPSILON: float = 1e-9

# Model Gemini Hierarchy
GEMINI_PRIMARY_MODEL: str = "gemini-3.6-flash"
GEMINI_FALLBACK_MODEL: str = "gemini-3.5-flash-lite"

# Backward Compatibility Constants Aliases (Crypto -> IDX Sync)
TOKOCRYPTO_FEE_ROUNDTRIP_PCT: float = IDX_FEE_ROUNDTRIP_PCT
TOKOCRYPTO_MIN_PRICE_USDT: float = HARD_LIMITS["IDX_MIN_PRICE_IDR"]
TOKOCRYPTO_MIN_24H_VOLUME_USDT: float = HARD_LIMITS["IDX_MIN_24H_TURNOVER_IDR"]
TOKOCRYPTO_MAX_STALENESS_SEC: float = HARD_LIMITS["IDX_MAX_STALENESS_SEC"]
PREDIKSI_CRYPTO_LOG_PATH: str = PREDIKSI_IDX_LOG_PATH


# ==============================================================================
# DEFENSIVE HELPER FUNCTIONS
# ==============================================================================
def _ensure_polars_df_auto(data: Any, default_cols: Optional[List[str]] = None) -> pl.DataFrame:
    """Sanitasi input internal untuk menjamin objek bertipe pl.DataFrame."""
    if data is None:
        cols = default_cols or ["symbol", "close"]
        return pl.DataFrame(schema={col: pl.Utf8 for col in cols})
    if isinstance(data, list):
        if not data:
            cols = default_cols or ["symbol", "close"]
            return pl.DataFrame(schema={col: pl.Utf8 for col in cols})
        return pl.DataFrame(data)
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
    except ImportError:
        pass
    return pl.DataFrame(data)


class StructuredJSONLogger:
    """Formatter & Handler Logging Terstruktur JSON tanpa kebocoran Rahasia."""

    def __init__(self, logger_name: str = "IDX.AutonomousEngine"):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_event(self, level: str, event_type: str, data: Dict[str, Any]) -> None:
        # Masking API key jika ada di data
        sanitized_data = self._sanitize_dict(data)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": "IDX.AutonomousEngine",
            "level": level.upper(),
            "event_type": event_type,
            "data": sanitized_data,
        }
        msg = json.dumps(payload, default=self._json_serializer)
        if level.upper() == "CRITICAL":
            self.logger.critical(msg)
        elif level.upper() == "ERROR":
            self.logger.error(msg)
        elif level.upper() == "WARNING":
            self.logger.warning(msg)
        else:
            self.logger.info(msg)

    def _sanitize_dict(self, d: Any) -> Any:
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                if any(sec in k.lower() for sec in ["api_key", "token", "secret", "password"]):
                    new_dict[k] = "[REDACTED]"
                else:
                    new_dict[k] = self._sanitize_dict(v)
            return new_dict
        elif isinstance(d, list):
            return [self._sanitize_dict(x) for x in d]
        return d

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, ZoneInfo)):
            return str(obj)
        elif math.isinf(obj) if isinstance(obj, float) else False:
            return "INF"
        return str(obj)


json_logger = StructuredJSONLogger()


# ==============================================================================
# GEMINI RESPONSE VALIDATOR & CLAMPING ENGINE
# ==============================================================================
class GeminiResponseValidator:
    """Validator & Clamping Engine terisolasi untuk Respon AI Gemini."""

    @staticmethod
    def validate_and_clamp(raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """Memvalidasi tipe data, jangkauan numerik, dan melakukan clamping ke HARD_LIMITS."""
        fallback = {
            "diagnostic": "Fallback: Respon AI tidak valid atau mengalami kegagalan validasi skema.",
            "market_regime": "NORMAL",
            "system_state": "NORMAL",
            "recommendation": "Pertahankan ambang risiko kuantitatif standar.",
            "risk_scale": 1.0,
            "confidence_threshold": 0.50,
            "position_scale": 0.25,
            "retraining_required": False,
            "risk_reasons": ["AI_VALIDATION_FALLBACK_TRIGGERED"],
            "anomalies": [],
            "validation_status": "FALLBACK"
        }

        if not isinstance(raw_response, dict):
            return fallback

        try:
            diagnostic = str(raw_response.get("diagnostic", fallback["diagnostic"]))
            market_regime = str(raw_response.get("market_regime", "NORMAL")).upper()
            if market_regime not in {"NORMAL", "HIGH_VOLATILITY", "SEVERE_VOLATILITY", "BEARISH_STRESS"}:
                market_regime = "NORMAL"

            system_state = str(raw_response.get("system_state", "NORMAL")).upper()
            if system_state not in {"NORMAL", "CAUTION", "RISK_REDUCTION", "RETRAIN", "HALT"}:
                system_state = "NORMAL"

            recommendation = str(raw_response.get("recommendation", fallback["recommendation"]))

            # Validasi & Clamp risk_scale
            raw_risk_scale = raw_response.get("risk_scale", 1.0)
            if not isinstance(raw_risk_scale, (int, float)) or math.isnan(raw_risk_scale) or math.isinf(raw_risk_scale):
                risk_scale = 1.0
            else:
                risk_scale = max(HARD_LIMITS["MIN_RISK_SCALE"], min(HARD_LIMITS["MAX_RISK_SCALE"], float(raw_risk_scale)))

            # Validasi & Clamp confidence_threshold
            raw_conf = raw_response.get("confidence_threshold", 0.50)
            if not isinstance(raw_conf, (int, float)) or math.isnan(raw_conf) or math.isinf(raw_conf):
                conf_thresh = 0.50
            else:
                conf_thresh = max(HARD_LIMITS["MIN_CONFIDENCE"], min(HARD_LIMITS["MAX_CONFIDENCE"], float(raw_conf)))

            # Validasi & Clamp position_scale
            raw_pos = raw_response.get("position_scale", 0.25)
            if not isinstance(raw_pos, (int, float)) or math.isnan(raw_pos) or math.isinf(raw_pos):
                pos_scale = 0.25
            else:
                pos_scale = max(HARD_LIMITS["MIN_POSITION_PCT"], min(HARD_LIMITS["MAX_POSITION_PCT"], float(raw_pos)))

            retraining_req = bool(raw_response.get("retraining_required", False))

            risk_reasons = raw_response.get("risk_reasons", [])
            if not isinstance(risk_reasons, list):
                risk_reasons = [str(risk_reasons)]
            else:
                risk_reasons = [str(r) for r in risk_reasons]

            anomalies = raw_response.get("anomalies", [])
            if not isinstance(anomalies, list):
                anomalies = [str(anomalies)]
            else:
                anomalies = [str(a) for a in anomalies]

            return {
                "diagnostic": diagnostic,
                "market_regime": market_regime,
                "system_state": system_state,
                "recommendation": recommendation,
                "risk_scale": risk_scale,
                "confidence_threshold": conf_thresh,
                "position_scale": pos_scale,
                "retraining_required": retraining_req,
                "risk_reasons": risk_reasons,
                "anomalies": anomalies,
                "validation_status": "VALIDATED"
            }
        except Exception as err:
            fallback["risk_reasons"] = [f"Validation Exception: {str(err)}"]
            return fallback


# ==============================================================================
# PERSISTENCE ENGINE (SQLITE)
# ==============================================================================
class AutonomousSQLiteStore:
    """Persistensi Metrik Otonom & Gemini Diagnostics Terpisah."""

    def __init__(self, db_path: str = AUTONOMOUS_DB_PATH):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    # Tabel Historis Metrik
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS metrics_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            evaluated_signals INTEGER,
                            prediction_mae REAL,
                            execution_mae REAL,
                            composite_mae REAL,
                            win_rate REAL,
                            profit_factor REAL,
                            sharpe_ratio REAL,
                            sortino_ratio REAL,
                            calmar_ratio REAL,
                            cvar_95 REAL,
                            kelly_fraction REAL,
                            max_streak_loss INTEGER,
                            brier_score REAL,
                            ece_calibration REAL,
                            psi_drift REAL,
                            peak_equity REAL,
                            current_drawdown_pct REAL,
                            risk_config_json TEXT
                        )
                    """)
                    # Tabel Diagnostik Gemini
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS gemini_diagnostics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            primary_model TEXT,
                            used_model TEXT,
                            latency_sec REAL,
                            is_healthy INTEGER,
                            validation_status TEXT,
                            market_regime TEXT,
                            system_state TEXT,
                            risk_scale_proposed REAL,
                            risk_scale_final REAL,
                            diagnostic TEXT,
                            recommendation TEXT,
                            raw_response_json TEXT
                        )
                    """)
                    conn.commit()
            except Exception as err:
                json_logger.log_event("ERROR", "SQLITE_INIT_FAILED", {"error": str(err)})

    def record_metrics(self, metrics: Dict[str, Any], risk_config: Dict[str, Any], drawdown_pct: float) -> None:
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO metrics_history (
                            timestamp, evaluated_signals, prediction_mae, execution_mae, composite_mae,
                            win_rate, profit_factor, sharpe_ratio, sortino_ratio, calmar_ratio,
                            cvar_95, kelly_fraction, max_streak_loss, brier_score, ece_calibration,
                            psi_drift, peak_equity, current_drawdown_pct, risk_config_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now(timezone.utc).isoformat(),
                        int(metrics.get("total_evaluated_signals", 0)),
                        float(metrics.get("prediction_mae", 0.0)),
                        float(metrics.get("execution_mae", 0.0)),
                        float(metrics.get("rolling_mae", 0.0)),
                        float(metrics.get("win_rate", 0.0)),
                        None if math.isinf(metrics.get("profit_factor", 1.0)) else float(metrics.get("profit_factor", 1.0)),
                        float(metrics.get("sharpe_ratio", 0.0)),
                        float(metrics.get("sortino_ratio", 0.0)),
                        float(metrics.get("calmar_ratio", 0.0)),
                        float(metrics.get("cvar_95", 0.0)),
                        float(metrics.get("kelly_fraction", 0.0)),
                        int(metrics.get("max_consecutive_loss", 0)),
                        float(metrics.get("brier_score", 0.0)),
                        float(metrics.get("ece_calibration", 0.0)),
                        float(metrics.get("psi_drift_score", 0.0)),
                        float(metrics.get("peak_equity_idr", 10_000_000.0)),
                        float(drawdown_pct),
                        json.dumps(risk_config)
                    ))
                    conn.commit()
            except Exception as err:
                json_logger.log_event("ERROR", "SQLITE_RECORD_FAILED", {"error": str(err)})

    def record_gemini_diagnostic(self, diag_res: Dict[str, Any], latency: float, used_model: str, is_healthy: bool) -> None:
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO gemini_diagnostics (
                            timestamp, primary_model, used_model, latency_sec, is_healthy,
                            validation_status, market_regime, system_state,
                            risk_scale_proposed, risk_scale_final, diagnostic,
                            recommendation, raw_response_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now(timezone.utc).isoformat(),
                        GEMINI_PRIMARY_MODEL,
                        used_model,
                        latency,
                        1 if is_healthy else 0,
                        diag_res.get("validation_status", "UNKNOWN"),
                        diag_res.get("market_regime", "NORMAL"),
                        diag_res.get("system_state", "NORMAL"),
                        float(diag_res.get("proposed_risk_scale", 1.0)),
                        float(diag_res.get("risk_scale", 1.0)),
                        diag_res.get("diagnostic", ""),
                        diag_res.get("recommendation", ""),
                        json.dumps(diag_res)
                    ))
                    conn.commit()
            except Exception as err:
                json_logger.log_event("ERROR", "SQLITE_RECORD_GEMINI_FAILED", {"error": str(err)})


# ==============================================================================
# GEMINI 3.6 FLASH META-DIAGNOSTIC ENGINE (INTERACTIONS API)
# ==============================================================================
class GeminiAutonomousDiagnosticEngine:
    """
    Meta-Diagnostic Sub-Engine berbasis SDK google-genai & Interactions API.
    Memasok rekomendasi adaptif kualitatif yang tunduk pada Hard Guardrails.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self.is_healthy = False
        self.active_model = GEMINI_PRIMARY_MODEL

        if HAS_GEMINI_SDK and self.api_key:
            try:
                # Inisialisasi SDK google-genai resmi dengan api_version v1
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options={"api_version": "v1"}
                )
                json_logger.log_event("INFO", "GEMINI_CLIENT_INIT_SUCCESS", {"primary_model": GEMINI_PRIMARY_MODEL})
            except Exception as err:
                json_logger.log_event("WARNING", "GEMINI_CLIENT_INIT_FAILED", {"error": str(err)})
        else:
            json_logger.log_event("WARNING", "GEMINI_DISABLED", {"reason": "API Key missing or SDK not installed."})

    def check_health(self) -> Tuple[bool, float, str]:
        """Melakukan ping ringan ke Gemini untuk memverifikasi kesiapan model."""
        if not self.client:
            self.is_healthy = False
            return False, 0.0, "CLIENT_UNAVAILABLE"

        start_time = time.time()
        for model_candidate in [GEMINI_PRIMARY_MODEL, GEMINI_FALLBACK_MODEL]:
            try:
                # Memanggil Interactions API sesuai spesifikasi master
                if hasattr(self.client, "interactions") and callable(getattr(self.client, "interactions", None)):
                    res = self.client.interactions.create(
                        model=model_candidate,
                        input="Health check. Respond with 'OK'."
                    )
                    _ = getattr(res, "output_text", None) or str(res)
                else:
                    # Fallback ke generate_content SDK baru
                    res = self.client.models.generate_content(
                        model=model_candidate,
                        contents="Health check. Respond with 'OK'."
                    )
                latency = time.time() - start_time
                self.is_healthy = True
                self.active_model = model_candidate
                json_logger.log_event("INFO", "GEMINI_HEALTH_CHECK_PASS", {
                    "model": model_candidate, "latency_sec": round(latency, 3)
                })
                return True, latency, model_candidate
            except Exception as err:
                json_logger.log_event("WARNING", "GEMINI_HEALTH_CHECK_MODEL_FAILED", {
                    "model": model_candidate, "error": str(err)
                })

        self.is_healthy = False
        json_logger.log_event("ERROR", "GEMINI_HEALTH_CHECK_FAILED", {"reason": "All models unresponsive."})
        return False, time.time() - start_time, "ALL_MODELS_FAILED"

    def _execute_api_call_with_retry(self, prompt: str) -> Tuple[Optional[str], str]:
        """Panggilan API dengan retry exponential backoff untuk transient error."""
        max_attempts = 3
        backoff = 2.0

        models_to_try = [GEMINI_PRIMARY_MODEL, GEMINI_FALLBACK_MODEL]

        for model_id in models_to_try:
            for attempt in range(1, max_attempts + 1):
                try:
                    if hasattr(self.client, "interactions") and callable(getattr(self.client, "interactions", None)):
                        response = self.client.interactions.create(
                            model=model_id,
                            input=prompt
                        )
                        text = getattr(response, "output_text", None) or str(response)
                    else:
                        response = self.client.models.generate_content(
                            model=model_id,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json"
                            ) if 'types' in globals() else None
                        )
                        text = response.text

                    if text and len(text.strip()) > 0:
                        return text.strip(), model_id

                except Exception as err:
                    err_str = str(err)
                    is_transient = any(code in err_str for code in ["429", "500", "502", "503", "504", "timeout", "Connection"])
                    json_logger.log_event("WARNING", "GEMINI_API_ATTEMPT_FAILED", {
                        "model": model_id, "attempt": attempt, "error": err_str, "is_transient": is_transient
                    })
                    if not is_transient or attempt == max_attempts:
                        break
                    time.sleep(backoff * (2 ** (attempt - 1)))

        return None, "NONE"

    def run_meta_diagnostic(self, metrics: Dict[str, Any], risk_config: Dict[str, Any], drawdown_pct: float) -> Dict[str, Any]:
        """Melakukan analisis diagnostik terstruktur melalui Gemini AI Meta-Layer."""
        start_time = time.time()
        
        # Validasi awal kesehatan client
        if not self.client:
            return GeminiResponseValidator.validate_and_clamp({})

        prompt = f"""
        Anda adalah Chief Risk Officer & Quantitative Architect BEI (IDX).
        Evaluasi metrik kuantitatif sistem perdagangan otomatis berikut dan berikan rekomendasi diagnostik adaptif.

        === RINGKASAN METRIK PERFORMA ===
        - Total Sinyal Evaluasi: {metrics.get('total_evaluated_signals', 0)}
        - Rolling Composite MAE: {metrics.get('rolling_mae', 0.0):.4f}
        - Win Rate: {metrics.get('win_rate', 0.0)*100:.1f}%
        - Profit Factor: {metrics.get('profit_factor', 1.0):.2f}
        - Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}
        - Sortino Ratio: {metrics.get('sortino_ratio', 0.0):.2f}
        - Calmar Ratio: {metrics.get('calmar_ratio', 0.0):.2f}
        - ECE Calibration Error: {metrics.get('ece_calibration', 0.0):.4f}
        - Brier Score: {metrics.get('brier_score', 0.0):.4f}
        - Population Stability Index (PSI Drift): {metrics.get('psi_drift_score', 0.0):.4f}
        - Current Drawdown: {drawdown_pct*100:.2f}%

        === BATAS KETAT (HARD LIMITS) ===
        - Max Risk Scale Allowed: {HARD_LIMITS['MAX_RISK_SCALE']}
        - Min Risk Scale Allowed: {HARD_LIMITS['MIN_RISK_SCALE']}
        - Max Position Allowed: {HARD_LIMITS['MAX_POSITION_PCT']*100}%

        === OUTPUT MANDATORY JSON SCHEMA ===
        Kembalikan HANYA objek JSON murni tanpa markdown dengan kunci:
        {{
            "diagnostic": "Kesimpulan singkat kesehatan data & model.",
            "market_regime": "NORMAL | HIGH_VOLATILITY | SEVERE_VOLATILITY | BEARISH_STRESS",
            "system_state": "NORMAL | CAUTION | RISK_REDUCTION | RETRAIN | HALT",
            "recommendation": "Langkah operasional konkret.",
            "risk_scale": 1.0,
            "confidence_threshold": 0.55,
            "position_scale": 0.25,
            "retraining_required": false,
            "risk_reasons": ["Alasan 1"],
            "anomalies": []
        }}
        """

        raw_text, used_model = self._execute_api_call_with_retry(prompt)
        latency = time.time() - start_time

        if not raw_text:
            json_logger.log_event("ERROR", "GEMINI_DIAGNOSTIC_EXECUTION_FAILED", {"latency_sec": round(latency, 3)})
            res = GeminiResponseValidator.validate_and_clamp({})
            res["proposed_risk_scale"] = 1.0
            return res

        try:
            # Cleaning JSON Markdown
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(cleaned_text)
            validated_res = GeminiResponseValidator.validate_and_clamp(parsed)
            validated_res["proposed_risk_scale"] = parsed.get("risk_scale", 1.0)
            
            # Catat Audit Trail
            json_logger.log_event("INFO", "GEMINI_DECISION", {
                "used_model": used_model,
                "latency_sec": round(latency, 3),
                "proposed_risk_scale": validated_res["proposed_risk_scale"],
                "final_risk_scale": validated_res["risk_scale"],
                "system_state": validated_res["system_state"],
                "validation_status": validated_res["validation_status"]
            })
            return validated_res

        except Exception as err:
            json_logger.log_event("WARNING", "GEMINI_OUTPUT_PARSING_FAILED", {"error": str(err), "raw_text": raw_text})
            res = GeminiResponseValidator.validate_and_clamp({})
            res["proposed_risk_scale"] = 1.0
            return res


# ==============================================================================
# MAIN SELF-LEARNING ENGINE CLASS
# ==============================================================================
class IDXSelfLearningEngine:
    """
    Autonomous Self-Learning Engine v2026.Q3.v18.0
    Mengintegrasikan Evaluasi Polars, Metrik Kuantitatif, State Machine Hierarchy,
    dan Meta-Diagnostic Layer Gemini dengan Absolute Hard Guardrails.
    """

    SYMBOL_COLS = ["symbol", "pair", "ticker", "asset", "code"]
    ENTRY_COLS = ["entry_price", "price", "buy_price", "signal_price", "close"]
    TP_COLS = ["take_profit", "target_tp", "target_tp_adj", "tp_price", "take_profit_price"]
    SL_COLS = ["stop_loss", "target_sl", "target_sl_adj", "sl_price", "stop_loss_price"]
    PRED_RETURN_COLS = ["predicted_return", "expected_return", "pred_return", "target_return", "return_5d"]
    PROBABILITY_COLS = ["probability", "confidence", "prediction_probability", "score"]
    TIMESTAMP_COLS = ["timestamp", "created_at", "date", "datetime", "time"]

    def __init__(
        self,
        log_path: str = PREDIKSI_IDX_LOG_PATH,
        db_path: str = AUTONOMOUS_DB_PATH,
        rolling_window_size: int = 100,
        default_evaluation_days: int = 5,
        timezone_str: str = DEFAULT_TIMEZONE,
        max_allowed_mae_pct: float = 0.10,
        max_drawdown_kill_pct: float = HARD_LIMITS["MAX_DRAWDOWN_KILL_PCT"],
        tpsl_conflict_mode: str = "WORST_CASE",  # Default Deterministic Conservative
        initial_equity_idr: float = 10_000_000.0,
        gemini_api_key: Optional[str] = None
    ):
        self.log_path = Path(log_path)
        self.db_store = AutonomousSQLiteStore(db_path)
        self.gemini_engine = GeminiAutonomousDiagnosticEngine(api_key=gemini_api_key)
        self.rolling_window_size = max(10, rolling_window_size)
        self.default_evaluation_days = max(1, default_evaluation_days)

        try:
            self.tz = ZoneInfo(timezone_str)
        except Exception:
            self.tz = ZoneInfo("Asia/Jakarta")

        self.max_allowed_mae_pct = max(0.01, max_allowed_mae_pct)
        self.max_drawdown_kill_pct = min(HARD_LIMITS["MAX_DRAWDOWN_KILL_PCT"], max(0.01, max_drawdown_kill_pct))
        self.tpsl_conflict_mode = tpsl_conflict_mode.upper()
        self._lock = threading.RLock()

        # Internal Equity Curve Tracking
        self.initial_equity_idr = initial_equity_idr
        self.current_equity_idr = initial_equity_idr
        self.peak_equity_idr = initial_equity_idr

    def _resolve_column(self, df_cols: List[str], candidates: List[str]) -> Optional[str]:
        cols_lower = {c.lower(): c for c in df_cols}
        for col in candidates:
            if col.lower() in cols_lower:
                return cols_lower[col.lower()]
        return None

    def _normalize_ticker(self, symbol: Any) -> str:
        if symbol is None or (isinstance(symbol, float) and math.isnan(symbol)):
            return ""
        sym = str(symbol).strip().upper().replace("_", ".").replace("/", ".")
        if not sym:
            return ""
        if not sym.endswith(".JK"):
            sym = f"{sym}.JK"
        return sym

    def load_and_normalize_logs(self) -> pl.DataFrame:
        """Memuat & menormalisasi log prediksi secara memori-efisien & thread-safe."""
        with self._lock:
            try:
                if not self.log_path.exists():
                    json_logger.log_event("WARNING", "LOG_FILE_NOT_FOUND", {"path": str(self.log_path)})
                    return pl.DataFrame()

                if self.log_path.suffix == ".parquet":
                    pldf = pl.read_parquet(self.log_path)
                else:
                    pldf = pl.read_csv(self.log_path, infer_schema_length=10000)

                if pldf.is_empty():
                    json_logger.log_event("WARNING", "LOG_FILE_EMPTY", {"path": str(self.log_path)})
                    return pl.DataFrame()

                cols = pldf.columns
                sym_col = self._resolve_column(cols, self.SYMBOL_COLS)
                entry_col = self._resolve_column(cols, self.ENTRY_COLS)
                tp_col = self._resolve_column(cols, self.TP_COLS)
                sl_col = self._resolve_column(cols, self.SL_COLS)
                pred_col = self._resolve_column(cols, self.PRED_RETURN_COLS)
                prob_col = self._resolve_column(cols, self.PROBABILITY_COLS)
                ts_col = self._resolve_column(cols, self.TIMESTAMP_COLS)

                if not all([sym_col, entry_col, tp_col, sl_col]):
                    json_logger.log_event("ERROR", "LOG_SCHEMA_INVALID", {"columns": cols})
                    return pl.DataFrame()

                rename_dict = {
                    sym_col: "symbol",
                    entry_col: "entry_price",
                    tp_col: "target_tp",
                    sl_col: "target_sl",
                }
                if pred_col:
                    rename_dict[pred_col] = "predicted_return"
                if prob_col:
                    rename_dict[prob_col] = "probability"
                if ts_col:
                    rename_dict[ts_col] = "timestamp"

                pldf = pldf.rename(rename_dict)

                pldf = pldf.with_columns([
                    pl.col("entry_price").cast(pl.Float64, strict=False),
                    pl.col("target_tp").cast(pl.Float64, strict=False),
                    pl.col("target_sl").cast(pl.Float64, strict=False),
                ])

                if "predicted_return" in pldf.columns:
                    pldf = pldf.with_columns(pl.col("predicted_return").cast(pl.Float64, strict=False))
                else:
                    pldf = pldf.with_columns(
                        ((pl.col("target_tp") - pl.col("entry_price")) / (pl.col("entry_price") + EPSILON)).alias("predicted_return")
                    )

                if "probability" in pldf.columns:
                    pldf = pldf.with_columns(pl.col("probability").cast(pl.Float64, strict=False))
                else:
                    pldf = pldf.with_columns(pl.lit(0.53).alias("probability"))

                norm_symbols = [self._normalize_ticker(s) for s in pldf["symbol"].to_list()]
                pldf = pldf.with_columns(pl.Series("symbol", norm_symbols))

                if "timestamp" not in pldf.columns:
                    pldf = pldf.with_columns(pl.lit(datetime.now(timezone.utc).isoformat()).alias("timestamp"))

                return pldf.filter(
                    pl.col("symbol").is_not_null() &
                    (pl.col("symbol") != "") &
                    pl.col("entry_price").is_not_null() &
                    (pl.col("entry_price") >= HARD_LIMITS["IDX_MIN_PRICE_IDR"])
                )

            except Exception as err:
                json_logger.log_event("ERROR", "LOG_LOAD_FAILED", {"error": str(err)})
                return pl.DataFrame()

    def evaluate_temporal_outcomes_polars(
        self,
        logs_pldf: pl.DataFrame,
        candles_dict: Dict[str, Any],
        evaluation_days: Optional[int] = None,
    ) -> pl.DataFrame:
        """Evaluasi sekuensial TP/SL berbasis Polars Vectorization & Deterministic Fee Model."""
        with self._lock:
            logs_pldf = _ensure_polars_df_auto(logs_pldf)
            if logs_pldf.is_empty() or not candles_dict:
                return pl.DataFrame()

            eval_horizon = evaluation_days or self.default_evaluation_days
            evaluated_rows = []

            norm_candles = {}
            for k, v in candles_dict.items():
                norm_k = self._normalize_ticker(k)
                norm_candles[norm_k] = _ensure_polars_df_auto(v)

            for row in logs_pldf.iter_rows(named=True):
                symbol = row["symbol"]
                entry_p = float(row["entry_price"])
                tp_p = float(row["target_tp"])
                sl_p = float(row["target_sl"])
                pred_ret = float(row["predicted_return"])

                if symbol not in norm_candles or entry_p < HARD_LIMITS["IDX_MIN_PRICE_IDR"]:
                    continue

                c_df = norm_candles[symbol]
                if c_df.is_empty():
                    continue

                future_c = c_df.head(eval_horizon)
                if future_c.is_empty():
                    continue

                is_hit_tp = False
                is_hit_sl = False
                exit_price = entry_p
                hit_day = 0

                high_col = self._resolve_column(future_c.columns, ["high", "High"]) or "high"
                low_col = self._resolve_column(future_c.columns, ["low", "Low"]) or "low"
                close_col = self._resolve_column(future_c.columns, ["close", "Close"]) or "close"

                highs = future_c[high_col].to_list()
                lows = future_c[low_col].to_list()
                closes = future_c[close_col].to_list()

                for i, (hp, lp) in enumerate(zip(highs, lows), start=1):
                    hp, lp = float(hp), float(lp)

                    if hp >= tp_p and lp <= sl_p:
                        # Deterministic Conservative Default: WORST_CASE (Stop Loss First)
                        if self.tpsl_conflict_mode == "BEST_CASE":
                            is_hit_tp = True
                            exit_price = tp_p
                        elif self.tpsl_conflict_mode == "RANDOMIZED":
                            if random.random() < 0.5:
                                is_hit_sl = True
                                exit_price = sl_p
                            else:
                                is_hit_tp = True
                                exit_price = tp_p
                        else:  # WORST_CASE Default
                            is_hit_sl = True
                            exit_price = sl_p
                        hit_day = i
                        break
                    elif hp >= tp_p:
                        is_hit_tp = True
                        exit_price = tp_p
                        hit_day = i
                        break
                    elif lp <= sl_p:
                        is_hit_sl = True
                        exit_price = sl_p
                        hit_day = i
                        break

                if not is_hit_tp and not is_hit_sl:
                    exit_price = float(closes[-1])
                    hit_day = len(closes)

                future_close_return = (float(closes[-1]) - entry_p) / entry_p

                # Realized Net Return (BEI Asymmetric Friction Fee Model)
                net_buy_price = entry_p * (1.0 + IDX_BUY_FEE_PCT)
                net_sell_price = exit_price * (1.0 - IDX_SELL_FEE_PCT)
                realized_net_return = (net_sell_price - net_buy_price) / (net_buy_price + EPSILON)

                prediction_error = abs(pred_ret - future_close_return)
                execution_error = abs(future_close_return - realized_net_return)
                composite_abs_error = 0.6 * prediction_error + 0.4 * execution_error

                record = dict(row)
                record.update({
                    "is_hit_tp": is_hit_tp,
                    "is_hit_sl": is_hit_sl,
                    "exit_price": exit_price,
                    "holding_days": hit_day,
                    "future_close_return": future_close_return,
                    "realized_return": realized_net_return,
                    "prediction_error": prediction_error,
                    "execution_error": execution_error,
                    "abs_error": composite_abs_error,
                })
                evaluated_rows.append(record)

            if not evaluated_rows:
                return pl.DataFrame()

            return pl.DataFrame(evaluated_rows)

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """Kalkulasi Population Stability Index (PSI) untuk Deteksi Data Drift."""
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return 0.0

        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref_clean, quantiles)
        bins = np.unique(bins)

        if len(bins) < 2:
            return 0.0

        bins[0] -= 1e-5
        bins[-1] += 1e-5

        ref_counts, _ = np.histogram(ref_clean, bins=bins)
        cur_counts, _ = np.histogram(cur_clean, bins=bins)

        ref_pct = ref_counts / max(1, len(ref_clean))
        cur_pct = cur_counts / max(1, len(cur_clean))

        ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)

        psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi_value)

    def _calculate_brier_and_ece(self, probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 5) -> Tuple[float, float]:
        """Menghitung Brier Score dan Expected Calibration Error (ECE)."""
        if len(probs) == 0:
            return 0.0, 0.0

        brier = float(np.mean((probs - outcomes) ** 2))

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total_samples = len(probs)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            if i == n_bins - 1:
                in_bin = (probs >= bin_lower) & (probs <= bin_upper)
            else:
                in_bin = (probs >= bin_lower) & (probs < bin_upper)

            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(outcomes[in_bin])
                avg_confidence_in_bin = np.mean(probs[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * (np.sum(in_bin) / total_samples)

        return brier, float(ece)

    def calculate_rolling_metrics_suite(self, evaluated_pldf: pl.DataFrame) -> Dict[str, Any]:
        """Suite Statistik Lengkap: Sharpe, Sortino, Calmar, Kelly, CVaR 95%, PSI Drift, & Calibration."""
        with self._lock:
            evaluated_pldf = _ensure_polars_df_auto(evaluated_pldf)
            if evaluated_pldf.is_empty() or "realized_return" not in evaluated_pldf.columns:
                return {
                    "rolling_mae": 0.0,
                    "prediction_mae": 0.0,
                    "execution_mae": 0.0,
                    "win_rate": 0.50,
                    "profit_factor": 1.0,
                    "expectancy": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "calmar_ratio": 0.0,
                    "cvar_95": 0.0,
                    "kelly_fraction": 0.0,
                    "max_consecutive_loss": 0,
                    "brier_score": 0.0,
                    "ece_calibration": 0.0,
                    "psi_drift_score": 0.0,
                    "total_evaluated_signals": 0,
                }

            recent_pldf = evaluated_pldf.tail(self.rolling_window_size)
            total_signals = len(recent_pldf)

            returns = recent_pldf["realized_return"].to_numpy()
            pred_errs = recent_pldf["prediction_error"].to_numpy() if "prediction_error" in recent_pldf.columns else returns
            exec_errs = recent_pldf["execution_error"].to_numpy() if "execution_error" in recent_pldf.columns else returns
            comp_errs = recent_pldf["abs_error"].to_numpy() if "abs_error" in recent_pldf.columns else returns

            pred_mae = float(np.mean(pred_errs))
            exec_mae = float(np.mean(exec_errs))
            rolling_mae = float(np.mean(comp_errs))

            wins = returns[returns > 0]
            losses = returns[returns < 0]
            win_rate = len(wins) / total_signals if total_signals > 0 else 0.50

            avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
            avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0

            expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

            total_gross_wins = np.sum(wins)
            total_gross_losses = np.abs(np.sum(losses))
            if total_gross_losses > EPSILON:
                profit_factor = float(total_gross_wins / total_gross_losses)
            elif total_gross_wins > 0:
                profit_factor = float("inf")
            else:
                profit_factor = 1.0

            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns))
            downside_std = float(np.std(losses)) if len(losses) > 0 else EPSILON

            sharpe_ratio = (mean_ret / (std_ret + EPSILON)) * math.sqrt(252)
            sortino_ratio = (mean_ret / (downside_std + EPSILON)) * math.sqrt(252)

            max_streak = 0
            current_streak = 0
            for r in returns:
                if r < 0:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0

            win_loss_ratio = (avg_win / (avg_loss + EPSILON))
            kelly_fraction = max(0.0, win_rate - ((1.0 - win_rate) / (win_loss_ratio + EPSILON)))

            var_95 = float(np.percentile(returns, 5))
            cvar_95_slice = returns[returns <= var_95]
            cvar_95 = float(np.mean(cvar_95_slice)) if len(cvar_95_slice) > 0 else var_95

            cagr = mean_ret * 252
            actual_dd = (self.peak_equity_idr - self.current_equity_idr) / (self.peak_equity_idr + EPSILON)
            max_dd = max(0.02, actual_dd)
            calmar_ratio = cagr / (max_dd + EPSILON)

            probs = recent_pldf["probability"].to_numpy() if "probability" in recent_pldf.columns else np.full(total_signals, 0.53)
            binary_outcomes = (returns > 0).astype(float)
            brier_score, ece = self._calculate_brier_and_ece(probs, binary_outcomes)

            historical_returns = evaluated_pldf["realized_return"].to_numpy()
            psi_drift = self._calculate_psi(historical_returns, returns)

            metrics = {
                "rolling_mae": rolling_mae,
                "prediction_mae": pred_mae,
                "execution_mae": exec_mae,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "cvar_95": cvar_95,
                "kelly_fraction": kelly_fraction,
                "max_consecutive_loss": max_streak,
                "brier_score": brier_score,
                "ece_calibration": ece,
                "psi_drift_score": psi_drift,
                "total_evaluated_signals": total_signals,
            }

            json_logger.log_event("INFO", "METRICS_SUITE_COMPUTED", metrics)
            return metrics

    def update_internal_equity_and_drawdown(self, metrics: Dict[str, Any]) -> float:
        """Memperbarui Internal Equity Curve & Menghitung Mark-to-Market Drawdown."""
        with self._lock:
            expectancy = metrics.get("expectancy", 0.0)
            total_signals = metrics.get("total_evaluated_signals", 0)

            if total_signals > 0:
                self.current_equity_idr *= (1.0 + expectancy)

            if self.current_equity_idr > self.peak_equity_idr:
                self.peak_equity_idr = self.current_equity_idr

            drawdown_pct = (self.peak_equity_idr - self.current_equity_idr) / (self.peak_equity_idr + EPSILON)

            metrics["peak_equity_idr"] = self.peak_equity_idr
            metrics["current_equity_idr"] = self.current_equity_idr
            metrics["internal_drawdown_pct"] = drawdown_pct

            return float(drawdown_pct)

    def update_adaptive_risk_thresholds(
        self,
        metrics: Dict[str, Any],
        ai_validated_res: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Skalasi Risiko Adaptif Kuantitatif Terpadu yang Diclamp oleh Hard Limits."""
        with self._lock:
            mae = metrics.get("rolling_mae", 0.05)
            win_rate = metrics.get("win_rate", 0.50)
            ece = metrics.get("ece_calibration", 0.0)
            psi = metrics.get("psi_drift_score", 0.0)

            ai_risk_scale = ai_validated_res.get("risk_scale", 1.0)
            ai_conf_proposed = ai_validated_res.get("confidence_threshold", 0.50)
            ai_pos_proposed = ai_validated_res.get("position_scale", 0.25)

            mae_ratio = mae / (self.max_allowed_mae_pct + EPSILON)
            mae_factor = min(1.5, max(0.8, mae_ratio))

            calibration_penalty = 1.10 if ece > 0.20 else 1.0
            drift_penalty = 1.15 if psi >= 0.25 else 1.0

            # Hitung Kuantitatif
            quant_confidence = HARD_LIMITS["MIN_CONFIDENCE"] * mae_factor * calibration_penalty * drift_penalty
            
            # Gabungkan dengan Proposal AI (AI Proposes, Python Clamps)
            final_confidence = min(HARD_LIMITS["MAX_CONFIDENCE"], max(HARD_LIMITS["MIN_CONFIDENCE"], max(quant_confidence, ai_conf_proposed)))

            if win_rate < 0.35:
                position_scale = 0.60
            elif win_rate > 0.55:
                position_scale = 1.20
            else:
                position_scale = 1.00

            # Terapkan AI Risk Scale Multiplier
            final_pos_scale = position_scale * ai_risk_scale
            quant_max_position = (HARD_LIMITS["MAX_POSITION_PCT"] * final_pos_scale) / (mae_factor * calibration_penalty)

            # Enforce Hard Limits Floor & Ceiling
            final_max_position = min(
                HARD_LIMITS["MAX_POSITION_PCT"],
                max(HARD_LIMITS["MIN_POSITION_PCT"], min(quant_max_position, ai_pos_proposed))
            )

            risk_config = {
                "min_required_confidence": float(final_confidence),
                "max_position_pct_idr": float(final_max_position),
                "max_position_pct_usdt": float(final_max_position),
                "mae_scaling_factor": float(mae_factor),
                "calibration_penalty": float(calibration_penalty),
                "drift_penalty": float(drift_penalty),
                "ai_risk_scale": float(ai_risk_scale),
                "hard_limits": HARD_LIMITS
            }

            json_logger.log_event("INFO", "ADAPTIVE_RISK_UPDATED", risk_config)
            return risk_config

    def resolve_state_hierarchy(self, quant_retrain: bool, quant_halt: bool, ai_state: str) -> Tuple[str, bool, bool]:
        """State Machine Hierarchy: HALT > RETRAIN > RISK_REDUCTION > CAUTION > NORMAL."""
        prio_map = {"HALT": 5, "RETRAIN": 4, "RISK_REDUCTION": 3, "CAUTION": 2, "NORMAL": 1}

        quant_state = "NORMAL"
        if quant_halt:
            quant_state = "HALT"
        elif quant_retrain:
            quant_state = "RETRAIN"

        q_prio = prio_map.get(quant_state, 1)
        ai_prio = prio_map.get(ai_state, 1)

        # Python Hard Guardrail Dominance: AI tidak bisa menuntun HALT kembali ke NORMAL
        final_prio = max(q_prio, ai_prio)

        prio_to_state = {5: "HALT", 4: "RETRAIN", 3: "RISK_REDUCTION", 2: "CAUTION", 1: "NORMAL"}
        final_state = prio_to_state[final_prio]

        final_halt = (final_state == "HALT")
        final_retrain = (final_state in ["HALT", "RETRAIN"])

        return final_state, final_retrain, final_halt

    def check_composite_retraining_trigger(self, metrics: Dict[str, Any]) -> bool:
        """Konsensus Multi-Faktor Retraining Trigger (MAE + PF + WR + PSI + Brier)."""
        with self._lock:
            mae = metrics.get("rolling_mae", 0.0)
            pf = metrics.get("profit_factor", 1.0)
            wr = metrics.get("win_rate", 1.0)
            psi = metrics.get("psi_drift_score", 0.0)
            brier = metrics.get("brier_score", 0.0)
            total_signals = metrics.get("total_evaluated_signals", 0)

            if total_signals < 5:
                return False

            triggers = [
                mae > self.max_allowed_mae_pct,
                pf < 1.0,
                wr < 0.35,
                psi >= 0.25,
                brier > 0.30,
            ]

            retrain_required = sum(triggers) >= 3

            if retrain_required:
                json_logger.log_event("WARNING", "RETRAIN_TRIGGERED", {
                    "reason": "Composite Multi-Factor consensus threshold met.",
                    "mae": mae, "profit_factor": pf, "win_rate": wr, "psi": psi, "brier": brier
                })

            return retrain_required

    def check_emergency_kill_switch(self, metrics: Dict[str, Any], current_drawdown_pct: float) -> Tuple[bool, str]:
        """Pemeriksaan Sakelar Pembunuh Darurat (Emergency Kill-Switch)."""
        with self._lock:
            mae = metrics.get("rolling_mae", 0.0)
            win_rate = metrics.get("win_rate", 1.0)
            total_signals = metrics.get("total_evaluated_signals", 0)

            if current_drawdown_pct >= HARD_LIMITS["MAX_DRAWDOWN_KILL_PCT"]:
                reason = f"KILL-SWITCH: Drawdown ({current_drawdown_pct*100:.2f}%) melampaui HARD_LIMIT ({HARD_LIMITS['MAX_DRAWDOWN_KILL_PCT']*100:.2f}%)."
                json_logger.log_event("CRITICAL", "KILL_SWITCH_DRAWDOWN", {"reason": reason})
                return True, reason

            if total_signals >= 10 and mae > (self.max_allowed_mae_pct * 2.0):
                reason = f"KILL-SWITCH: MAE ({mae:.4f}) melampaui batas kritis ({self.max_allowed_mae_pct*2.0:.4f})."
                json_logger.log_event("CRITICAL", "KILL_SWITCH_MAE", {"reason": reason})
                return True, reason

            if total_signals >= 15 and win_rate < 0.15:
                reason = f"KILL-SWITCH: Win Rate ({win_rate*100:.1f}%) di bawah batas minimal (15%)."
                json_logger.log_event("CRITICAL", "KILL_SWITCH_WINRATE", {"reason": reason})
                return True, reason

            return False, "SYSTEM OPERATIONAL: Normal condition."

    def run_autonomous_cycle(
        self,
        candles_dict: Dict[str, Any],
        external_drawdown_pct: Optional[float] = None,
        evaluation_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Eksekusi Terpadu Siklus Autonomous Feedback Loop Saham IDX."""
        with self._lock:
            json_logger.log_event("INFO", "AUTONOMOUS_CYCLE_START", {"mode": self.tpsl_conflict_mode})

            # 1. Health Check Gemini Sebelum pemrosesan
            is_healthy, latency, used_model = self.gemini_engine.check_health()

            # 2. Muat Log Histori
            logs_pldf = self.load_and_normalize_logs()
            if logs_pldf.is_empty():
                return {
                    "kill_switch_triggered": False,
                    "status": "No log data available.",
                    "risk_config": {"min_required_confidence": 0.50, "max_position_pct_idr": 0.25},
                }

            # 3. Evaluasi TP/SL
            evaluated_pldf = self.evaluate_temporal_outcomes_polars(logs_pldf, candles_dict, evaluation_days)

            # 4. Hitung Statistik Kuantitatif
            metrics = self.calculate_rolling_metrics_suite(evaluated_pldf)

            # 5. Hitung Equity & Drawdown
            internal_drawdown = self.update_internal_equity_and_drawdown(metrics)
            effective_drawdown = external_drawdown_pct if external_drawdown_pct is not None else internal_drawdown

            # 6. Jalankan Meta-Diagnostic Layer Gemini (jika Sehat)
            ai_diag = self.gemini_engine.run_meta_diagnostic(metrics, {}, effective_drawdown)

            # 7. Update Konfigurasi Risiko Adaptif (Diclamp oleh Hard Limits)
            risk_config = self.update_adaptive_risk_thresholds(metrics, ai_diag)

            # 8. Evaluasi Triggers & State Hierarchy
            quant_retrain = self.check_composite_retraining_trigger(metrics)
            quant_kill, kill_reason = self.check_emergency_kill_switch(metrics, effective_drawdown)

            final_state, final_retrain, final_kill = self.resolve_state_hierarchy(
                quant_retrain, quant_kill, ai_diag.get("system_state", "NORMAL")
            )

            # 9. Rekam Hasil ke SQLite Persistence
            self.db_store.record_metrics(metrics, risk_config, effective_drawdown)
            self.db_store.record_gemini_diagnostic(ai_diag, latency, used_model, is_healthy)

            json_logger.log_event("INFO", "AUTONOMOUS_CYCLE_COMPLETE", {
                "evaluated_count": len(evaluated_pldf),
                "system_state": final_state,
                "kill_switch": final_kill,
                "retrain": final_retrain,
                "ai_diagnostic": ai_diag.get("diagnostic")
            })

            return {
                "system_state": final_state,
                "kill_switch_triggered": final_kill,
                "kill_switch_reason": kill_reason if final_kill else "OPERATIONAL",
                "retraining_triggered": final_retrain,
                "effective_drawdown_pct": effective_drawdown,
                "metrics": metrics,
                "risk_config": risk_config,
                "ai_diagnostic": ai_diag,
                "evaluated_signals_count": len(evaluated_pldf),
            }


# ==============================================================================
# ALIAS COMPATIBILITY & FACADE CLASS
# ==============================================================================
CryptoSelfLearningEngine = IDXSelfLearningEngine


class UnifiedAutonomousEngine:
    """Facade class terpusat sebagai single point entry dari main.py."""

    def __init__(self, log_path: str = PREDIKSI_IDX_LOG_PATH, db_path: str = AUTONOMOUS_DB_PATH, gemini_api_key: Optional[str] = None):
        self.engine = IDXSelfLearningEngine(log_path=log_path, db_path=db_path, gemini_api_key=gemini_api_key)

    def execute_evaluation(
        self,
        candles_dict: Dict[str, Any],
        current_drawdown_pct: Optional[float] = None,
        evaluation_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.engine.run_autonomous_cycle(
            candles_dict=candles_dict,
            external_drawdown_pct=current_drawdown_pct,
            evaluation_days=evaluation_days,
        )


if __name__ == "__main__":
    json_logger.log_event("INFO", "MODULE_SELF_TEST_START", {"version": "v2026.Q3.v18.0"})
    mock_facade = UnifiedAutonomousEngine()
    mock_candles = {
        "BBCA.JK": pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc).isoformat()],
            "high": [10200.0],
            "low": [9800.0],
            "close": [10000.0],
        })
    }
    res = mock_facade.execute_evaluation(mock_candles, current_drawdown_pct=0.02)
    print("\nHasil Evaluasi Otonom Terpadu (JSON Risk Config):")
    print(json.dumps(res["risk_config"], indent=2))

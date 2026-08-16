"""
=============================================================================
IDX Quantitative Scalping Engine — Main Orchestrator
Version       : 2026.Q3.v28.0 (Gemini God Entity — Full Module Integration)
Compliance    : Indonesia Stock Exchange (IDX) — Fail-Closed Safety
Architecture  : Gemini as central decision entity coordinating all modules
=============================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import traceback
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_SDK = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    HAS_GEMINI_SDK = False

WIB_TZ = ZoneInfo("Asia/Jakarta")

from data import (
    load_and_prepare_market_data,
    sanitize_ticker_list,
    DataLoader,
)

try:
    from machine_learning import UnifiedModelEngine
    HAS_ML = True
except ImportError:
    UnifiedModelEngine = None  # type: ignore
    HAS_ML = False

try:
    from features import UnifiedFeatureEngine, extract_all_features, compute_features
    HAS_FEATURES = True
except ImportError:
    UnifiedFeatureEngine = None  # type: ignore
    extract_all_features = None  # type: ignore
    compute_features = None  # type: ignore
    HAS_FEATURES = False

try:
    from prediction import UnifiedPredictionEngine
    HAS_PREDICTION = True
except ImportError:
    UnifiedPredictionEngine = None  # type: ignore
    HAS_PREDICTION = False

try:
    from signal_idx import UnifiedSignalEngine
    HAS_SIGNAL = True
except ImportError:
    UnifiedSignalEngine = None  # type: ignore
    HAS_SIGNAL = False

try:
    from risk import UnifiedRiskEngine
    HAS_RISK = True
except ImportError:
    UnifiedRiskEngine = None  # type: ignore
    HAS_RISK = False

try:
    from portfolio import UnifiedPortfolioEngine
    HAS_PORTFOLIO = True
except ImportError:
    UnifiedPortfolioEngine = None  # type: ignore
    HAS_PORTFOLIO = False

try:
    from simulation import simulate_execution, run_backtest
    HAS_SIMULATION = True
except ImportError:
    simulate_execution = None  # type: ignore
    run_backtest = None  # type: ignore
    HAS_SIMULATION = False

try:
    from reporting import UnifiedReportingEngine, broadcast_signals
    HAS_REPORTING = True
except ImportError:
    UnifiedReportingEngine = None  # type: ignore
    broadcast_signals = None  # type: ignore
    HAS_REPORTING = False

try:
    from storage import UnifiedStorageEngine
    HAS_STORAGE = True
except ImportError:
    UnifiedStorageEngine = None  # type: ignore
    HAS_STORAGE = False

try:
    from gemini_universe_analyzer import (
        verify_gemini_health,
        get_dynamic_trading_parameters,
        analyze_top_candidates_with_gemini,
        generate_market_narrative,
        get_active_model,
        get_client,
    )
    HAS_GEMINI_ANALYZER = True
except ImportError:
    verify_gemini_health = None  # type: ignore
    get_dynamic_trading_parameters = None  # type: ignore
    analyze_top_candidates_with_gemini = None  # type: ignore
    generate_market_narrative = None  # type: ignore
    get_active_model = None  # type: ignore
    get_client = None  # type: ignore
    HAS_GEMINI_ANALYZER = False

try:
    from self_learning import UnifiedSelfLearningEngine
    HAS_SELF_LEARNING = True
except ImportError:
    UnifiedSelfLearningEngine = None  # type: ignore
    HAS_SELF_LEARNING = False

try:
    from validation import GeminiValidationDiagnosticEngine
    HAS_VALIDATION = True
except ImportError:
    GeminiValidationDiagnosticEngine = None  # type: ignore
    HAS_VALIDATION = False

try:
    from evaluation import UnifiedEvaluationEngine
    HAS_EVALUATION = True
except ImportError:
    UnifiedEvaluationEngine = None  # type: ignore
    HAS_EVALUATION = False

try:
    from monitoring import UnifiedMonitoringEngine, HealthCheckEngine
    HAS_MONITORING = True
except ImportError:
    UnifiedMonitoringEngine = None  # type: ignore
    HealthCheckEngine = None  # type: ignore
    HAS_MONITORING = False

try:
    from autonomous_engine_idx import UnifiedAutonomousEngine
    HAS_AUTONOMOUS = True
except ImportError:
    UnifiedAutonomousEngine = None  # type: ignore
    HAS_AUTONOMOUS = False

try:
    import research  # noqa: F401
    HAS_RESEARCH = True
except ImportError:
    HAS_RESEARCH = False

try:
    from logger import get_logger
    logger = get_logger("IDX.Main")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("IDX.Main")

# Simulation-only system: live broker / live order paths are intentionally absent.
SIMULATION_ONLY: bool = True
PORTFOLIO_STATE_FILE: str = "portfolio_state.json"  # simulated portfolio persistence

# Multi-horizon labels mapped to daily-bar holding windows (existing data is daily OHLCV)
HORIZON_DAYS = {"SHORT": 1, "MEDIUM": 5, "LONG": 20}
VALID_HORIZONS = ("SHORT", "MEDIUM", "LONG")


def normalize_horizon(raw) -> str:
    """Map expected_holding_days / horizon string → SHORT|MEDIUM|LONG."""
    if raw is None:
        return "SHORT"
    s = str(raw).strip().upper()
    if s in VALID_HORIZONS:
        return s
    try:
        d = float(s)
        if d <= 2:
            return "SHORT"
        if d <= 10:
            return "MEDIUM"
        return "LONG"
    except Exception:
        return "SHORT"


HARD_SAFETY_BOUNDS: Dict[str, Dict[str, float]] = {
    "min_adtv_idr": {"min": 1_000_000_000.0, "max": 50_000_000_000.0, "default": 5_000_000_000.0},
    "min_confidence": {"min": 0.55, "max": 0.95, "default": 0.72},
    "min_rrr": {"min": 1.0, "max": 5.0, "default": 1.20},
    "risk_scale": {"min": 0.05, "max": 0.50, "default": 0.20},
}

DEFAULT_BLUECHIP_UNIVERSE = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
    "ASII.JK", "UNVR.JK", "ICBP.JK", "INDF.JK", "GOTO.JK",
]
DEFAULT_UNIVERSE_FILE = "universe.json"
DEFAULT_MODEL_PATH = "models/idx_scalping_model.joblib"
PRIMARY_MODEL = "gemini-2.0-flash"
FALLBACK_MODEL = "gemini-2.0-flash-lite"


def apply_hard_safety_clamps(raw_config: Dict[str, Any], log=logger) -> Dict[str, Any]:
    clamped: Dict[str, Any] = {}
    for key, bounds in HARD_SAFETY_BOUNDS.items():
        val = raw_config.get(key, bounds["default"])
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = bounds["default"]
        clamped_val = max(bounds["min"], min(bounds["max"], val))
        if clamped_val != val:
            log.warning(
                f"🛡️ [HARD_CLAMP] {key}: Gemini proposed {val} → clamped to {clamped_val} "
                f"(range {bounds['min']}–{bounds['max']})"
            )
        clamped[key] = clamped_val
    clamped["configured_by"] = raw_config.get("configured_by", "GEMINI_GOD+HARD_CLAMP")
    return clamped


class GeminiGodEntity:
    """Central decision entity. Hard clamps remain non-negotiable."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        ).strip()
        self.primary_model = PRIMARY_MODEL
        self.fallback_model = FALLBACK_MODEL
        self.active_model: Optional[str] = None
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        if not HAS_GEMINI_SDK or not self.api_key:
            logger.warning("⚠️ [GOD] Gemini SDK/API key unavailable — DEGRADED mode.")
            return
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info("🧠 [GOD] Gemini client initialized.")
        except Exception as e:
            logger.error(f"❌ [GOD] Client init failed: {e}")
            self.client = None

    def awaken(self) -> bool:
        if HAS_GEMINI_ANALYZER and verify_gemini_health is not None:
            try:
                ok = bool(verify_gemini_health())
                if ok and get_active_model is not None:
                    self.active_model = get_active_model() or self.primary_model
                logger.info(f"🧠 [GOD_AWAKE] analyzer_health={ok} model={self.active_model}")
                return ok
            except Exception as e:
                logger.warning(f"⚠️ [GOD] analyzer health failed: {e}")
        if self.client is None:
            return False
        for model in (self.primary_model, self.fallback_model):
            try:
                self.client.models.generate_content(model=model, contents="ping")
                self.active_model = model
                logger.info(f"🧠 [GOD_AWAKE] bound model={model}")
                return True
            except Exception as e:
                logger.warning(f"⚠️ [GOD] model {model} unreachable: {e}")
        self.active_model = None
        return False

    def propose_trading_parameters(self) -> Dict[str, Any]:
        if HAS_GEMINI_ANALYZER and get_dynamic_trading_parameters is not None:
            try:
                params = get_dynamic_trading_parameters() or {}
                if isinstance(params, dict) and params:
                    params["configured_by"] = "GEMINI_UNIVERSE_ANALYZER"
                    return params
            except Exception as e:
                logger.warning(f"⚠️ [GOD] dynamic params failed: {e}")
        defaults = {
            "min_adtv_idr": HARD_SAFETY_BOUNDS["min_adtv_idr"]["default"],
            "min_confidence": HARD_SAFETY_BOUNDS["min_confidence"]["default"],
            "min_rrr": HARD_SAFETY_BOUNDS["min_rrr"]["default"],
            "risk_scale": HARD_SAFETY_BOUNDS["risk_scale"]["default"],
            "configured_by": "SAFETY_DEFAULT",
        }
        if self.client is None or self.active_model is None:
            return defaults
        try:
            prompt = (
                "You are Chief Risk Officer for IDX scalping. "
                "Return ONLY JSON keys: min_adtv_idr, min_confidence, min_rrr, risk_scale. "
                "Conservative intraday for liquid BEI stocks."
            )
            resp = self.client.models.generate_content(model=self.active_model, contents=prompt)
            text = (getattr(resp, "text", None) or "").strip()
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                raw = json.loads(text[start : end + 1])
                raw["configured_by"] = f"GEMINI_GOD:{self.active_model}"
                return raw
        except Exception as e:
            logger.warning(f"⚠️ [GOD] inline param proposal failed: {e}")
        return defaults

    def analyze_candidates(self, top_report: str, relaxation: bool = False) -> Any:
        if HAS_GEMINI_ANALYZER and analyze_top_candidates_with_gemini is not None:
            try:
                return analyze_top_candidates_with_gemini(top_report, relaxation_mode=relaxation)
            except Exception as e:
                logger.warning(f"⚠️ [GOD] candidate analysis failed: {e}")
        return None

    def narrate(self, portfolio_state: Dict[str, Any], top_signals: Optional[List[Dict[str, Any]]] = None) -> str:
        if HAS_GEMINI_ANALYZER and generate_market_narrative is not None:
            try:
                narrative = generate_market_narrative(portfolio_state, top_signals=top_signals)
                if narrative:
                    return str(narrative)
            except Exception as e:
                logger.warning(f"⚠️ [GOD] narrative failed: {e}")
        n = len(top_signals or [])
        return f"IDX dry-run scan complete. Valid signals: {n}. Mode: simulation only."


class ProductionOrchestrator:
    def __init__(self, dry_run: bool = True, self_learning: bool = False) -> None:
        self.dry_run = True
        self.self_learning_flag = bool(self_learning)
        self.api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
        self.god = GeminiGodEntity(api_key=self.api_key)
        self.trading_config = apply_hard_safety_clamps({}, logger)
        self.universe: List[str] = []
        self.market_data: pl.DataFrame = pl.DataFrame()
        self.features_data: pl.DataFrame = pl.DataFrame()
        self.predictions_data: pl.DataFrame = pl.DataFrame()
        self.signals_data: pl.DataFrame = pl.DataFrame()
        self.orders: List[Dict[str, Any]] = []
        self.portfolio_state: Dict[str, Any] = {}
        self.risk_output: Any = None
        self.execution_id = datetime.now(WIB_TZ).strftime("%Y%m%d_%H%M%S")
        self.ml_engine = None
        self.signal_engine = None
        self.risk_engine = None
        self.portfolio_engine = None
        self.storage_engine = None
        self._log_module_matrix()

    def _log_module_matrix(self) -> None:
        matrix = {
            "data": True,
            "features": HAS_FEATURES,
            "machine_learning": HAS_ML,
            "prediction": HAS_PREDICTION,
            "signal_idx": HAS_SIGNAL,
            "risk": HAS_RISK,
            "portfolio": HAS_PORTFOLIO,
            "simulation": HAS_SIMULATION,
            "reporting": HAS_REPORTING,
            "storage": HAS_STORAGE,
            "gemini_universe_analyzer": HAS_GEMINI_ANALYZER,
            "self_learning": HAS_SELF_LEARNING,
            "validation": HAS_VALIDATION,
            "evaluation": HAS_EVALUATION,
            "monitoring": HAS_MONITORING,
            "autonomous_engine_idx": HAS_AUTONOMOUS,
            "research": HAS_RESEARCH,
            "gemini_sdk": HAS_GEMINI_SDK,
        }
        online = sum(1 for v in matrix.values() if v)
        logger.info(f"📦 [MODULE_MATRIX] {online}/{len(matrix)} online → {matrix}")

    def _step_0_god_awaken(self) -> None:
        logger.info("▶ [STEP 0] Gemini God Entity awaken")
        ok = self.god.awaken()
        if not ok:
            logger.warning("⚠️ [GOD_DEGRADED] Continuing with safety defaults only.")

    def _step_1_universe(self) -> None:
        logger.info("▶ [STEP 1] Universe sync")
        if os.path.isfile(DEFAULT_UNIVERSE_FILE):
            try:
                with open(DEFAULT_UNIVERSE_FILE, encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    self.universe = sanitize_ticker_list(raw)
                elif isinstance(raw, dict):
                    self.universe = sanitize_ticker_list(raw.get("tickers") or raw.get("universe") or [])
            except Exception as e:
                logger.warning(f"⚠️ universe.json read failed: {e}")
        if not self.universe:
            self.universe = sanitize_ticker_list(DEFAULT_BLUECHIP_UNIVERSE)
            logger.info(f"ℹ️ Using default bluechip universe ({len(self.universe)} tickers)")
        logger.info(f"✔ Universe size={len(self.universe)}")

    def _step_2_data(self) -> None:
        logger.info("▶ [STEP 2] Data layer")
        try:
            try:
                result = load_and_prepare_market_data(symbols=self.universe, use_cache=True)
            except TypeError:
                loader = DataLoader()
                result = loader.load_and_prepare_market_data(symbols=self.universe, use_cache=True)
            if isinstance(result, pl.DataFrame):
                self.market_data = result
            elif isinstance(result, dict) and "data" in result:
                self.market_data = result["data"]
            else:
                self.market_data = pl.DataFrame()
            logger.info(f"✔ Market data rows={self.market_data.height} cols={self.market_data.width}")
        except Exception as e:
            logger.error(f"❌ [STEP_2] Data load failed: {e}")
            self.market_data = pl.DataFrame()

    def _step_3_god_config(self) -> None:
        logger.info("▶ [STEP 3] Gemini God proposes trading config → hard clamp")
        proposed = self.god.propose_trading_parameters()
        self.trading_config = apply_hard_safety_clamps(proposed, logger)
        logger.info(f"🤖 [CONFIG] {self.trading_config}")

    def _step_4_features(self) -> None:
        logger.info("▶ [STEP 4] Feature engineering")
        if self.market_data.height == 0:
            logger.warning("⚠️ Skip features — empty market data")
            return
        if not HAS_FEATURES:
            logger.warning("⚠️ features.py offline — pass-through")
            self.features_data = self.market_data
            return
        try:
            if compute_features is not None:
                self.features_data = compute_features(self.market_data)
            elif extract_all_features is not None:
                self.features_data = extract_all_features(self.market_data)
            elif UnifiedFeatureEngine is not None:
                eng = UnifiedFeatureEngine()
                if hasattr(eng, "transform"):
                    self.features_data = eng.transform(self.market_data)
                elif hasattr(eng, "build"):
                    self.features_data = eng.build(self.market_data)
                else:
                    self.features_data = self.market_data
            else:
                self.features_data = self.market_data
            if not isinstance(self.features_data, pl.DataFrame):
                self.features_data = self.market_data
            logger.info(f"✔ Features rows={self.features_data.height}")
        except Exception as e:
            logger.error(f"❌ Features failed: {e}")
            self.features_data = self.market_data

    def _step_5_ml(self) -> None:
        logger.info("▶ [STEP 5] Machine learning predict+calibrate")
        src = self.features_data if self.features_data.height else self.market_data
        if src.height == 0:
            logger.warning("⚠️ Skip ML — no data")
            return
        if not HAS_ML or UnifiedModelEngine is None:
            logger.warning("⚠️ machine_learning.py offline — RULE_BASED fallback if features allow")
            rule_df = self._build_rule_based_from_features()
            if rule_df is not None and rule_df.height > 0:
                self.predictions_data = rule_df
                logger.info(f"✔ [RULE_BASED] Offline-ML fallback rows={rule_df.height}")
            return
        try:
            if self.ml_engine is None:
                try:
                    if hasattr(UnifiedModelEngine, "load_model") and os.path.isfile(DEFAULT_MODEL_PATH):
                        self.ml_engine = UnifiedModelEngine.load_model(DEFAULT_MODEL_PATH)
                except Exception:
                    self.ml_engine = None
                if self.ml_engine is None:
                    self.ml_engine = UnifiedModelEngine(gemini_api_key=self.api_key)
            self.predictions_data = self.ml_engine.predict_and_calibrate(src)
            if "model_status" in self.predictions_data.columns:
                bad = self.predictions_data.filter(
                    pl.col("model_status").is_in(["MODEL_ERROR", "MODEL_NOT_READY"])
                )
                if bad.height:
                    logger.warning(f"🛡️ Fail-closed: {bad.height} MODEL_ERROR rows noted")
            if self.predictions_data.height and "model_status" in self.predictions_data.columns:
                statuses = self.predictions_data["model_status"].unique().to_list()
                logger.info(f"✔ Predictions rows={self.predictions_data.height} model_status={statuses}")
                if all(str(s) in ("MODEL_NOT_READY", "MODEL_ERROR", "MODEL_INVALID") for s in statuses):
                    logger.warning(
                        "🛡️ [MODEL_FAILURE] No ready model — ML BUY blocked; "
                        "attempting RULE_BASED fallback from existing technical features"
                    )
                    rule_df = self._build_rule_based_from_features()
                    if rule_df is not None and rule_df.height > 0:
                        self.predictions_data = rule_df
                        logger.info(
                            f"✔ [RULE_BASED] Fallback candidates rows={rule_df.height} "
                            f"(signal_source=RULE_BASED, model_status stays MODEL_NOT_READY)"
                        )
                    else:
                        logger.warning("🛡️ [RULE_BASED] No valid rule candidates — remain NO_SIGNAL")
            else:
                logger.info(f"✔ Predictions rows={self.predictions_data.height}")
        except Exception as e:
            logger.error(f"❌ [MODEL_FAILURE] ML pipeline failed: {e}")
            self.predictions_data = pl.DataFrame()
            rule_df = self._build_rule_based_from_features()
            if rule_df is not None and rule_df.height > 0:
                self.predictions_data = rule_df
                logger.info(f"✔ [RULE_BASED] Post-error fallback rows={rule_df.height}")

    def _model_pipeline_ready(self) -> bool:
        """True only when ML produced at least one non-failure model_status."""
        df = self.predictions_data
        if df is None or df.height == 0:
            return False
        if "model_status" not in df.columns:
            return "signal_source" not in df.columns or not (
                df["signal_source"].cast(pl.Utf8).str.to_uppercase() == "RULE_BASED"
            ).any()
        statuses = {str(s).upper() for s in df["model_status"].unique().to_list()}
        return bool(statuses - {"MODEL_NOT_READY", "MODEL_ERROR", "MODEL_INVALID", "NONE", ""})

    def _build_rule_based_from_features(self) -> pl.DataFrame:
        """Deterministic first-run signal path when ML is MODEL_NOT_READY.

        Uses existing technical feature columns produced by features.IDXTechnicalAndTrendBuilder
        and regime logic aligned with prediction.QuantMarketRegimeEngine.
        Does NOT invent probability=0.5 and does NOT convert MODEL_NOT_READY → ML BUY.
        Emits signal_source=RULE_BASED only when indicator agreement clears gates.
        Multi-horizon SHORT/MEDIUM/LONG with ATR-based TP/SL geometry.
        """
        src = self.features_data if self.features_data is not None and self.features_data.height else self.market_data
        if src is None or src.height == 0:
            return pl.DataFrame()

        work = src
        tcol = "ticker" if "ticker" in work.columns else ("asset" if "asset" in work.columns else ("symbol" if "symbol" in work.columns else None))
        if tcol is None:
            return pl.DataFrame()
        pcol = "close" if "close" in work.columns else ("current_price" if "current_price" in work.columns else None)
        if pcol is None:
            return pl.DataFrame()

        # Latest bar per symbol (no look-ahead)
        try:
            time_c = next((c for c in ("timestamp", "date", "datetime") if c in work.columns), None)
            if time_c:
                work = work.sort([tcol, time_c]).group_by(tcol).tail(1)
            else:
                work = work.unique(subset=[tcol], keep="last")
        except Exception:
            work = work.unique(subset=[tcol], keep="last")

        min_conf = float(self.trading_config.get("min_confidence", 0.72))
        min_rrr = float(self.trading_config.get("min_rrr", 1.20))
        min_adtv = float(self.trading_config.get("min_adtv_idr", 5_000_000_000.0))
        # First-run rule path: slightly softer score floor than production ML, but confidence
        # must still clear trading_config min_confidence (fail-closed on weak agreement).
        min_rule_score = max(0.58, min_conf - 0.10)

        # Horizon ATR geometry (same family as SignalGenerator ATR TP/SL)
        horizon_cfg = {
            "SHORT":  {"days": 1,  "tp_atr": 1.5, "sl_atr": 1.0},
            "MEDIUM": {"days": 5,  "tp_atr": 2.5, "sl_atr": 1.2},
            "LONG":   {"days": 20, "tp_atr": 4.0, "sl_atr": 1.5},
        }

        def _col(row: Dict[str, Any], names: List[str], default: float = float("nan")) -> float:
            for n in names:
                if n in row and row[n] is not None:
                    try:
                        v = float(row[n])
                        if v == v:  # not NaN
                            return v
                    except (TypeError, ValueError):
                        continue
            return default

        records: List[Dict[str, Any]] = []
        ts_now = datetime.now(WIB_TZ).isoformat()

        for row in work.to_dicts():
            ticker = str(row.get(tcol) or "").strip()
            if not ticker:
                continue
            if not ticker.endswith(".JK"):
                ticker = f"{ticker}.JK"

            price = _col(row, [pcol, "close", "current_price", "entry_price"], 0.0)
            if price < 50.0:
                continue

            rsi = _col(row, ["feature_rsi", "f_rsi_14", "rsi_14", "rsi"], float("nan"))
            macd_h = _col(row, ["feature_macd_histogram", "f_macd_histogram", "macd_hist"], float("nan"))
            ema_fast_d = _col(row, ["feature_ema_fast_distance", "f_ema_20_dist"], float("nan"))
            ema_slow_d = _col(row, ["feature_ema_slow_distance", "f_ema_50_dist"], float("nan"))
            vel = _col(row, ["feature_trend_velocity", "f_trend_velocity"], float("nan"))
            vol_ratio = _col(row, ["volume_ratio", "f_volume_ratio", "vol_ratio"], float("nan"))
            atr = _col(row, ["feature_atr", "f_atr_14", "atr_14", "atr"], float("nan"))
            if atr != atr or atr <= 0:
                atr = price * 0.015
            atr = max(atr, price * 0.005)

            vol_idr = _col(
                row,
                ["volume_24h_idr", "volume_idr", "adtv_20", "adtv20", "f_adtv_20d_idr", "median_turnover_20d"],
                float("nan"),
            )
            if vol_idr != vol_idr or vol_idr <= 0:
                # share volume × price fallback
                sh = _col(row, ["volume", "adv_20", "adv20"], float("nan"))
                if sh == sh and sh > 0:
                    vol_idr = sh * price
            # Liquidity gate: reject illiquid when volume known and below floor
            if vol_idr == vol_idr and vol_idr > 0 and vol_idr < min_adtv:
                continue

            # --- Deterministic component scores (aligned with QuantMarketRegimeEngine family) ---
            components: List[Tuple[str, float]] = []
            # RSI: healthy momentum band 48–68 preferred; oversold bounce 35–48 mild credit; overbought >75 reject component
            if rsi == rsi:
                if 48.0 <= rsi <= 68.0:
                    components.append(("rsi", 0.55 + (rsi - 48.0) / 40.0))  # ~0.55–1.05 → clip later
                elif 35.0 <= rsi < 48.0:
                    components.append(("rsi", 0.45 + (rsi - 35.0) / 50.0))
                elif rsi > 75.0:
                    components.append(("rsi", 0.20))
                else:
                    components.append(("rsi", 0.35))
            if macd_h == macd_h:
                components.append(("macd", 0.70 if macd_h > 0 else 0.30))
            if ema_fast_d == ema_fast_d:
                components.append(("ema_fast", 0.72 if ema_fast_d > 0 else 0.28))
            if ema_slow_d == ema_slow_d:
                components.append(("ema_slow", 0.70 if ema_slow_d > 0 else 0.30))
            if vel == vel:
                components.append(("velocity", 0.68 if vel > 0 else 0.32))
            if vol_ratio == vol_ratio:
                components.append(("volume", 0.65 if vol_ratio >= 1.0 else 0.40))

            if len(components) < 2:
                # Insufficient technical evidence → fail-closed
                continue

            scores = [min(1.0, max(0.0, s)) for _, s in components]
            rule_score = float(sum(scores) / len(scores))
            bullish_votes = sum(1 for s in scores if s >= 0.55)
            agreement = bullish_votes / float(len(scores))
            # Confidence from agreement strength × rule_score (never a flat 0.5)
            confidence = float(min(0.95, max(0.0, rule_score * (0.55 + 0.45 * agreement))))

            if rule_score < min_rule_score or confidence < min_conf:
                continue
            if bullish_votes < max(2, (len(scores) + 1) // 2):
                continue

            reason_bits = [f"{n}={s:.2f}" for n, s in components]
            reason = (
                f"RULE_BASED technical agreement score={rule_score:.3f} "
                f"conf={confidence:.3f} votes={bullish_votes}/{len(scores)} | "
                + ", ".join(reason_bits)
            )
            ts = str(row.get("timestamp") or row.get("date") or ts_now)

            for hz, cfg in horizon_cfg.items():
                tp = price + cfg["tp_atr"] * atr
                sl = price - cfg["sl_atr"] * atr
                if sl <= 0 or tp <= price or sl >= price:
                    continue
                rrr = (tp - price) / (price - sl)
                if rrr < min_rrr:
                    continue

                records.append({
                    "ticker": ticker,
                    "asset": ticker,
                    "symbol": ticker,
                    "close": price,
                    "current_price": price,
                    "entry_price": price,
                    "take_profit": tp,
                    "stop_loss": sl,
                    "tp_price": tp,
                    "sl_price": sl,
                    "optimized_take_profit": tp,
                    "optimized_stop_loss": sl,
                    "atr_14": atr,
                    "feature_atr": atr,
                    "raw_score": rule_score,
                    "prediction_probability": confidence,
                    "probability": confidence,
                    "prediction_confidence": confidence,
                    "confidence": confidence,
                    "calibrated_prob": confidence,
                    "risk_reward_ratio": rrr,
                    "optimized_risk_reward": rrr,
                    "expected_return": (tp - price) / price,
                    "side": "BUY",
                    "direction": "BUY",
                    "candidate_signal": "BUY",
                    "signal_direction": 1,
                    "horizon": hz,
                    "expected_holding_days": float(cfg["days"]),
                    "model_status": "MODEL_NOT_READY",  # never rewrite as ML-OK
                    "signal_source": "RULE_BASED",
                    "reason": reason,
                    "signal_reason": reason,
                    "signal_explanation": reason,
                    "signal_explanation_text": reason,
                    "timestamp": ts,
                    "volume_24h_idr": vol_idr if vol_idr == vol_idr else 1e12,
                    "volume_idr": vol_idr if vol_idr == vol_idr else 1e12,
                    "is_valid_execution": True,
                    "signal_valid": True,
                })

        if not records:
            return pl.DataFrame()
        return pl.DataFrame(records)

    def _step_6_prediction_engine(self) -> None:
        logger.info("▶ [STEP 6] UnifiedPredictionEngine (TP/SL/regime/rank)")
        if not HAS_PREDICTION or UnifiedPredictionEngine is None:
            logger.warning("⚠️ prediction.py offline")
            return
        src = self.predictions_data if self.predictions_data.height else self.features_data
        if src.height == 0:
            return
        # RULE_BASED candidates already carry Entry/TP/SL/RRR/horizon — do not re-map via ML calibrators
        if "signal_source" in src.columns:
            try:
                if src.filter(pl.col("signal_source").cast(pl.Utf8).str.to_uppercase() == "RULE_BASED").height == src.height:
                    logger.info("✔ [RULE_BASED] Skip UnifiedPredictionEngine re-map (geometry already set)")
                    return
            except Exception:
                pass
        try:
            eng = UnifiedPredictionEngine(master_config={"gemini_config": {"api_key": self.api_key}})
            if hasattr(eng, "activate_all"):
                try:
                    eng.activate_all()
                except Exception:
                    pass
            work = src
            if "current_price" not in work.columns and "close" in work.columns:
                work = work.with_columns(pl.col("close").alias("current_price"))
            if "raw_score" not in work.columns:
                for c in ("prediction_probability", "probability", "confidence"):
                    if c in work.columns:
                        work = work.with_columns(pl.col(c).alias("raw_score"))
                        break
                if "raw_score" not in work.columns:
                    logger.warning("⚠️ prediction engine needs raw_score — skip")
                    return
            out = eng.run_prediction_pipeline(work)
            if isinstance(out, pl.DataFrame) and out.height:
                self.predictions_data = out
                logger.info(f"✔ Prediction engine rows={out.height}")
        except Exception as e:
            logger.warning(f"⚠️ Prediction engine soft-fail: {e}")

    def _step_7_god_candidates(self) -> None:
        logger.info("▶ [STEP 7] Gemini God deep-dive top candidates")
        src = self.predictions_data if self.predictions_data.height else pl.DataFrame()
        if src.height == 0:
            logger.info("ℹ️ No candidates for God analysis")
            return
        try:
            cols = [c for c in ("ticker", "asset", "prediction_probability", "probability", "close", "raw_score") if c in src.columns]
            head = src.select(cols).head(10) if cols else src.head(10)
            report = head.write_csv() if hasattr(head, "write_csv") else str(head)
            analysis = self.god.analyze_candidates(report, relaxation=False)
            if analysis:
                logger.info(f"🧠 [GOD_ANALYSIS] {str(analysis)[:500]}")
        except Exception as e:
            logger.warning(f"⚠️ God candidate analysis soft-fail: {e}")

    def _step_8_signals(self) -> None:
        logger.info("▶ [STEP 8] Signal gateway pipeline")
        src = self.predictions_data
        if src is None or src.height == 0:
            logger.info("ℹ️ No predictions → NO_SIGNAL")
            self.signals_data = pl.DataFrame()
            return
        if not HAS_SIGNAL or UnifiedSignalEngine is None:
            logger.warning("⚠️ signal_idx offline — minimal orders from predictions")
            self.orders = self._predictions_to_orders(src)
            return
        try:
            init_kwargs = {
                "custom_configs": {
                    "generator": {
                        "min_24h_volume_idr": float(self.trading_config["min_adtv_idr"]),
                        "min_risk_reward_ratio": float(self.trading_config["min_rrr"]),
                    },
                    "confidence": {
                        "min_prediction_confidence": float(self.trading_config["min_confidence"]),
                    },
                }
            }
            try:
                self.signal_engine = UnifiedSignalEngine(**init_kwargs, gemini_api_key=self.api_key)
            except TypeError:
                self.signal_engine = UnifiedSignalEngine(**init_kwargs)
            if hasattr(self.signal_engine, "execute_pipeline"):
                self.signals_data = self.signal_engine.execute_pipeline(src, run_ai_diagnostics=True)
            elif hasattr(self.signal_engine, "generate_signals"):
                self.signals_data = self.signal_engine.generate_signals(src)
            else:
                self.signals_data = pl.DataFrame()
            self.orders = self._signals_to_orders(self.signals_data)
            logger.info(f"✔ Signals rows={getattr(self.signals_data, 'height', 0)} orders={len(self.orders)}")
        except Exception as e:
            logger.error(f"❌ Signal pipeline failed: {e}")
            self.orders = self._predictions_to_orders(src)

    def _step_9_risk(self) -> None:
        logger.info("▶ [STEP 9] Risk engine")
        if not HAS_RISK or UnifiedRiskEngine is None:
            logger.warning("⚠️ risk.py offline")
            return
        src = self.market_data if self.market_data.height else self.predictions_data
        if src.height == 0:
            return
        try:
            self.risk_engine = UnifiedRiskEngine()
            if hasattr(self.risk_engine, "evaluate_market_risk"):
                self.risk_output = self.risk_engine.evaluate_market_risk(
                    src,
                    pipeline_timestamp=datetime.now(timezone.utc).isoformat(),
                    execution_id=self.execution_id,
                )
                logger.info(f"✔ Risk evaluated: {type(self.risk_output).__name__}")
        except Exception as e:
            logger.warning(f"⚠️ Risk soft-fail: {e}")

    def _step_10_portfolio_simulation(self) -> None:
        """Simulated portfolio: load → process_trading_signals(latest_prices) → TP/SL → simulate_execution → save."""
        logger.info("▶ [STEP 10] Portfolio simulation sync (SIMULATION_ONLY)")
        latest_prices = self._latest_prices_from_market()

        if HAS_PORTFOLIO and UnifiedPortfolioEngine is not None:
            try:
                try:
                    self.portfolio_engine = UnifiedPortfolioEngine(
                        config={"dry_run": True, "simulation_only": True},
                        state_file=PORTFOLIO_STATE_FILE,
                        gemini_api_key=self.api_key,
                    )
                except TypeError:
                    try:
                        self.portfolio_engine = UnifiedPortfolioEngine(
                            config={"dry_run": True, "simulation_only": True},
                            state_file=PORTFOLIO_STATE_FILE,
                        )
                    except TypeError:
                        self.portfolio_engine = UnifiedPortfolioEngine(config={"dry_run": True})

                if hasattr(self.portfolio_engine, "load_portfolio_state"):
                    try:
                        loaded = self.portfolio_engine.load_portfolio_state()
                        if isinstance(loaded, dict) and loaded:
                            self.portfolio_state = loaded
                            logger.info(f"✔ Portfolio state LOADED | positions={loaded.get('active_positions_count', len(loaded.get('positions') or {}))}")
                    except Exception as e:
                        logger.warning(f"⚠️ load_portfolio_state: {e}")

                # Build signals payload from orders (list[dict]) — compatible with process_trading_signals
                signals_payload: Any = self.orders if self.orders else []
                if self.signals_data is not None and getattr(self.signals_data, "height", 0) > 0 and not signals_payload:
                    # fallback: convert valid signal rows
                    signals_payload = self._signals_to_orders(self.signals_data)

                if signals_payload and hasattr(self.portfolio_engine, "process_trading_signals"):
                    if not latest_prices:
                        # derive from orders entry prices
                        for o in signals_payload:
                            t = str(o.get("ticker") or o.get("symbol") or "")
                            px = float(o.get("entry_price") or 0.0)
                            if t and px > 0:
                                latest_prices[t] = px
                                latest_prices[t.replace(".JK", "")] = px
                    try:
                        result = self.portfolio_engine.process_trading_signals(
                            signals_payload,
                            latest_prices=latest_prices,
                            top_n=max(1, len(signals_payload)),
                        )
                        logger.info(f"✔ process_trading_signals OK | type={type(result).__name__} | orders_in={len(signals_payload)}")
                        if isinstance(result, dict):
                            self.portfolio_state = {**self.portfolio_state, **result}
                    except TypeError as te:
                        logger.error(f"❌ process_trading_signals signature error: {te}")
                    except Exception as e:
                        logger.warning(f"⚠️ process_trading_signals soft-fail: {e}")

                # Refresh state summary
                try:
                    if hasattr(self.portfolio_engine, "load_portfolio_state"):
                        st = self.portfolio_engine.load_portfolio_state()
                        if isinstance(st, dict):
                            self.portfolio_state = {**self.portfolio_state, **st}
                except Exception:
                    pass

                npos = len(getattr(self.portfolio_engine, "positions", {}) or {})
                logger.info(f"✔ Simulated portfolio positions ACTIVE count={npos}")

                if hasattr(self.portfolio_engine, "save_portfolio_state"):
                    try:
                        ok = self.portfolio_engine.save_portfolio_state()
                        logger.info(f"{'✅' if ok else '⚠️'} Portfolio state SAVED → {PORTFOLIO_STATE_FILE}")
                    except Exception as e:
                        logger.warning(f"⚠️ save_portfolio_state: {e}")

            except Exception as e:
                logger.warning(f"⚠️ Portfolio init soft-fail: {e}")

        # Existing simulation module — paper path only
        if HAS_SIMULATION and simulate_execution is not None:
            try:
                sig_df = self.signals_data if self.signals_data is not None and self.signals_data.height else None
                if sig_df is None and self.orders:
                    sig_df = pl.DataFrame(self.orders)
                mkt = self.market_data if self.market_data is not None and self.market_data.height else None
                if mkt is not None and sig_df is not None and sig_df.height > 0:
                    sim_out = simulate_execution(mkt, sig_df, config={"simulation_only": True})
                    self.simulation_result = sim_out
                    logger.info(f"✔ simulation.simulate_execution rows={getattr(sim_out, 'height', 'n/a')}")
                else:
                    logger.info("ℹ️ simulation skipped (need market_df + signals_df)")
            except Exception as e:
                logger.warning(f"⚠️ simulation.simulate_execution soft-fail: {e}")
        else:
            logger.info("ℹ️ simulation.py not available")

        # TP/SL monitoring via existing execute_sell (simulation only)
        try:
            self._apply_simulated_tpsl_exits(latest_prices)
        except Exception as e:
            logger.warning(f"⚠️ TP/SL monitor: {e}")

        if not self.portfolio_state:
            self.portfolio_state = {
                "equity": 10_000_000.0,
                "cash": 10_000_000.0,
                "positions": {},
                "return_pct": 0.0,
                "exposure_pct": 0.0,
                "mode": "simulation",
                "simulation_only": True,
            }
        else:
            self.portfolio_state["mode"] = "simulation"
            self.portfolio_state["simulation_only"] = True

    def _step_11_storage(self) -> None:
        logger.info("▶ [STEP 11] Persist signals/predictions")
        if not HAS_STORAGE or UnifiedStorageEngine is None:
            logger.warning("⚠️ storage.py offline")
            return
        try:
            self.storage_engine = UnifiedStorageEngine()
            if self.signals_data is not None and getattr(self.signals_data, "height", 0) and hasattr(self.storage_engine, "persist_signals"):
                try:
                    self.storage_engine.persist_signals(self.signals_data)
                except Exception as e:
                    logger.warning(f"⚠️ persist_signals: {e}")
            if self.predictions_data is not None and getattr(self.predictions_data, "height", 0) and hasattr(self.storage_engine, "persist_predictions"):
                try:
                    self.storage_engine.persist_predictions(self.predictions_data)
                except Exception as e:
                    logger.warning(f"⚠️ persist_predictions: {e}")
            logger.info("✔ Storage step done")
        except Exception as e:
            logger.warning(f"⚠️ Storage soft-fail: {e}")

    def _step_12_telegram(self) -> None:
        logger.info("▶ [STEP 12] Telegram broadcast via reporting")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        narrative = self.god.narrate(self.portfolio_state, top_signals=self.orders[:5] if self.orders else None)
        logger.info(f"🗞️ [NARRATIVE] {narrative[:300]}")
        if not HAS_REPORTING or UnifiedReportingEngine is None:
            logger.warning("⚠️ reporting.py offline")
            return
        if not token or not chat_id:
            logger.warning("⚠️ TELEGRAM secrets unset — skip send")
            return
        try:
            engine = UnifiedReportingEngine(
                config={
                    "TELEGRAM_BOT_TOKEN": token,
                    "TELEGRAM_CHAT_ID": chat_id,
                    "REPORTING_MIN_CONFIDENCE": float(self.trading_config["min_confidence"]),
                    "INITIAL_CAPITAL_IDR": float(self.portfolio_state.get("equity", 10_000_000.0)),
                    "MARKET_NARRATIVE": narrative,
                },
                mode="dry_run",
            )
            ok = engine.send_telegram_broadcast(
                orders=self.orders if self.orders else None,
                portfolio_data=self.portfolio_state,
            )
            logger.info(f"{'✅' if ok else '⚠️'} Telegram result={ok} orders={len(self.orders)}")
        except Exception as e:
            logger.error(f"❌ Telegram failed: {e}")

    def _step_13_monitoring_eval(self) -> None:
        logger.info("▶ [STEP 13] Monitoring / evaluation hooks")
        if HAS_MONITORING and UnifiedMonitoringEngine is not None:
            try:
                mon = UnifiedMonitoringEngine()
                if hasattr(mon, "execute_full_audit"):
                    try:
                        # best-effort; may require richer inputs
                        mon.execute_full_audit() if mon.execute_full_audit.__code__.co_argcount <= 1 else None
                    except TypeError:
                        logger.info("✔ Monitoring online (execute_full_audit needs extra args — construct OK)")
                    except Exception as e:
                        logger.info(f"✔ Monitoring online; audit soft-skip: {e}")
                logger.info(f"✔ Monitoring engine: {type(mon).__name__}")
            except Exception as e:
                logger.warning(f"⚠️ Monitoring soft-fail: {e}")
        if HAS_EVALUATION and UnifiedEvaluationEngine is not None:
            try:
                ev = UnifiedEvaluationEngine()
                # Prefer business methods over construct-only
                called = False
                for method_name in ("run_full_evaluation", "execute_pipeline_evaluation", "evaluate_pipeline", "evaluate"):
                    if not hasattr(ev, method_name):
                        continue
                    fn = getattr(ev, method_name)
                    try:
                        # supply predictions if signature allows kwargs
                        if self.predictions_data is not None and self.predictions_data.height > 0:
                            try:
                                self.evaluation_result = fn(predictions_df=self.predictions_data)
                            except TypeError:
                                try:
                                    self.evaluation_result = fn(self.predictions_data)
                                except TypeError:
                                    self.evaluation_result = fn()
                        else:
                            self.evaluation_result = fn()
                        called = True
                        logger.info(f"✔ Evaluation via .{method_name}() → {type(self.evaluation_result).__name__}")
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ evaluation.{method_name}: {e}")
                if not called:
                    logger.info(f"✔ Evaluation engine online: {type(ev).__name__} (no compatible call this run)")
            except Exception as e:
                logger.warning(f"⚠️ Evaluation soft-fail: {e}")
        if HAS_AUTONOMOUS and UnifiedAutonomousEngine is not None:
            logger.info("✔ autonomous_engine_idx available (not auto-started)")

    def _step_14_self_learning(self) -> None:
        """
        Continuous adaptation model:
        - Every run: log feedback snapshot (signals → portfolio outcome) for later learning.
        - Full UnifiedSelfLearningEngine: on --self-learning OR post-market cron OR
          IDX_ALWAYS_LEARN=1.
        Model retrain remains relatively expensive → prefer daily post-market for heavy fit.
        """
        always = os.getenv("IDX_ALWAYS_LEARN", "").strip() in ("1", "true", "True", "yes")
        do_full = bool(self.self_learning_flag or always)

        # Light continuous memory: append run telemetry for adaptation over time
        try:
            os.makedirs("storage", exist_ok=True)
            feedback = {
                "ts": datetime.now(WIB_TZ).isoformat(),
                "execution_id": self.execution_id,
                "n_orders": len(self.orders),
                "trading_config": self.trading_config,
                "portfolio_equity": self.portfolio_state.get("equity"),
                "portfolio_return_pct": self.portfolio_state.get("return_pct"),
                "simulation_only": True,
            }
            with open("storage/learning_feedback.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback, default=str) + "\n")
            logger.info("✔ [LEARN_LIGHT] feedback snapshot appended → storage/learning_feedback.jsonl")
        except Exception as e:
            logger.warning(f"⚠️ light feedback log failed: {e}")

        if not do_full:
            logger.info(
                "▶ [STEP 14] Full self-learning deferred "
                "(enable via --self-learning, post-market cron, or IDX_ALWAYS_LEARN=1)"
            )
            return

        logger.info("▶ [STEP 14] Full self-learning / adaptation engine")
        if not HAS_SELF_LEARNING or UnifiedSelfLearningEngine is None:
            logger.warning("⚠️ self_learning.py offline")
            return
        try:
            eng = UnifiedSelfLearningEngine()
            ran = False
            for method in ("run", "adapt", "execute", "learn", "run_adaptation_cycle"):
                if hasattr(eng, method):
                    try:
                        getattr(eng, method)()
                        logger.info(f"✔ Self-learning via .{method}()")
                        ran = True
                        break
                    except TypeError:
                        continue
                    except Exception as e:
                        logger.warning(f"⚠️ self-learning {method}: {e}")
            if not ran:
                logger.warning("⚠️ Self-learning engine found but no callable cycle method succeeded")
        except Exception as e:
            logger.warning(f"⚠️ Self-learning soft-fail: {e}")

    def _predictions_to_orders(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        """Convert prediction/signal rows into portfolio-compatible order dicts.

        Prefer signal-engine validated rows. Fail-closed on bad geometry / low conf / low RRR.
        Does NOT invent BUY from MODEL_NOT_READY unless row is already is_valid_execution.
        """
        orders: List[Dict[str, Any]] = []
        if df is None or df.height == 0:
            return orders
        min_conf = float(self.trading_config["min_confidence"])
        min_rrr = float(self.trading_config["min_rrr"])
        work = df

        ticker_col = "ticker" if "ticker" in work.columns else ("asset" if "asset" in work.columns else ("symbol" if "symbol" in work.columns else None))
        if ticker_col is None:
            return orders

        for row in work.to_dicts():
            # Already validated by signal engine?
            prevalidated = bool(row.get("is_valid_execution") or row.get("signal_valid"))
            signal_source = str(row.get("signal_source") or "ML").strip().upper()
            status = str(row.get("model_status") or "")

            # ML fail-closed: MODEL_NOT_READY never becomes ML BUY.
            # RULE_BASED is a separate deterministic path and must carry full contract.
            if signal_source != "RULE_BASED" and (not prevalidated) and status in ("MODEL_ERROR", "MODEL_NOT_READY"):
                continue
            if signal_source == "RULE_BASED" and status in ("MODEL_ERROR",) and not prevalidated:
                continue

            side = str(row.get("side") or row.get("direction") or row.get("candidate_signal") or "BUY").upper()
            if side not in ("BUY", "LONG", "1"):
                # only simulated long entries in this system
                if not prevalidated:
                    continue
                if side not in ("BUY", "LONG"):
                    continue
            side = "BUY"

            # --- mandatory probability/confidence (fail-closed on NaN/invalid) ---
            raw_conf = row.get("prediction_confidence")
            if raw_conf is None:
                raw_conf = row.get("prediction_probability")
            if raw_conf is None:
                raw_conf = row.get("confidence")
            if raw_conf is None:
                raw_conf = row.get("probability")
            try:
                conf = float(raw_conf) if raw_conf is not None else float("nan")
            except (TypeError, ValueError):
                conf = float("nan")
            if conf != conf:  # NaN
                continue
            if conf > 1.0:
                conf /= 100.0
            if conf <= 0.0 or conf > 1.0:
                continue
            if conf < min_conf and not prevalidated:
                continue
            if conf < min_conf and prevalidated:
                # even READY rows must meet absolute floor after dilution
                continue

            entry = float(row.get("entry_price") or row.get("current_price") or row.get("close") or 0.0)
            tp = float(
                row.get("optimized_take_profit")
                or row.get("tp_price")
                or row.get("take_profit")
                or row.get("target_price")
                or 0.0
            )
            sl = float(
                row.get("optimized_stop_loss")
                or row.get("sl_price")
                or row.get("stop_loss")
                or 0.0
            )
            if entry <= 0 or tp <= entry or sl <= 0 or sl >= entry:
                continue
            rrr = float(row.get("optimized_risk_reward") or row.get("risk_reward_ratio") or row.get("calculated_risk_reward") or 0.0)
            if rrr <= 0:
                rrr = (tp - entry) / (entry - sl)
            if rrr < min_rrr:
                continue

            t = str(row.get(ticker_col) or row.get("symbol") or row.get("asset") or "").strip()
            if not t:
                continue
            if not t.endswith(".JK"):
                t = f"{t}.JK"

            reason = str(
                row.get("signal_explanation_text")
                or row.get("signal_explanation")
                or row.get("signal_reason")
                or row.get("final_validator_reason")
                or row.get("reason")
                or ""
            ).strip()
            if not reason:
                continue  # missing reason → reject

            horizon = normalize_horizon(row.get("horizon") or row.get("expected_holding_days") or "SHORT")
            exp_ret = float(row.get("expected_return") or row.get("calculated_expected_value_pct") or 0.0)
            ts = str(row.get("timestamp") or row.get("date") or "")
            if not ts:
                continue  # missing timestamp → reject

            orders.append({
                "asset": t,
                "ticker": t,
                "symbol": t,
                "direction": "BUY",
                "side": "BUY",
                "horizon": horizon,
                "entry_price": entry,
                "tp_price": tp,
                "sl_price": sl,
                "take_profit": tp,
                "stop_loss": sl,
                "optimized_take_profit": tp,
                "optimized_stop_loss": sl,
                "probability": conf,
                "confidence": conf,
                "prediction_probability": conf,
                "ranking_score": conf,
                "risk_reward_ratio": rrr,
                "optimized_risk_reward": rrr,
                "expected_return": exp_ret,
                "reason": reason,
                "timestamp": ts,
                "is_valid_execution": bool(prevalidated),
                "signal_source": signal_source if signal_source in ("ML", "RULE_BASED") else "ML",
                "model_status": status or ("MODEL_NOT_READY" if signal_source == "RULE_BASED" else "OK"),
                "volume_24h_idr": float(row.get("volume_24h_idr") or row.get("f_adtv_20d_idr") or row.get("volume_idr") or row.get("adtv_20d_idr") or 1e12),
                "volume_idr": float(row.get("volume_idr") or row.get("volume_24h_idr") or 1e12),
            })
        return orders

    def _signals_to_orders(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        """Bridge signal engine output → orders. Prefer is_valid_execution / signal_valid rows."""
        if df is None or df.height == 0:
            return []
        work = df
        # Prefer explicit BUY + valid execution flags when present
        if "side" in work.columns:
            work = work.filter(pl.col("side").cast(pl.Utf8).str.to_uppercase().is_in(["BUY", "LONG"]))
        if "is_valid_execution" in work.columns:
            valid = work.filter(pl.col("is_valid_execution").fill_null(False) == True)
            if valid.height > 0:
                work = valid
        elif "signal_valid" in work.columns:
            valid = work.filter(pl.col("signal_valid").fill_null(False) == True)
            if valid.height > 0:
                work = valid
        orders = self._predictions_to_orders(work)
        logger.info(f"🔗 [SIGNAL→ORDERS] input_rows={df.height} filtered={work.height} orders={len(orders)}")
        return orders

    def _latest_prices_from_market(self) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        src = self.market_data if self.market_data is not None and self.market_data.height else self.features_data
        if src is None or src.height == 0:
            return prices
        tcol = "ticker" if "ticker" in src.columns else ("asset" if "asset" in src.columns else None)
        if tcol is None or "close" not in src.columns:
            return prices
        try:
            last = src.sort("timestamp" if "timestamp" in src.columns else ("date" if "date" in src.columns else tcol)).group_by(tcol).tail(1)
            for row in last.to_dicts():
                t = str(row.get(tcol, ""))
                if t and not t.endswith(".JK"):
                    t = f"{t}.JK"
                px = float(row.get("close") or 0.0)
                if t and px > 0:
                    prices[t] = px
                    # also bare symbol key
                    prices[t.replace(".JK", "")] = px
        except Exception as e:
            logger.warning(f"⚠️ latest_prices build failed: {e}")
        return prices

    def _apply_simulated_tpsl_exits(self, prices: Dict[str, float]) -> None:
        """Use existing portfolio execute_sell when TP/SL touched by post-entry prices. Simulation only.

        Conservative rule: if a single price point would hit both TP and SL (ambiguous),
        prefer CLOSED_SL (fail-safe, non-optimistic).
        """
        if not self.portfolio_engine or not prices:
            return
        try:
            positions = dict(getattr(self.portfolio_engine, "positions", {}) or {})
        except Exception:
            return
        for sym, pos in list(positions.items()):
            if str(pos.get("status", "ACTIVE")).upper() not in ("ACTIVE", "OPEN", ""):
                continue
            base = str(pos.get("base_symbol") or sym.split("|")[0])
            px = float(
                prices.get(sym)
                or prices.get(base)
                or prices.get(base.replace(".JK", ""))
                or prices.get(sym.replace(".JK", ""))
                or pos.get("current_price")
                or 0.0
            )
            if px <= 0:
                continue
            tp = float(pos.get("tp_price") or pos.get("take_profit") or 0.0)
            sl = float(pos.get("sl_price") or pos.get("stop_loss") or 0.0)
            entry = float(pos.get("avg_price") or pos.get("buy_price") or 0.0)
            hit_tp = tp > 0 and px >= tp
            hit_sl = sl > 0 and px <= sl
            if hit_tp and hit_sl:
                reason = "CLOSED_SL"  # conservative / ambiguous → not optimistic TP
            elif hit_tp:
                reason = "CLOSED_TP"
            elif hit_sl:
                reason = "CLOSED_SL"
            else:
                continue
            try:
                eng = self.portfolio_engine.execution_engine
                res = eng.execute_sell(sym, lots_to_sell=None, current_price=px, reason=reason, volume_24h_idr=float("inf"))
                if res:
                    if getattr(eng, "transaction_history", None):
                        eng.transaction_history[-1]["exit_reason"] = reason
                        eng.transaction_history[-1]["status"] = reason
                        eng.transaction_history[-1]["horizon"] = pos.get("horizon", "SHORT")
                    logger.info(
                        f"🏁 [{reason}] {sym} horizon={pos.get('horizon')} @ {px:.2f} "
                        f"(entry={entry:.2f} tp={tp:.2f} sl={sl:.2f})"
                    )
            except Exception as e:
                logger.warning(f"⚠️ TP/SL exit failed for {sym}: {e}")

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info("🚀 IDX SIMULATION PIPELINE — Gemini God Entity | SIMULATION_ONLY (no live trading)")
        logger.info(f"   execution_id={self.execution_id} dry_run={self.dry_run}")
        logger.info("=" * 60)
        try:
            self._step_0_god_awaken()
            self._step_1_universe()
            self._step_2_data()
            self._step_3_god_config()
            self._step_4_features()
            self._step_5_ml()
            self._step_6_prediction_engine()
            self._step_7_god_candidates()
            self._step_8_signals()
            self._step_9_risk()
            self._step_10_portfolio_simulation()
            self._step_11_storage()
            self._step_12_telegram()
            self._step_13_monitoring_eval()
            self._step_14_self_learning()
            logger.info("=" * 60)
            logger.info(
                f"🎉 PIPELINE COMPLETE | orders={len(self.orders)} | "
                f"configured_by={self.trading_config.get('configured_by')}"
            )
            logger.info("=" * 60)
        except Exception as e:
            msg = str(e)
            code = "PIPELINE_FAILURE"
            low = msg.lower()
            if "data" in low or "bei" in low or "cache" in low or "yfinance" in low:
                code = "DATA_FAILURE"
            elif "model" in low:
                code = "MODEL_FAILURE"
            elif "predict" in low:
                code = "PREDICTION_FAILURE"
            elif "signal" in low:
                code = "SIGNAL_FAILURE"
            elif "risk" in low:
                code = "RISK_FAILURE"
            elif "portfolio" in low:
                code = "PORTFOLIO_FAILURE"
            elif "simulat" in low:
                code = "SIMULATION_FAILURE"
            elif "report" in low or "telegram" in low:
                code = "REPORTING_FAILURE"
            logger.error(f"💥 [{code}] Fatal pipeline error: {e}\n{traceback.format_exc()}")
            sys.exit(1)

    def run_health_check(self) -> None:
        logger.info("🩺 Health-check mode")
        self.god.awaken()
        if HAS_MONITORING and HealthCheckEngine is not None:
            try:
                hc = HealthCheckEngine()
                if hasattr(hc, "run"):
                    hc.run()
                logger.info("✔ HealthCheckEngine finished")
            except Exception as e:
                logger.warning(f"⚠️ HealthCheckEngine: {e}")
        self._log_module_matrix()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Simulated Scalping Engine — Gemini God Orchestrator (SIMULATION ONLY, no live trading)")
    parser.add_argument("--dry-run", action="store_true", help="Paper/simulation run (always on; live trading removed)")
    parser.add_argument("--self-learning", action="store_true", help="Picu self-learning engine")
    parser.add_argument("--bootstrap-universe", action="store_true", help="Tulis universe.json default")
    parser.add_argument("--reset-dryrun", action="store_true", help="Reset state simulasi lokal saja")
    parser.add_argument("--health-check", action="store_true", help="Jalankan health check modul")
    args = parser.parse_args()

    if args.bootstrap_universe:
        sanitised = sanitize_ticker_list(DEFAULT_BLUECHIP_UNIVERSE)
        with open(DEFAULT_UNIVERSE_FILE, "w", encoding="utf-8") as f:
            json.dump(sanitised, f, indent=2)
        print(f"✅ universe.json restored with {len(sanitised)} tickers")
        sys.exit(0)

    if args.reset_dryrun:
        import pathlib
        removed = []
        for rel in ("data/storage_dryrun.db", "storage_dryrun.db", "autonomous_engine.db"):
            p = pathlib.Path(rel)
            if p.is_file():
                try:
                    p.unlink()
                    removed.append(str(p))
                except OSError as e:
                    print(f"⚠️ cannot remove {p}: {e}")
        print(f"✅ [RESET-DRYRUN] removed={removed or 'none'}")
        print("ℹ️ Production portfolio NOT touched.")
        sys.exit(0)

    orch = ProductionOrchestrator(dry_run=True, self_learning=args.self_learning)
    if args.health_check:
        orch.run_health_check()
    else:
        if not args.dry_run and not args.self_learning:
            logger.info("🛡️ SIMULATION_ONLY=True — live trading paths removed; paper portfolio only.")
        orch.run()

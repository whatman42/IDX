"""
=============================================================================
IDX Quantitative Scalping Engine — Main Orchestrator (S.E.A. v8.0)
Version       : 2026.Q3.v28.1 (Transactional Evolution)
Compliance    : IDX — Fail-Closed Safety, Pipeline State Machine
=============================================================================
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

WIB_TZ = ZoneInfo("Asia/Jakarta")

# ============================================================================
# DYNAMIC LOADER
# ============================================================================
try:
    from sea_loader import set_agent, load_module_dynamically
    HAS_SEA_LOADER = True
except ImportError:
    set_agent = None
    load_module_dynamically = None
    HAS_SEA_LOADER = False
    print("❌ sea_loader.py not found. Please create it.")
    sys.exit(1)

try:
    from sea import GodEntity, VERSION as AGENT_VERSION
    HAS_SEA = True
except ImportError:
    GodEntity = None
    AGENT_VERSION = "0.0.0"
    HAS_SEA = False
    print("❌ sea.py not found.")
    sys.exit(1)

# ============================================================================
# STATIC MODULES
# ============================================================================
try:
    from data import load_and_prepare_market_data, sanitize_ticker_list, DataLoader
    HAS_DATA = True
except ImportError:
    load_and_prepare_market_data = None
    sanitize_ticker_list = None
    DataLoader = None
    HAS_DATA = False
    print("⚠️ data.py not found. Market data loading will fail.")

try:
    from logger import get_logger
    logger = get_logger("IDX.Main")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("IDX.Main")

# ============================================================================
# CONSTANTS
# ============================================================================
SIMULATION_ONLY: bool = True
PORTFOLIO_STATE_FILE: str = "portfolio_state.json"
DEFAULT_BLUECHIP_UNIVERSE = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
    "ASII.JK", "UNVR.JK", "ICBP.JK", "INDF.JK", "GOTO.JK",
]
DEFAULT_UNIVERSE_FILE = "universe.json"

VALID_HORIZONS = ("SHORT", "MEDIUM", "LONG")

def normalize_horizon(raw) -> str:
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

# ============================================================================
# ORCHESTRATOR WITH PIPELINE STATE
# ============================================================================
class ProductionOrchestrator:
    def __init__(self, dry_run: bool = True, self_learning: bool = False, enable_self_evolving: bool = False):
        self.dry_run = True
        self.self_learning_flag = bool(self_learning)
        self.enable_self_evolving = bool(enable_self_evolving)
        self.api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()

        # Pipeline state machine
        self.pipeline_state = "INIT"   # INIT, DATA_LOADED, FEATURES_READY, ML_READY,
                                       # SIGNALS_READY, PORTFOLIO_SIMULATED, EVOLUTION_COMPLETE,
                                       # DATA_FAILURE, PIPELINE_FAILURE

        self.trading_config: Dict[str, Any] = {}
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

        # S.E.A. Agent (full authority, but transactional)
        self.self_evolving_agent = None
        if self.enable_self_evolving and HAS_SEA and GodEntity is not None:
            project_root = os.getcwd()
            self.self_evolving_agent = GodEntity(base_dir=project_root)
            if HAS_SEA_LOADER and set_agent is not None:
                set_agent(self.self_evolving_agent)
            logger.info(f"🤖 [SEA_MASTER] sea.py v{AGENT_VERSION} initialized (Transactional mode)")

            # Ensure critical modules exist and are valid
            self._ensure_modules()
        elif self.enable_self_evolving:
            logger.warning("⚠️ sea.py not available — disable --enable-self-evolving")

        self._log_module_matrix()

    # ------------------------------------------------------------------------
    # Module validation & loading
    # ------------------------------------------------------------------------
    def _ensure_modules(self) -> None:
        if not self.self_evolving_agent:
            return
        required = ["risk", "features", "portfolio", "prediction", "signal_idx"]
        for mod in required:
            success, code, err = self.self_evolving_agent.generate_module_code(mod)
            if not success:
                logger.error(f"❌ Module {mod} generation failed: {err}")
                if mod in ["risk", "portfolio"]:
                    raise RuntimeError(f"Critical module {mod} invalid.")
            else:
                logger.info(f"✅ Module {mod} is valid (len={len(code)})")

    def _load_module_with_contract(self, module_name: str, required_symbols: List[str]):
        if not HAS_SEA_LOADER or load_module_dynamically is None:
            return None
        mod = load_module_dynamically(module_name)
        if not mod:
            return None
        # If agent available, verify contract
        if self.self_evolving_agent:
            if not self.self_evolving_agent.validate_module_contract(module_name, required_symbols):
                logger.error(f"❌ Contract validation failed for {module_name}")
                return None
        return mod

    def _log_module_matrix(self) -> None:
        matrix = {
            "data": HAS_DATA,
            "sea (agent)": HAS_SEA,
            "sea_loader": HAS_SEA_LOADER,
        }
        if self.self_evolving_agent:
            for mod in ["risk", "features", "portfolio", "prediction", "signal_idx"]:
                valid = self.self_evolving_agent.memory.is_module_valid(mod)
                matrix[mod] = valid
        online = sum(1 for v in matrix.values() if v)
        logger.info(f"📦 [MODULE_MATRIX] {online}/{len(matrix)} online → {matrix}")

    # ------------------------------------------------------------------------
    # Pipeline steps with state machine
    # ------------------------------------------------------------------------
    def _step_0_sea_awaken(self) -> None:
        logger.info("▶ [STEP 0] S.E.A. Master awaken")
        if self.self_evolving_agent:
            try:
                self.self_evolving_agent.boot()
                logger.info("🧠 [SEA_MASTER] GodEntity booted.")
            except Exception as e:
                logger.warning(f"⚠️ [SEA_MASTER] Boot failed: {e}")
                self.pipeline_state = "PIPELINE_FAILURE"
        else:
            logger.warning("⚠️ [SEA_MASTER] Agent disabled.")

    def _step_1_universe(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 1] Universe sync")
        if os.path.isfile(DEFAULT_UNIVERSE_FILE):
            try:
                with open(DEFAULT_UNIVERSE_FILE, encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    self.universe = sanitize_ticker_list(raw) if HAS_DATA else raw
                elif isinstance(raw, dict):
                    self.universe = sanitize_ticker_list(raw.get("tickers") or raw.get("universe") or []) if HAS_DATA else []
            except Exception as e:
                logger.warning(f"⚠️ universe.json read failed: {e}")
        if not self.universe:
            self.universe = sanitize_ticker_list(DEFAULT_BLUECHIP_UNIVERSE) if HAS_DATA else DEFAULT_BLUECHIP_UNIVERSE
            logger.info(f"ℹ️ Default universe ({len(self.universe)} tickers)")
        logger.info(f"✔ Universe size={len(self.universe)}")

    def _step_2_data(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 2] Data layer")
        if not HAS_DATA:
            logger.error("❌ data.py not available.")
            self.pipeline_state = "DATA_FAILURE"
            return
        try:
            raw_result = load_and_prepare_market_data(symbols=self.universe, use_cache=True)
            # Normalize result
            if isinstance(raw_result, pl.DataFrame):
                self.market_data = raw_result
            elif isinstance(raw_result, dict):
                if "data" in raw_result and isinstance(raw_result["data"], pl.DataFrame):
                    self.market_data = raw_result["data"]
                elif "df" in raw_result and isinstance(raw_result["df"], pl.DataFrame):
                    self.market_data = raw_result["df"]
                else:
                    self.market_data = pl.DataFrame()
            else:
                self.market_data = pl.DataFrame()

            if self.market_data is not None and self.market_data.height > 0:
                logger.info(f"✔ Market data rows={self.market_data.height} cols={self.market_data.width}")
                self.pipeline_state = "DATA_LOADED"
                if self.self_evolving_agent:
                    self.self_evolving_agent.memory.set("last_market_data_shape",
                                                        {"rows": self.market_data.height, "cols": self.market_data.width})
            else:
                logger.error("❌ Market data is empty despite loader success.")
                self.pipeline_state = "DATA_FAILURE"
        except Exception as e:
            logger.error(f"❌ Data load failed: {e}")
            self.market_data = pl.DataFrame()
            self.pipeline_state = "DATA_FAILURE"

    def _step_3_sea_config(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 3] S.E.A. Master proposes trading config")
        if self.self_evolving_agent:
            try:
                proposed = self.self_evolving_agent.propose_trading_config()
                self.trading_config = proposed
                self.self_evolving_agent.memory.set("trading_config", proposed)
            except Exception as e:
                logger.error(f"❌ Config proposal failed: {e}")
                self.trading_config = {
                    "min_adtv_idr": 5_000_000_000.0,
                    "min_confidence": 0.72,
                    "min_rrr": 1.20,
                    "risk_scale": 0.20,
                    "configured_by": "FALLBACK"
                }
        else:
            self.trading_config = {
                "min_adtv_idr": 5_000_000_000.0,
                "min_confidence": 0.72,
                "min_rrr": 1.20,
                "risk_scale": 0.20,
                "configured_by": "FALLBACK"
            }
        logger.info(f"🤖 [CONFIG] {self.trading_config}")

    def _step_4_features(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 4] Feature engineering")
        if self.market_data.height == 0:
            logger.warning("⚠️ No market data, skip features.")
            return
        # Try dynamic features
        mod = self._load_module_with_contract("features", ["UnifiedFeatureEngine", "transform"])
        if mod and hasattr(mod, "UnifiedFeatureEngine"):
            try:
                eng = mod.UnifiedFeatureEngine(sea_agent=self.self_evolving_agent)
                if hasattr(eng, "transform"):
                    self.features_data = eng.transform(self.market_data)
                elif hasattr(eng, "build"):
                    self.features_data = eng.build(self.market_data)
                else:
                    self.features_data = self.market_data
                if self.features_data.height > 0:
                    self.pipeline_state = "FEATURES_READY"
                    logger.info(f"✔ Features rows={self.features_data.height}")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Features failed: {e}")
        # Fallback
        self.features_data = self.market_data
        if self.features_data.height > 0:
            self.pipeline_state = "FEATURES_READY"
        logger.info("ℹ️ Features: using raw market data as fallback")

    def _step_5_ml(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 5] Machine learning")
        src = self.features_data if self.features_data.height else self.market_data
        if src.height == 0:
            logger.warning("⚠️ No data for ML.")
            return
        # Try dynamic ML
        mod = self._load_module_with_contract("machine_learning", ["UnifiedModelEngine"])
        if mod and hasattr(mod, "UnifiedModelEngine"):
            try:
                if self.ml_engine is None:
                    self.ml_engine = mod.UnifiedModelEngine(sea_agent=self.self_evolving_agent)
                if hasattr(self.ml_engine, "predict_and_calibrate"):
                    self.predictions_data = self.ml_engine.predict_and_calibrate(src)
                    if self.predictions_data.height > 0:
                        self.pipeline_state = "ML_READY"
                        logger.info(f"✔ Predictions rows={self.predictions_data.height}")
                        return
            except Exception as e:
                logger.warning(f"⚠️ ML failed: {e}")
        # Fallback: simple rule-based
        self._build_rule_based_from_features()

    def _build_rule_based_from_features(self) -> None:
        src = self.features_data if self.features_data.height else self.market_data
        if src.height == 0:
            self.predictions_data = pl.DataFrame()
            return
        try:
            # Simple placeholder rules
            if "rsi_14" in src.columns and "macd_hist" in src.columns:
                cond = (pl.col("rsi_14") > 50) & (pl.col("macd_hist") > 0)
                filtered = src.filter(cond)
                if filtered.height > 0:
                    self.predictions_data = filtered.with_columns([
                        pl.lit("BUY").alias("candidate_signal"),
                        pl.lit(0.65).alias("prediction_probability"),
                        pl.lit("MODEL_NOT_READY").alias("model_status"),
                        pl.lit("RULE_BASED").alias("signal_source"),
                    ])
                    self.pipeline_state = "ML_READY"
                    logger.info(f"✔ Rule-based predictions rows={self.predictions_data.height}")
                    return
        except Exception as e:
            logger.warning(f"Rule-based fallback failed: {e}")
        self.predictions_data = pl.DataFrame()

    def _step_6_prediction_engine(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 6] Prediction engine")
        src = self.predictions_data if self.predictions_data.height else self.features_data
        if src.height == 0:
            return
        mod = self._load_module_with_contract("prediction", ["UnifiedPredictionEngine", "run_prediction_pipeline"])
        if mod and hasattr(mod, "UnifiedPredictionEngine"):
            try:
                eng = mod.UnifiedPredictionEngine(sea_agent=self.self_evolving_agent)
                if hasattr(eng, "run_prediction_pipeline"):
                    out = eng.run_prediction_pipeline(src)
                    if isinstance(out, pl.DataFrame) and out.height:
                        self.predictions_data = out
                        logger.info(f"✔ Prediction engine rows={out.height}")
                        return
            except Exception as e:
                logger.warning(f"⚠️ Prediction engine failed: {e}")

    def _step_7_sea_analysis(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 7] S.E.A. Master deep-dive")
        src = self.predictions_data if self.predictions_data.height else pl.DataFrame()
        if src.height == 0 or not self.self_evolving_agent:
            return
        try:
            cols = [c for c in ("ticker", "asset", "prediction_probability", "probability", "close") if c in src.columns]
            head = src.select(cols).head(10) if cols else src.head(10)
            report = head.write_csv() if hasattr(head, "write_csv") else str(head)
            analysis = self.self_evolving_agent.analyze_candidates_deep(report)
            if analysis:
                logger.info(f"🧠 [SEA_ANALYSIS] {str(analysis)[:500]}")
        except Exception as e:
            logger.warning(f"⚠️ Analysis soft-fail: {e}")

    def _step_8_signals(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 8] Signal gateway")
        src = self.predictions_data
        if src is None or src.height == 0:
            self.signals_data = pl.DataFrame()
            self.orders = []
            return
        mod = self._load_module_with_contract("signal_idx", ["UnifiedSignalEngine", "execute_pipeline"])
        if mod and hasattr(mod, "UnifiedSignalEngine"):
            try:
                init_kwargs = {
                    "custom_configs": {
                        "generator": {
                            "min_24h_volume_idr": float(self.trading_config.get("min_adtv_idr", 5e9)),
                            "min_risk_reward_ratio": float(self.trading_config.get("min_rrr", 1.2)),
                        },
                        "confidence": {
                            "min_prediction_confidence": float(self.trading_config.get("min_confidence", 0.72)),
                        }
                    }
                }
                try:
                    self.signal_engine = mod.UnifiedSignalEngine(**init_kwargs, sea_agent=self.self_evolving_agent)
                except TypeError:
                    self.signal_engine = mod.UnifiedSignalEngine(**init_kwargs)
                if hasattr(self.signal_engine, "execute_pipeline"):
                    self.signals_data = self.signal_engine.execute_pipeline(src, run_ai_diagnostics=True)
                elif hasattr(self.signal_engine, "generate_signals"):
                    self.signals_data = self.signal_engine.generate_signals(src)
                else:
                    self.signals_data = pl.DataFrame()
                self.orders = self._signals_to_orders(self.signals_data)
                if self.orders:
                    self.pipeline_state = "SIGNALS_READY"
                logger.info(f"✔ Signals rows={getattr(self.signals_data, 'height', 0)} orders={len(self.orders)}")
                return
            except Exception as e:
                logger.error(f"❌ Signal pipeline failed: {e}")
        # Fallback: convert predictions directly
        self.orders = self._predictions_to_orders(src)
        if self.orders:
            self.pipeline_state = "SIGNALS_READY"

    def _step_9_risk(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 9] Risk engine")
        src = self.market_data if self.market_data.height else self.predictions_data
        if src.height == 0:
            return
        mod = self._load_module_with_contract("risk", ["UnifiedRiskEngine", "evaluate_market_risk"])
        if mod and hasattr(mod, "UnifiedRiskEngine"):
            try:
                self.risk_engine = mod.UnifiedRiskEngine(sea_agent=self.self_evolving_agent)
                if hasattr(self.risk_engine, "evaluate_market_risk"):
                    self.risk_output = self.risk_engine.evaluate_market_risk(
                        src,
                        pipeline_timestamp=datetime.now(timezone.utc).isoformat(),
                        execution_id=self.execution_id,
                    )
                    logger.info(f"✔ Risk evaluated: {type(self.risk_output).__name__}")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Risk soft-fail: {e}")

    def _step_10_portfolio_simulation(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 10] Portfolio simulation")
        latest_prices = self._latest_prices_from_market()
        mod = self._load_module_with_contract("portfolio", ["UnifiedPortfolioEngine", "process_trading_signals"])
        if mod and hasattr(mod, "UnifiedPortfolioEngine"):
            try:
                try:
                    self.portfolio_engine = mod.UnifiedPortfolioEngine(
                        config={"dry_run": True, "simulation_only": True},
                        state_file=PORTFOLIO_STATE_FILE,
                        sea_agent=self.self_evolving_agent,
                    )
                except TypeError:
                    self.portfolio_engine = mod.UnifiedPortfolioEngine(
                        config={"dry_run": True, "simulation_only": True},
                        state_file=PORTFOLIO_STATE_FILE,
                    )
                if hasattr(self.portfolio_engine, "load_portfolio_state"):
                    try:
                        loaded = self.portfolio_engine.load_portfolio_state()
                        if isinstance(loaded, dict) and loaded:
                            self.portfolio_state = loaded
                    except Exception as e:
                        logger.warning(f"⚠️ load_portfolio_state: {e}")
                if self.orders and hasattr(self.portfolio_engine, "process_trading_signals"):
                    try:
                        result = self.portfolio_engine.process_trading_signals(
                            self.orders,
                            latest_prices=latest_prices,
                            top_n=len(self.orders)
                        )
                        if isinstance(result, dict):
                            self.portfolio_state = {**self.portfolio_state, **result}
                        logger.info(f"✔ process_trading_signals OK")
                    except Exception as e:
                        logger.warning(f"⚠️ process_trading_signals: {e}")
                if hasattr(self.portfolio_engine, "save_portfolio_state"):
                    try:
                        self.portfolio_engine.save_portfolio_state()
                        logger.info(f"✅ Portfolio state saved")
                    except Exception as e:
                        logger.warning(f"⚠️ save_portfolio_state: {e}")
                self.pipeline_state = "PORTFOLIO_SIMULATED"
                return
            except Exception as e:
                logger.warning(f"⚠️ Portfolio soft-fail: {e}")
        # Fallback
        self.portfolio_state = {
            "equity": 10_000_000.0,
            "cash": 10_000_000.0,
            "positions": {},
            "return_pct": 0.0,
            "exposure_pct": 0.0,
            "mode": "simulation",
        }

    def _step_11_storage(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 11] Storage persist")
        mod = self._load_module_with_contract("storage", ["UnifiedStorageEngine", "persist_signals"])
        if mod and hasattr(mod, "UnifiedStorageEngine"):
            try:
                self.storage_engine = mod.UnifiedStorageEngine(sea_agent=self.self_evolving_agent)
                if self.signals_data is not None and self.signals_data.height:
                    if hasattr(self.storage_engine, "persist_signals"):
                        self.storage_engine.persist_signals(self.signals_data)
                if self.predictions_data is not None and self.predictions_data.height:
                    if hasattr(self.storage_engine, "persist_predictions"):
                        self.storage_engine.persist_predictions(self.predictions_data)
                logger.info("✔ Storage done")
            except Exception as e:
                logger.warning(f"⚠️ Storage soft-fail: {e}")

    def _step_12_sea_telegram(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 12] Telegram broadcast")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            logger.warning("⚠️ Telegram secrets unset.")
            return
        narrative = self.self_evolving_agent.narrate_market(self.portfolio_state, self.orders[:5] if self.orders else None) if self.self_evolving_agent else "No narrative"
        logger.info(f"🗞️ [NARRATIVE] {narrative[:300]}")
        mod = self._load_module_with_contract("reporting", ["UnifiedReportingEngine", "send_telegram_broadcast"])
        if mod and hasattr(mod, "UnifiedReportingEngine"):
            try:
                engine = mod.UnifiedReportingEngine(
                    config={
                        "TELEGRAM_BOT_TOKEN": token,
                        "TELEGRAM_CHAT_ID": chat_id,
                        "REPORTING_MIN_CONFIDENCE": float(self.trading_config.get("min_confidence", 0.72)),
                        "INITIAL_CAPITAL_IDR": float(self.portfolio_state.get("equity", 10_000_000.0)),
                        "MARKET_NARRATIVE": narrative,
                    },
                    mode="dry_run",
                )
                if hasattr(engine, "send_telegram_broadcast"):
                    ok = engine.send_telegram_broadcast(
                        orders=self.orders if self.orders else None,
                        portfolio_data=self.portfolio_state,
                    )
                    logger.info(f"{'✅' if ok else '⚠️'} Telegram result={ok}")
            except Exception as e:
                logger.warning(f"⚠️ Telegram soft-fail: {e}")

    def _step_13_monitoring_eval(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 13] Monitoring / evaluation")
        mod = self._load_module_with_contract("monitoring", ["UnifiedMonitoringEngine", "execute_full_audit"])
        if mod and hasattr(mod, "UnifiedMonitoringEngine"):
            try:
                mon = mod.UnifiedMonitoringEngine(sea_agent=self.self_evolving_agent)
                if hasattr(mon, "execute_full_audit"):
                    try:
                        mon.execute_full_audit()
                    except Exception as e:
                        logger.info(f"ℹ️ Audit soft-skip: {e}")
                logger.info(f"✔ Monitoring engine: {type(mon).__name__}")
            except Exception as e:
                logger.warning(f"⚠️ Monitoring soft-fail: {e}")

    def _step_14_self_learning(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            return
        logger.info("▶ [STEP 14] Self-learning")
        always = os.getenv("IDX_ALWAYS_LEARN", "").strip() in ("1", "true", "True", "yes")
        do_full = bool(self.self_learning_flag or always)
        # Log feedback
        try:
            os.makedirs("storage", exist_ok=True)
            feedback = {
                "ts": datetime.now(WIB_TZ).isoformat(),
                "execution_id": self.execution_id,
                "n_orders": len(self.orders),
                "trading_config": self.trading_config,
                "portfolio_equity": self.portfolio_state.get("equity"),
            }
            with open("storage/learning_feedback.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback, default=str) + "\n")
            logger.info("✔ Feedback snapshot appended")
        except Exception as e:
            logger.warning(f"⚠️ Feedback log failed: {e}")
        if not do_full:
            logger.info("ℹ️ Full self-learning deferred (use --self-learning)")
            return
        mod = self._load_module_with_contract("self_learning", ["UnifiedSelfLearningEngine"])
        if mod and hasattr(mod, "UnifiedSelfLearningEngine"):
            try:
                eng = mod.UnifiedSelfLearningEngine(sea_agent=self.self_evolving_agent)
                for method in ("run", "adapt", "learn"):
                    if hasattr(eng, method):
                        try:
                            getattr(eng, method)()
                            logger.info(f"✔ Self-learning via .{method}()")
                            break
                        except Exception as e:
                            logger.warning(f"⚠️ {method}: {e}")
            except Exception as e:
                logger.warning(f"⚠️ Self-learning soft-fail: {e}")

    def _step_15_autonomous_agent(self) -> None:
        if self.pipeline_state in ("DATA_FAILURE", "PIPELINE_FAILURE"):
            logger.warning("⛔ Skipping evolution due to DATA_FAILURE.")
            return
        if not self.enable_self_evolving or self.self_evolving_agent is None:
            logger.info("▶ [STEP 15] Autonomous agent disabled.")
            return
        logger.info("▶ [STEP 15] S.E.A. Master with FULL AUTHORITY (Transactional)")
        try:
            agent = self.self_evolving_agent
            if not getattr(agent, "_booted", False):
                agent.boot()
            obs = agent.observe()
            goal = (
                f"Execute deep quantitative trading analysis, optimize the IDX scalping engine, "
                f"and self-evolve (execution_id={self.execution_id}). "
                f"You have FULL AUTHORITY to modify sea.py, main.py, and modules. "
                f"Optimize based on Sharpe, profit, speed, memory."
            )
            agent.set_goal(goal)
            plan = agent.plan(goal)
            agent.memory.set("last_trading_execution_id", self.execution_id)
            agent.memory.set("last_trading_observation", obs)
            agent.memory.set("last_trading_plan", plan)
            logger.info(f"🔎 [AUTONOMOUS] Observation files={obs.get('repository', {}).get('file_count')}")
            logger.info(f"🧠 [AUTONOMOUS] Plan tasks={len(plan.get('tasks', []))}")
            max_iter = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
            result = agent.run_loop(max_iterations=max_iter)
            logger.info(f"⚡ [AUTONOMOUS] Evolution completed: iterations={result.get('iterations')}")
            self.pipeline_state = "EVOLUTION_COMPLETE"
        except Exception as e:
            logger.warning(f"⚠️ [STEP 15] Autonomous soft-fail: {e}")

    # =========================================================================
    # Helpers
    # =========================================================================
    def _predictions_to_orders(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        orders = []
        if df is None or df.height == 0:
            return orders
        min_conf = float(self.trading_config.get("min_confidence", 0.72))
        min_rrr = float(self.trading_config.get("min_rrr", 1.20))
        ticker_col = "ticker" if "ticker" in df.columns else ("asset" if "asset" in df.columns else ("symbol" if "symbol" in df.columns else None))
        if ticker_col is None:
            return orders
        for row in df.to_dicts():
            side = str(row.get("side") or row.get("candidate_signal") or "").upper()
            if side not in ("BUY", "LONG"):
                continue
            conf = float(row.get("prediction_probability") or row.get("probability") or 0.0)
            if conf < min_conf:
                continue
            entry = float(row.get("entry_price") or row.get("current_price") or row.get("close") or 0.0)
            tp = float(row.get("tp_price") or row.get("take_profit") or 0.0)
            sl = float(row.get("sl_price") or row.get("stop_loss") or 0.0)
            if entry <= 0 or tp <= entry or sl <= 0 or sl >= entry:
                continue
            rrr = (tp - entry) / (entry - sl) if (entry - sl) > 0 else 0
            if rrr < min_rrr:
                continue
            t = str(row.get(ticker_col) or "").strip()
            if not t.endswith(".JK"):
                t = f"{t}.JK"
            orders.append({
                "ticker": t,
                "symbol": t,
                "direction": "BUY",
                "entry_price": entry,
                "tp_price": tp,
                "sl_price": sl,
                "probability": conf,
                "risk_reward_ratio": rrr,
                "horizon": normalize_horizon(row.get("horizon")),
                "timestamp": str(row.get("timestamp") or datetime.now(WIB_TZ).isoformat()),
                "reason": str(row.get("reason") or row.get("signal_reason") or "Rule-based"),
            })
        return orders

    def _signals_to_orders(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        if df is None or df.height == 0:
            return []
        if "is_valid_execution" in df.columns:
            df = df.filter(pl.col("is_valid_execution").fill_null(False) == True)
        elif "signal_valid" in df.columns:
            df = df.filter(pl.col("signal_valid").fill_null(False) == True)
        return self._predictions_to_orders(df)

    def _latest_prices_from_market(self) -> Dict[str, float]:
        prices = {}
        src = self.market_data if self.market_data.height else self.features_data
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
                    prices[t.replace(".JK", "")] = px
        except Exception as e:
            logger.warning(f"⚠️ latest_prices: {e}")
        return prices

    # =========================================================================
    # Main run
    # =========================================================================
    def run(self) -> None:
        logger.info("=" * 60)
        logger.info("🚀 IDX PIPELINE — S.E.A. Master (Transactional)")
        logger.info(f"   execution_id={self.execution_id}")
        logger.info("=" * 60)
        try:
            self._step_0_sea_awaken()
            if self.pipeline_state == "PIPELINE_FAILURE":
                logger.critical("⛔ Pipeline aborted due to agent failure.")
                return
            self._step_1_universe()
            self._step_2_data()
            if self.pipeline_state == "DATA_FAILURE":
                logger.critical("⛔ Pipeline halted due to DATA_FAILURE.")
                return
            self._step_3_sea_config()
            self._step_4_features()
            self._step_5_ml()
            self._step_6_prediction_engine()
            self._step_7_sea_analysis()
            self._step_8_signals()
            self._step_9_risk()
            self._step_10_portfolio_simulation()
            self._step_11_storage()
            self._step_12_sea_telegram()
            self._step_13_monitoring_eval()
            self._step_14_self_learning()
            self._step_15_autonomous_agent()
            logger.info("=" * 60)
            logger.info(f"🎉 PIPELINE COMPLETE | orders={len(self.orders)} | state={self.pipeline_state}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"💥 Pipeline error: {e}\n{traceback.format_exc()}")
            self.pipeline_state = "PIPELINE_FAILURE"
            sys.exit(1)

# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Orchestrator (S.E.A. v8.0)")
    parser.add_argument("--dry-run", action="store_true", help="Simulation only")
    parser.add_argument("--self-learning", action="store_true")
    parser.add_argument("--bootstrap-universe", action="store_true")
    parser.add_argument("--reset-dryrun", action="store_true")
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument("--enable-self-evolving", action="store_true")
    args = parser.parse_args()

    if args.bootstrap_universe:
        if HAS_DATA and sanitize_ticker_list is not None:
            universe = sanitize_ticker_list(DEFAULT_BLUECHIP_UNIVERSE)
        else:
            universe = DEFAULT_BLUECHIP_UNIVERSE
        with open(DEFAULT_UNIVERSE_FILE, "w", encoding="utf-8") as f:
            json.dump(universe, f, indent=2)
        print(f"✅ universe.json created with {len(universe)} tickers")
        sys.exit(0)

    if args.reset_dryrun:
        for rel in ("data/storage_dryrun.db", "storage_dryrun.db", "autonomous_engine.db"):
            if os.path.exists(rel):
                try:
                    os.unlink(rel)
                    print(f"🗑️ Removed {rel}")
                except Exception as e:
                    print(f"⚠️ Could not remove {rel}: {e}")
        sys.exit(0)

    orch = ProductionOrchestrator(
        dry_run=True,
        self_learning=args.self_learning,
        enable_self_evolving=args.enable_self_evolving,
    )

    if args.health_check:
        orch._log_module_matrix()
        if orch.self_evolving_agent:
            orch.self_evolving_agent.boot()
            print("✅ Health check passed.")
        else:
            print("⚠️ Agent not available.")
    else:
        orch.run()

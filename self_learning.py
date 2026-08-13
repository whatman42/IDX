"""
=============================================================================
IDX Stock Quantitative Trading System - Self-Learning & Adaptive Engine
FileName      : self_learning.py
Directory     : Flat Directory (Root Level with main.py, research.py, risk.py)
Version       : 2026.Q3.v3.2.2 (Production Ready - Gemini AI Integrated)
Compliance    : IDX Trading Rules (IDR Pairs, 0.3% Fee, WIB Timezone)
=============================================================================
"""

import datetime
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

# Standar Zona Waktu Pasar Saham Indonesia (WIB)
WIB_TZ: ZoneInfo = ZoneInfo("Asia/Jakarta")

# Model Baseline Gemini SDK
PRIMARY_MODEL: str = "gemini-2.5-flash"
FALLBACK_MODEL: str = "gemini-2.0-flash"

# Import Google GenAI Client
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# =============================================================================
# INTEGRASI LOGGING INTERNAL
# =============================================================================
try:
    from logger import get_logger
    logger = get_logger("IDX.SelfLearning")
except ImportError:
    logger = logging.getLogger("IDX.SelfLearning")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# =============================================================================
# KONSTANTA BURSA EFEK INDONESIA (IDX)
# =============================================================================
IDX_FEE_ROUNDTRIP_PCT: float = 0.003          # Fee Transaksi Roundtrip IDX (0.3%)
IDX_MIN_PRICE_IDR: float = 50.0               # Batas Minimal Harga Saham IDX (Rp 50)
IDX_MAX_STALENESS_SEC: float = 172800.0       # Toleransi Keusangan Data Candle (48 Jam)

# =============================================================================
# HELPER TIMESTAMP
# =============================================================================
def _get_wib_timestamp_str() -> str:
    """Mengembalikan timestamp ISO dengan Zona Waktu Indonesia Barat (WIB)."""
    return datetime.datetime.now(WIB_TZ).isoformat()

# =============================================================================
# EXCEPTIONS
# =============================================================================
try:
    from exceptions import SelfLearningError, DataValidationError, ModelDriftError, StorageError
except ImportError:
    class SelfLearningError(Exception): pass
    class DataValidationError(SelfLearningError): pass
    class ModelDriftError(SelfLearningError): pass
    class StorageError(SelfLearningError): pass

# =============================================================================
# ENUMS
# =============================================================================
class DriftSeverity(Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AdaptationStrategy(Enum):
    WEIGHT_REBALANCING = "WEIGHT_REBALANCING"
    HYPERPARAMETER_MUTATION = "HYPERPARAMETER_MUTATION"
    REGIME_SWITCHING = "REGIME_SWITCHING"
    MODEL_RETRAIN_TRIGGER = "MODEL_RETRAIN_TRIGGER"
    NO_ACTION = "NO_ACTION"

# =============================================================================
# DATA TRANSFER OBJECTS (DTOs)
# =============================================================================
@dataclass(frozen=True)
class PerformanceFeedback:
    strategy_id: str
    realized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    trade_count: int
    execution_timestamp: str

@dataclass(frozen=True)
class DriftReport:
    is_drift_detected: bool
    drift_score: float
    severity: DriftSeverity
    recommended_action: AdaptationStrategy
    telemetry: Dict[str, Any]

@dataclass(frozen=True)
class AdaptationDecision:
    decision_id: str
    strategy_weights: Dict[str, float]
    adaptation_strategy: AdaptationStrategy
    ai_explanation: str
    timestamp: str
    is_applied: bool

# =============================================================================
# 1. ONLINE PERFORMANCE MONITOR (CONCEPT DRIFT / ALPHA DECAY DETECTOR)
# =============================================================================
class OnlinePerformanceMonitor:
    """Memonitor performa real-time dan mendeteksi alpha decay/concept drift."""

    def __init__(
        self,
        baseline_sharpe: float = 1.5,
        baseline_win_rate: float = 55.0,
        drift_window: int = 30,
        z_threshold: float = 2.0
    ) -> None:
        self.baseline_sharpe = baseline_sharpe
        self.baseline_win_rate = baseline_win_rate
        self.drift_window = drift_window
        self.z_threshold = z_threshold
        self._history: List[PerformanceFeedback] = []
        self._version = "3.2.2"

    def record_feedback(self, feedback: PerformanceFeedback) -> None:
        self._history.append(feedback)
        if len(self._history) > 500:
            self._history = self._history[-500:]

    def detect_drift(self, pipeline_timestamp: str) -> DriftReport:
        if len(self._history) < self.drift_window:
            return DriftReport(
                is_drift_detected=False,
                drift_score=0.0,
                severity=DriftSeverity.NONE,
                recommended_action=AdaptationStrategy.NO_ACTION,
                telemetry={"status": "INSUFFICIENT_DATA", "samples": len(self._history)}
            )

        recent = self._history[-self.drift_window:]
        recent_sharpes = [f.sharpe_ratio for f in recent]
        recent_win_rates = [f.win_rate_pct for f in recent]

        mean_sharpe = float(np.mean(recent_sharpes))
        std_sharpe = float(np.std(recent_sharpes)) + 1e-8
        mean_win_rate = float(np.mean(recent_win_rates))

        # Z-Score terhadap baseline
        z_sharpe = (self.baseline_sharpe - mean_sharpe) / std_sharpe
        win_rate_drop = max(0.0, self.baseline_win_rate - mean_win_rate)

        drift_score = float(np.clip((0.6 * max(0.0, z_sharpe) / self.z_threshold) + (0.4 * win_rate_drop / 20.0), 0.0, 1.0))

        is_drift = drift_score >= 0.5
        severity = DriftSeverity.NONE
        action = AdaptationStrategy.NO_ACTION

        if drift_score >= 0.85:
            severity = DriftSeverity.CRITICAL
            action = AdaptationStrategy.MODEL_RETRAIN_TRIGGER
        elif drift_score >= 0.65:
            severity = DriftSeverity.HIGH
            action = AdaptationStrategy.REGIME_SWITCHING
        elif drift_score >= 0.50:
            severity = DriftSeverity.MEDIUM
            action = AdaptationStrategy.WEIGHT_REBALANCING
        elif drift_score >= 0.30:
            severity = DriftSeverity.LOW
            action = AdaptationStrategy.NO_ACTION

        telemetry = {
            "schema_version": self._version,
            "mean_recent_sharpe": mean_sharpe,
            "mean_recent_win_rate": mean_win_rate,
            "z_sharpe": z_sharpe,
            "timestamp": pipeline_timestamp
        }

        return DriftReport(
            is_drift_detected=is_drift,
            drift_score=drift_score,
            severity=severity,
            recommended_action=action,
            telemetry=telemetry
        )

# =============================================================================
# 2. REINFORCEMENT STRATEGY ADAPTER (UCB1 / THOMPSON SAMPLING)
# =============================================================================
class ReinforcementStrategyAdapter:
    """Multi-Armed Bandit (UCB1 & Thompson Sampling) untuk rebalancing bobot strategi secara dinamis."""

    def __init__(self, strategy_ids: List[str], exploration_factor: float = 1.0) -> None:
        self.strategy_ids = strategy_ids
        self.exploration_factor = exploration_factor
        self.counts = {s_id: 1 for s_id in strategy_ids}
        self.rewards = {s_id: 1.0 for s_id in strategy_ids}
        self._lock = threading.RLock()

    def update_reward(self, strategy_id: str, reward: float) -> None:
        with self._lock:
            if strategy_id in self.rewards:
                self.counts[strategy_id] += 1
                # Exponential moving average update for dynamic rewards
                self.rewards[strategy_id] = 0.85 * self.rewards[strategy_id] + 0.15 * reward

    def get_rebalanced_weights(self) -> Dict[str, float]:
        with self._lock:
            total_counts = sum(self.counts.values())
            ucb_scores = {}

            for s_id in self.strategy_ids:
                mean_reward = self.rewards[s_id]
                bonus = self.exploration_factor * math.sqrt((2.0 * math.log(total_counts)) / self.counts[s_id])
                ucb_scores[s_id] = max(0.01, mean_reward + bonus)

            total_score = sum(ucb_scores.values())
            normalized_weights = {s_id: float(score / total_score) for s_id, score in ucb_scores.items()}
            return normalized_weights

# =============================================================================
# 3. SQLITE PERSISTENT STORE
# =============================================================================
class SQLiteSelfLearningStore:
    """Penyimpanan persistent SQLite WAL mode untuk feedback performa dan audit trail."""

    def __init__(self, db_path: str = "./checkpoints/self_learning.sqlite") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS performance_feedback (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            strategy_id TEXT NOT NULL,
                            realized_return_pct REAL NOT NULL,
                            sharpe_ratio REAL NOT NULL,
                            max_drawdown_pct REAL NOT NULL,
                            win_rate_pct REAL NOT NULL,
                            trade_count INTEGER NOT NULL,
                            timestamp TEXT NOT NULL
                        )
                    """)
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS adaptation_audit (
                            decision_id TEXT PRIMARY KEY,
                            adaptation_strategy TEXT NOT NULL,
                            weights_json TEXT NOT NULL,
                            ai_explanation TEXT NOT NULL,
                            applied BOOLEAN NOT NULL,
                            timestamp TEXT NOT NULL
                        )
                    """)
            except sqlite3.Error as e:
                raise StorageError(f"Gagal menginisialisasi Self-Learning SQLite DB: {e}")

    def save_feedback(self, feedback: PerformanceFeedback) -> None:
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT INTO performance_feedback 
                        (strategy_id, realized_return_pct, sharpe_ratio, max_drawdown_pct, win_rate_pct, trade_count, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        feedback.strategy_id, feedback.realized_return_pct, feedback.sharpe_ratio,
                        feedback.max_drawdown_pct, feedback.win_rate_pct, feedback.trade_count, feedback.execution_timestamp
                    ))
            except sqlite3.Error as e:
                logger.error(f"Gagal menyimpan feedback: {e}")

    def save_adaptation_decision(self, decision: AdaptationDecision) -> None:
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT INTO adaptation_audit 
                        (decision_id, adaptation_strategy, weights_json, ai_explanation, applied, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        decision.decision_id, decision.adaptation_strategy.value,
                        json.dumps(decision.strategy_weights), decision.ai_explanation,
                        decision.is_applied, decision.timestamp
                    ))
            except sqlite3.Error as e:
                logger.error(f"Gagal menyimpan keputusan adaptasi: {e}")

# =============================================================================
# 4. GEMINI AI SELF-LEARNING REASONING ENGINE
# =============================================================================
class GeminiSelfLearningReasoningEngine:
    """
    Sub-Engine Google Gemini AI untuk analisis post-mortem & rekomendasi penyesuaian strategi.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None

        if HAS_GEMINI_SDK and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("🤖 [GEMINI_SELF_LEARNING] Gemini Client terhubung untuk Self-Learning Engine.")
            except Exception as e:
                logger.warning(f"⚠️ Gagal inisialisasi Gemini Client pada Self-Learning Engine: {e}")

    def explain_and_recommend(self, drift_report: DriftReport, current_weights: Dict[str, float]) -> str:
        if not self.client:
            return "Adaptasi berbasis aturan otomatis (Gemini SDK tidak aktif)."

        prompt = f"""
        Anda adalah Chief AI Quant Analyst untuk perdagangan saham IDX (BEI).
        Analisis laporan degradasi performa (Model Drift) dan berikan rekomendasi penyesuaian bobot/hyperparameter strategi:

        === REPORT DRIFT PERFORMA ===
        - Drift Detected: {drift_report.is_drift_detected}
        - Drift Score: {drift_report.drift_score:.4f}
        - Severity: {drift_report.severity.value}
        - Rekomendasi Aksi: {drift_report.recommended_action.value}
        - Telemetri Performa: {json.dumps(drift_report.telemetry)}

        === BOBOT STRATEGI SAAT INI ===
        {json.dumps(current_weights, indent=2)}

        Tugas:
        Berikan penjelasan teknis ringkas (maksimal 2 kalimat) mengapa performa meluruh dan rekomendasi penyesuaian bobot portofolio.
        """

        for model_target in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                response = self.client.models.generate_content(
                    model=model_target,
                    contents=prompt,
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"⚠️ Error Gemini API ({model_target}) pada Self-Learning Reasoning: {e}")

        return "Terdeteksi peluruhan performa. Melakukan rebalancing bobot portofolio berbasis UCB1 Bandit."

# =============================================================================
# 5. UNIFIED SELF-LEARNING ENGINE (FACADE CLASS)
# =============================================================================
class UnifiedSelfLearningEngine:
    """
    Facade Class utama yang mengintegrasikan Drift Detection, Reinforcement Learning Adapter,
    SQLite Storage, dan Gemini AI Diagnostics (Main.py ready).
    """

    def __init__(
        self,
        strategy_ids: Optional[List[str]] = None,
        db_path: str = "./checkpoints/self_learning.sqlite",
        gemini_api_key: Optional[str] = None
    ) -> None:
        self.strategy_ids = strategy_ids or ["MOMENTUM_ALPHA", "MEAN_REVERSION", "BREAKOUT_VOLATILITY"]
        self.monitor = OnlinePerformanceMonitor()
        self.adapter = ReinforcementStrategyAdapter(strategy_ids=self.strategy_ids)
        self.store = SQLiteSelfLearningStore(db_path=db_path)
        self.gemini_engine = GeminiSelfLearningReasoningEngine(api_key=gemini_api_key)
        self._version = "3.2.2"

    def process_trade_feedback(
        self,
        strategy_id: str,
        realized_return_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        win_rate_pct: float,
        trade_count: int = 1
    ) -> AdaptationDecision:
        """Siklus lengkap pemrosesan umpan balik perdagangan dan penyesuaian strategi."""
        timestamp = _get_wib_timestamp_str()

        feedback = PerformanceFeedback(
            strategy_id=strategy_id,
            realized_return_pct=realized_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate_pct=win_rate_pct,
            trade_count=trade_count,
            execution_timestamp=timestamp
        )

        # 1. Simpan Feedback ke DB
        self.store.save_feedback(feedback)
        self.monitor.record_feedback(feedback)

        # 2. Update Reward Multi-Armed Bandit
        # Reward dihitung dari kombinasi Sharpe & Win Rate
        reward = max(0.01, (sharpe_ratio * 0.6) + (win_rate_pct / 100.0 * 0.4))
        self.adapter.update_reward(strategy_id, reward)

        # 3. Deteksi Drift
        drift_report = self.monitor.detect_drift(timestamp)

        # 4. Evaluasi Rebalancing
        current_weights = self.adapter.get_rebalanced_weights()
        decision_id = f"ADAPT-{uuid.uuid4().hex[:8].upper()}"

        ai_explanation = self.gemini_engine.explain_and_recommend(drift_report, current_weights)

        decision = AdaptationDecision(
            decision_id=decision_id,
            strategy_weights=current_weights,
            adaptation_strategy=drift_report.recommended_action,
            ai_explanation=ai_explanation,
            timestamp=timestamp,
            is_applied=drift_report.is_drift_detected or drift_report.recommended_action != AdaptationStrategy.NO_ACTION
        )

        # 5. Audit Keputusan
        self.store.save_adaptation_decision(decision)

        return decision

    def get_current_strategy_weights(self) -> Dict[str, float]:
        """Mengembalikan bobot strategi adaptif terbaru saat ini."""
        return self.adapter.get_rebalanced_weights()

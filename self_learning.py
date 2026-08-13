"""
=============================================================================
IDX Stock Quantitative Trading System - Autonomous Self-Learning Engine
FileName      : self_learning.py
Directory     : Flat Directory (Root Level with main.py, research.py, risk.py)
Version       : 2026.Q3.v3.3.0 (Production Ready - Autonomous Closed-Loop)
Compliance    : IDX Trading Rules (IDR Pairs, 0.3% Fee, WIB Timezone)
=============================================================================
"""

import datetime
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
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
# ENUMS & HARD LIMITS
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

class AdaptationStatus(Enum):
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    RISK_CHECKED = "RISK_CHECKED"
    APPLIED = "APPLIED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"

# Hard limits pencegah runaway learning
MAX_WEIGHT_CHANGE_PER_CYCLE: float = 0.10
MIN_STRATEGY_WEIGHT: float = 0.05
MAX_STRATEGY_WEIGHT: float = 0.60
MIN_CONFIDENCE_THRESHOLD: float = 0.70
MIN_SAMPLE_COUNT_RELIABILITY: int = 15

# =============================================================================
# PROTOCOLS / INTERFACES
# =============================================================================
class StrategyManagerProtocol(Protocol):
    """Interface untuk menghubungkan Self-Learning Engine dengan Trading Engine utama."""
    def apply_strategy_weights(self, weights: Dict[str, float]) -> bool: ...
    def get_active_weights(self) -> Dict[str, float]: ...

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
    transaction_cost_pct: float
    execution_timestamp: str

@dataclass(frozen=True)
class StrategyBaseline:
    strategy_id: str
    mean_sharpe: float
    std_sharpe: float
    mean_win_rate: float
    std_win_rate: float
    sample_size: int

@dataclass(frozen=True)
class StrategyDriftResult:
    strategy_id: str
    z_sharpe: float
    z_win_rate: float
    drift_score: float
    severity: DriftSeverity
    recommended_action: AdaptationStrategy

@dataclass(frozen=True)
class MultiStrategyDriftReport:
    is_drift_detected: bool
    overall_severity: DriftSeverity
    strategy_drifts: Dict[str, StrategyDriftResult]
    recommended_action: AdaptationStrategy
    timestamp: str

@dataclass(frozen=True)
class StructuredAIProposal:
    action: str
    confidence: float
    proposed_weights: Dict[str, float]
    reason_codes: List[str]
    risk_level: str
    raw_response: str

@dataclass(frozen=True)
class AdaptationDecision:
    decision_id: str
    status: AdaptationStatus
    adaptation_strategy: AdaptationStrategy
    current_weights: Dict[str, float]
    proposed_weights: Dict[str, float]
    validated_weights: Dict[str, float]
    ai_proposal: Optional[StructuredAIProposal]
    rejection_reason: Optional[str]
    timestamp: str

# =============================================================================
# 1. SQLITE PERSISTENT STORE (BANDIT STATE & AUDIT TRAIL)
# =============================================================================
class SQLiteSelfLearningStore:
    """Penyimpanan persistent SQLite WAL mode untuk state UCB1, baseline, dan audit lifecycle."""

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
                            transaction_cost_pct REAL NOT NULL,
                            timestamp TEXT NOT NULL
                        )
                    """)
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS strategy_baselines (
                            strategy_id TEXT PRIMARY KEY,
                            mean_sharpe REAL NOT NULL,
                            std_sharpe REAL NOT NULL,
                            mean_win_rate REAL NOT NULL,
                            std_win_rate REAL NOT NULL,
                            sample_size INTEGER NOT NULL,
                            updated_at TEXT NOT NULL
                        )
                    """)
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS bandit_state (
                            strategy_id TEXT PRIMARY KEY,
                            pull_count INTEGER NOT NULL,
                            reward_ema REAL NOT NULL,
                            updated_at TEXT NOT NULL
                        )
                    """)
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS adaptation_lifecycle_audit (
                            decision_id TEXT PRIMARY KEY,
                            status TEXT NOT NULL,
                            adaptation_strategy TEXT NOT NULL,
                            current_weights_json TEXT NOT NULL,
                            proposed_weights_json TEXT NOT NULL,
                            validated_weights_json TEXT NOT NULL,
                            ai_confidence REAL,
                            rejection_reason TEXT,
                            ai_proposal_json TEXT,
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
                        (strategy_id, realized_return_pct, sharpe_ratio, max_drawdown_pct, win_rate_pct, trade_count, transaction_cost_pct, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        feedback.strategy_id, feedback.realized_return_pct, feedback.sharpe_ratio,
                        feedback.max_drawdown_pct, feedback.win_rate_pct, feedback.trade_count,
                        feedback.transaction_cost_pct, feedback.execution_timestamp
                    ))
            except sqlite3.Error as e:
                logger.error(f"Gagal menyimpan feedback: {e}")

    def load_bandit_state(self, strategy_ids: List[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
        counts = {s_id: 1 for s_id in strategy_ids}
        rewards = {s_id: 1.0 for s_id in strategy_ids}
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT strategy_id, pull_count, reward_ema FROM bandit_state")
                    for row in cursor.fetchall():
                        s_id, p_count, r_ema = row[0], row[1], row[2]
                        if s_id in strategy_ids:
                            counts[s_id] = max(1, int(p_count))
                            rewards[s_id] = float(r_ema)
            except sqlite3.Error as e:
                logger.error(f"Gagal memuat bandit state: {e}")
        return counts, rewards

    def save_bandit_state(self, counts: Dict[str, int], rewards: Dict[str, float]) -> None:
        timestamp = _get_wib_timestamp_str()
        with self._lock:
            try:
                with self._get_connection() as conn:
                    for s_id in counts.keys():
                        conn.execute("""
                            INSERT INTO bandit_state (strategy_id, pull_count, reward_ema, updated_at)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(strategy_id) DO UPDATE SET
                                pull_count = excluded.pull_count,
                                reward_ema = excluded.reward_ema,
                                updated_at = excluded.updated_at
                        """, (s_id, counts[s_id], rewards[s_id], timestamp))
            except sqlite3.Error as e:
                logger.error(f"Gagal menyimpan bandit state: {e}")

    def get_baseline(self, strategy_id: str) -> Optional[StrategyBaseline]:
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT strategy_id, mean_sharpe, std_sharpe, mean_win_rate, std_win_rate, sample_size FROM strategy_baselines WHERE strategy_id = ?",
                        (strategy_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        return StrategyBaseline(row[0], row[1], row[2], row[3], row[4], row[5])
            except sqlite3.Error as e:
                logger.error(f"Gagal memuat baseline {strategy_id}: {e}")
        return None

    def save_baseline(self, baseline: StrategyBaseline) -> None:
        timestamp = _get_wib_timestamp_str()
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT INTO strategy_baselines (strategy_id, mean_sharpe, std_sharpe, mean_win_rate, std_win_rate, sample_size, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(strategy_id) DO UPDATE SET
                            mean_sharpe = excluded.mean_sharpe,
                            std_sharpe = excluded.std_sharpe,
                            mean_win_rate = excluded.mean_win_rate,
                            std_win_rate = excluded.std_win_rate,
                            sample_size = excluded.sample_size,
                            updated_at = excluded.updated_at
                    """, (
                        baseline.strategy_id, baseline.mean_sharpe, baseline.std_sharpe,
                        baseline.mean_win_rate, baseline.std_win_rate, baseline.sample_size, timestamp
                    ))
            except sqlite3.Error as e:
                logger.error(f"Gagal menyimpan baseline {baseline.strategy_id}: {e}")

    def save_adaptation_audit(self, decision: AdaptationDecision) -> None:
        with self._lock:
            try:
                with self._get_connection() as conn:
                    ai_prop_json = json.dumps(decision.ai_proposal.__dict__) if decision.ai_proposal else None
                    ai_conf = decision.ai_proposal.confidence if decision.ai_proposal else None

                    conn.execute("""
                        INSERT INTO adaptation_lifecycle_audit 
                        (decision_id, status, adaptation_strategy, current_weights_json, proposed_weights_json, validated_weights_json, ai_confidence, rejection_reason, ai_proposal_json, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(decision_id) DO UPDATE SET
                            status = excluded.status,
                            validated_weights_json = excluded.validated_weights_json,
                            rejection_reason = excluded.rejection_reason,
                            timestamp = excluded.timestamp
                    """, (
                        decision.decision_id, decision.status.value, decision.adaptation_strategy.value,
                        json.dumps(decision.current_weights), json.dumps(decision.proposed_weights),
                        json.dumps(decision.validated_weights), ai_conf, decision.rejection_reason,
                        ai_prop_json, decision.timestamp
                    ))
            except sqlite3.Error as e:
                logger.error(f"Gagal mencatat audit adaptasi {decision.decision_id}: {e}")

# =============================================================================
# 2. PER-STRATEGY STATISTICAL DRIFT DETECTOR
# =============================================================================
class PerStrategyDriftDetector:
    """Mendeteksi alpha decay / concept drift secara terpisah per strategi berdasarkan Z-Score Distribusi Baseline."""

    def __init__(self, store: SQLiteSelfLearningStore, drift_window: int = 20) -> None:
        self.store = store
        self.drift_window = drift_window
        self._buffers: Dict[str, List[PerformanceFeedback]] = {}
        self._lock = threading.RLock()

    def record_feedback(self, feedback: PerformanceFeedback) -> None:
        with self._lock:
            if feedback.strategy_id not in self._buffers:
                self._buffers[feedback.strategy_id] = []
            self._buffers[feedback.strategy_id].append(feedback)
            if len(self._buffers[feedback.strategy_id]) > 200:
                self._buffers[feedback.strategy_id] = self._buffers[feedback.strategy_id][-200:]

    def evaluate_drift(self, pipeline_timestamp: str) -> MultiStrategyDriftReport:
        with self._lock:
            strategy_drifts: Dict[str, StrategyDriftResult] = {}
            max_drift_score = 0.0
            overall_severity = DriftSeverity.NONE
            highest_action = AdaptationStrategy.NO_ACTION

            for s_id, history in self._buffers.items():
                if len(history) < 5:
                    continue

                recent = history[-self.drift_window:]
                recent_n = sum(f.trade_count for f in recent)

                # Ambil atau buat baseline historis
                baseline = self.store.get_baseline(s_id)
                if not baseline:
                    # Default baseline OOS konservatif
                    baseline = StrategyBaseline(
                        strategy_id=s_id, mean_sharpe=1.5, std_sharpe=0.5,
                        mean_win_rate=55.0, std_win_rate=8.0, sample_size=100
                    )
                    self.store.save_baseline(baseline)

                recent_sharpes = [f.sharpe_ratio for f in recent if np.isfinite(f.sharpe_ratio)]
                recent_win_rates = [f.win_rate_pct for f in recent if np.isfinite(f.win_rate_pct)]

                if not recent_sharpes or not recent_win_rates:
                    continue

                mean_recent_sharpe = float(np.mean(recent_sharpes))
                mean_recent_win_rate = float(np.mean(recent_win_rates))

                # Formula Z-Score yang benar: Z = (mean_recent - mean_base) / (std_base / sqrt(N_recent))
                se_sharpe = baseline.std_sharpe / math.sqrt(max(1, len(recent_sharpes)))
                z_sharpe = (mean_recent_sharpe - baseline.mean_sharpe) / max(se_sharpe, 1e-6)

                se_win_rate = baseline.std_win_rate / math.sqrt(max(1, len(recent_win_rates)))
                z_win_rate = (mean_recent_win_rate - baseline.mean_win_rate) / max(se_win_rate, 1e-6)

                # Hitung Drift Score (hanya jika terjadi penurunan / Z negatif)
                sharpe_drop_score = max(0.0, -z_sharpe) / 3.0
                win_rate_drop_score = max(0.0, -z_win_rate) / 3.0

                drift_score = float(np.clip((0.6 * sharpe_drop_score) + (0.4 * win_rate_drop_score), 0.0, 1.0))

                severity = DriftSeverity.NONE
                action = AdaptationStrategy.NO_ACTION

                if drift_score >= 0.80:
                    severity = DriftSeverity.CRITICAL
                    action = AdaptationStrategy.MODEL_RETRAIN_TRIGGER
                elif drift_score >= 0.60:
                    severity = DriftSeverity.HIGH
                    action = AdaptationStrategy.REGIME_SWITCHING
                elif drift_score >= 0.40:
                    severity = DriftSeverity.MEDIUM
                    action = AdaptationStrategy.WEIGHT_REBALANCING
                elif drift_score >= 0.20:
                    severity = DriftSeverity.LOW
                    action = AdaptationStrategy.NO_ACTION

                strategy_drifts[s_id] = StrategyDriftResult(
                    strategy_id=s_id, z_sharpe=z_sharpe, z_win_rate=z_win_rate,
                    drift_score=drift_score, severity=severity, recommended_action=action
                )

                if drift_score > max_drift_score:
                    max_drift_score = drift_score
                    overall_severity = severity
                    highest_action = action

            is_drift_detected = max_drift_score >= 0.40

            return MultiStrategyDriftReport(
                is_drift_detected=is_drift_detected,
                overall_severity=overall_severity,
                strategy_drifts=strategy_drifts,
                recommended_action=highest_action,
                timestamp=pipeline_timestamp
            )

# =============================================================================
# 3. PERSISTENT & RISK-ADJUSTED BANDIT ADAPTER
# =============================================================================
class PersistentBanditAdapter:
    """Multi-Armed Bandit (UCB1) dengan State SQLite Persistent & Reward Komprehensif."""

    def __init__(self, strategy_ids: List[str], store: SQLiteSelfLearningStore, exploration_factor: float = 0.8) -> None:
        self.strategy_ids = strategy_ids
        self.store = store
        self.exploration_factor = exploration_factor
        self._lock = threading.RLock()

        # Load persistent state dari SQLite
        self.counts, self.rewards = self.store.load_bandit_state(self.strategy_ids)

    def calculate_holistic_reward(self, feedback: PerformanceFeedback) -> float:
        """
        Komputasi Reward Komprehensif:
        Reward = (Sharpe * 0.35) + (WinRate * 0.25) - (Drawdown_Penalty * 0.25) - (Fee_Penalty * 0.15)
        Dengan pembobotan keandalan sampel (Sample Reliability Scaling).
        """
        # Sanitasi Input
        s_ratio = float(np.clip(feedback.sharpe_ratio, -3.0, 5.0))
        win_rate = float(np.clip(feedback.win_rate_pct / 100.0, 0.0, 1.0))
        max_dd = float(np.clip(feedback.max_drawdown_pct / 100.0, 0.0, 1.0))
        tx_cost = float(np.clip(feedback.transaction_cost_pct / 100.0, 0.0, 0.10))

        sharpe_part = max(0.0, s_ratio / 3.0)
        win_rate_part = win_rate
        dd_penalty = math.exp(max_dd * 4.0) - 1.0
        fee_penalty = tx_cost * 10.0

        raw_reward = (0.35 * sharpe_part) + (0.25 * win_rate_part) - (0.25 * dd_penalty) - (0.15 * fee_penalty)

        # Sample reliability shrinkage based on trade count
        reliability_factor = 1.0 - math.exp(-feedback.trade_count / float(MIN_SAMPLE_COUNT_RELIABILITY))
        adjusted_reward = float(np.clip(raw_reward * reliability_factor, 0.01, 1.0))

        return adjusted_reward

    def update_reward(self, feedback: PerformanceFeedback) -> None:
        with self._lock:
            s_id = feedback.strategy_id
            if s_id in self.rewards:
                reward = self.calculate_holistic_reward(feedback)
                self.counts[s_id] += max(1, feedback.trade_count)

                # Exponential Moving Average Update
                alpha = 0.15
                self.rewards[s_id] = float(np.clip((1.0 - alpha) * self.rewards[s_id] + alpha * reward, 0.01, 1.0))

                # Persist state ke SQLite
                self.store.save_bandit_state(self.counts, self.rewards)

    def get_rebalanced_weights(self) -> Dict[str, float]:
        with self._lock:
            total_counts = sum(self.counts.values())
            ucb_scores = {}

            for s_id in self.strategy_ids:
                mean_reward = self.rewards[s_id]
                bonus = self.exploration_factor * math.sqrt((2.0 * math.log(max(1, total_counts))) / max(1, self.counts[s_id]))
                ucb_scores[s_id] = max(0.01, mean_reward + bonus)

            total_score = sum(ucb_scores.values())
            normalized_weights = {s_id: float(score / total_score) for s_id, score in ucb_scores.items()}
            return normalized_weights

# =============================================================================
# 4. STRUCTURED GEMINI AI REASONING ENGINE
# =============================================================================
class GeminiSelfLearningReasoningEngine:
    """
    Sub-Engine Google Gemini AI dengan Strict Structured JSON Output, Retry Mechanism, dan Model Fallback.
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

    def propose_adaptation(
        self,
        drift_report: MultiStrategyDriftReport,
        current_weights: Dict[str, float],
        bandit_weights: Dict[str, float]
    ) -> Optional[StructuredAIProposal]:
        if not self.client:
            return None

        prompt = f"""
        Anda adalah Chief AI Quant Analyst & Portfolio Risk Controller untuk perdagangan saham IDX (BEI).
        Analisis laporan Model Drift dan usulan bobot UCB1 Bandit berikut, lalu berikan keputusan penyesuaian bobot strategi dalam FORMAT JSON MURNI:

        === REPORT DRIFT STRATEGI ===
        - Overall Drift Detected: {drift_report.is_drift_detected}
        - Overall Severity: {drift_report.overall_severity.value}
        - Recommended Action: {drift_report.recommended_action.value}
        - Detail Per-Strategi:
        {json.dumps({s_id: d.__dict__ for s_id, d in drift_report.strategy_drifts.items()}, indent=2, default=str)}

        === BOBOT STRATEGI SAAT INI ===
        {json.dumps(current_weights, indent=2)}

        === USULAN BOBOT UCB1 BANDIT ===
        {json.dumps(bandit_weights, indent=2)}

        Syarat Mutlak Response:
        Kembalikan HANYA JSON murni tanpa markdown triple backticks.
        JSON harus mengikuti struktur persis berikut:
        {{
            "action": "WEIGHT_REBALANCING",
            "confidence": 0.85,
            "proposed_weights": {json.dumps(bandit_weights)},
            "reason_codes": ["MOMENTUM_STABLE", "MEAN_REVERSION_DRIFT_HIGH"],
            "risk_level": "MEDIUM"
        }}
        """

        for model_target in [PRIMARY_MODEL, FALLBACK_MODEL]:
            for attempt in range(2):  # Exponential retry
                try:
                    response = self.client.models.generate_content(
                        model=model_target,
                        contents=prompt,
                    )
                    if response and hasattr(response, "text") and response.text:
                        raw_text = response.text.strip()
                        if raw_text.startswith("```json"):
                            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                        elif raw_text.startswith("```"):
                            raw_text = raw_text.split("```")[1].split("```")[0].strip()

                        parsed = json.loads(raw_text)

                        return StructuredAIProposal(
                            action=str(parsed.get("action", "NO_ACTION")),
                            confidence=float(parsed.get("confidence", 0.0)),
                            proposed_weights={k: float(v) for k, v in parsed.get("proposed_weights", {}).items()},
                            reason_codes=list(parsed.get("reason_codes", [])),
                            risk_level=str(parsed.get("risk_level", "HIGH")),
                            raw_response=raw_text
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Retry {attempt+1} Gemini API ({model_target}) Gagal: {e}")
                    time.sleep(1.0 * (2 ** attempt))

        return None

# =============================================================================
# 5. DECISION VALIDATOR & HARD SAFETY LIMITS
# =============================================================================
class AdaptationValidator:
    """Validator Independen & Gatekeeper Hard Limits untuk mencegah runaway AI/learning errors."""

    @staticmethod
    def validate_and_clip_proposal(
        current_weights: Dict[str, float],
        proposed_weights: Dict[str, float],
        ai_proposal: Optional[StructuredAIProposal]
    ) -> Tuple[bool, Dict[str, float], Optional[str]]:
        """
        Memvalidasi usulan perubahan bobot berdasarkan batas aman:
        1. AI Confidence Threshold (>= 0.70 jika ada usulan AI).
        2. Maximum Weight Change per Cycle (<= 0.10).
        3. Weight Bounds (0.05 <= weight <= 0.60).
        4. Validasi Sum of Weights = 1.0 (dengan Normalisasi).
        5. Sanitasi NaN/Inf.
        """
        if not proposed_weights:
            return False, current_weights, "Proposed weights empty."

        # Cek sanitasi NaN / Inf
        for k, v in proposed_weights.items():
            if not np.isfinite(v) or v < 0.0:
                return False, current_weights, f"Invalid weight detected for {k}: {v}"

        # Cek Confidence AI jika ada
        if ai_proposal and ai_proposal.confidence < MIN_CONFIDENCE_THRESHOLD:
            return False, current_weights, f"AI confidence too low ({ai_proposal.confidence:.2f} < {MIN_CONFIDENCE_THRESHOLD})."

        validated_weights = {}
        for s_id, curr_w in current_weights.items():
            target_w = proposed_weights.get(s_id, curr_w)

            # Max change per cycle constraint (Clipping delta)
            delta = target_w - curr_w
            clipped_delta = float(np.clip(delta, -MAX_WEIGHT_CHANGE_PER_CYCLE, MAX_WEIGHT_CHANGE_PER_CYCLE))
            new_w = curr_w + clipped_delta

            # Absolute bounds clipping
            bounded_w = float(np.clip(new_w, MIN_STRATEGY_WEIGHT, MAX_STRATEGY_WEIGHT))
            validated_weights[s_id] = bounded_w

        # Normalisasi agar total bobot persis = 1.0
        total_w = sum(validated_weights.values())
        if total_w <= 0.0:
            return False, current_weights, "Total weight sum non-positive."

        normalized_weights = {k: float(v / total_w) for k, v in validated_weights.items()}

        return True, normalized_weights, None

# =============================================================================
# 6. UNIFIED SELF-LEARNING ENGINE (FACADE CLASS WITH CLOSED LOOP)
# =============================================================================
class UnifiedSelfLearningEngine:
    """
    Unified Facade Class yang mengelola seluruh siklus hidup pembelajaran mandiri (Self-Learning Lifecycle).
    Menghubungkan Feedback -> Drift -> Persistent Bandit -> Gemini AI -> Safety Gate -> Execution Engine.
    """

    def __init__(
        self,
        strategy_ids: Optional[List[str]] = None,
        db_path: str = "./checkpoints/self_learning.sqlite",
        gemini_api_key: Optional[str] = None
    ) -> None:
        self.strategy_ids = strategy_ids or ["MOMENTUM_ALPHA", "MEAN_REVERSION", "BREAKOUT_VOLATILITY"]
        self.store = SQLiteSelfLearningStore(db_path=db_path)
        self.drift_detector = PerStrategyDriftDetector(store=self.store)
        self.bandit_adapter = PersistentBanditAdapter(strategy_ids=self.strategy_ids, store=self.store)
        self.gemini_engine = GeminiSelfLearningReasoningEngine(api_key=gemini_api_key)
        self.strategy_manager: Optional[StrategyManagerProtocol] = None
        self._version = "3.3.0"

    def register_strategy_manager(self, manager: StrategyManagerProtocol) -> None:
        """Menghubungkan Trading Engine aktif agar bobot teradaptasi dapat langsung diterapkan."""
        self.strategy_manager = manager
        logger.info("🔗 [SELF_LEARNING] Strategy Manager terhubung untuk eksekusi bobot otomatis.")

    def process_trade_feedback(
        self,
        strategy_id: str,
        realized_return_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        win_rate_pct: float,
        trade_count: int = 1,
        transaction_cost_pct: float = 0.30
    ) -> AdaptationDecision:
        """
        Siklus Transaksional Lengkap Self-Learning Loop:
        1. Record Feedback & Update State
        2. Evaluate Per-Strategy Drift
        3. Generate Bandit Weights & Gemini AI Proposal
        4. Validate Safety Limits (Decision Gate)
        5. Apply Weights to Trading Engine
        6. Verify Application & Record Audit Trail
        """
        timestamp = _get_wib_timestamp_str()
        decision_id = f"ADAPT-{uuid.uuid4().hex[:8].upper()}"

        # Sanitasi Data Feedback Input
        if not np.isfinite(realized_return_pct) or not np.isfinite(sharpe_ratio) or trade_count < 1:
            raise DataValidationError("Feedback input mengandung nilai non-finite atau trade_count invalid.")

        feedback = PerformanceFeedback(
            strategy_id=strategy_id,
            realized_return_pct=realized_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate_pct=win_rate_pct,
            trade_count=trade_count,
            transaction_cost_pct=transaction_cost_pct,
            execution_timestamp=timestamp
        )

        # 1. Simpan Feedback & Update UCB1 Bandit State
        self.store.save_feedback(feedback)
        self.drift_detector.record_feedback(feedback)
        self.bandit_adapter.update_reward(feedback)

        # 2. Deteksi Drift Per-Strategi
        drift_report = self.drift_detector.evaluate_drift(timestamp)

        # 3. Dapatkan Bobot Saat Ini & Usulan UCB1
        current_weights = self.strategy_manager.get_active_weights() if self.strategy_manager else self.bandit_adapter.get_rebalanced_weights()
        bandit_weights = self.bandit_adapter.get_rebalanced_weights()

        # 4. Gemini AI Autonomous Proposal
        ai_proposal = None
        if drift_report.is_drift_detected or drift_report.recommended_action != AdaptationStrategy.NO_ACTION:
            ai_proposal = self.gemini_engine.propose_adaptation(drift_report, current_weights, bandit_weights)

        target_proposed_weights = ai_proposal.proposed_weights if (ai_proposal and ai_proposal.proposed_weights) else bandit_weights

        # Lifecycle State 1: PROPOSED
        decision = AdaptationDecision(
            decision_id=decision_id,
            status=AdaptationStatus.PROPOSED,
            adaptation_strategy=drift_report.recommended_action,
            current_weights=current_weights,
            proposed_weights=target_proposed_weights,
            validated_weights=current_weights,
            ai_proposal=ai_proposal,
            rejection_reason=None,
            timestamp=timestamp
        )
        self.store.save_adaptation_audit(decision)

        # 5. Validation & Safety Gate
        is_valid, validated_weights, rejection_reason = AdaptationValidator.validate_and_clip_proposal(
            current_weights, target_proposed_weights, ai_proposal
        )

        if not is_valid:
            decision = AdaptationDecision(
                decision_id=decision_id,
                status=AdaptationStatus.REJECTED,
                adaptation_strategy=drift_report.recommended_action,
                current_weights=current_weights,
                proposed_weights=target_proposed_weights,
                validated_weights=current_weights,
                ai_proposal=ai_proposal,
                rejection_reason=rejection_reason,
                timestamp=timestamp
            )
            self.store.save_adaptation_audit(decision)
            logger.warning(f"⚠️ [ADAPTATION_REJECTED] {decision_id}: {rejection_reason}")
            return decision

        # Lifecycle State 2: VALIDATED
        decision = AdaptationDecision(
            decision_id=decision_id,
            status=AdaptationStatus.VALIDATED,
            adaptation_strategy=drift_report.recommended_action,
            current_weights=current_weights,
            proposed_weights=target_proposed_weights,
            validated_weights=validated_weights,
            ai_proposal=ai_proposal,
            rejection_reason=None,
            timestamp=timestamp
        )
        self.store.save_adaptation_audit(decision)

        # 6. Apply Adaptation & Post-Verification
        if self.strategy_manager:
            apply_success = self.strategy_manager.apply_strategy_weights(validated_weights)
            if apply_success:
                # Verifikasi Post-Application
                active_now = self.strategy_manager.get_active_weights()
                verified = all(math.isclose(active_now.get(k, 0.0), v, abs_tol=1e-4) for k, v in validated_weights.items())
                final_status = AdaptationStatus.VERIFIED if verified else AdaptationStatus.FAILED
            else:
                final_status = AdaptationStatus.FAILED
        else:
            final_status = AdaptationStatus.APPLIED  # Simulation mode tanpa active manager

        # Lifecycle State 3: VERIFIED / APPLIED
        final_decision = AdaptationDecision(
            decision_id=decision_id,
            status=final_status,
            adaptation_strategy=drift_report.recommended_action,
            current_weights=current_weights,
            proposed_weights=target_proposed_weights,
            validated_weights=validated_weights,
            ai_proposal=ai_proposal,
            rejection_reason=None if final_status in [AdaptationStatus.VERIFIED, AdaptationStatus.APPLIED] else "Execution manager failed to apply weights.",
            timestamp=_get_wib_timestamp_str()
        )
        self.store.save_adaptation_audit(final_decision)
        logger.info(f"✅ [ADAPTATION_SUCCESS] {decision_id} Status: {final_status.value} Weights: {validated_weights}")

        return final_decision

    def get_current_strategy_weights(self) -> Dict[str, float]:
        """Mengembalikan bobot strategi teradaptasi saat ini."""
        return self.bandit_adapter.get_rebalanced_weights()

"""
=============================================================================
IDX Quantitative Signal Analysis System - Consolidated Research Module
FileName      : research.py
Directory     : Flat Directory (Root Level with main.py)
Version       : 2026.Q3.v1.2.2 (Production Ready - Gemini AI Integrated)
Compliance    : IDX Trading Rules (IDR Pairs, 0.3% Fee, 12H Max Staleness)
=============================================================================
"""

import concurrent.futures
import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union

import numpy as np
import polars as pl

# Standar Zona Waktu Pasar Saham Indonesia (WIB)
WIB_TZ: Final[ZoneInfo] = ZoneInfo("Asia/Jakarta")

# Model Baseline Gemini SDK
PRIMARY_MODEL: Final[str] = "gemini-2.5-flash"
FALLBACK_MODEL: Final[str] = "gemini-2.0-flash"

# Import Google GenAI Client
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# Dynamic Optuna Import Handler with Graceful Fallback
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Dynamic SciPy Import Handler for Exact Statistical Distribution Testing
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# =============================================================================
# INTEGRATION LOGGING & EXCEPTION HANDLERS
# =============================================================================
try:
    from logger import logger, get_logger
    logger = get_logger("IDX.Research")
except ImportError:
    logger = logging.getLogger("IDX.Research")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s][IDX.RESEARCH] %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# =============================================================================
# BURSA EFEK INDONESIA (IDX) & RETRO-COMPATIBILITY CONSTANTS
# =============================================================================
IDX_FEE_ROUNDTRIP_PCT: float = 0.003          # Fee transaksi roundtrip IDX (0.3%)
IDX_MIN_PRICE_IDR: float = 50.0               # Batas harga minimal saham IDX (Rp 50)
IDX_MIN_24H_VOLUME_IDR: float = 1_000_000_000.0 # Likuiditas harian minimal Rp 1 Miliar
IDX_MAX_STALENESS_SEC: float = 43200.0        # Batas usia data candle harian (12 Jam)

# Alias Konstanta untuk Mempertahankan Kompatibilitas Sistem Upstream/Downstream
TOKOCRYPTO_FEE_ROUNDTRIP_PCT: float = IDX_FEE_ROUNDTRIP_PCT
TOKOCRYPTO_MIN_PRICE_USDT: float = IDX_MIN_PRICE_IDR
TOKOCRYPTO_MIN_24H_VOLUME_USDT: float = IDX_MIN_24H_VOLUME_IDR
TOKOCRYPTO_MAX_STALENESS_SEC: float = IDX_MAX_STALENESS_SEC

# =============================================================================
# DEFENSIVE SANITIZATION HELPERS
# =============================================================================
def _ensure_polars_df_research(data: Any, default_cols: Optional[List[str]] = None) -> pl.DataFrame:
    """Sanitasi input internal untuk menjamin objek bertipe pl.DataFrame (Polars 1.0+ Compliant)."""
    if data is None:
        cols = default_cols or ["timestamp", "metric_name", "value"]
        return pl.DataFrame(schema={col: pl.String for col in cols})
    if isinstance(data, list):
        if not data:
            cols = default_cols or ["timestamp", "metric_name", "value"]
            return pl.DataFrame(schema={col: pl.String for col in cols})
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


def _get_wib_timestamp_str() -> str:
    """Mengembalikan timestamp berformat WIB (Asia/Jakarta)."""
    return datetime.now(WIB_TZ).strftime("%Y-%m-%d %H:%M:%S WIB")

# =============================================================================
# LOCAL EXCEPTIONS (FLAT DIRECTORY COMPLIANCE)
# =============================================================================
try:
    from exceptions import ResearchError, ValidationError, StorageError, FrameworkConfigurationError
except ImportError:
    class ResearchError(Exception): pass
    class ValidationError(ResearchError): pass
    class StorageError(ResearchError): pass
    class FrameworkConfigurationError(ResearchError): pass

# =============================================================================
# ENUMS
# =============================================================================
class ControllerState(Enum):
    INITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()

class MetricName(Enum):
    SHARPE_RATIO = "SHARPE_RATIO"
    TOTAL_RETURN = "TOTAL_RETURN"
    LATENCY_P95 = "LATENCY_P95"
    EXECUTION_SLIPPAGE = "EXECUTION_SLIPPAGE"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    QUEUE_DELAY = "QUEUE_DELAY"

class MetricDirection(Enum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"

class GapSeverity(Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class StrategyType(Enum):
    FEATURE_MUTATION = "FEATURE_MUTATION"
    HYPERPARAMETER_OPTIMIZATION = "HYPERPARAMETER_OPTIMIZATION"
    ENSEMBLE_REBALANCING = "ENSEMBLE_REBALANCING"
    REGULARIZATION_TIGHTENING = "REGULARIZATION_TIGHTENING"
    PRUNING_COMPRESSION = "PRUNING_COMPRESSION"

class ParameterType(Enum):
    CATEGORICAL = auto()
    INT = auto()
    FLOAT = auto()

class DeploymentStatus(Enum):
    SKIPPED = "SKIPPED"
    DEPLOYED = "DEPLOYED"
    SIMULATED = "SIMULATED"
    FAILED = "FAILED"
    FAILED_RECOVERY = "FAILED_RECOVERY"

class MarketRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    UNKNOWN = "UNKNOWN"

# =============================================================================
# DATA TRANSFER OBJECTS (DTOs)
# =============================================================================
@dataclass(frozen=True)
class ResearchConfig:
    cycle_interval_seconds: int = 3600
    max_consecutive_failures: int = 3
    max_experiment_duration_seconds: int = 7200
    max_candidate_models: int = 5
    dry_run: bool = False
    auto_deploy: bool = True
    random_seed: int = 42
    checkpoint_dir: str = "./checkpoints/research"
    gemini_api_key: Optional[str] = None

@dataclass(frozen=True)
class ScheduleDecision:
    should_run: bool
    reason: str = ""
    priority: str = "NORMAL"

@dataclass(frozen=True)
class BaselineMetric:
    metric_name: MetricName
    value: float
    std_dev: float
    sample_size: int
    version: str
    generated_at: datetime

@dataclass(frozen=True)
class GapThresholds:
    metric_name: MetricName
    direction: MetricDirection
    low_trigger_pct: float = 2.0
    medium_trigger_pct: float = 5.0
    high_trigger_pct: float = 10.0
    statistical_significance_z: float = 1.96

@dataclass(frozen=True)
class MetricGap:
    metric_name: MetricName
    current_mean: float
    current_median: float
    current_p95: float
    current_mad: float
    baseline_value: float
    absolute_gap: float
    relative_gap_pct: float
    severity: GapSeverity
    severity_score: float
    is_significant: bool
    priority_score: float

@dataclass(frozen=True)
class GapReport:
    is_optimal: bool
    detected_gaps: List[MetricGap]
    critical_count: int
    degraded_count: int
    optimal_count: int
    sample_size_evaluated: int
    generated_at: datetime
    market_regime: MarketRegime = MarketRegime.UNKNOWN

@dataclass(frozen=True)
class ParameterBound:
    param_type: ParameterType
    choices: Optional[List[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    log_scale: bool = False

@dataclass(frozen=True)
class StrategyBlueprint:
    strategy_id: str
    strategy_type: StrategyType
    target_metric: MetricName
    priority_score: float
    search_space: Dict[str, ParameterBound]
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class ResearchPlan:
    plan_id: str
    created_at: datetime
    source_cycle_id: str
    blueprints: List[StrategyBlueprint]
    total_estimated_complexity: float
    is_actionable: bool
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    ai_research_hypothesis: Optional[str] = None

@dataclass(frozen=True)
class ExperimentCandidate:
    candidate_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    train_duration: float
    validation_duration: float
    seed: int
    score: float

@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    plan_id: str
    candidate_models: List[ExperimentCandidate]
    best_candidate: Optional[ExperimentCandidate]
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    statistical_test: str
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    passed: bool
    execution_time: float
    random_seed: int
    artifacts: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class InstitutionalQuantMetrics:
    profit_factor: float = 1.0
    win_rate_pct: float = 50.0
    calmar_ratio: float = 1.0
    cvar_95_pct: float = 2.0
    max_drawdown_pct: float = 10.0

@dataclass(frozen=True)
class DeploymentDecision:
    decision_id: str
    experiment_id: str
    candidate_id: Optional[str]
    plan_id: str
    status: DeploymentStatus
    approved_at: datetime
    rationale: str
    rollback_token: Optional[str] = None

@dataclass(frozen=True)
class CycleResult:
    cycle_id: str
    success: bool
    duration_seconds: float
    plan_id: Optional[str] = None
    deployment_status: DeploymentStatus = DeploymentStatus.SKIPPED
    metrics: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None

@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    purged_train_count: int
    val_count: int
    embargo_hours: int

@dataclass(frozen=True)
class MetaExperimentRecord:
    record_id: str
    experiment_id: str
    plan_id: str
    strategy_type: str
    target_metric: str
    market_regime: MarketRegime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    composite_score: float
    created_at: datetime

class CancellationToken:
    def __init__(self, event: threading.Event) -> None:
        self._event = event

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise ResearchError("Operation cleanly aborted via CancellationToken request.")

# =============================================================================
# HELPER: GEMINI AI RESEARCH REASONING ENGINE
# =============================================================================
class GeminiResearchReasoningEngine:
    """
    Sub-Engine integrasi Google Gemini untuk memformulasi hipotesis
    optimasi kuantitatif berdasarkan laporan gap performa sistem.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None

        if HAS_GEMINI_SDK and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("🤖 [GEMINI_RESEARCH_INIT] Gemini Client terhubung pada Research Engine.")
            except Exception as e:
                logger.warning(f"⚠️ Gagal inisialisasi Gemini Client pada Research Engine: {e}")

    def generate_research_hypothesis(self, gap_report: GapReport) -> str:
        if not self.client or gap_report.is_optimal:
            return "Kondisi performa model optimal. Tidak diperlukan hipotesis baru."

        prompt = f"""
        Anda adalah Chief Quantitative Researcher untuk sistem perdagangan saham IDX (BEI).
        Analisis Laporan Kesenjangan Performa (Performance Gap Report) berikut dan berikan 1 hipotesis optimasi kuantitatif ringkas (maksimal 2 kalimat):

        Rezim Pasar: {gap_report.market_regime.value}
        Jumlah Gap Kritis: {gap_report.critical_count}
        Detail Gap Terdeteksi:
        {[f"- {g.metric_name.value}: Gap {g.relative_gap_pct:.2f}% (Baseline: {g.baseline_value}, Current: {g.current_mean})" for g in gap_report.detected_gaps]}

        Berikan fokus pada perbaikan hyperparameter atau regulasi yang paling relevan.
        """

        # PERBAIKAN: Dual-tier fallback untuk ketahanan kuota API Gemini
        for model_target in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                response = self.client.models.generate_content(
                    model=model_target,
                    contents=prompt,
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"⚠️ Error Gemini API ({model_target}) pada Research Reasoning: {e}")

        return "Gagal menghasilkan hipotesis AI."

# =============================================================================
# ABSTRACT PROTOCOLS (DEPENDENCY INVERSION)
# =============================================================================
class IActiveRegistry(Protocol):
    def promote_to_active(self, candidate_id: str, plan_id: str) -> None: ...
    def revert_to_previous(self, rollback_token: str) -> None: ...

class IModelEvaluator(Protocol):
    def evaluate_candidate(self, parameters: Dict[str, Any], metric: MetricName) -> Tuple[List[float], float, float]: ...
    def get_champion_samples(self, metric: MetricName) -> List[float]: ...

class IWalkForwardEvaluator(Protocol):
    def evaluate_fold(self, parameters: Dict[str, Any], df_train: pl.DataFrame, df_val: pl.DataFrame) -> Dict[str, float]: ...

class ITelemetryProvider(Protocol):
    def fetch_recent_metrics(self, window_seconds: int) -> pl.DataFrame: ...
    def detect_market_regime(self) -> MarketRegime: ...

class IBaselineRegistry(Protocol):
    def get_baseline(self, metric_name: MetricName) -> Optional[BaselineMetric]: ...
    def get_dynamic_baseline(self, metric_name: MetricName, window_days: int = 90) -> Optional[BaselineMetric]: ...

class IStateStore(Protocol):
    def load_consecutive_failures(self) -> int: ...
    def save_consecutive_failures(self, count: int) -> None: ...

class IScheduleRepository(Protocol):
    def get_last_cycle_start_time(self) -> Optional[datetime]: ...
    def save_cycle_start(self, cycle_id: str, start_time: datetime) -> None: ...
    def save_cycle_result(self, result: CycleResult) -> None: ...

class IMetaLearningStore(Protocol):
    def save_record(self, record: MetaExperimentRecord) -> None: ...
    def get_warmstart_priors(self, strategy_type: str, target_metric: str, market_regime: MarketRegime, top_k: int = 5) -> List[Dict[str, Any]]: ...

# =============================================================================
# FALLBACK IMPLEMENTATIONS
# =============================================================================
class DummyTelemetryProvider:
    def fetch_recent_metrics(self, window_seconds: int) -> pl.DataFrame:
        dates = [datetime.now(timezone.utc) - timedelta(hours=i) for i in range(200)]
        values = [1.5 + (0.1 * math.sin(i / 10.0)) for i in range(200)]
        return pl.DataFrame({"timestamp": dates, "metric_name": ["SHARPE_RATIO"] * 200, "value": values})

    def detect_market_regime(self) -> MarketRegime:
        return MarketRegime.BULL

class DummyBaselineRegistry:
    def get_baseline(self, metric_name: MetricName) -> Optional[BaselineMetric]:
        return BaselineMetric(
            metric_name=metric_name, value=1.5, std_dev=0.1, sample_size=100, version="v1", generated_at=datetime.now(timezone.utc)
        )

    def get_dynamic_baseline(self, metric_name: MetricName, window_days: int = 90) -> Optional[BaselineMetric]:
        return self.get_baseline(metric_name)

class DummyModelEvaluator:
    def evaluate_candidate(self, parameters: Dict[str, Any], metric: MetricName) -> Tuple[List[float], float, float]:
        return ([1.65] * 20, 0.1, 0.1)

    def get_champion_samples(self, metric: MetricName) -> List[float]:
        return [1.5] * 20

    def evaluate_fold(self, parameters: Dict[str, Any], df_train: pl.DataFrame, df_val: pl.DataFrame) -> Dict[str, float]:
        lr = parameters.get("learning_rate", 0.01)
        return {"SHARPE_RATIO": 1.70 - (lr * 2.0), "TOTAL_RETURN": 0.15, "MAX_DRAWDOWN": 0.08, "LATENCY_P95": 35.0}

class DummyActiveRegistry:
    def promote_to_active(self, candidate_id: str, plan_id: str) -> None: pass
    def revert_to_previous(self, rollback_token: str) -> None: pass

class DummyStateStore:
    def load_consecutive_failures(self) -> int: return 0
    def save_consecutive_failures(self, count: int) -> None: pass

# =============================================================================
# PURGED WALK-FORWARD CROSS-VALIDATION ENGINE
# =============================================================================
class PurgedWalkForwardCV:
    def __init__(
        self,
        n_splits: int = 5,
        label_holding_hours: int = 24,
        embargo_hours: int = 12,
        is_expanding_window: bool = True
    ) -> None:
        self.n_splits = max(2, n_splits)
        self.label_holding_hours = label_holding_hours
        self.embargo_hours = embargo_hours
        self.is_expanding_window = is_expanding_window

    def generate_folds(self, df_telemetry: pl.DataFrame, time_column: str = "timestamp") -> List[WalkForwardFold]:
        df_telemetry = _ensure_polars_df_research(df_telemetry)
        if df_telemetry.is_empty() or time_column not in df_telemetry.columns:
            return []

        # PERBAIKAN: Sanitasi format tipe data datetime
        if df_telemetry.schema[time_column] == pl.String:
            df_sorted = df_telemetry.with_columns(
                pl.col(time_column).str.to_datetime(strict=False)
            ).sort(time_column)
        else:
            df_sorted = df_telemetry.sort(time_column)

        min_time = df_sorted[time_column].min()
        max_time = df_sorted[time_column].max()

        if not isinstance(min_time, datetime) or not isinstance(max_time, datetime):
            return []

        total_duration = max_time - min_time
        segment_duration = total_duration / self.n_splits

        folds: List[WalkForwardFold] = []
        label_delta = timedelta(hours=self.label_holding_hours)

        for i in range(self.n_splits - 1):
            train_start = min_time if self.is_expanding_window else min_time + (segment_duration * i)
            raw_train_end = min_time + (segment_duration * (i + 1))
            val_start = raw_train_end
            val_end = val_start + segment_duration

            purged_train_end = val_start - label_delta

            train_sub = df_sorted.filter((pl.col(time_column) >= train_start) & (pl.col(time_column) <= purged_train_end))
            val_sub = df_sorted.filter((pl.col(time_column) >= val_start) & (pl.col(time_column) <= val_end))

            folds.append(
                WalkForwardFold(
                    fold_index=i + 1, train_start=train_start, train_end=purged_train_end,
                    val_start=val_start, val_end=val_end, purged_train_count=train_sub.height,
                    val_count=val_sub.height, embargo_hours=self.embargo_hours
                )
            )

        return folds

    def split_dataframe(self, df: pl.DataFrame, fold: WalkForwardFold, time_column: str = "timestamp") -> Tuple[pl.DataFrame, pl.DataFrame]:
        df = _ensure_polars_df_research(df)
        df_train = df.filter((pl.col(time_column) >= fold.train_start) & (pl.col(time_column) <= fold.train_end))
        df_val = df.filter((pl.col(time_column) >= fold.val_start) & (pl.col(time_column) <= fold.val_end))
        return df_train, df_val

# =============================================================================
# SQLITE META-LEARNING STORE
# =============================================================================
class SQLiteMetaLearningStore(IMetaLearningStore):
    def __init__(self, db_path: str = "./checkpoints/meta_learning.sqlite") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS meta_experiments (
                        record_id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        plan_id TEXT NOT NULL,
                        strategy_type TEXT NOT NULL,
                        target_metric TEXT NOT NULL,
                        market_regime TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        metrics_json TEXT NOT NULL,
                        composite_score REAL NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_meta_query 
                    ON meta_experiments (strategy_type, target_metric, market_regime, composite_score DESC);
                """)
        except sqlite3.Error as e:
            raise StorageError(f"Gagal menginisialisasi SQLite Meta Learning Store: {e}")

    def save_record(self, record: MetaExperimentRecord) -> None:
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO meta_experiments (
                    record_id, experiment_id, plan_id, strategy_type, target_metric,
                    market_regime, parameters_json, metrics_json, composite_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id, record.experiment_id, record.plan_id, record.strategy_type,
                    record.target_metric, record.market_regime.value, json.dumps(record.parameters),
                    json.dumps(record.metrics), float(record.composite_score), record.created_at.isoformat()
                )
            )
            conn.execute("COMMIT;")
        except sqlite3.Error as e:
            conn.execute("ROLLBACK;")
            raise StorageError(f"Gagal menyimpan catatan meta-learning: {e}")
        finally:
            conn.close()

    def get_warmstart_priors(self, strategy_type: str, target_metric: str, market_regime: MarketRegime, top_k: int = 5) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT parameters_json FROM meta_experiments
                WHERE strategy_type = ? AND target_metric = ? AND market_regime = ?
                ORDER BY composite_score DESC LIMIT ?
                """,
                (strategy_type, target_metric, market_regime.value, top_k)
            )
            priors: List[Dict[str, Any]] = []
            for r in cursor.fetchall():
                try:
                    priors.append(json.loads(r[0]))
                except json.JSONDecodeError:
                    continue
            return priors
        except sqlite3.Error:
            return []
        finally:
            conn.close()

# =============================================================================
# DOMAIN COMPONENTS
# =============================================================================
class PerformanceGapAnalyzer:
    def __init__(
        self,
        telemetry_provider: ITelemetryProvider,
        baseline_registry: IBaselineRegistry,
        thresholds: List[GapThresholds],
        evaluation_window_seconds: int = 86400
    ) -> None:
        self._telemetry_provider = telemetry_provider
        self._baseline_registry = baseline_registry
        self._threshold_maps: Dict[MetricName, GapThresholds] = {t.metric_name: t for t in thresholds}
        self._window_seconds = evaluation_window_seconds

    def analyze_gaps(self, cancel_token: CancellationToken) -> GapReport:
        cancel_token.raise_if_cancelled()
        try:
            regime = self._telemetry_provider.detect_market_regime()
            df_raw = self._telemetry_provider.fetch_recent_metrics(self._window_seconds)
            df_raw = _ensure_polars_df_research(df_raw)
            cancel_token.raise_if_cancelled()

            if df_raw.is_empty():
                return self._create_empty_report(regime)

            required_cols = {"timestamp", "metric_name", "value"}
            if not required_cols.issubset(set(df_raw.columns)):
                raise ValidationError(f"Telemetry dataframe missing columns: {required_cols - set(df_raw.columns)}")

            df_working = df_raw.select(["metric_name", "value"]).filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            if df_working.is_empty():
                return self._create_empty_report(regime)

            # PERBAIKAN: Mengganti pl.col().count() terdepresiasi dengan pl.len()
            aggregated = df_working.group_by("metric_name").agg([
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().alias("std_dev"),
                pl.len().alias("count"),
                pl.col("value").median().alias("median"),
                pl.col("value").quantile(0.95).alias("p95"),
            ])

            med_by_metric = df_working.group_by("metric_name").agg(pl.col("value").median().alias("med_val"))
            mad_df = (
                df_working.join(med_by_metric, on="metric_name", how="left")
                .with_columns((pl.col("value") - pl.col("med_val")).abs().alias("abs_dev"))
                .group_by("metric_name").agg(pl.col("abs_dev").median().alias("mad"))
            )
            aggregated = aggregated.join(mad_df, on="metric_name", how="left")

            detected_gaps: List[MetricGap] = []
            optimal_count = degraded_count = critical_count = 0

            for idx, row in enumerate(aggregated.iter_rows(named=True)):
                if idx % 20 == 0:
                    cancel_token.raise_if_cancelled()

                try:
                    m_enum = MetricName(row["metric_name"])
                except ValueError:
                    continue

                baseline = self._baseline_registry.get_dynamic_baseline(m_enum) or self._baseline_registry.get_baseline(m_enum)
                if not baseline or m_enum not in self._threshold_maps:
                    optimal_count += 1
                    continue

                gap_dto = self._evaluate_two_sample_gap(m_enum, row, baseline, self._threshold_maps[m_enum])
                if gap_dto.is_significant and gap_dto.severity != GapSeverity.NONE:
                    detected_gaps.append(gap_dto)
                    if gap_dto.severity in (GapSeverity.HIGH, GapSeverity.CRITICAL):
                        critical_count += 1
                    else:
                        degraded_count += 1
                else:
                    optimal_count += 1

            detected_gaps.sort(key=lambda x: x.priority_score, reverse=True)
            total_samples = int(aggregated["count"].sum()) if not aggregated.is_empty() and aggregated["count"].sum() is not None else 0

            return GapReport(
                is_optimal=(len(detected_gaps) == 0), detected_gaps=detected_gaps, critical_count=critical_count,
                degraded_count=degraded_count, optimal_count=optimal_count, sample_size_evaluated=total_samples,
                generated_at=datetime.now(timezone.utc), market_regime=regime
            )
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Gap analyzer fault: {str(e)}", exc_info=True)
            raise ResearchError(f"Statistical gap evaluation failure: {str(e)}")

    def _evaluate_two_sample_gap(self, metric_name: MetricName, stats_row: dict, baseline: BaselineMetric, rules: GapThresholds) -> MetricGap:
        EPSILON = 1e-9
        cur_mean = float(stats_row["mean"]) if stats_row["mean"] is not None else 0.0
        cur_std = float(stats_row["std_dev"]) if stats_row["std_dev"] is not None else 0.0
        cur_n = int(stats_row["count"]) if stats_row["count"] is not None else 0

        abs_gap = (baseline.value - cur_mean) if rules.direction == MetricDirection.HIGHER_IS_BETTER else (cur_mean - baseline.value)
        denom = baseline.value if abs(baseline.value) > EPSILON else EPSILON
        rel_gap_pct = (abs_gap / abs(denom)) * 100.0

        if rel_gap_pct <= 0:
            return MetricGap(
                metric_name=metric_name, current_mean=cur_mean, current_median=float(stats_row["median"] or 0),
                current_p95=float(stats_row["p95"] or 0), current_mad=float(stats_row["mad"] or 0), baseline_value=baseline.value,
                absolute_gap=0.0, relative_gap_pct=0.0, severity=GapSeverity.NONE, severity_score=0.0, is_significant=False, priority_score=0.0
            )

        is_significant = False
        if cur_n > 1 and baseline.sample_size > 1:
            pooled_se = ((cur_std ** 2) / cur_n + (baseline.std_dev ** 2) / baseline.sample_size) ** 0.5
            if (abs_gap / (pooled_se + EPSILON)) >= rules.statistical_significance_z:
                is_significant = True
        elif cur_n > 0 and rel_gap_pct > rules.low_trigger_pct:
            is_significant = True

        max_expected = rules.high_trigger_pct * 1.5
        severity_score = min(max(0.0, rel_gap_pct / (max_expected + EPSILON)), 2.0)
        severity = GapSeverity.NONE

        if is_significant:
            if rel_gap_pct >= rules.high_trigger_pct:
                severity = GapSeverity.CRITICAL if rel_gap_pct >= max_expected else GapSeverity.HIGH
            elif rel_gap_pct >= rules.medium_trigger_pct:
                severity = GapSeverity.MEDIUM
            elif rel_gap_pct >= rules.low_trigger_pct:
                severity = GapSeverity.LOW

        return MetricGap(
            metric_name=metric_name, current_mean=cur_mean, current_median=float(stats_row["median"] or 0),
            current_p95=float(stats_row["p95"] or 0), current_mad=float(stats_row["mad"] or 0), baseline_value=baseline.value,
            absolute_gap=abs_gap, relative_gap_pct=rel_gap_pct, severity=severity, severity_score=severity_score,
            is_significant=is_significant, priority_score=severity_score * (1.5 if severity == GapSeverity.CRITICAL else 1.0)
        )

    def _create_empty_report(self, regime: MarketRegime) -> GapReport:
        return GapReport(
            is_optimal=True, detected_gaps=[], critical_count=0, degraded_count=0, optimal_count=0,
            sample_size_evaluated=0, generated_at=datetime.now(timezone.utc), market_regime=regime
        )


class ImprovementPlanner:
    """Mengubah kesenjangan performa (gaps) menjadi cetak biru strategi optimasi dinamis dengan Google Gemini AI Reasoning."""
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_engine = GeminiResearchReasoningEngine(api_key=gemini_api_key)

    def generate_plan(self, gap_report: GapReport, cancel_token: CancellationToken) -> ResearchPlan:
        cancel_token.raise_if_cancelled()
        plan_id = f"PLN-{uuid.uuid4().hex[:8].upper()}"

        if gap_report.is_optimal or not gap_report.detected_gaps:
            return ResearchPlan(
                plan_id=plan_id, created_at=datetime.now(timezone.utc), source_cycle_id="AUTO",
                blueprints=[], total_estimated_complexity=0.0, is_actionable=False, market_regime=gap_report.market_regime
            )

        ai_hypothesis = self.gemini_engine.generate_research_hypothesis(gap_report)

        blueprints: List[StrategyBlueprint] = []
        for gap in gap_report.detected_gaps:
            cancel_token.raise_if_cancelled()
            bp_id = f"BP-{secrets.token_hex(3).upper()}"
            strategy_type, search_space = self._resolve_search_space(gap.metric_name, gap_report.market_regime)

            blueprints.append(
                StrategyBlueprint(
                    strategy_id=bp_id, strategy_type=strategy_type, target_metric=gap.metric_name,
                    priority_score=gap.priority_score, search_space=search_space,
                    metadata={
                        "target_gap_pct": f"{gap.relative_gap_pct:.2f}",
                        "regime": gap_report.market_regime.value,
                        "ai_hypothesis": ai_hypothesis
                    }
                )
            )

        return ResearchPlan(
            plan_id=plan_id, created_at=datetime.now(timezone.utc), source_cycle_id="AUTO",
            blueprints=blueprints, total_estimated_complexity=float(len(blueprints) * 10), is_actionable=True,
            market_regime=gap_report.market_regime, ai_research_hypothesis=ai_hypothesis
        )

    def _resolve_search_space(self, metric: MetricName, regime: MarketRegime) -> Tuple[StrategyType, Dict[str, ParameterBound]]:
        if metric in (MetricName.LATENCY_P95, MetricName.QUEUE_DELAY):
            return StrategyType.PRUNING_COMPRESSION, {
                "num_leaves": ParameterBound(ParameterType.INT, low=15, high=63),
                "max_depth": ParameterBound(ParameterType.INT, low=3, high=8),
                "n_estimators": ParameterBound(ParameterType.INT, low=50, high=200)
            }
        elif metric == MetricName.EXECUTION_SLIPPAGE or regime == MarketRegime.HIGH_VOLATILITY:
            return StrategyType.REGULARIZATION_TIGHTENING, {
                "l1_regularization": ParameterBound(ParameterType.FLOAT, low=1e-3, high=5.0, log_scale=True),
                "l2_regularization": ParameterBound(ParameterType.FLOAT, low=1e-3, high=10.0, log_scale=True),
                "min_child_samples": ParameterBound(ParameterType.INT, low=20, high=100)
            }
        else:
            return StrategyType.HYPERPARAMETER_OPTIMIZATION, {
                "learning_rate": ParameterBound(ParameterType.FLOAT, low=0.001, high=0.1, log_scale=True),
                "num_leaves": ParameterBound(ParameterType.INT, low=15, high=127),
                "l2_regularization": ParameterBound(ParameterType.FLOAT, low=1e-4, high=10.0, log_scale=True),
                "feature_fraction": ParameterBound(ParameterType.FLOAT, low=0.5, high=1.0)
            }


class ExperimentRunner:
    def __init__(
        self,
        evaluator: IModelEvaluator,
        meta_store: Optional[IMetaLearningStore] = None,
        wf_cv: Optional[PurgedWalkForwardCV] = None,
        trials_per_blueprint: int = 15,
        random_seed: int = 42
    ) -> None:
        self._evaluator = evaluator
        self._meta_store = meta_store or SQLiteMetaLearningStore()
        self._wf_cv = wf_cv or PurgedWalkForwardCV()
        self._trials_per_blueprint = trials_per_blueprint
        self._random_seed = random_seed
        self._min_required_samples = 15

    def run_experiment(self, plan: ResearchPlan, cancel_token: CancellationToken) -> ExperimentResult:
        cancel_token.raise_if_cancelled()
        start_time = time.perf_counter()
        experiment_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"
        logs: List[str] = [f"[{_get_wib_timestamp_str()}] Experiment {experiment_id} started."]

        if plan.ai_research_hypothesis:
            logs.append(f"AI Research Hypothesis: {plan.ai_research_hypothesis}")

        if not plan.is_actionable or not plan.blueprints:
            logs.append("No actionable blueprints. Aborting experiment execution.")
            return self._create_empty_result(experiment_id, plan.plan_id, start_time, logs)

        all_candidates: List[ExperimentCandidate] = []
        best_candidate: Optional[ExperimentCandidate] = None
        best_overall_score = -float("inf")
        primary_metric = plan.blueprints[0].target_metric
        best_samples: List[float] = []

        for bp in sorted(plan.blueprints, key=lambda x: x.priority_score, reverse=True):
            cancel_token.raise_if_cancelled()
            logs.append(f"Executing blueprint {bp.strategy_id} targeting {bp.target_metric.value}")

            raw_champ = self._evaluator.get_champion_samples(bp.target_metric) or []
            champion_samples = [float(x) for x in raw_champ if x is not None and np.isfinite(x)]
            if len(champion_samples) < self._min_required_samples:
                continue

            priors = self._meta_store.get_warmstart_priors(bp.strategy_type.value, bp.target_metric.value, plan.market_regime, top_k=3)

            if HAS_OPTUNA:
                sampler = optuna.samplers.TPESampler(seed=self._random_seed)
                study = optuna.create_study(direction="maximize", sampler=sampler)
                for prior in priors:
                    study.enqueue_trial(prior)

                def objective(trial: optuna.Trial) -> float:
                    cancel_token.raise_if_cancelled()
                    params = self._suggest_parameters(trial, bp.search_space)
                    samples, train_dur, val_dur = self._evaluator.evaluate_candidate(params, bp.target_metric)
                    clean = [float(x) for x in (samples or []) if x is not None and np.isfinite(x)]
                    if len(clean) < self._min_required_samples:
                        raise optuna.exceptions.TrialPruned("Insufficient candidate samples.")

                    metrics_map = {bp.target_metric.value: sum(clean) / len(clean)}
                    score = self._compute_composite_score(metrics_map, bp.target_metric)

                    cand_id = f"CAND-{secrets.token_hex(6).upper()}"
                    trial.set_user_attr("candidate_id", cand_id)
                    trial.set_user_attr("train_dur", train_dur)
                    trial.set_user_attr("val_dur", val_dur)
                    trial.set_user_attr("clean_samples", clean)
                    return score

                study.optimize(objective, n_trials=self._trials_per_blueprint, n_jobs=1)
                completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if completed:
                    bp_candidates = self._extract_candidates(study, bp.target_metric)
                    all_candidates.extend(bp_candidates)
                    top_trial = study.best_trial
                    top_cand = next((c for c in bp_candidates if c.candidate_id == top_trial.user_attrs.get("candidate_id")), None)
                    if top_cand and top_cand.score > best_overall_score:
                        best_overall_score = top_cand.score
                        best_candidate = top_cand
                        best_samples = top_trial.user_attrs.get("clean_samples", [])
            else:
                bp_best, samples_found, bp_cands = self._fallback_random_search(bp, cancel_token)
                all_candidates.extend(bp_cands)
                if bp_best and bp_best.score > best_overall_score:
                    best_overall_score = bp_best.score
                    best_candidate = bp_best
                    best_samples = samples_found

        if not best_candidate or not best_samples:
            return self._create_empty_result(experiment_id, plan.plan_id, start_time, logs)

        raw_champ = self._evaluator.get_champion_samples(primary_metric) or []
        champion_samples = [float(x) for x in raw_champ if x is not None and np.isfinite(x)]
        p_value, effect_size, ci = self._compute_welchs_test_institutional(champion_samples, best_samples)
        passed = (p_value < 0.05) and (effect_size >= 0.5)

        logs.append(f"Welch's t-test: p={p_value:.4f}, Hedges' g={effect_size:.2f} | Passed Validation: {passed}")

        meta_record = MetaExperimentRecord(
            record_id=f"REC-{secrets.token_hex(6).upper()}", experiment_id=experiment_id, plan_id=plan.plan_id,
            strategy_type=plan.blueprints[0].strategy_type.value, target_metric=primary_metric.value,
            market_regime=plan.market_regime, parameters=best_candidate.parameters, metrics=best_candidate.metrics,
            composite_score=best_candidate.score, created_at=datetime.now(timezone.utc)
        )
        self._meta_store.save_record(meta_record)

        champ_mean = float(sum(champion_samples)/len(champion_samples)) if champion_samples else 0.0

        return ExperimentResult(
            experiment_id=experiment_id, plan_id=plan.plan_id, candidate_models=all_candidates,
            best_candidate=best_candidate, champion_metrics={primary_metric.value: champ_mean},
            challenger_metrics={primary_metric.value: best_candidate.score}, statistical_test="Welch's t-test (Hedges' g)",
            p_value=p_value, effect_size=effect_size, confidence_interval=ci, passed=passed,
            execution_time=time.perf_counter() - start_time, random_seed=self._random_seed,
            artifacts={"optimizer": "Optuna-TPE" if HAS_OPTUNA else "Randomized-Search-Fallback"}, logs=logs
        )

    def _compute_composite_score(self, metrics: Dict[str, float], target_metric: MetricName) -> float:
        sharpe = metrics.get(MetricName.SHARPE_RATIO.value, 1.0)
        drawdown = abs(metrics.get(MetricName.MAX_DRAWDOWN.value, 0.10))
        latency = metrics.get(MetricName.LATENCY_P95.value, 0.0)

        dd_penalty = math.exp(max(0.0, drawdown - 0.15) * 10.0) - 1.0
        composite = (sharpe * 1.5) - (drawdown * 2.0) - dd_penalty
        if target_metric in (MetricName.LATENCY_P95, MetricName.QUEUE_DELAY):
            composite -= (latency / 100.0)
        return float(composite)

    def _suggest_parameters(self, trial: Any, search_space: Dict[str, ParameterBound]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key, bound in search_space.items():
            if bound.param_type == ParameterType.CATEGORICAL and bound.choices:
                params[key] = trial.suggest_categorical(key, bound.choices)
            elif bound.param_type == ParameterType.INT:
                params[key] = trial.suggest_int(key, int(bound.low or 0), int(bound.high or 100), log=bound.log_scale)
            elif bound.param_type == ParameterType.FLOAT:
                params[key] = trial.suggest_float(key, float(bound.low or 0.0), float(bound.high or 1.0), log=bound.log_scale)
        return params

    def _extract_candidates(self, study: Any, metric: MetricName) -> List[ExperimentCandidate]:
        candidates = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                candidates.append(
                    ExperimentCandidate(
                        candidate_id=t.user_attrs["candidate_id"], parameters=t.params,
                        metrics={metric.value: float(t.value or 0.0)}, train_duration=float(t.user_attrs.get("train_dur", 0.0)),
                        validation_duration=float(t.user_attrs.get("val_dur", 0.0)), seed=self._random_seed + t.number, score=float(t.value or 0.0)
                    )
                )
        return candidates

    def _fallback_random_search(self, bp: StrategyBlueprint, cancel_token: CancellationToken, num_samples: int = 25) -> Tuple[Optional[ExperimentCandidate], List[float], List[ExperimentCandidate]]:
        rng = np.random.default_rng(self._random_seed)
        candidates: List[ExperimentCandidate] = []
        best_cand: Optional[ExperimentCandidate] = None
        best_samples: List[float] = []
        best_score = -float("inf")

        for i in range(num_samples):
            cancel_token.raise_if_cancelled()
            sampled_params: Dict[str, Any] = {}
            for key, bound in bp.search_space.items():
                if bound.param_type == ParameterType.CATEGORICAL and bound.choices:
                    sampled_params[key] = rng.choice(bound.choices)
                elif bound.param_type == ParameterType.INT:
                    sampled_params[key] = int(rng.integers(int(bound.low or 0), int(bound.high or 100) + 1))
                elif bound.param_type == ParameterType.FLOAT:
                    if bound.log_scale:
                        sampled_params[key] = float(np.exp(rng.uniform(np.log(bound.low or 1e-4), np.log(bound.high or 1.0))))
                    else:
                        sampled_params[key] = float(rng.uniform(bound.low or 0.0, bound.high or 1.0))

            samples, t_dur, v_dur = self._evaluator.evaluate_candidate(sampled_params, bp.target_metric)
            clean = [float(x) for x in (samples or []) if x is not None and np.isfinite(x)]
            if not clean:
                continue

            score = self._compute_composite_score({bp.target_metric.value: sum(clean) / len(clean)}, bp.target_metric)
            cand_id = f"CAND-RND-{secrets.token_hex(4).upper()}"
            candidate = ExperimentCandidate(
                candidate_id=cand_id, parameters=sampled_params, metrics={bp.target_metric.value: score},
                train_duration=t_dur, validation_duration=v_dur, seed=self._random_seed + i, score=score
            )
            candidates.append(candidate)
            if score > best_score:
                best_score = score
                best_cand = candidate
                best_samples = clean

        return best_cand, best_samples, candidates

    def _compute_welchs_test_institutional(self, champion: List[float], challenger: List[float]) -> Tuple[float, float, Tuple[float, float]]:
        champ_clean = [float(x) for x in champion if x is not None and np.isfinite(x)]
        chall_clean = [float(x) for x in challenger if x is not None and np.isfinite(x)]

        n1, n2 = len(champ_clean), len(chall_clean)
        if n1 < 2 or n2 < 2:
            return 1.0, 0.0, (0.0, 0.0)

        mean1, mean2 = sum(champ_clean) / n1, sum(chall_clean) / n2
        var1 = sum((x - mean1) ** 2 for x in champ_clean) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in chall_clean) / (n2 - 1)

        se_sq1, se_sq2 = var1 / n1, var2 / n2
        se = math.sqrt(se_sq1 + se_sq2) + 1e-9
        df_num = (se_sq1 + se_sq2) ** 2
        df_den = (se_sq1 ** 2) / (n1 - 1) + (se_sq2 ** 2) / (n2 - 1)
        df = df_num / max(1e-9, df_den)

        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, n1 + n2 - 2)) + 1e-9
        cohens_d = (mean2 - mean1) / pooled_std

        # PERBAIKAN: Guardrail pembagian dengan nol pada koreksi Hedges' g
        denom_j = max(1e-5, abs(4.0 * df - 1.0))
        j_factor = 1.0 - (3.0 / denom_j)
        hedges_g = cohens_d * j_factor

        diff = mean2 - mean1
        if HAS_SCIPY:
            t_stat, p_val = stats.ttest_ind(chall_clean, champ_clean, equal_var=False)
            p_val = float(p_val) if np.isfinite(p_val) else 1.0
            t_crit = float(stats.t.ppf(0.975, df=df))
        else:
            t_stat = diff / se
            z = 1.95996
            t_crit = z + (z**3 + z) / (4 * df) + (5 * z**3 + 16 * z**3 + 3 * z) / (96 * df**2)
            z_stat = math.sqrt(max(0.0, (df - 0.5) * math.log(1 + (t_stat**2) / df)))
            p_val = math.erfc(z_stat / math.sqrt(2.0))

        margin = t_crit * se
        return min(max(0.0, float(p_val)), 1.0), float(hedges_g), (diff - margin, diff + margin)

    def _create_empty_result(self, exp_id: str, plan_id: str, t_start: float, logs: List[str]) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=exp_id, plan_id=plan_id, candidate_models=[], best_candidate=None, champion_metrics={},
            challenger_metrics={}, statistical_test="None", p_value=1.0, effect_size=0.0, confidence_interval=(0.0, 0.0),
            passed=False, execution_time=time.perf_counter() - t_start, random_seed=self._random_seed, logs=logs
        )


class InstitutionalDeploymentGate:
    def __init__(self, registry: IActiveRegistry, db_path: str = "./checkpoints/research_history.sqlite", dry_run: bool = False) -> None:
        self._registry = registry
        self.db_path = db_path
        self._dry_run = dry_run
        self._hmac_secret = secrets.token_bytes(32)
        self._init_audit_table()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def evaluate_challenger(
        self,
        experiment_result: ExperimentResult,
        quant_metrics: Optional[InstitutionalQuantMetrics] = None
    ) -> DeploymentDecision:
        decision_id = f"DEC-{secrets.token_hex(4).upper()}"

        if not experiment_result.passed or not experiment_result.best_candidate:
            return self._create_rejected_decision(decision_id, experiment_result, "Rejected: Failed statistical boundary.", DeploymentStatus.SKIPPED)

        metrics = quant_metrics or InstitutionalQuantMetrics()
        if metrics.profit_factor < 1.3:
            return self._create_rejected_decision(decision_id, experiment_result, "Quant Reject: Profit Factor < 1.3", DeploymentStatus.FAILED)
        if metrics.max_drawdown_pct > 15.0:
            return self._create_rejected_decision(decision_id, experiment_result, "Quant Reject: Max Drawdown > 15%", DeploymentStatus.FAILED)
        if metrics.cvar_95_pct > 5.0:
            return self._create_rejected_decision(decision_id, experiment_result, "Quant Reject: Tail Risk CVaR (95%) Exceeds 5%", DeploymentStatus.FAILED)

        candidate_id = experiment_result.best_candidate.candidate_id
        plan_id = experiment_result.plan_id
        approved_time = datetime.now(timezone.utc)
        rollback_token = self._generate_hmac_token(candidate_id, plan_id, approved_time)
        status = DeploymentStatus.SIMULATED if self._dry_run else DeploymentStatus.DEPLOYED
        rationale = f"Approved. p-value: {experiment_result.p_value:.4f}, Hedges' g: {experiment_result.effect_size:.2f}."

        return DeploymentDecision(
            decision_id=decision_id, experiment_id=experiment_result.experiment_id, candidate_id=candidate_id,
            plan_id=plan_id, status=status, approved_at=approved_time, rationale=rationale, rollback_token=rollback_token
        )

    def deploy(self, decision: DeploymentDecision) -> None:
        if decision.status == DeploymentStatus.SKIPPED:
            return
        if decision.status == DeploymentStatus.FAILED:
            raise ValidationError(f"Deployment blocked: {decision.rationale}")

        try:
            self._save_audit_log_transactional(decision)
            if decision.status == DeploymentStatus.DEPLOYED and not self._dry_run and decision.candidate_id:
                self._registry.promote_to_active(candidate_id=decision.candidate_id, plan_id=decision.plan_id)
        except Exception as e:
            logger.critical(f"Deployment crashed: {str(e)}")
            self._update_audit_status_transactional(decision.decision_id, DeploymentStatus.FAILED_RECOVERY)
            raise ResearchError(f"Deployment orchestration failure: {str(e)}")

    def rollback(self, decision: DeploymentDecision) -> None:
        if not decision.rollback_token:
            return
        try:
            if not self._dry_run:
                self._registry.revert_to_previous(decision.rollback_token)
            self._update_audit_status_transactional(decision.decision_id, DeploymentStatus.FAILED_RECOVERY)
        except Exception as e:
            raise ResearchError(f"System deployment recovery failure: {str(e)}")

    def _generate_hmac_token(self, candidate_id: str, plan_id: str, timestamp: datetime) -> str:
        payload = f"{candidate_id}:{plan_id}:{timestamp.isoformat()}"
        mac = hmac.new(self._hmac_secret, payload.encode('utf-8'), hashlib.sha256)
        return f"RBK-{mac.hexdigest()[:24].upper()}"

    def _init_audit_table(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS deployment_audit (
                        decision_id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        candidate_id TEXT,
                        plan_id TEXT,
                        status TEXT NOT NULL,
                        approved_at TEXT NOT NULL,
                        rationale TEXT NOT NULL,
                        rollback_token TEXT
                    )
                """)
        except sqlite3.Error as e:
            raise StorageError(f"Failed initializing deployment audit table: {e}")

    def _save_audit_log_transactional(self, d: DeploymentDecision) -> None:
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO deployment_audit 
                (decision_id, experiment_id, candidate_id, plan_id, status, approved_at, rationale, rollback_token)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (d.decision_id, d.experiment_id, d.candidate_id, d.plan_id, d.status.value, d.approved_at.isoformat(), d.rationale, d.rollback_token)
            )
            conn.execute("COMMIT;")
        except sqlite3.Error as e:
            conn.execute("ROLLBACK;")
            raise StorageError(f"Atomic audit log write failed: {e}")
        finally:
            conn.close()

    def _update_audit_status_transactional(self, decision_id: str, status: DeploymentStatus) -> None:
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("UPDATE deployment_audit SET status = ? WHERE decision_id = ?", (status.value, decision_id))
            conn.execute("COMMIT;")
        except sqlite3.Error as e:
            conn.execute("ROLLBACK;")
            raise StorageError(f"Atomic audit update failed: {e}")
        finally:
            conn.close()

    def _create_rejected_decision(self, decision_id: str, res: ExperimentResult, rationale: str, status: DeploymentStatus) -> DeploymentDecision:
        return DeploymentDecision(
            decision_id=decision_id, experiment_id=res.experiment_id, candidate_id=res.best_candidate.candidate_id if res.best_candidate else None,
            plan_id=res.plan_id, status=status, approved_at=datetime.now(timezone.utc), rationale=rationale, rollback_token=None
        )

# =============================================================================
# SCHEDULER & REPOSITORY INFRASTRUCTURE
# =============================================================================
class SQLiteScheduleRepository(IScheduleRepository):
    def __init__(self, db_path: str = "./checkpoints/research_history.sqlite") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS research_cycles (
                        cycle_id TEXT PRIMARY KEY,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        success BOOLEAN,
                        duration_seconds REAL,
                        plan_id TEXT,
                        deployment_status TEXT,
                        metrics JSON,
                        exception TEXT
                    )
                """)
        except sqlite3.Error as e:
            raise StorageError(f"Failed initializing schedule table: {e}")

    def get_last_cycle_start_time(self) -> Optional[datetime]:
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT start_time FROM research_cycles ORDER BY start_time DESC LIMIT 1")
                row = cursor.fetchone()
                return datetime.fromisoformat(row[0]) if row else None
        except sqlite3.Error as e:
            raise StorageError(f"Database query failed: {e}")

    def save_cycle_start(self, cycle_id: str, start_time: datetime) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute("INSERT INTO research_cycles (cycle_id, start_time) VALUES (?, ?)", (cycle_id, start_time.isoformat()))
        except sqlite3.Error as e:
            raise StorageError(f"Failed inserting cycle start: {e}")

    def save_cycle_result(self, result: CycleResult) -> None:
        end_time = datetime.now(timezone.utc).isoformat()
        metrics_json = json.dumps(result.metrics) if result.metrics else None
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE research_cycles 
                    SET end_time = ?, success = ?, duration_seconds = ?, plan_id = ?, 
                        deployment_status = ?, metrics = ?, exception = ?
                    WHERE cycle_id = ?
                    """,
                    (end_time, result.success, result.duration_seconds, result.plan_id, result.deployment_status.value, metrics_json, result.exception, result.cycle_id)
                )
        except sqlite3.Error as e:
            raise StorageError(f"Failed updating cycle result: {e}")


class ResearchScheduler:
    def __init__(self, config: ResearchConfig, repository: IScheduleRepository) -> None:
        self.config = config
        self.repository = repository
        self._lock = threading.RLock()
        self._active_cycle_id: Optional[str] = None

    def evaluate_schedule(self) -> ScheduleDecision:
        with self._lock:
            if self._active_cycle_id:
                return ScheduleDecision(should_run=False, reason=f"Cycle {self._active_cycle_id} active.")

            try:
                last_start = self.repository.get_last_cycle_start_time()
            except StorageError:
                return ScheduleDecision(should_run=False, reason="Storage query error.")

            if not last_start:
                return ScheduleDecision(should_run=True, reason="First execution initialization.", priority="HIGH")

            elapsed = (datetime.now(timezone.utc) - last_start).total_seconds()
            if elapsed >= self.config.cycle_interval_seconds:
                return ScheduleDecision(should_run=True, reason=f"Interval met ({elapsed:.0f}s >= {self.config.cycle_interval_seconds}s).")

            return ScheduleDecision(should_run=False, reason="Interval not yet met.")

    def record_cycle_start(self, cycle_id: str) -> None:
        with self._lock:
            self._active_cycle_id = cycle_id
            self.repository.save_cycle_start(cycle_id, datetime.now(timezone.utc))

    def record_cycle_end(self, result: CycleResult) -> None:
        with self._lock:
            try:
                self.repository.save_cycle_result(result)
            finally:
                if self._active_cycle_id == result.cycle_id:
                    self._active_cycle_id = None

# =============================================================================
# RESEARCH CONTROLLER (FINITE STATE MACHINE)
# =============================================================================
class ResearchController:
    VALID_TRANSITIONS: Dict[ControllerState, Set[ControllerState]] = {
        ControllerState.INITIALIZED: {ControllerState.RUNNING},
        ControllerState.RUNNING: {ControllerState.PAUSED, ControllerState.STOPPED, ControllerState.ERROR},
        ControllerState.PAUSED: {ControllerState.RUNNING, ControllerState.STOPPED, ControllerState.ERROR},
        ControllerState.ERROR: {ControllerState.INITIALIZED, ControllerState.STOPPED},
        ControllerState.STOPPED: set(),
    }

    def __init__(
        self,
        config: ResearchConfig,
        scheduler: ResearchScheduler,
        gap_analyzer: PerformanceGapAnalyzer,
        planner: ImprovementPlanner,
        experiment_runner: ExperimentRunner,
        deployment_gate: InstitutionalDeploymentGate,
        state_store: IStateStore,
    ) -> None:
        self.config = config
        self._scheduler = scheduler
        self._gap_analyzer = gap_analyzer
        self._planner = planner
        self._experiment_runner = experiment_runner
        self._deployment_gate = deployment_gate
        self._state_store = state_store

        self._state: ControllerState = ControllerState.INITIALIZED
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._cancel_token = CancellationToken(self._stop_event)
        self._thread: Optional[threading.Thread] = None
        self._consecutive_failures = self._state_store.load_consecutive_failures()

    @property
    def state(self) -> ControllerState:
        with self._state_lock:
            return self._state

    def _transition_to(self, new_state: ControllerState) -> None:
        with self._state_lock:
            if new_state not in self.VALID_TRANSITIONS[self._state]:
                raise FrameworkConfigurationError(f"Invalid transition: {self._state.name} -> {new_state.name}")
            logger.info(f"FSM State Transition: {self._state.name} -> {new_state.name}")
            self._state = new_state
            if new_state == ControllerState.PAUSED:
                self._pause_event.clear()
            elif new_state == ControllerState.RUNNING:
                self._pause_event.set()

    def start(self, daemon: bool = True) -> None:
        if self.state in (ControllerState.RUNNING, ControllerState.PAUSED):
            return
        self._stop_event.clear()
        self._transition_to(ControllerState.RUNNING)
        self._thread = threading.Thread(target=self._research_loop, daemon=daemon, name="ResearchLoop")
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop_event.set()
        self._pause_event.set()
        try:
            self._transition_to(ControllerState.STOPPED)
        except FrameworkConfigurationError:
            pass
        if threading.current_thread() is not self._thread and self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _research_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break
                decision = self._scheduler.evaluate_schedule()
                if decision.should_run:
                    self._execute_cycle(decision)
                else:
                    self._stop_event.wait(timeout=60.0)
            except Exception as e:
                logger.error(f"Fatal error in research loop: {str(e)}", exc_info=True)
                self._transition_to(ControllerState.ERROR)
                break

    def _execute_cycle(self, decision: ScheduleDecision) -> None:
        cycle_id = f"RES-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        start_time = time.time()
        self._scheduler.record_cycle_start(cycle_id)
        result = CycleResult(cycle_id=cycle_id, success=False, duration_seconds=0.0)

        try:
            gap_report = self._gap_analyzer.analyze_gaps(self._cancel_token)
            if self._cancel_token.is_cancelled:
                return

            if gap_report.is_optimal:
                result = CycleResult(cycle_id=cycle_id, success=True, duration_seconds=time.time()-start_time, deployment_status=DeploymentStatus.SKIPPED)
                return

            plan = self._planner.generate_plan(gap_report, self._cancel_token)
            if self._cancel_token.is_cancelled or not plan.is_actionable:
                return

            result = CycleResult(cycle_id=cycle_id, success=False, duration_seconds=0.0, plan_id=plan.plan_id)

            exp_res = self._experiment_runner.run_experiment(plan, self._cancel_token)
            if self._cancel_token.is_cancelled:
                return

            dep_decision = self._deployment_gate.evaluate_challenger(exp_res)
            if dep_decision.status in (DeploymentStatus.DEPLOYED, DeploymentStatus.SIMULATED):
                if not self.config.dry_run and self.config.auto_deploy:
                    self._deployment_gate.deploy(dep_decision)
                    result = CycleResult(cycle_id=cycle_id, success=True, duration_seconds=time.time()-start_time, plan_id=plan.plan_id, deployment_status=DeploymentStatus.DEPLOYED)
                else:
                    result = CycleResult(cycle_id=cycle_id, success=True, duration_seconds=time.time()-start_time, plan_id=plan.plan_id, deployment_status=DeploymentStatus.SIMULATED)
            else:
                result = CycleResult(cycle_id=cycle_id, success=True, duration_seconds=time.time()-start_time, plan_id=plan.plan_id, deployment_status=DeploymentStatus.SKIPPED)

            if self._consecutive_failures > 0:
                self._consecutive_failures = 0
                self._state_store.save_consecutive_failures(0)

        except ResearchError as re:
            if self._cancel_token.is_cancelled:
                logger.info(f"[{cycle_id}] Research cycle cleanly aborted via cancellation token.")
                result = CycleResult(cycle_id=cycle_id, success=False, duration_seconds=time.time()-start_time, exception="Aborted", deployment_status=DeploymentStatus.SKIPPED)
                return
            logger.error(f"[{cycle_id}] Research cycle domain error: {str(re)}")
            self._consecutive_failures += 1
            self._state_store.save_consecutive_failures(self._consecutive_failures)
            result = CycleResult(cycle_id=cycle_id, success=False, duration_seconds=time.time()-start_time, exception=str(re), deployment_status=DeploymentStatus.FAILED)
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                self._transition_to(ControllerState.ERROR)
        except Exception as e:
            logger.error(f"[{cycle_id}] Unexpected cycle failure: {str(e)}", exc_info=True)
            self._consecutive_failures += 1
            self._state_store.save_consecutive_failures(self._consecutive_failures)
            result = CycleResult(cycle_id=cycle_id, success=False, duration_seconds=time.time()-start_time, exception=str(e), deployment_status=DeploymentStatus.FAILED)
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                self._transition_to(ControllerState.ERROR)
        finally:
            if not self._cancel_token.is_cancelled:
                self._scheduler.record_cycle_end(result)

# =============================================================================
# FACADE CLASS (SINGLE POINT OF ENTRY)
# =============================================================================
class UnifiedResearchEngine:
    """
    Unified High-Level Facade Class yang mengelola seluruh Autonomous Research Lifecycle.
    Digunakan oleh main.py untuk integrasi tanpa overhead.
    """
    def __init__(
        self,
        config: Optional[ResearchConfig] = None,
        telemetry_provider: Optional[ITelemetryProvider] = None,
        baseline_registry: Optional[IBaselineRegistry] = None,
        model_evaluator: Optional[IModelEvaluator] = None,
        active_registry: Optional[IActiveRegistry] = None,
        meta_store: Optional[IMetaLearningStore] = None,
        state_store: Optional[IStateStore] = None,
        db_path: str = "./checkpoints/research_history.sqlite"
    ) -> None:
        self.config = config or ResearchConfig()
        self.db_path = db_path

        self._state_store = state_store or DummyStateStore()
        telemetry_provider = telemetry_provider or DummyTelemetryProvider()
        baseline_registry = baseline_registry or DummyBaselineRegistry()
        model_evaluator = model_evaluator or DummyModelEvaluator()
        active_registry = active_registry or DummyActiveRegistry()
        meta_store = meta_store or SQLiteMetaLearningStore()

        self.repository = SQLiteScheduleRepository(db_path=self.db_path)
        self.scheduler = ResearchScheduler(config=self.config, repository=self.repository)
        
        default_thresholds = [
            GapThresholds(MetricName.SHARPE_RATIO, MetricDirection.HIGHER_IS_BETTER),
            GapThresholds(MetricName.TOTAL_RETURN, MetricDirection.HIGHER_IS_BETTER),
            GapThresholds(MetricName.MAX_DRAWDOWN, MetricDirection.LOWER_IS_BETTER)
        ]

        self.gap_analyzer = PerformanceGapAnalyzer(telemetry_provider, baseline_registry, default_thresholds)
        self.planner = ImprovementPlanner(gemini_api_key=self.config.gemini_api_key)
        self.wf_cv = PurgedWalkForwardCV()
        self.experiment_runner = ExperimentRunner(
            evaluator=model_evaluator, meta_store=meta_store, wf_cv=self.wf_cv, random_seed=self.config.random_seed
        )
        self.deployment_gate = InstitutionalDeploymentGate(active_registry, db_path=self.db_path, dry_run=self.config.dry_run)

        self.controller = ResearchController(
            config=self.config,
            scheduler=self.scheduler,
            gap_analyzer=self.gap_analyzer,
            planner=self.planner,
            experiment_runner=self.experiment_runner,
            deployment_gate=self.deployment_gate,
            state_store=self._state_store
        )

    def start_autonomous_loop(self) -> None:
        if self.controller:
            self.controller.start()
            logger.info("Unified Research Engine autonomous loop started.")

    def stop_autonomous_loop(self) -> None:
        if self.controller:
            self.controller.stop()
            logger.info("Unified Research Engine autonomous loop stopped.")

    def run_manual_cycle(self) -> ScheduleDecision:
        decision = self.scheduler.evaluate_schedule()
        if self.controller and (decision.should_run or self.config.dry_run):
            logger.info("Running manual research evaluation cycle...")
            self.controller._execute_cycle(decision)
        return decision

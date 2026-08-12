"""
IDX Synchronized Data Engine - Consolidated Quantitative Data Pipeline v2026.8.7 (Patched)
=======================================================================================
Production-Grade Ultra-Low Latency Pipeline for Indonesian Stock Exchange (IDX).
Patched for Rule Compliance: Parallel Corporate Actions, Classified Errors, and Continuity-Safe Outliers.
"""

import concurrent.futures
import datetime
import hashlib
import io
import json
import logging
import os
import random
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import requests
import yfinance as yf
from cryptography.fernet import Fernet

# ==============================================================================
# YFINANCE TZCACHE LOCATION OVERRIDE
# ==============================================================================
try:
    tz_cache_dir = os.path.join(tempfile.gettempdir(), "yf_tz_cache")
    os.makedirs(tz_cache_dir, exist_ok=True)
    yf.set_tz_cache_location(tz_cache_dir)
except Exception:
    pass

# ==============================================================================
# ENVIRONMENT DETECTION, CONSTANTS & RUN_ID
# ==============================================================================
WIB_TZ = ZoneInfo("Asia/Jakarta")
UTC_TZ = datetime.timezone.utc

RUN_ID: str = datetime.datetime.now(WIB_TZ).strftime("%Y%m%d-%H%M%S")
IS_CI_ENV: bool = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"

DEFAULT_MAX_WORKERS: int = 8
DEFAULT_BATCH_SIZE: int = 5 if IS_CI_ENV else 10
INTER_BATCH_DELAY_SEC: float = 0.3 if IS_CI_ENV else 0.1
FETCH_TIMEOUT_SEC: float = float(os.getenv("IDX_FETCH_TIMEOUT_SEC", "15.0"))

IDX_FEE_ROUNDTRIP_PCT: float = 0.003             # 0.3% Average Roundtrip Transaction Fee
IDX_MIN_PRICE_IDR: float = 50.0                  # IDX Regular Board Floor Price (Rp 50)
IDX_MIN_24H_TURNOVER_IDR: float = 100_000_000.0  # Rp 100 Juta Minimal Daily Turnover Baseline
IDX_MAX_HOLIDAY_STALENESS_SEC: float = 432000.0  # 120 Hours (Supports Extended Holidays)

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

def build_resilient_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HTTP_HEADERS)
    return session

SHARED_SESSION = build_resilient_session()

logger = logging.getLogger("IDX-DataEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[%(asctime)s][{RUN_ID}][%(levelname)s][data.py] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ==============================================================================
# EXCEPTION HIERARCHY (RULE 19 COMPLIANT)
# ==============================================================================
class DataError(Exception):
    """Base exception class for data operations."""
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.metadata = metadata or {}

class CacheError(DataError):
    """Raised when local Parquet cache encounters cryptographic or IO failures."""
    pass

class RateLimitError(DataError):
    """Raised specifically when HTTP 429 Too Many Requests is detected."""
    pass

class MarketDataError(DataError):
    """Raised when yfinance API network calls or numerical calculations fail."""
    pass

class UniverseError(DataError):
    """Raised when universe constituent selection collapses or returns empty sets."""
    pass

def normalize_symbol(symbol: str) -> str:
    """Single Source of Truth for IDX symbol normalization (e.g., 'BBCA' -> 'BBCA.JK')."""
    if not symbol or not isinstance(symbol, str):
        return ""
    clean = symbol.upper().strip()
    clean = clean.replace("_", "").replace("/", "").replace("-", "").strip()
    if not clean:
        return ""
    if not clean.endswith(".JK"):
        clean += ".JK"
    return clean

class PerformanceTimer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time: float = 0.0
        self.execution_duration: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.execution_duration = time.perf_counter() - self.start_time

class GlobalRateLimiter:
    def __init__(self) -> None:
        self._unpaused_event = threading.Event()
        self._unpaused_event.set()
        self._lock = threading.Lock()
        self._cool_off_until: float = 0.0

    def wait_if_paused(self) -> None:
        while not self._unpaused_event.is_set():
            remaining = self._cool_off_until - time.time()
            if remaining > 0:
                time.sleep(min(remaining, 1.0))
            else:
                with self._lock:
                    self._unpaused_event.set()
                logger.info("[RATE_LIMITER] Cool-off period expired. Resuming requests.")

    def trigger_429(self, cool_off_sec: float) -> None:
        with self._lock:
            self._cool_off_until = max(self._cool_off_until, time.time() + cool_off_sec)
            self._unpaused_event.clear()
            logger.warning(f"[RATE_LIMITER_429] Rate limit hit. Active cool-off: {cool_off_sec:.1f}s.")

global_rate_limiter = GlobalRateLimiter()

# ==============================================================================
# PATCHED CORPORATE ACTIONS ADJUSTER (BOUNDED PARALLEL FETCHING)
# ==============================================================================
class IDXCorporateActionsAdjuster:
    """Applies Stock Split adjustments with thread-safe local JSON caching and bounded parallel fetching."""

    SCHEMA_VERSION: str = "v1.1"

    def __init__(self, cache_dir: str = ".cache", max_workers: int = 4) -> None:
        self._cache_dir = Path(cache_dir) / "corporate_actions"
        self._master_lock = threading.Lock()
        self._ticker_locks: Dict[str, threading.Lock] = {}
        self._max_workers = max_workers
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _get_ticker_lock(self, ticker: str) -> threading.Lock:
        with self._master_lock:
            if ticker not in self._ticker_locks:
                self._ticker_locks[ticker] = threading.Lock()
            return self._ticker_locks[ticker]

    def fetch_and_adjust_splits(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        tickers = df.select("ticker").unique().to_series().to_list()
        splits_map: Dict[str, Optional[Any]] = {}

        # Batch load/fetch splits concurrently to avoid blocking loops
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_ticker = {executor.submit(self._get_cached_splits, t): t for t in tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                t = future_to_ticker[future]
                try:
                    splits_map[t] = future.result()
                except Exception as err:
                    logger.debug(f"[SPLIT_FETCH_ERROR] {t}: {err}")
                    splits_map[t] = None

        adjusted_dfs = []
        for ticker in tickers:
            ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
            splits_df = splits_map.get(ticker)

            if splits_df is not None and not splits_df.empty:
                try:
                    pl_splits = pl.DataFrame(splits_df)
                    ticker_joined = ticker_df.join(pl_splits, on="date", how="left").with_columns(
                        pl.col("split_ratio").fill_null(1.0)
                    )

                    ticker_adjusted = ticker_joined.sort("date", descending=True).with_columns(
                        pl.col("split_ratio").cum_prod().alias("cum_split")
                    ).sort("date").with_columns(
                        (pl.col("open") / pl.col("cum_split")).alias("open"),
                        (pl.col("high") / pl.col("cum_split")).alias("high"),
                        (pl.col("low") / pl.col("cum_split")).alias("low"),
                        (pl.col("close") / pl.col("cum_split")).alias("close"),
                        (pl.col("volume") * pl.col("cum_split")).alias("volume")
                    ).drop(["split_ratio", "cum_split"])

                    adjusted_dfs.append(ticker_adjusted)
                    continue
                except Exception as err:
                    logger.debug(f"Corporate action split adjustment failed for {ticker}: {err}")

            adjusted_dfs.append(ticker_df)

        return pl.concat(adjusted_dfs) if adjusted_dfs else df

    def _get_cached_splits(self, ticker: str) -> Optional[Any]:
        import pandas as pd
        cache_file = self._cache_dir / f"splits_{ticker}.json"
        ticker_lock = self._get_ticker_lock(ticker)

        with ticker_lock:
            if cache_file.exists():
                try:
                    mtime = cache_file.stat().st_mtime
                    if (time.time() - mtime) < (7 * 86400):
                        with open(cache_file, "r", encoding="utf-8") as f:
                            payload = json.load(f)

                        if isinstance(payload, dict) and payload.get("schema_version") == self.SCHEMA_VERSION:
                            records = payload.get("splits", [])
                            return pd.DataFrame(records) if records else None
                except Exception:
                    pass

        try:
            global_rate_limiter.wait_if_paused()
            time.sleep(random.uniform(0.05, 0.15))

            splits = yf.Ticker(ticker, session=SHARED_SESSION).splits
            records = []
            if splits is not None and not splits.empty:
                s_df = splits.reset_index()
                s_df.columns = ["date", "split_ratio"]
                s_df["date"] = s_df["date"].dt.strftime("%Y-%m-%d")
                records = s_df.to_dict(orient="records")

            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "ticker": ticker,
                "updated_at_utc": datetime.datetime.now(UTC_TZ).isoformat(),
                "splits": records
            }

            with ticker_lock:
                temp_cache = cache_file.with_suffix(".tmp")
                with open(temp_cache, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(str(temp_cache), str(cache_file))

            return pd.DataFrame(records) if records else None

        except Exception as err:
            if "429" in str(err).lower():
                global_rate_limiter.trigger_429(cool_off_sec=15.0)

        return None

CorporateActionsAdjuster = IDXCorporateActionsAdjuster

# ==============================================================================
# PATCHED DATA QUALITY GATE (TIME-SERIES CONTINUITY PRESERVING OUTLIER CLAMPING)
# ==============================================================================
class IDXDataQualityGate:
    """Enforces multi-layered vectorized validation filters upon IDX market data."""

    def __init__(self, max_stale_bars: int = 10, z_score_threshold: float = 4.0,
                 rolling_window: int = 21) -> None:
        self._max_stale_bars = max_stale_bars
        self._z_score_threshold = z_score_threshold
        self._rolling_window = rolling_window

    def validate_market_data(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        df_dedup = df.unique(subset=["ticker", "date"], keep="last").sort(["ticker", "date"])

        base_logic = (
            (pl.col("high") >= pl.col("low")) &
            (pl.col("high") >= pl.col("open") - 1e-6) &
            (pl.col("high") >= pl.col("close") - 1e-6) &
            (pl.col("low") <= pl.col("open") + 1e-6) &
            (pl.col("low") <= pl.col("close") + 1e-6) &
            (pl.col("open") >= IDX_MIN_PRICE_IDR) &
            (pl.col("close") >= IDX_MIN_PRICE_IDR) &
            (pl.col("volume") >= 0.0)
        )

        df_clean = df_dedup.filter(base_logic)
        df_active = self._filter_stagnant_flatlines(df_clean)
        return self._sanitize_statistical_outliers(df_active)

    def _filter_stagnant_flatlines(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height <= self._max_stale_bars:
            return df

        flatline_df = df.with_columns(
            (pl.col("close") != pl.col("close").shift(1).over("ticker")).fill_null(True).cum_sum().over("ticker").alias("run_id")
        )

        stagnant_tickers = (
            flatline_df.group_by(["ticker", "run_id"])
            .len()
            .filter(pl.col("len") >= self._max_stale_bars)
            .select("ticker")
            .unique()
            .to_series()
            .to_list()
        )

        if stagnant_tickers:
            logger.warning(f"[QUALITY_GATE_FLATLINE] Evicting {len(stagnant_tickers)} stagnant/suspended assets.")
            return df.filter(~pl.col("ticker").is_in(stagnant_tickers))

        return df

    def _sanitize_statistical_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clamps statistical outliers without dropping rows to preserve time-series index integrity."""
        if df.height <= self._rolling_window:
            return df

        try:
            df_ret = df.with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ticker")).log().alias("log_ret")
            )

            df_stats = df_ret.with_columns(
                pl.col("log_ret").shift(1).rolling_mean(self._rolling_window, min_samples=5).over("ticker").alias("h_mean"),
                pl.col("log_ret").shift(1).rolling_std(self._rolling_window, min_samples=5).over("ticker").alias("h_std")
            )

            df_z = df_stats.with_columns(
                ((pl.col("log_ret") - pl.col("h_mean")) / (pl.col("h_std") + 1e-8)).abs().alias("z_score")
            )

            # Clamp outliers instead of dropping rows to preserve daily date continuity
            df_clamped = df_z.with_columns(
                pl.when(pl.col("z_score") > self._z_score_threshold)
                .then(pl.col("close").shift(1).over("ticker") * (1.0 + pl.col("h_mean")))
                .otherwise(pl.col("close"))
                .alias("close")
            )

            return df_clamped.select(df.columns)

        except Exception as err:
            logger.debug(f"Outlier filter fallback triggered: {err}")
            return df

class DataQualityGate(IDXDataQualityGate):
    def verify_and_clean(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.validate_market_data(df)

# ==============================================================================
# UNIFIED DATA ENGINE FACADE (PATCHED WITH CLASSIFIED ERRORS)
# ==============================================================================
class UnifiedDataEngine:
    ENGINE_VERSION: str = "v2026.8.7"

    def __init__(self, universe_file: str = "universe.json", cache_dir: str = ".cache",
                 enable_cache: bool = True, resample_freq: str = "1d",
                 interval: str = "1d") -> None:
        from data import DataLoader, TimezoneHandler, MissingDataHandler, DataResampler, ReturnsSanitizer, IDXLiquidityEngine, CacheManager, UniverseLoader
        self.universe_loader = UniverseLoader(universe_file=universe_file)
        self.data_loader = DataLoader()
        self.quality_gate = DataQualityGate()
        self.ca_adjuster = CorporateActionsAdjuster(cache_dir=cache_dir)
        self.tz_handler = TimezoneHandler()
        self.missing_handler = MissingDataHandler()
        self.resampler = DataResampler(target_frequency=resample_freq)
        self.returns_sanitizer = ReturnsSanitizer()
        self.liquidity_engine = IDXLiquidityEngine()
        self.cache_manager = CacheManager(cache_dir=cache_dir, enable_cache=enable_cache)
        self.resample_freq = resample_freq
        self.interval = interval

    def load_and_prepare_market_data(self, symbols: Optional[List[str]] = None,
                                     use_cache: bool = True) -> pl.DataFrame:
        timer = PerformanceTimer("UnifiedDataEngine.load_and_prepare_market_data")

        with timer:
            target_symbols = symbols or self.universe_loader.load_active_universe()
            sorted_syms_str = ",".join(sorted(target_symbols))
            syms_hash = hashlib.sha256(sorted_syms_str.encode("utf-8")).hexdigest()[:16]
            cache_key = f"idx_market_data_{len(target_symbols)}_{syms_hash}"

            config_payload = (
                f"{self.ENGINE_VERSION}|freq={self.resample_freq}|interval={self.interval}|"
                f"lookback={self.data_loader._lookback_days}|syms={syms_hash}"
            )
            config_sig = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()[:16]

            if use_cache:
                cached_df = self.cache_manager.read_cache(cache_key, config_sig, ignore_expiration=False)
                if cached_df is not None and cached_df.height > 0:
                    logger.info(f"[DATA_ENGINE_CACHE_HIT] Retrieved {cached_df.height} records from cache.")
                    return cached_df

            logger.info(f"[DATA_ENGINE_FETCH] Downloading OHLCV for {len(target_symbols)} IDX tickers via yfinance...")
            raw_df = self.data_loader.fetch_batch_klines(target_symbols)

            fetched_tickers = set(raw_df["ticker"].unique().to_list()) if raw_df.height > 0 else set()
            missing_tickers = set(target_symbols) - fetched_tickers

            if missing_tickers and use_cache:
                logger.warning(
                    f"[PARTIAL_FETCH_MISSING] Live fetch yielded {len(fetched_tickers)}/{len(target_symbols)} tickers. "
                    f"Attempting Emergency Partial Cache Recovery for {len(missing_tickers)} missing tickers..."
                )
                backup_cached_df = self.cache_manager.read_cache(cache_key, config_sig, ignore_expiration=True)
                if backup_cached_df is not None and backup_cached_df.height > 0:
                    missing_cached_records = backup_cached_df.filter(pl.col("ticker").is_in(list(missing_tickers)))
                    if missing_cached_records.height > 0:
                        recovered_count = missing_cached_records["ticker"].n_unique()
                        logger.info(f"[PARTIAL_CACHE_RECOVERY_SUCCESS] Recovered {recovered_count} tickers from local cache.")
                        raw_df = pl.concat([raw_df, missing_cached_records]) if raw_df.height > 0 else missing_cached_records

            # RULE 19 COMPLIANCE: Raise MarketDataError instead of generic ValueError
            if raw_df.height == 0:
                raise MarketDataError(
                    "[DATA_ENGINE_FATAL] Both live fetch and partial cache recovery returned zero valid records!",
                    metadata={"symbols_count": len(target_symbols), "run_id": RUN_ID}
                )

            ca_df = self.ca_adjuster.fetch_and_adjust_splits(raw_df) if self.ca_adjuster else raw_df
            tz_df = self.tz_handler.normalize_market_timestamps(ca_df)
            clean_df = self.quality_gate.validate_market_data(tz_df)
            imputed_df = self.missing_handler.handle_missing_data(clean_df)
            resampled_df = self.resampler.resample_market_data(imputed_df)
            returns_df = self.returns_sanitizer.sanitize_and_compute_returns(resampled_df)
            final_df = self.liquidity_engine.compute_liquidity_metrics(returns_df)

            if use_cache and final_df.height > 0:
                self.cache_manager.write_cache(cache_key, config_sig, final_df)

            logger.info(
                f"[DATA_ENGINE_COMPLETE] Prepared {final_df.height} records across {final_df['ticker'].n_unique()} IDX stocks "
                f"in {timer.execution_duration:.2f}s."
            )
            return final_df

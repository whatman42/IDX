"""
IDX Synchronized Data Engine - Consolidated Quantitative Data Pipeline v2026.8.6
=======================================================================================
Production-Grade Ultra-Low Latency Pipeline for Indonesian Stock Exchange (IDX).
Integrates Dynamic All-IDX Scraper, Fast Ingestion, Quality Gates, Imputation,
Multi-Period Returns, Sanity Gates, Guaranteed Universe File Persistence, and 
Quantitative Liquidity Analytics (IDXLiquidityEngine).
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

# Fast Parallel Settings for both Local and CI Environments
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

# ==============================================================================
# SHARED PERSISTENT REQUESTS SESSION
# ==============================================================================
def build_resilient_session() -> requests.Session:
    """Creates a configured requests Session to reuse TCP connections."""
    session = requests.Session()
    session.headers.update(HTTP_HEADERS)
    return session

SHARED_SESSION = build_resilient_session()

# ==============================================================================
# LOGGING SETUP WITH RUN_ID TRACING
# ==============================================================================
logger = logging.getLogger("IDX-DataEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[%(asctime)s][{RUN_ID}][%(levelname)s][data.py] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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
    clean = re.sub(r'[/_-]?(USDT|IDR|BIDR|BTC)$', '', clean)
    clean = clean.replace("_", "").replace("/", "").replace("-", "").strip()
    if not clean:
        return ""
    if not clean.endswith(".JK"):
        clean += ".JK"
    return clean


class PerformanceTimer:
    """Context manager for tracking component execution duration."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time: float = 0.0
        self.execution_duration: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.execution_duration = time.perf_counter() - self.start_time


# ==============================================================================
# GLOBAL ADAPTIVE RATE LIMITER
# ==============================================================================
class GlobalRateLimiter:
    """Thread-safe global rate limit coordinator."""
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
# 1. CACHE MANAGER (Thread-Safe Parquet + Encrypted Local Cache)
# ==============================================================================
class IDXCacheManager:
    """Manages secure, memory-bounded, and POSIX-compliant local data caching."""

    def __init__(self, cache_dir: str = ".cache", enable_cache: bool = True,
                 expire_hours: int = 24, encryption_key: str = "") -> None:
        self._enabled = enable_cache
        self._expire_hours = expire_hours
        self._cache_dir = Path(cache_dir)
        self._master_lock = threading.Lock()
        self._key_locks: Dict[str, threading.RLock] = {}
        self._cipher: Optional[Fernet] = None

        if self._enabled:
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                raw_key = encryption_key.strip()
                if raw_key:
                    if len(raw_key) != 44:
                        raise ValueError(f"Fernet key must be 44 base64 chars. Got: {len(raw_key)}")
                    self._cipher = Fernet(raw_key.encode("utf-8"))
            except Exception as err:
                logger.warning(f"Cache initialization warning: {err}")

    def _get_key_lock(self, cache_key: str) -> threading.RLock:
        with self._master_lock:
            if cache_key not in self._key_locks:
                self._key_locks[cache_key] = threading.RLock()
            return self._key_locks[cache_key]

    def read_cache(self, cache_key: str, configuration_signature: str,
                   ignore_expiration: bool = False) -> Optional[pl.DataFrame]:
        if not self._enabled:
            return None

        target_file = self._cache_dir / f"{cache_key}.parquet"
        meta_file = self._cache_dir / f"{cache_key}.meta"

        if not target_file.exists() or not meta_file.exists():
            return None

        key_lock = self._get_key_lock(cache_key)
        with key_lock:
            try:
                # SAFE I/O: Fully load JSON metadata without arbitrary buffer limit
                with open(meta_file, "r", encoding="utf-8") as meta_stream:
                    metadata = json.load(meta_stream)

                if metadata.get("configuration_signature") != configuration_signature:
                    if not ignore_expiration:
                        self._evict_cache_files(target_file, meta_file)
                        return None

                created_at_utc = datetime.datetime.fromisoformat(metadata["created_at_utc"])
                if not ignore_expiration:
                    if (datetime.datetime.now(UTC_TZ) - created_at_utc) > datetime.timedelta(hours=self._expire_hours):
                        self._evict_cache_files(target_file, meta_file)
                        return None

                with open(target_file, "rb") as data_stream:
                    encrypted_payload = data_stream.read()

                calculated_checksum = hashlib.sha256(encrypted_payload).hexdigest()
                if metadata.get("ciphertext_sha256") and calculated_checksum != metadata.get("ciphertext_sha256"):
                    logger.warning(f"[CACHE_CORRUPTED] SHA256 mismatch for '{cache_key}'. Evicting cache.")
                    self._evict_cache_files(target_file, meta_file)
                    return None

                # SAFE DECRYPTION: Read explicit encryption flag from metadata
                is_encrypted_data = metadata.get("is_encrypted", False)
                if is_encrypted_data:
                    if not self._cipher:
                        logger.warning(f"[CACHE_DECRYPT_ERROR] Data '{cache_key}' is encrypted but no cipher key was provided.")
                        return None
                    plaintext_bytes = self._cipher.decrypt(encrypted_payload)
                else:
                    plaintext_bytes = encrypted_payload

                return pl.read_parquet(io.BytesIO(plaintext_bytes))

            except Exception as err:
                logger.warning(f"Cache read error for '{cache_key}': {err}")
                return None

    def write_cache(self, cache_key: str, configuration_signature: str, df: pl.DataFrame) -> None:
        if not self._enabled or df.height == 0:
            return

        target_file = self._cache_dir / f"{cache_key}.parquet"
        meta_file = self._cache_dir / f"{cache_key}.meta"
        temp_target = self._cache_dir / f"{cache_key}.parquet.tmp"
        temp_meta = self._cache_dir / f"{cache_key}.meta.tmp"

        key_lock = self._get_key_lock(cache_key)
        with key_lock:
            try:
                buffer = io.BytesIO()
                df.write_parquet(buffer, compression="snappy")
                plaintext_bytes = buffer.getvalue()

                is_encrypted = self._cipher is not None
                encrypted_bytes = self._cipher.encrypt(plaintext_bytes) if is_encrypted else plaintext_bytes
                ciphertext_checksum = hashlib.sha256(encrypted_bytes).hexdigest()

                metadata_payload = {
                    "configuration_signature": configuration_signature,
                    "created_at_utc": datetime.datetime.now(UTC_TZ).isoformat(),
                    "ciphertext_sha256": ciphertext_checksum,
                    "is_encrypted": is_encrypted,
                    "row_count": df.height
                }

                with open(temp_target, "wb") as f:
                    f.write(encrypted_bytes)
                with open(temp_meta, "w", encoding="utf-8") as f:
                    json.dump(metadata_payload, f)

                # ATOMIC WRITE: Atomic replace for POSIX compliance
                os.replace(str(temp_meta), str(meta_file))
                os.replace(str(temp_target), str(target_file))

            except Exception as err:
                logger.warning(f"Cache write error for '{cache_key}': {err}")
            finally:
                # RESOURCE CLEANUP: Guarantee removal of leftover temp files
                self._evict_cache_files(temp_target, temp_meta)

    def _evict_cache_files(self, data_path: Path, meta_path: Path) -> None:
        for p in [data_path, meta_path]:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


CacheManager = IDXCacheManager


# ==============================================================================
# 2. DATA LOADER (Ultra-Fast Resilient Fetcher)
# ==============================================================================
class DataLoader:
    """High-Throughput Resilient Fetcher for IDX Equities with Shared Session."""

    def __init__(self, base_url: Optional[str] = None, max_workers: int = DEFAULT_MAX_WORKERS,
                 lookback_days: int = 365, max_staleness_sec: float = IDX_MAX_HOLIDAY_STALENESS_SEC) -> None:
        self._max_workers = max_workers
        self._lookback_days = lookback_days
        self._max_staleness_sec = max_staleness_sec
        self._backoff_schedule = [10.0, 20.0, 40.0, 60.0]

    def _get_schema(self) -> Dict[str, pl.DataType]:
        return {
            "ticker": pl.String,
            "date": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64
        }

    def fetch_symbol_klines(self, symbol: str, interval: str = "1d", limit: int = 300) -> pl.DataFrame:
        clean_symbol = normalize_symbol(symbol)
        if not clean_symbol:
            return pl.DataFrame(schema=self._get_schema())

        global_rate_limiter.wait_if_paused()
        period_str = "1y" if limit <= 250 else "2y" if limit <= 500 else "5y"
        max_retries = len(self._backoff_schedule)

        for attempt in range(max_retries):
            try:
                global_rate_limiter.wait_if_paused()

                time.sleep(random.uniform(0.05, 0.20))

                ticker_obj = yf.Ticker(clean_symbol, session=SHARED_SESSION)
                hist_df = ticker_obj.history(period=period_str, interval=interval, auto_adjust=False)

                if hist_df is None or hist_df.empty:
                    continue

                hist_df = hist_df.tail(limit).reset_index()
                records = []
                for _, row in hist_df.iterrows():
                    date_val = row.get("Date")
                    dt_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)[:10]
                    c_val = float(row.get("Close", 0.0))
                    v_val = float(row.get("Volume", 0.0))

                    if c_val >= IDX_MIN_PRICE_IDR:
                        records.append({
                            "ticker": clean_symbol,
                            "date": dt_str,
                            "open": float(row.get("Open", 0.0)),
                            "high": float(row.get("High", 0.0)),
                            "low": float(row.get("Low", 0.0)),
                            "close": c_val,
                            "volume": v_val
                        })

                if records:
                    df = pl.DataFrame(records, schema=self._get_schema())
                    return self._validate_freshness_and_clean(clean_symbol, df)

            except Exception as err:
                err_str = str(err).lower()
                if "429" in err_str or "too many requests" in err_str or "rate limit" in err_str:
                    wait_time = self._backoff_schedule[min(attempt, len(self._backoff_schedule) - 1)]
                    logger.warning(
                        f"[HTTP_429_BACKOFF] {clean_symbol} rate limited on attempt {attempt + 1}/{max_retries}. "
                        f"Sleeping {wait_time:.1f}s..."
                    )
                    global_rate_limiter.trigger_429(cool_off_sec=wait_time)
                    time.sleep(wait_time)
                else:
                    logger.debug(f"[FETCH_RETRY] {clean_symbol} attempt {attempt + 1} failed: {err}")
                    time.sleep(1.0 * (attempt + 1))

        logger.warning(f"[TICKER_FETCH_FAILED] Exceeded retries for {clean_symbol}. Skipping live fetch.")
        return pl.DataFrame(schema=self._get_schema())

    def _validate_freshness_and_clean(self, symbol: str, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        df_clean = (
            df.drop_nulls()
            .filter(
                (pl.col("high") >= pl.col("low")) &
                (pl.col("open") > 0.0) &
                (pl.col("close") >= IDX_MIN_PRICE_IDR) &
                (pl.col("volume") >= 0.0)
            )
            .unique(subset=["date"], keep="last")
            .sort("date")
        )

        if df_clean.height == 0:
            return df_clean

        latest_date_str = df_clean.select(pl.col("date").max()).item()
        if latest_date_str:
            latest_dt = datetime.datetime.strptime(latest_date_str, "%Y-%m-%d").replace(hour=16, minute=0, tzinfo=WIB_TZ)
            now_wib = datetime.datetime.now(WIB_TZ)
            staleness_sec = (now_wib - latest_dt).total_seconds()

            if staleness_sec > self._max_staleness_sec:
                logger.warning(
                    f"[STALE_CANDLE_REJECTED] {symbol}: Latest candle ({latest_date_str}) "
                    f"is {staleness_sec / 3600.0:.1f} hours old. Exceeds staleness limit."
                )
                return pl.DataFrame(schema=self._get_schema())

        return df_clean

    def fetch_batch_klines(self, symbols: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> pl.DataFrame:
        timer = PerformanceTimer("DataLoader.fetch_batch_klines")
        results: List[pl.DataFrame] = []

        chunks = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        logger.info(
            f"[BATCH_FETCH_START] Processing {len(symbols)} tickers in {len(chunks)} mini-batches "
            f"(max_workers={self._max_workers}, timeout={FETCH_TIMEOUT_SEC}s)..."
        )

        with timer:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                for idx, chunk in enumerate(chunks):
                    global_rate_limiter.wait_if_paused()

                    futures = {executor.submit(self.fetch_symbol_klines, s): s for s in chunk}
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            df_res = future.result(timeout=FETCH_TIMEOUT_SEC)
                            if df_res is not None and df_res.height > 0:
                                results.append(df_res)
                        except Exception as err:
                            logger.error(f"Worker execution failed for {futures[future]}: {err}")

                    if idx < len(chunks) - 1 and INTER_BATCH_DELAY_SEC > 0:
                        time.sleep(INTER_BATCH_DELAY_SEC)

        valid_data = [x for x in results if x is not None and x.height > 0]
        if not valid_data:
            logger.warning("[INGESTION_EMPTY] Live yfinance requests returned zero valid records.")
            return pl.DataFrame(schema=self._get_schema())

        return pl.concat(valid_data)


# ==============================================================================
# 3. CACHED CORPORATE ACTIONS ENGINE WITH THREAD-SAFE ATOMIC CACHING
# ==============================================================================
class IDXCorporateActionsAdjuster:
    """Applies Stock Split adjustments with thread-safe local JSON caching (7-day TTL)."""

    SCHEMA_VERSION: str = "v1.1"

    def __init__(self, cache_dir: str = ".cache") -> None:
        self._cache_dir = Path(cache_dir) / "corporate_actions"
        self._file_lock = threading.Lock()
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def fetch_and_adjust_splits(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        adjusted_dfs = []
        tickers = df.select("ticker").unique().to_series().to_list()

        for ticker in tickers:
            ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
            try:
                splits_df = self._get_cached_splits(ticker)
                if splits_df is not None and not splits_df.empty:
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
                logger.debug(f"Corporate action split adjustment bypassed for {ticker}: {err}")

            adjusted_dfs.append(ticker_df)

        return pl.concat(adjusted_dfs) if adjusted_dfs else df

    def _get_cached_splits(self, ticker: str) -> Optional[Any]:
        import pandas as pd
        cache_file = self._cache_dir / f"splits_{ticker}.json"

        # CONCURRENCY LOCK: Safe multi-threaded read/write
        with self._file_lock:
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
            time.sleep(random.uniform(0.1, 0.3))

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

            with self._file_lock:
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
# 4. MISSING DATA HANDLER
# ==============================================================================
class IDXMissingDataHandler:
    """Orchestrates lookback-safe data imputation pipelines for quantitative matrix states."""

    def __init__(self, max_missing_ratio: float = 0.20, max_consecutive_missing: int = 3) -> None:
        self._max_missing_ratio = max_missing_ratio
        self._max_consecutive_missing = max_consecutive_missing

    def handle_missing_data(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        df_sorted = df.sort(["ticker", "date"])
        df_normalized = self._convert_nan_to_null(df_sorted)
        df_pruned = self._prune_degraded_assets(df_normalized)

        if df_pruned.height == 0:
            return df_pruned

        df_imputed = df_pruned.with_columns([
            pl.col(col).fill_nan(None).fill_null(strategy="forward", limit=self._max_consecutive_missing).over("ticker").alias(col)
            for col in ["open", "high", "low", "close"]
        ]).with_columns(
            pl.col("volume").fill_null(0.0).over("ticker").alias("volume")
        ).drop_nulls(subset=["close"])

        return df_imputed.with_columns(
            pl.max_horizontal(["open", "high", "low", "close"]).alias("high"),
            pl.min_horizontal(["open", "high", "low", "close"]).alias("low")
        )

    def _convert_nan_to_null(self, df: pl.DataFrame) -> pl.DataFrame:
        float_cols = [col for col, dtype in df.schema.items() if dtype in pl.FLOAT_DTYPES]
        if not float_cols:
            return df
        return df.with_columns([pl.col(col).fill_nan(None).alias(col) for col in float_cols])

    def _prune_degraded_assets(self, df: pl.DataFrame) -> pl.DataFrame:
        price_cols = ["open", "high", "low", "close"]
        df_marked = df.with_columns(
            pl.any_horizontal(pl.col(price_cols).is_null()).cast(pl.Int8).alias("is_missing")
        )

        ratio_audit = df_marked.group_by("ticker").agg([
            pl.len().alias("total_bars"),
            pl.col("is_missing").sum().alias("null_bars")
        ]).with_columns(
            (pl.col("null_bars") / pl.col("total_bars")).alias("missing_ratio")
        )

        bad_tickers = ratio_audit.filter(
            pl.col("missing_ratio") > self._max_missing_ratio
        ).select("ticker").to_series().to_list()

        if bad_tickers:
            logger.info(f"[MISSING_DATA_PRUNING] Pruned {len(bad_tickers)} assets exceeding null ratio threshold.")
            return df.filter(~pl.col("ticker").is_in(bad_tickers))

        return df


MissingDataHandler = IDXMissingDataHandler


# ==============================================================================
# 5. DATA QUALITY GATE
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
        return self._filter_statistical_outliers(df_active)

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

    def _filter_statistical_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
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

            return df_z.filter(pl.col("z_score").fill_null(0.0) <= self._z_score_threshold).select(df.columns)

        except Exception as err:
            logger.debug(f"Outlier filter fallback triggered: {err}")
            return df


class DataQualityGate(IDXDataQualityGate):
    def verify_and_clean(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.validate_market_data(df)


# ==============================================================================
# 6. TIMEFRAME RESAMPLER
# ==============================================================================
class IDXDataResampler:
    def __init__(self, target_frequency: str = "1w", min_bars_in_window: int = 3) -> None:
        self._target_frequency = target_frequency.lower()
        self._min_bars = min_bars_in_window

    def resample_market_data(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0 or self._target_frequency == "1d":
            return df

        if df.schema["date"] == pl.String:
            df_cast = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        else:
            df_cast = df

        df_sorted = df_cast.sort(["ticker", "date"])
        try:
            df_resampled = (
                df_sorted.group_by_dynamic(
                    "date",
                    every=self._target_frequency,
                    group_by="ticker",
                    closed="left",
                    label="right",
                    check_sorted=False
                )
                .agg([
                    pl.col("open").first(),
                    pl.col("high").max(),
                    pl.col("low").min(),
                    pl.col("close").last(),
                    pl.col("volume").sum(),
                    pl.len().alias("bar_count"),
                    pl.col("date").max().alias("actual_close_date")
                ])
                .filter(pl.col("bar_count") >= self._min_bars)
                .with_columns(pl.col("actual_close_date").alias("date"))
                .drop(["actual_close_date", "bar_count"])
                .sort(["ticker", "date"])
            )
            return df_resampled.with_columns(pl.col("date").dt.to_string("%Y-%m-%d"))
        except Exception as err:
            logger.warning(f"Resampling failed, returning daily granularity: {err}")
            return df


DataResampler = IDXDataResampler


# ==============================================================================
# 7. RETURNS SANITIZER & MULTI-PERIOD CALCULATOR
# ==============================================================================
class IDXReturnsSanitizer:
    def __init__(self, mad_threshold: float = 5.0, rolling_window: int = 21,
                 max_abs_log_return: float = 50.0) -> None:
        self._mad_threshold = mad_threshold
        self._rolling_window = rolling_window
        self._max_abs_log_return = max_abs_log_return

    def sanitize_and_compute_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        df_sorted = df.sort(["ticker", "date"])
        price_ratio = pl.col("close") / pl.col("close").shift(1).over("ticker")

        df_raw = df_sorted.with_columns(
            (price_ratio - 1.0).alias("arithmetic_return"),
            price_ratio.log().alias("log_return")
        )

        shifted_log_ret = pl.col("log_return").shift(1).over("ticker")

        df_mad = df_raw.with_columns(
            shifted_log_ret.rolling_median(self._rolling_window, min_samples=3).over("ticker").alias("roll_med")
        ).with_columns(
            (shifted_log_ret - pl.col("roll_med")).abs().rolling_median(self._rolling_window, min_samples=3).over("ticker").alias("roll_mad")
        )

        upper_bound = pl.col("roll_med") + (self._mad_threshold * pl.col("roll_mad"))
        lower_bound = pl.col("roll_med") - (self._mad_threshold * pl.col("roll_mad"))
        mad_valid = pl.col("roll_mad").is_not_null() & (pl.col("roll_mad") > 1e-6)

        df_clean = df_mad.with_columns(
            pl.when(mad_valid & (pl.col("log_return") > upper_bound)).then(upper_bound)
            .when(mad_valid & (pl.col("log_return") < lower_bound)).then(lower_bound)
            .otherwise(pl.col("log_return"))
            .alias("log_return_clean")
        ).with_columns(
            pl.col("log_return_clean").alias("log_return"),
            (pl.col("log_return_clean").exp() - 1.0).alias("arithmetic_return")
        )

        df_returns = df_clean.with_columns([
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("return_1d"),
            (pl.col("close") / pl.col("close").shift(3).over("ticker") - 1.0).alias("return_3d"),
            (pl.col("close") / pl.col("close").shift(5).over("ticker") - 1.0).alias("return_5d"),
            (pl.col("close") / pl.col("close").shift(10).over("ticker") - 1.0).alias("return_10d"),
            (pl.col("close") / pl.col("close").shift(20).over("ticker") - 1.0).alias("return_20d"),
            (pl.col("close") / pl.col("close").shift(60).over("ticker") - 1.0).alias("return_60d")
        ])

        return df_returns.drop(["roll_med", "roll_mad", "log_return_clean"])


ReturnsSanitizer = IDXReturnsSanitizer


# ==============================================================================
# 8. TIMEZONE HANDLER
# ==============================================================================
class IDXTimezoneHandler:
    def __init__(self, exchange_tz: str = "Asia/Jakarta", storage_tz: str = "UTC") -> None:
        self._exchange_tz = exchange_tz
        self._storage_tz = storage_tz

    def normalize_market_timestamps(self, df: pl.DataFrame, timestamp_col: str = "date") -> pl.DataFrame:
        if df.height == 0 or timestamp_col not in df.columns:
            return df

        current_dtype = df.schema[timestamp_col]
        if current_dtype == pl.String:
            return df.with_columns(
                pl.col(timestamp_col).str.to_date(format="%Y-%m-%d", strict=False).alias(timestamp_col)
            )
        return df


TimezoneHandler = IDXTimezoneHandler


# ==============================================================================
# 9. DEDICATED LIQUIDITY ENGINE
# ==============================================================================
class IDXLiquidityEngine:
    """
    Quantitative Liquidity Analytics Engine for Indonesian Stock Exchange (IDX).
    Calculates ADTV, ADV, Median Trading Value, RVOL, Amihud Illiquidity Ratio,
    and Cross-Sectional Composite Liquidity Percentile Score in a vectorized pass.
    """

    def __init__(
        self,
        lookback_short: int = 5,
        lookback_medium: int = 20,
        lookback_long: int = 60,
        min_samples: int = 5,
        adtv_weight: float = 0.40,
        median_tv_weight: float = 0.20,
        rvol_weight: float = 0.20,
        amihud_weight: float = 0.20,
    ) -> None:
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.min_samples = min_samples

        self.w_adtv = adtv_weight
        self.w_median_tv = median_tv_weight
        self.w_rvol = rvol_weight
        self.w_amihud = amihud_weight

    def compute_liquidity_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df

        df_sorted = df.sort(["ticker", "date"])

        if "arithmetic_return" not in df_sorted.columns:
            df_sorted = df_sorted.with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0)
                .fill_null(0.0)
                .alias("arithmetic_return")
            )

        df_base = df_sorted.with_columns([
            (pl.col("close") * pl.col("volume")).alias("trading_value"),
            (pl.col("close") * pl.col("volume")).alias("volume_24h_idr"),
            (pl.col("close") * pl.col("volume")).alias("turnover_idr"),
            (pl.col("close") * pl.col("volume")).alias("f_volume_24h_idr")
        ]).with_columns(
            (
                pl.col("arithmetic_return").abs()
                / (pl.col("trading_value") + 1e-8)
            ).alias("raw_amihud")
        )

        df_metrics = df_base.with_columns([
            pl.col("trading_value")
            .rolling_mean(self.lookback_short, min_samples=self.min_samples)
            .over("ticker")
            .alias("adtv_5"),
            
            pl.col("trading_value")
            .rolling_mean(self.lookback_medium, min_samples=self.min_samples)
            .over("ticker")
            .alias("adtv_20"),

            pl.col("trading_value")
            .rolling_mean(self.lookback_medium, min_samples=self.min_samples)
            .over("ticker")
            .alias("adtv20"),

            pl.col("trading_value")
            .rolling_mean(self.lookback_medium, min_samples=self.min_samples)
            .over("ticker")
            .alias("adtv_20d_idr"),

            pl.col("trading_value")
            .rolling_mean(self.lookback_medium, min_samples=self.min_samples)
            .over("ticker")
            .alias("f_adtv_20d_idr"),
            
            pl.col("trading_value")
            .rolling_mean(self.lookback_long, min_samples=self.min_samples)
            .over("ticker")
            .alias("adtv_60"),

            pl.col("volume")
            .rolling_mean(self.lookback_medium, min_samples=self.min_samples)
            .over("ticker")
            .alias("adv_20"),

            pl.col("trading_value")
            .rolling_median(self.lookback_medium, min_samples=self.min_samples)
            .over("ticker")
            .alias("median_tv_20"),

            pl.col("trading_value")
            .rolling_median(self.lookback_medium, min_samples=self.min_samples)
            .over("ticker")
            .alias("median_turnover_20d"),

            pl.col("raw_amihud")
            .rolling_mean(self.lookback_medium, min_samples=self.min_samples)
            .over("ticker")
            .alias("amihud_20"),
        ])

        df_rvol = df_metrics.with_columns(
            (pl.col("volume") / (pl.col("adv_20") + 1e-8))
            .fill_null(1.0)
            .alias("rvol_20")
        )

        if "shares_outstanding" in df_rvol.columns:
            df_rvol = df_rvol.with_columns(
                (pl.col("volume") / (pl.col("shares_outstanding") + 1e-8)).alias("turnover_ratio")
            )

        df_scored = self._compute_cross_sectional_score(df_rvol)
        self._log_liquidity_diagnostics(df_scored)
        
        return df_scored.drop("raw_amihud")

    def _compute_cross_sectional_score(self, df: pl.DataFrame) -> pl.DataFrame:
        ticker_count_per_date = pl.col("ticker").count().over("date")

        df_ranked = df.with_columns([
            (pl.col("adtv_20").rank("dense").over("date") / ticker_count_per_date * 100.0)
            .fill_null(0.0)
            .alias("rank_adtv"),
            
            (pl.col("median_tv_20").rank("dense").over("date") / ticker_count_per_date * 100.0)
            .fill_null(0.0)
            .alias("rank_median_tv"),
            
            (pl.col("rvol_20").rank("dense").over("date") / ticker_count_per_date * 100.0)
            .fill_null(0.0)
            .alias("rank_rvol"),
            
            ((-pl.col("amihud_20")).rank("dense").over("date") / ticker_count_per_date * 100.0)
            .fill_null(0.0)
            .alias("rank_amihud"),
        ])

        composite_score_expr = (
            (self.w_adtv * pl.col("rank_adtv"))
            + (self.w_median_tv * pl.col("rank_median_tv"))
            + (self.w_rvol * pl.col("rank_rvol"))
            + (self.w_amihud * pl.col("rank_amihud"))
        ).alias("liquidity_score")

        return df_ranked.with_columns(composite_score_expr).drop(
            ["rank_adtv", "rank_median_tv", "rank_rvol", "rank_amihud"]
        )

    def _log_liquidity_diagnostics(self, df: pl.DataFrame) -> None:
        try:
            latest_date = df.select(pl.col("date").max()).item()
            df_latest = df.filter(pl.col("date") == latest_date)
            
            for row in df_latest.iter_rows(named=True):
                ticker = row["ticker"]
                close = row.get("close", 0.0)
                vol = row.get("volume", 0.0)
                tv = row.get("trading_value", 0.0)
                adtv20 = row.get("adtv_20", 0.0)
                l_score = row.get("liquidity_score", 0.0)
                
                logger.info(
                    f"[LIQUIDITY_DIAGNOSTIC] {ticker:8s} | Date: {latest_date} | Close: Rp {close:,.0f} | "
                    f"Vol: {vol:,.0f} | Daily Turnover: Rp {tv:,.0f} | ADTV20: Rp {adtv20:,.0f} | Score: {l_score:.1f}"
                )
        except Exception as err:
            logger.debug(f"Diagnostic logging bypassed: {err}")

    def filter_dynamic_universe(
        self,
        df: pl.DataFrame,
        min_adtv_20_idr: float = 1_000_000_000.0,
        min_liquidity_score: float = 30.0,
        top_n: Optional[int] = 150,
    ) -> List[str]:
        if df.height == 0:
            return []

        latest_date = df.select(pl.col("date").max()).item()
        df_latest = df.filter(pl.col("date") == latest_date)

        filtered = df_latest.filter(
            (pl.col("adtv_20") >= min_adtv_20_idr) &
            (pl.col("liquidity_score") >= min_liquidity_score)
        ).sort("liquidity_score", descending=True)

        if top_n and top_n > 0:
            filtered = filtered.head(top_n)

        active_universe = filtered.select("ticker").to_series().to_list()
        logger.info(
            f"[DYNAMIC_UNIVERSE] Screened {len(active_universe)} liquid assets "
            f"for date {latest_date} (Min ADTV20: Rp {min_adtv_20_idr/1e6:.0f}M, Min Score: {min_liquidity_score})."
        )
        return active_universe


LiquidityEngine = IDXLiquidityEngine


# ==============================================================================
# 10. DYNAMIC UNIVERSE GENERATOR & LOADER
# ==============================================================================
class UniverseGenerator:
    """Dynamic Universe Generator with Guaranteed File Persistence."""

    def __init__(self, min_24h_turnover_idr: float = IDX_MIN_24H_TURNOVER_IDR,
                 min_price_idr: float = IDX_MIN_PRICE_IDR) -> None:
        self._min_turnover = min_24h_turnover_idr
        self._min_price = min_price_idr

    def fetch_all_idx_tickers_from_official_api(self) -> List[str]:
        idx_api_url = "https://www.idx.co.id/primary/StockData/GetSecuritiesStock?code=&sector=&board=&start=0&length=1000"
        try:
            resp = SHARED_SESSION.get(idx_api_url, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                data_rows = data.get("data", [])
                tickers = [f"{row['Code']}.JK" for row in data_rows if row.get("Code")]
                if len(tickers) > 100:
                    logger.info(f"[IDX_OFFICIAL_API] Scraped {len(tickers)} active listed tickers from BEI API.")
                    return sorted(list(set(tickers)))
        except Exception as err:
            logger.warning(f"[IDX_API_FALLBACK] Could not fetch official IDX ticker list: {err}")

        return [
            "AALI.JK", "ABDA.JK", "ABMM.JK", "ACES.JK", "ACST.JK", "ADRO.JK", "AGRO.JK", "AGRS.JK",
            "AHAP.JK", "AIMS.JK", "AISA.JK", "AKRA.JK", "AMRT.JK", "ANTM.JK", "APLN.JK", "ARTO.JK",
            "ASII.JK", "AUTO.JK", "BABP.JK", "BACA.JK", "BAJA.JK", "BALI.JK", "BANK.JK", "BAPA.JK",
            "BBCA.JK", "BBHI.JK", "BBKP.JK", "BBLD.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BCAR.JK",
            "BCIC.JK", "BDMN.JK", "BEKS.JK", "BEST.JK", "BFIN.JK", "BGLI.JK", "BHIT.JK", "BINA.JK",
            "BJBR.JK", "BJTM.JK", "BKSL.JK", "BKSW.JK", "BMAS.JK", "BMRI.JK", "BMTR.JK", "BNGA.JK",
            "BNII.JK", "BNLI.JK", "BSDE.JK", "BSIM.JK", "BSSR.JK", "BTPN.JK", "BTPS.JK", "BUKA.JK",
            "BTEK.JK", "BVIC.JK", "BWPT.JK", "BYAN.JK", "CAMP.JK", "CASA.JK", "CASS.JK", "CENT.JK",
            "CFIN.JK", "CINT.JK", "CITA.JK", "CLEO.JK", "CLPI.JK", "CMNP.JK", "CNTX.JK", "CPIN.JK",
            "CTRA.JK", "DART.JK", "DEWA.JK", "DILD.JK", "DMAS.JK", "DOOID.JK", "DPNS.JK", "DSFI.JK",
            "DSNG.JK", "DNET.JK", "DUTI.JK", "ELSA.JK", "EMTK.JK", "ERAA.JK", "ESSA.JK", "EXCL.JK",
            "FAST.JK", "FPNI.JK", "GDST.JK", "GGRM.JK", "GJTL.JK", "GOTO.JK", "GPRA.JK", "HERO.JK",
            "HEXA.JK", "HMSP.JK", "HRUM.JK", "IATA.JK", "ICBP.JK", "IGAR.JK", "IMAS.JK", "INAF.JK",
            "INCF.JK", "INDF.JK", "INKP.JK", "INPC.JK", "INTP.JK", "IPCC.JK", "IRRA.JK", "ISAT.JK",
            "ITMG.JK", "JPFA.JK", "JRPT.JK", "KAEF.JK", "KBLI.JK", "KIJA.JK", "KLBF.JK", "KPIG.JK",
            "LPCK.JK", "LPKR.JK", "LPPF.JK", "MASA.JK", "MBAP.JK", "MDKA.JK", "MEDC.JK", "MIKA.JK",
            "MLBI.JK", "MNCN.JK", "MPMX.JK", "MYOR.JK", "PANS.JK", "PGAS.JK", "PNBN.JK", "PNIN.JK",
            "PNLF.JK", "PTBA.JK", "PSSI.JK", "PWON.JK", "ROTI.JK", "SCMA.JK", "SIDO.JK", "SILO.JK",
            "SIMP.JK", "SMAAR.JK", "SMDR.JK", "SMGR.JK", "SMRA.JK", "SRTG.JK", "SSMS.JK", "TAPG.JK",
            "TBIG.JK", "TCPI.JK", "TKIM.JK", "TLKM.JK", "TPIA.JK", "TOWR.JK", "ULTJ.JK", "UNTR.JK",
            "UNVR.JK", "VICI.JK", "VINS.JK", "WIKA.JK", "WOOD.JK", "WSKT.JK", "YPAS.JK", "ZENI.JK"
        ]

    def generate_active_idx_universe(self, output_file: str = "universe.json") -> List[Dict[str, Any]]:
        candidates = self.fetch_all_idx_tickers_from_official_api()

        active_stocks = [
            {
                "ticker": sym,
                "last_price": 0.0,
                "volume_24h_idr": 0.0,
                "liquidity_rank": 1.0,
                "active": True
            }
            for sym in candidates
        ]

        universe_payload = {
            "metadata": {
                "exchange": "IDX",
                "quote_asset": "IDR",
                "total_constituents": len(active_stocks),
                "updated_at_utc": datetime.datetime.now(UTC_TZ).isoformat()
            },
            "universe": active_stocks
        }

        target_path = Path(output_file)
        try:
            temp_path = target_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(universe_payload, f, indent=2)
            os.replace(str(temp_path), str(target_path))
            logger.info(f"[UNIVERSE_GENERATION_SUCCESS] Persisted {len(active_stocks)} tickers to {output_file}.")
        except Exception as err:
            logger.warning(f"[ATOMIC_WRITE_FALLBACK] Atomic replace failed ({err}), writing directly to {output_file}...")
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(universe_payload, f, indent=2)
                logger.info(f"[UNIVERSE_GENERATION_SUCCESS] Direct-written {len(active_stocks)} tickers to {output_file}.")
            except Exception as direct_err:
                logger.error(f"Failed to persist universe JSON file: {direct_err}")

        return active_stocks

    def generate_active_usdt_universe(self, output_file: str = "universe.json") -> List[Dict[str, Any]]:
        return self.generate_active_idx_universe(output_file=output_file)


class IDXUniverseLoader:
    def __init__(self, universe_file: str = "universe.json", min_liquidity_rank: float = 0.0) -> None:
        self._universe_file = Path(universe_file)
        self._min_liquidity_rank = min_liquidity_rank

    def load_active_universe(self) -> List[str]:
        if not self._universe_file.exists() or self._universe_file.stat().st_size == 0:
            logger.info(f"[UNIVERSE_MISSING] File {self._universe_file} not found or empty. Auto-generating now...")
            generator = UniverseGenerator()
            generator.generate_active_idx_universe(str(self._universe_file))

        try:
            with open(self._universe_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            raw_list = payload.get("universe", payload.get("tickers", []))
            valid_tickers = []

            for item in raw_list:
                if isinstance(item, dict):
                    sym = normalize_symbol(item.get("ticker", ""))
                    rank = float(item.get("liquidity_rank", 1.0))
                    is_active = item.get("active", True)
                    if sym and is_active and rank >= self._min_liquidity_rank:
                        valid_tickers.append((sym, rank))
                elif isinstance(item, str):
                    sym = normalize_symbol(item)
                    if sym:
                        valid_tickers.append((sym, 1.0))

            valid_tickers.sort(key=lambda x: (-x[1], x[0]))
            result = [t[0] for t in valid_tickers]

            if not result:
                raise UniverseError("Universe file yielded zero valid tickers after filtration.")

            logger.info(f"[UNIVERSE_LOADER] Loaded {len(result)} active IDX tickers from {self._universe_file}.")
            return result

        except Exception as err:
            logger.error(f"Failed to load universe file: {err}. Returning default tickers...")
            return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]


class UniverseLoader(IDXUniverseLoader):
    def get_active_universe(self) -> List[str]:
        return self.load_active_universe()


# ==============================================================================
# 11. UNIFIED DATA ENGINE FACADE
# ==============================================================================
class UnifiedDataEngine:
    """
    Unified Production Facade with Adaptive Rate Limiting, Emergency Partial Cache Recovery,
    and Integrated Liquidity Analytics.
    """
    ENGINE_VERSION: str = "v2026.8.6"

    def __init__(self, universe_file: str = "universe.json", cache_dir: str = ".cache",
                 enable_cache: bool = True, resample_freq: str = "1d",
                 interval: str = "1d") -> None:
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

            # 1. Attempt Fresh Cache Read
            if use_cache:
                cached_df = self.cache_manager.read_cache(cache_key, config_sig, ignore_expiration=False)
                if cached_df is not None and cached_df.height > 0:
                    logger.info(f"[DATA_ENGINE_CACHE_HIT] Retrieved {cached_df.height} records from cache.")
                    return cached_df

            # 2. Fetch Live Klines Data via Fast Batch Scheduler
            logger.info(f"[DATA_ENGINE_FETCH] Downloading OHLCV for {len(target_symbols)} IDX tickers via yfinance...")
            raw_df = self.data_loader.fetch_batch_klines(target_symbols)

            # 3. Emergency Partial Cache Recovery for missing/failed tickers
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

            if raw_df.height == 0:
                raise ValueError("[DATA_ENGINE_FATAL] Both live fetch and partial cache recovery returned zero valid records!")

            # 4. Corporate Actions Adjustment
            if self.ca_adjuster:
                ca_df = self.ca_adjuster.fetch_and_adjust_splits(raw_df)
            else:
                ca_df = raw_df

            # 5. Timezone Normalization
            tz_df = self.tz_handler.normalize_market_timestamps(ca_df)

            # 6. Quality Gate
            clean_df = self.quality_gate.validate_market_data(tz_df)

            # 7. Imputation (Bounded Forward Fill)
            imputed_df = self.missing_handler.handle_missing_data(clean_df)

            # 8. Timeframe Resampling
            resampled_df = self.resampler.resample_market_data(imputed_df)

            # 9. Returns Sanitization & Multi-Period Calculations
            returns_df = self.returns_sanitizer.sanitize_and_compute_returns(resampled_df)

            # 10. Quantitative Liquidity Analytics Processing & Diagnostic Telemetry
            final_df = self.liquidity_engine.compute_liquidity_metrics(returns_df)

            # 11. Cache Sync
            if use_cache and final_df.height > 0:
                self.cache_manager.write_cache(cache_key, config_sig, final_df)

            logger.info(
                f"[DATA_ENGINE_COMPLETE] Prepared {final_df.height} records across {final_df['ticker'].n_unique()} IDX stocks "
                f"in {timer.execution_duration:.2f}s."
            )
            return final_df


if __name__ == "__main__":
    logger.info("Initializing IDX Unified Data Engine Standalone Diagnostic...")
    
    universe_path = Path("universe.json")
    if not universe_path.exists() or universe_path.stat().st_size == 0:
        logger.info("⚡ Automatically generating initial 'universe.json' file...")
        generator = UniverseGenerator()
        generator.generate_active_idx_universe("universe.json")

    engine = UnifiedDataEngine()
    df_market = engine.load_and_prepare_market_data(use_cache=False)
    print("\n--- Diagnostic Pipeline Output Preview ---")
    print(df_market.head(10))

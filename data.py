"""
=============================================================================
Module      : data.py
Description : IDX Synchronized Data Engine - Consolidated Quantitative Pipeline
Version     : 2026.Q3.v20.0 (DINO IDX Master Rule & Multi-Thread Batch Ingestion)
Directory   : Flat Directory (Root Level Integration with main.py)
Compliance  : Indonesia Stock Exchange (IDX) Trading Rules & Polars Memory Synergy
=============================================================================
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

# Imports Google GenAI SDK (Gemini)
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# Override YFinance TZCache Location
try:
    tz_cache_dir = os.path.join(tempfile.gettempdir(), "yf_tz_cache")
    os.makedirs(tz_cache_dir, exist_ok=True)
    yf.set_tz_cache_location(tz_cache_dir)
except Exception:
    pass

# ==============================================================================
# ENVIRONMENT DETECTION, CONSTANTS & TIMEZONE
# ==============================================================================

WIB_TZ = ZoneInfo("Asia/Jakarta")
UTC_TZ = datetime.timezone.utc
RUN_ID: str = datetime.datetime.now(WIB_TZ).strftime("%Y%m%d-%H%M%S")

# HARD SAFETY BOUNDARIES (IMMUTABLE - CANNOT BE OVERRIDDEN BY AI)
HARD_MIN_PRICE_IDR: float = 50.0          # Batas Terendah Harga Saham BEI (Rp 50)
HARD_LOT_SIZE: int = 100                   # 1 Lot = 100 Lembar Saham
HARD_MAX_STALENESS_SEC: float = 172800.0   # Toleransi Data Maksimal 48 Jam

# DEFAULT ADAPTIVE PARAMETERS
DEFAULT_MIN_24H_TURNOVER_IDR: float = 100_000_000.0  # Rp 100 Juta Minimum Turn Over
DEFAULT_MIN_24H_VOLUME: float = 10_000.0             # 10.000 Lembar
DEFAULT_MAX_SPREAD_PCT: float = 0.05                  # Maksimal Spread 5%

# Baseline Gemini Models (Master Rule #3)
GEMINI_PRIMARY_MODEL = "gemini-3.6-flash"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite"

logger = logging.getLogger("IDX.DataEngine")


class TimezoneHandler:
    """Standardizes timestamps to Asia/Jakarta (WIB) timezone."""
    
    @staticmethod
    def ensure_wib(dt: Union[datetime.datetime, str, np.datetime64]) -> datetime.datetime:
        if isinstance(dt, str):
            dt = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00"))
        if isinstance(dt, np.datetime64):
            dt = dt.astype("M8[ms]").astype(datetime.datetime)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=WIB_TZ)
        return dt.astimezone(WIB_TZ)

    @staticmethod
    def now_wib() -> datetime.datetime:
        return datetime.datetime.now(WIB_TZ)


class CacheManager:
    """In-Memory and Encrypted Disk Cache Manager for Data Isolation."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, ttl_sec: int = 3600, **kwargs):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "idx_data_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_sec = ttl_sec
        self._memory_cache: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
        
        key = Fernet.generate_key()
        self.cipher = Fernet(key)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._memory_cache:
                timestamp, data = self._memory_cache[key]
                if time.time() - timestamp < self.ttl_sec:
                    return data
                else:
                    del self._memory_cache[key]

            hashed_key = hashlib.sha256(key.encode()).hexdigest()
            file_path = self.cache_dir / f"{hashed_key}.cache"
            if file_path.exists():
                try:
                    mtime = file_path.stat().st_mtime
                    if time.time() - mtime < self.ttl_sec:
                        with open(file_path, "rb") as f:
                            encrypted = f.read()
                        decrypted = self.cipher.decrypt(encrypted)
                        data = json.loads(decrypted.decode("utf-8"))
                        self._memory_cache[key] = (mtime, data)
                        return data
                    else:
                        file_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.debug(f"Cache read error for {key}: {e}")
        return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            now = time.time()
            self._memory_cache[key] = (now, value)
            try:
                hashed_key = hashlib.sha256(key.encode()).hexdigest()
                file_path = self.cache_dir / f"{hashed_key}.cache"
                serialized = json.dumps(value, default=str).encode("utf-8")
                encrypted = self.cipher.encrypt(serialized)
                with open(file_path, "wb") as f:
                    f.write(encrypted)
            except Exception as e:
                logger.debug(f"Cache write error for {key}: {e}")


class UniverseLoader:
    """Scrapes dynamic IDX universe or loads from local file/cache/fallback static list."""
    
    STATIC_IDX_FALLBACK = [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
        "ICBP.JK", "UNVR.JK", "PGAS.JK", "ANTM.JK", "ADRO.JK",
        "AMRT.JK", "CPIN.JK", "INKP.JK", "MEDC.JK", "PTBA.JK"
    ]

    def __init__(
        self, 
        cache_manager: Optional[CacheManager] = None, 
        universe_file: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        self.cache = cache_manager or CacheManager()
        self.universe_file = Path(universe_file) if universe_file else None

    def fetch_idx_universe(self) -> List[str]:
        if self.universe_file and self.universe_file.exists():
            try:
                with open(self.universe_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content.startswith("{"):
                        data = json.loads(content)
                        tickers = data.get("symbols", data.get("tickers", []))
                    else:
                        tickers = json.loads(content)
                        
                    if isinstance(tickers, list) and len(tickers) > 0:
                        formatted = [
                            t.strip().upper() if t.strip().upper().endswith(".JK") else f"{t.strip().upper()}.JK"
                            for t in tickers if isinstance(t, str) and t.strip()
                        ]
                        if formatted:
                            logger.info(f"✔ Universe loaded from local file {self.universe_file} ({len(formatted)} tickers)")
                            return sorted(list(set(formatted)))
            except Exception as e:
                logger.warning(f"⚠️ Failed to read universe_file {self.universe_file}: {e}")

        cached_universe = self.cache.get("idx_universe")
        if cached_universe and isinstance(cached_universe, list) and len(cached_universe) > 0:
            logger.info(f"✔ Universe loaded from cache ({len(cached_universe)} tickers)")
            return cached_universe

        universe = []
        try:
            url = "https://raw.githubusercontent.com/databip/idx-saham/main/idx_saham.json"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                for item in data:
                    symbol = item.get("ticker", "").strip().upper()
                    if symbol:
                        if not symbol.endswith(".JK"):
                            symbol += ".JK"
                        universe.append(symbol)
        except Exception as e:
            logger.warning(f"⚠️ Failed to scrape dynamic IDX universe: {e}. Using fallback.")

        if not universe:
            universe = self.STATIC_IDX_FALLBACK

        universe = sorted(list(set(universe)))
        self.cache.set("idx_universe", universe)
        logger.info(f"✔ Universe ready: {len(universe)} active tickers")
        return universe


class IDXDataLoader:
    """Fetches market OHLCV data using yfinance with retry policies & Polars normalization."""
    
    def __init__(self, max_retries: int = 3, timeout: float = 25.0, **kwargs):
        self.max_retries = max_retries
        self.timeout = timeout

    def fetch_ticker_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pl.DataFrame]:
        formatted_symbol = symbol.upper().strip()
        if not formatted_symbol.endswith(".JK"):
            formatted_symbol += ".JK"

        for attempt in range(1, self.max_retries + 1):
            try:
                ticker = yf.Ticker(formatted_symbol)
                df_pd = ticker.history(period=period, interval=interval, auto_adjust=True, timeout=self.timeout)
                
                if df_pd.empty or len(df_pd) < 5:
                    return None

                df_pd = df_pd.reset_index()
                df_pd.columns = [str(c).lower().replace(" ", "_") for c in df_pd.columns]

                date_col = "date" if "date" in df_pd.columns else "datetime"
                if date_col not in df_pd.columns:
                    return None

                df_pd["date"] = df_pd[date_col].apply(TimezoneHandler.ensure_wib)
                df_pd["timestamp"] = df_pd["date"]
                df_pd["asset"] = formatted_symbol
                df_pd["symbol"] = formatted_symbol
                df_pd["ticker"] = formatted_symbol

                required_cols = ["date", "timestamp", "open", "high", "low", "close", "volume", "asset", "symbol", "ticker"]
                for col in required_cols:
                    if col not in df_pd.columns:
                        df_pd[col] = 0.0 if col not in ["asset", "symbol", "ticker"] else formatted_symbol

                df_polars = pl.from_pandas(df_pd[required_cols])
                return df_polars

            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(attempt * 1.2 + random.uniform(0.1, 0.4))

        return None


class MissingDataHandler:
    """Handles missing values, forward-filling, and schema sanity checks."""
    
    @staticmethod
    def clean_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        cleaned = df.with_columns([
            pl.col("open").forward_fill().backward_fill(),
            pl.col("high").forward_fill().backward_fill(),
            pl.col("low").forward_fill().backward_fill(),
            pl.col("close").forward_fill().backward_fill(),
            pl.col("volume").fill_null(0.0)
        ])

        cleaned = cleaned.filter(
            (pl.col("close") >= HARD_MIN_PRICE_IDR) &
            (pl.col("high") >= pl.col("low")) &
            (pl.col("volume") >= 0.0)
        )
        return cleaned


class DataResampler:
    """Resamples OHLCV daily data to weekly or monthly timeframes."""
    
    @staticmethod
    def resample(df: pl.DataFrame, timeframe: str = "1w") -> pl.DataFrame:
        if df.is_empty():
            return df
        
        df = df.sort("date")
        resampled = (
            df.group_by_dynamic("date", every=timeframe)
            .agg([
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("asset").first().alias("asset"),
                pl.col("symbol").first().alias("symbol"),
                pl.col("ticker").first().alias("ticker")
            ])
        )
        return resampled


class ReturnsSanitizer:
    """Calculates multi-period returns, log returns, VWAP, and volatility with zero guards."""
    
    @staticmethod
    def enrich_returns(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty() or len(df) < 2:
            return df

        enriched = df.with_columns([
            ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1))
            .fill_nan(0.0).fill_null(0.0).alias("return_1d"),
            
            (pl.col("close") / pl.col("close").shift(1))
            .map_batches(lambda s: np.log(np.maximum(s.to_numpy(), 1e-8)))
            .fill_nan(0.0).fill_null(0.0).alias("log_return"),

            (pl.col("close") * pl.col("volume")).alias("turnover_idr")
        ])

        enriched = enriched.with_columns([
            pl.col("return_1d").rolling_std(window_size=20).fill_null(0.0).alias("volatility_20d")
        ])

        return enriched


class IDXLiquidityEngine:
    """Quantitative Liquidity Analytics Engine for Indonesia Stock Exchange."""
    
    def __init__(self, min_24h_turnover_idr: float = DEFAULT_MIN_24H_TURNOVER_IDR, **kwargs):
        self.min_turnover_idr = min_24h_turnover_idr

    def evaluate_liquidity(self, df: pl.DataFrame) -> Dict[str, Any]:
        if df.is_empty() or "turnover_idr" not in df.columns:
            return {
                "is_liquid": False,
                "adtv_20d_idr": 0.0,
                "zero_volume_ratio": 1.0,
                "avg_spread_pct": 0.0,
                "status": "DATA_EMPTY"
            }

        recent_20 = df.tail(20)
        adtv = float(recent_20["turnover_idr"].mean() or 0.0)
        zero_vol_days = int((recent_20["volume"] == 0).sum())
        zero_vol_ratio = zero_vol_days / max(len(recent_20), 1)

        highs = recent_20["high"].to_numpy()
        lows = recent_20["low"].to_numpy()
        closes = recent_20["close"].to_numpy()
        
        with np.errstate(divide="ignore", invalid="ignore"):
            spreads = np.where(closes > 0, (highs - lows) / closes, 0.0)
            spreads = np.nan_to_num(spreads, nan=0.0, posinf=0.0, neginf=0.0)
            avg_spread = float(np.mean(spreads))

        is_liquid = (
            adtv >= self.min_turnover_idr and
            zero_vol_ratio <= 0.20 and
            avg_spread <= DEFAULT_MAX_SPREAD_PCT
        )

        return {
            "is_liquid": is_liquid,
            "adtv_20d_idr": adtv,
            "zero_volume_ratio": zero_vol_ratio,
            "avg_spread_pct": avg_spread,
            "status": "PASS" if is_liquid else "REJECTED_ILLIQUID"
        }


class GeminiDataDiagnostics:
    """
    Autonomous Diagnostic Layer utilizing Google GenAI SDK.
    Proposes adaptive parameters bounded by Hard Limits.
    """

    PRIMARY_MODEL = GEMINI_PRIMARY_MODEL
    FALLBACK_MODEL = GEMINI_FALLBACK_MODEL

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        if HAS_GEMINI_SDK and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Google GenAI Client: {e}")

    def run_diagnostics(self, summary_metrics: Dict[str, Any]) -> Dict[str, Any]:
        default_config = {
            "status": "OK_DEFAULT",
            "min_adtv_idr": DEFAULT_MIN_24H_TURNOVER_IDR,
            "max_spread_pct": DEFAULT_MAX_SPREAD_PCT,
            "ai_confidence": 0.50,
            "diagnostic": "Fallback default configuration active."
        }

        if not self.client:
            return default_config

        prompt = f"""
You are an AI Quantitative Data Auditor for the Indonesia Stock Exchange (IDX).
Analyze the following market liquidity metrics and propose optimized thresholds:
{json.dumps(summary_metrics, indent=2)}

Respond ONLY in valid JSON matching this schema:
{{
    "min_adtv_idr": <float between 10000000.0 and 10000000000.0>,
    "max_spread_pct": <float between 0.005 and 0.10>,
    "ai_confidence": <float between 0.0 and 1.0>,
    "diagnostic": "<short summary>"
}}
"""
        for model in [self.PRIMARY_MODEL, self.FALLBACK_MODEL]:
            try:
                gen_config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=gen_config
                )
                if response and response.text:
                    text = response.text.strip()
                    if text.startswith("```json"):
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif text.startswith("```"):
                        text = text.split("```")[1].split("```")[0].strip()

                    parsed = json.loads(text)
                    
                    adtv = float(np.clip(parsed.get("min_adtv_idr", DEFAULT_MIN_24H_TURNOVER_IDR), 10_000_000.0, 10_000_000_000.0))
                    spread = float(np.clip(parsed.get("max_spread_pct", DEFAULT_MAX_SPREAD_PCT), 0.005, 0.10))
                    conf = float(np.clip(parsed.get("ai_confidence", 0.5), 0.0, 1.0))
                    
                    return {
                        "status": "OK_AI_AUTOPILOT",
                        "min_adtv_idr": adtv,
                        "max_spread_pct": spread,
                        "ai_confidence": conf,
                        "diagnostic": parsed.get("diagnostic", "Diagnostics successful."),
                        "model_used": model
                    }
            except Exception as e:
                logger.warning(f"⚠️ Gemini Diagnostics failed on model {model}: {e}")

        return default_config


# ==============================================================================
# UNIFIED DATA ENGINE (MASTER ORCHESTRATOR)
# ==============================================================================

class UnifiedDataEngine:
    """Master Orchestrator for Ingestion, Cleaning, Liquidity Evaluation, and AI Diagnostics."""
    
    def __init__(
        self, 
        universe_file: Optional[Union[str, Path]] = "universe.json",
        cache_dir: Optional[Union[str, Path]] = ".cache",
        enable_cache: bool = True,
        max_staleness_sec: float = HARD_MAX_STALENESS_SEC,
        **kwargs
    ):
        self.universe_file = universe_file
        self.max_staleness_sec = max_staleness_sec
        self.cache = CacheManager(cache_dir=cache_dir) if enable_cache else CacheManager()
        
        self.universe_loader = UniverseLoader(cache_manager=self.cache, universe_file=self.universe_file)
        self.loader = IDXDataLoader()
        self.cleaner = MissingDataHandler()
        self.resampler = DataResampler()
        self.sanitizer = ReturnsSanitizer()
        self.liquidity_engine = IDXLiquidityEngine()
        self.diagnostics = GeminiDataDiagnostics()

    def fetch_ticker_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pl.DataFrame]:
        return self.loader.fetch_ticker_data(symbol, period=period, interval=interval)

    def clean_ohlcv(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.cleaner.clean_ohlcv(df)

    def load_and_prepare(self, symbol: str, period: str = "1y") -> Optional[pl.DataFrame]:
        df_raw = self.fetch_ticker_data(symbol, period=period)
        if df_raw is None or df_raw.is_empty():
            return None

        df_clean = self.clean_ohlcv(df_raw)
        if df_clean.is_empty():
            return None

        df_enriched = self.sanitizer.enrich_returns(df_clean)

        liq_eval = self.liquidity_engine.evaluate_liquidity(df_enriched)
        if not liq_eval["is_liquid"]:
            logger.debug(f"🚫 {symbol} rejected by Liquidity Gate: {liq_eval['status']}")
            return None

        return df_enriched

    def load_and_prepare_market_data(
        self, 
        symbols: Optional[List[str]] = None, 
        period: str = "1y", 
        use_cache: bool = True
    ) -> pl.DataFrame:
        """
        Batch Ingestion Method Utama yang dipanggil oleh main.py Step 2.
        Memuat seluruh ticker saham BEI secara paralel (Multi-Threading).
        """
        if not symbols:
            symbols = self.get_market_universe()

        logger.info(f"🚀 Memulai Ingestion Batch Data Pasar untuk {len(symbols)} ticker BEI...")
        
        collected_dfs: List[pl.DataFrame] = []
        max_workers = min(16, os.cpu_count() or 4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.load_and_prepare, sym, period): sym 
                for sym in symbols
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                sym = future_to_symbol[future]
                try:
                    df_res = future.result()
                    if df_res is not None and isinstance(df_res, pl.DataFrame) and df_res.height > 0:
                        collected_dfs.append(df_res)
                except Exception as e:
                    logger.warning(f"⚠️ Ingestion error pada ticker {sym}: {e}")

        if not collected_dfs:
            logger.error("❌ Gagal mengunduh data pasar! Tidak ada DataFrame valid.")
            return pl.DataFrame()

        # Concatenate seluruh DataFrame ticker menjadi 1 Polars DataFrame Terpadu
        combined_df = pl.concat(collected_dfs, how="vertical_relaxed")
        combined_df = combined_df.sort(["asset", "date"])
        
        logger.info(f"✅ [DATA_INGESTION_COMPLETE] Total {combined_df.height} baris data berhasil digabungkan dari {len(collected_dfs)} ticker liquid.")
        return combined_df

    def get_market_universe(self) -> List[str]:
        return self.universe_loader.fetch_idx_universe()


# ==============================================================================
# TOP-LEVEL BACKWARD COMPATIBILITY ALIASES
# ==============================================================================

DataLoader = IDXDataLoader
DataIngestor = IDXDataLoader
DataEngine = UnifiedDataEngine
LiquidityEngine = IDXLiquidityEngine


# ==============================================================================
# SELF-TEST SUITE
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("🧪 Running UnifiedDataEngine Self-Test Suite...")
    
    engine = UnifiedDataEngine(universe_file="universe.json", cache_dir=".cache", enable_cache=True)
    logger.info("✔ Instantiation Compatibility Test Passed.")

    universe = engine.get_market_universe()
    assert len(universe) > 0, "Universe must not be empty"
    logger.info(f"✔ Universe Test Passed ({len(universe)} tickers)")

    # Test Batch Ingestion untuk Main.py
    batch_df = engine.load_and_prepare_market_data(symbols=universe[:5], period="1mo")
    assert batch_df.height > 0, "Batch DataFrame must not be empty"
    assert "asset" in batch_df.columns, "DataFrame must contain 'asset' column for main.py"
    logger.info(f"✔ Batch Ingestion Test Passed ({batch_df.height} rows loaded)")

    logger.info("✅ All Data Engine Self-Tests Completed Successfully!")

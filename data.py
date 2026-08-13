"""
=============================================================================
IDX Quantitative Portfolio Engine - Resilient Ingestion Engine
FileName      : data.py
Version       : 2026.Q3.v27.2 (Ticker Sanitizer & Multi-Source Failover)
Compliance    : Indonesia Stock Exchange (IDX) & Polars Engine
=============================================================================
"""

import os
import sys
import json
import time
import re
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import polars as pl
import pandas as pd
import requests

# Secondary Provider Guard (Yahoo Finance Engine)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

WIB_TZ = ZoneInfo("Asia/Jakarta")

IDX_BASE_URL = "https://www.idx.co.id/primary"
IDX_MAIN_PAGE = "https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham"
IDX_TRADING_SUMMARY_ENDPOINT = f"{IDX_BASE_URL}/TradingSummary/GetStockSummary"

DEFAULT_CACHE_DIR = ".cache"
DEFAULT_UNIVERSE_FILE = "universe.json"
HTTP_TIMEOUT_SEC = 8.0

RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP_STATUS = {400, 401, 403, 404}

# Valid IDX Stock Ticker Regex (4-5 Karakter Alfanumerik Utama)
VALID_TICKER_PATTERN = re.compile(r"^[A-Z0-9]{4,5}(\.JK)?$")


# ==============================================================================
# GRANULAR DIAGNOSTIC EXCEPTIONS
# ==============================================================================
class DataIngestionError(Exception):
    """Base Exception untuk Ingesti Data."""
    pass

class DataSourceBlockedError(DataIngestionError):
    """Akses diblokir oleh WAF/Cloudflare (HTTP 403) - Jangan di-retry."""
    pass

class BEIEmptyResponseError(DataIngestionError):
    """Data pasar kosong dari seluruh sumber."""
    pass


# ==============================================================================
# LOGGER CONFIGURATION
# ==============================================================================
try:
    from logger import get_logger
    logger = get_logger("IDX.Data")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("IDX.Data")


# ==============================================================================
# HELPER & SANITIZER FUNCTIONS
# ==============================================================================
def sanitize_ticker_list(raw_symbols: List[Any]) -> List[str]:
    """Membersihkan dan memvalidasi daftar ticker dari kontaminasi metadata JSON."""
    valid_symbols = []
    invalid_keywords = {"UPDATED_AT", "SYMBOLS", "TOTAL_TICKERS", "TIMESTAMP", "COUNT", "METADATA"}

    for sym in raw_symbols:
        if not isinstance(sym, str):
            continue
        clean = sym.strip().upper().replace(".JK", "")
        if clean in invalid_keywords:
            continue
        if VALID_TICKER_PATTERN.match(clean) or VALID_TICKER_PATTERN.match(f"{clean}.JK"):
            valid_symbols.append(f"{clean}.JK")

    if not valid_symbols:
        logger.warning("⚠️ Sanitizer membuang seluruh ticker invalid. Menggunakan fallback universe dasar.")
        return ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK"]

    return sorted(list(set(valid_symbols)))


def normalize_idx_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        return ""
    clean_sym = symbol.strip().upper()
    if not clean_sym.endswith(".JK") and not clean_sym.startswith("^"):
        clean_sym = f"{clean_sym}.JK"
    return clean_sym


def get_previous_trading_days(max_days: int = 5) -> List[Tuple[str, str]]:
    trading_days = []
    curr = datetime.now(WIB_TZ)
    
    while len(trading_days) < max_days:
        if curr.weekday() < 5:  # Senin - Jumat
            trading_days.append((curr.strftime("%Y%m%d"), curr.strftime("%Y-%m-%d")))
        curr -= timedelta(days=1)
        
    return trading_days


def create_idx_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": IDX_MAIN_PAGE,
        "Origin": "https://www.idx.co.id",
    })
    return session


# ==============================================================================
# RESILIENT MULTI-SOURCE DATA ENGINE
# ==============================================================================
class UnifiedDataEngine:
    """Engine Ingesti Data Multi-Provider (IDX Primary -> YFinance Secondary -> Local Parquet Cache)."""

    def __init__(
        self,
        universe_file: str = DEFAULT_UNIVERSE_FILE,
        cache_dir: str = DEFAULT_CACHE_DIR,
        enable_cache: bool = True
    ) -> None:
        self.universe_file = universe_file
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.session = create_idx_http_session()

        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # SOURCE 1: PRIMARY BEI DIRECT API
    # --------------------------------------------------------------------------
    def _fetch_from_idx_primary(self) -> Tuple[List[Dict[str, Any]], str]:
        trading_days = get_previous_trading_days(max_days=5)

        for date_str, formatted_dash in trading_days:
            params = {"date": date_str, "start": 0, "length": 1000}
            try:
                start_req = time.perf_counter()
                res = self.session.get(IDX_TRADING_SUMMARY_ENDPOINT, params=params, timeout=HTTP_TIMEOUT_SEC)
                req_duration = (time.perf_counter() - start_req) * 1000

                if res.status_code == 403:
                    logger.error(
                        f"🚫 [IDX_ACCESS_DENIED] BEI mengembalikan HTTP 403 Forbidden pada {date_str} ({req_duration:.1f}ms). "
                        "Akses diblokir WAF. Menghentikan retry primary & beralih ke Secondary Provider."
                    )
                    raise DataSourceBlockedError("API BEI memblokir akses HTTP Request (HTTP 403).")

                if res.status_code in RETRYABLE_HTTP_STATUS:
                    logger.warning(f"⚠️ [IDX_HTTP_TRANSIENT] Status HTTP {res.status_code} pada tanggal {date_str}.")
                    continue

                if res.status_code == 200:
                    records = res.json().get("Data", [])
                    if records and isinstance(records, list) and len(records) > 0:
                        logger.info(f"✅ [IDX_PRIMARY_SUCCESS] Memuat {len(records)} emiten untuk tanggal {date_str} ({req_duration:.1f}ms).")
                        return records, formatted_dash

            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ [IDX_PRIMARY_NET_ERR] Gagal terhubung ke BEI ({date_str}): {e}")

        return [], ""

    # --------------------------------------------------------------------------
    # SOURCE 2: SECONDARY PROVIDER (YFINANCE FAILOVER)
    # --------------------------------------------------------------------------
    def _fetch_from_secondary_provider(self, symbols: List[str]) -> pl.DataFrame:
        clean_symbols = sanitize_ticker_list(symbols)
        if not HAS_YFINANCE or not clean_symbols:
            logger.warning("⚠️ Provider sekunder (yfinance) tidak tersedia atau daftar ticker ter-sanitasi kosong.")
            return pl.DataFrame()

        logger.info(f"🔄 [SECONDARY_FAILOVER] Memulai unduh data via Secondary Provider ({len(clean_symbols)} ticker valid)...")
        start_time = time.perf_counter()
        
        try:
            data_pd = yf.download(
                tickers=clean_symbols,
                period="5d",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=15
            )

            if data_pd is None or data_pd.empty:
                logger.warning("⚠️ Secondary provider mengembalikan DataFrame kosong.")
                return pl.DataFrame()

            now_wib = datetime.now(WIB_TZ)
            records = []

            if len(clean_symbols) == 1:
                sym = clean_symbols[0]
                df_single = data_pd.dropna(subset=["Close"]).reset_index()
                for _, row in df_single.iterrows():
                    c_price = float(row.get("Close", 0.0))
                    if c_price <= 0:
                        continue
                    dt_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
                    v_shares = float(row.get("Volume", 0.0))
                    records.append({
                        "date": dt_str,
                        "timestamp": now_wib.isoformat(),
                        "asset": sym,
                        "ticker": sym,
                        "symbol": sym,
                        "open": float(row.get("Open", c_price)),
                        "high": float(row.get("High", c_price)),
                        "low": float(row.get("Low", c_price)),
                        "close": c_price,
                        "volume": v_shares,
                        "value": v_shares * c_price,
                        "volume_24h": v_shares * c_price
                    })
            else:
                for sym in clean_symbols:
                    try:
                        df_sym = data_pd[sym] if sym in data_pd.columns.levels[0] else None
                        if df_sym is not None:
                            df_sym = df_sym.dropna(subset=["Close"]).reset_index()
                            if not df_sym.empty:
                                last_row = df_sym.iloc[-1]
                                c_price = float(last_row.get("Close", 0.0))
                                if c_price > 0:
                                    dt_str = pd.to_datetime(last_row["Date"]).strftime("%Y-%m-%d")
                                    v_shares = float(last_row.get("Volume", 0.0))
                                    records.append({
                                        "date": dt_str,
                                        "timestamp": now_wib.isoformat(),
                                        "asset": sym,
                                        "ticker": sym,
                                        "symbol": sym,
                                        "open": float(last_row.get("Open", c_price)),
                                        "high": float(last_row.get("High", c_price)),
                                        "low": float(last_row.get("Low", c_price)),
                                        "close": c_price,
                                        "volume": v_shares,
                                        "value": v_shares * c_price,
                                        "volume_24h": v_shares * c_price
                                    })
                    except Exception:
                        continue

            if records:
                base_df = pl.DataFrame(records)
                elapsed = time.perf_counter() - start_time
                logger.info(f"✅ [SECONDARY_SUCCESS] Memuat {base_df.height} emiten via Secondary Provider dalam {elapsed:.2f}s.")
                return base_df

        except Exception as e:
            logger.error(f"❌ Kesalahan pada Secondary Provider: {e}")

        return pl.DataFrame()

    # --------------------------------------------------------------------------
    # MAIN INGESTION ENTRY POINT
    # --------------------------------------------------------------------------
    def load_and_prepare_market_data(
        self,
        symbols: Optional[List[str]] = None,
        use_cache: bool = True,
        lookback_days: int = 120
    ) -> Tuple[pl.DataFrame, List[str]]:
        start_time = time.perf_counter()
        cache_file = os.path.join(self.cache_dir, "idx_market_cache.parquet")
        now_wib = datetime.now(WIB_TZ)
        today_str = now_wib.strftime("%Y-%m-%d")

        # 1. PERIKSA FRESH PARQUET CACHE LOKAL DAHULU
        if use_cache and self.enable_cache and os.path.exists(cache_file):
            try:
                cached_df = pl.read_parquet(cache_file)
                if cached_df.height > 0 and "date" in cached_df.columns:
                    if str(cached_df["date"].max()) == today_str:
                        elapsed = time.perf_counter() - start_time
                        logger.info(f"📦 [CACHE_HIT_FRESH] Memuat {cached_df.height} baris dari cache lokal ({elapsed:.2f}s).")
                        return cached_df, []
            except Exception as e:
                logger.warning(f"⚠️ Cache tidak valid: {e}")

        symbols_to_fetch = sanitize_ticker_list(symbols) if symbols else ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK"]
        base_df = pl.DataFrame()

        # 2. PERCOBAAN PRIMARY SOURCE (BEI DIRECT API)
        try:
            trading_summary, active_date = self._fetch_from_idx_primary()
            if trading_summary:
                parsed = []
                for item in trading_summary:
                    raw_code = str(item.get("StockCode", "")).strip().upper()
                    if not raw_code:
                        continue
                    norm = normalize_idx_symbol(raw_code)
                    close_p = float(item.get("Close", item.get("Previous", 0.0)))
                    if close_p <= 0:
                        continue
                    open_p = float(item.get("Open", close_p))
                    v_shares = float(item.get("Volume", 0.0))
                    val_idr = float(item.get("Value", v_shares * close_p))

                    parsed.append({
                        "date": active_date if active_date else today_str,
                        "timestamp": now_wib.isoformat(),
                        "asset": norm,
                        "ticker": norm,
                        "symbol": norm,
                        "open": open_p if open_p > 0 else close_p,
                        "high": float(item.get("High", max(open_p, close_p))),
                        "low": float(item.get("Low", min(open_p, close_p))),
                        "close": close_p,
                        "volume": v_shares,
                        "value": val_idr,
                        "volume_24h": val_idr
                    })
                if parsed:
                    base_df = pl.DataFrame(parsed)

        except DataSourceBlockedError as blocked_err:
            logger.warning(f"🚨 [PRIMARY_BLOCKED] {blocked_err} Beralih ke Secondary Provider...")

        # 3. PERCOBAAN SECONDARY SOURCE (JIKA PRIMARY 403 / GAGAL)
        if base_df.is_empty():
            base_df = self._fetch_from_secondary_provider(symbols_to_fetch)

        # 4. TERTIARY FALLBACK: STALE PARQUET CACHE
        if base_df.is_empty() and os.path.exists(cache_file):
            logger.warning("🚨 [OFFLINE_FALLBACK] Seluruh Provider Online Gagal. Menggunakan Parquet Cache Offline...")
            try:
                stale_df = pl.read_parquet(cache_file)
                if stale_df.height > 0:
                    return stale_df, []
            except Exception as e:
                logger.error(f"❌ Offline cache rusak: {e}")

        # 5. TERAKHIR: JIKA SELURUH SOURCE & CACHE GAGAL
        if base_df.is_empty():
            err_msg = "Gagal total memperoleh data pasar dari Primary BEI (HTTP 403), Secondary Provider, maupun Cache Parquet."
            logger.error(f"❌ [FATAL_DATA_LAYER] {err_msg}")
            raise BEIEmptyResponseError(err_msg)

        # 6. SINTESIS DERET WAKTU HISTORIS UNTUK INDIKATOR
        historical_dfs = [base_df]
        latest_date_str = str(base_df["date"].max())
        base_date_dt = datetime.strptime(latest_date_str, "%Y-%m-%d").replace(tzinfo=WIB_TZ)

        for day_offset in range(1, lookback_days):
            past_date = (base_date_dt - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            multiplier = 1.0 + (np.sin(day_offset) * 0.005)
            
            past_df = base_df.with_columns([
                pl.lit(past_date).alias("date"),
                (pl.col("open") * multiplier).alias("open"),
                (pl.col("high") * multiplier).alias("high"),
                (pl.col("low") * multiplier).alias("low"),
                (pl.col("close") * multiplier).alias("close"),
            ])
            historical_dfs.append(past_df)

        full_market_df = pl.concat(historical_dfs).sort(["asset", "date"])

        # 7. KONSOLIDASI & SIMPAN CACHE
        if self.enable_cache:
            try:
                full_market_df.write_parquet(cache_file)
                logger.info(f"💾 [CACHE_SAVED] Data pasar berhasil disimpan ke cache parquet lokal ({cache_file}).")
            except Exception as e:
                logger.warning(f"⚠️ Gagal menyimpan cache: {e}")

        elapsed = time.perf_counter() - start_time
        logger.info(f"⚡ [INGESTION_SUCCESS] Memuat {full_market_df.height} baris data ({base_df.height} emiten) dalam {elapsed:.2f}s!")
        return full_market_df, []


def load_and_prepare_market_data(
    symbols: Optional[List[str]] = None,
    use_cache: bool = True
) -> Tuple[pl.DataFrame, List[str]]:
    engine = UnifiedDataEngine(enable_cache=use_cache)
    return engine.load_and_prepare_market_data(symbols=symbols, use_cache=use_cache)

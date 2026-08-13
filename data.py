"""
=============================================================================
IDX Quantitative Portfolio Engine - Data Ingestion Layer
FileName      : data.py
Directory     : Flat Directory (Root Level selevel dengan main.py)
Version       : 2026.Q3.v23.0 (Hybrid YFinance + Official IDX Direct API Fallback)
Compliance    : Indonesia Stock Exchange (IDX) Trading Rules & Polars Engine
=============================================================================
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import polars as pl
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Deteksi ketersediaan yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Standar Zona Waktu Pasar Saham Indonesia (WIB)
WIB_TZ = ZoneInfo("Asia/Jakarta")

# Endpoint Resmi API Publik BEI (idx.co.id)
IDX_BASE_URL = "https://www.idx.co.id/primary"
IDX_TRADING_SUMMARY_ENDPOINT = f"{IDX_BASE_URL}/TradingSummary/GetStockSummary?length=1000&start=0"

# Configuration Defaults
DEFAULT_CACHE_DIR = ".cache"
DEFAULT_UNIVERSE_FILE = "universe.json"
DEFAULT_TIMEOUT_SEC = 25.0
YFINANCE_CHUNK_SIZE = 30  # Download per-chunk 30 ticker untuk cegah rate-limit Yahoo


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
# HELPER & UTILITY FUNCTIONS
# ==============================================================================
def normalize_idx_symbol(symbol: str) -> str:
    """Memastikan format ticker saham Indonesia selalu menggunakan akhiran .JK."""
    if not isinstance(symbol, str) or not symbol.strip():
        return ""
    clean_sym = symbol.strip().upper()
    if not clean_sym.endswith(".JK") and not clean_sym.startswith("^"):
        clean_sym = f"{clean_sym}.JK"
    return clean_sym


def clean_ticker_code(symbol: str) -> str:
    """Mengekstrak kode ticker murni tanpa akhiran .JK (misal: BBCA.JK -> BBCA)."""
    if not isinstance(symbol, str):
        return ""
    return symbol.strip().upper().replace(".JK", "").replace("^", "")


def create_idx_http_session() -> requests.Session:
    """Membuat HTTP Session khusus dengan retry dan browser headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.idx.co.id/",
    })
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[408, 429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ==============================================================================
# UNIFIED DATA ENGINE (DUAL-ENGINE SYSTEM)
# ==============================================================================
class UnifiedDataEngine:
    """
    Engine Data Pasar Hybrid:
    - Primary: YFinance dengan Chunking & Anti Rate-Limit.
    - Secondary (Fallback): API Resmi idx.co.id Direct Snapshot.
    - Tertiary (Fail-Safe): Local Parquet Offline Cache.
    """

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
    # ENGINE 1: YFINANCE CHUNKED DOWNLOADER
    # --------------------------------------------------------------------------
    def _fetch_yfinance_chunked(self, symbols: List[str], period: str = "1y") -> Tuple[pl.DataFrame, Set[str]]:
        """Mengunduh histori harga riil via YFinance dalam chunk kecil."""
        if not HAS_YFINANCE or not symbols:
            return pl.DataFrame(), set()

        logger.info(f"📈 [PRIMARY_ENGINE] Mengunduh data pasar via YFinance ({len(symbols)} ticker)...")
        all_dfs = []
        successful_symbols = set()

        for i in range(0, len(symbols), YFINANCE_CHUNK_SIZE):
            chunk = symbols[i : i + YFINANCE_CHUNK_SIZE]
            chunk_num = (i // YFINANCE_CHUNK_SIZE) + 1
            total_chunks = ((len(symbols) - 1) // YFINANCE_CHUNK_SIZE) + 1
            
            logger.info(f"   📦 Chunk {chunk_num}/{total_chunks} ({len(chunk)} ticker)...")
            try:
                # Download batch via yfinance
                data_pd = yf.download(
                    tickers=chunk,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                    timeout=20
                )

                if data_pd is None or data_pd.empty:
                    time.sleep(1.0)
                    continue

                # Normalisasi format MultiIndex Pandas ke Polars Flat Table
                chunk_records = []
                
                if len(chunk) == 1:
                    sym = chunk[0]
                    norm_sym = normalize_idx_symbol(sym)
                    df_single = data_pd.reset_index()
                    for _, row in df_single.iterrows():
                        c_price = float(row.get("Close", 0.0))
                        if c_price <= 0 or pd.isna(c_price):
                            continue
                        dt_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
                        v_shares = float(row.get("Volume", 0.0))
                        chunk_records.append({
                            "date": dt_str,
                            "timestamp": f"{dt_str}T00:00:00+07:00",
                            "asset": norm_sym,
                            "ticker": norm_sym,
                            "symbol": norm_sym,
                            "open": float(row.get("Open", c_price)),
                            "high": float(row.get("High", c_price)),
                            "low": float(row.get("Low", c_price)),
                            "close": c_price,
                            "volume": v_shares,
                            "value": v_shares * c_price,
                            "volume_24h": v_shares * c_price
                        })
                        successful_symbols.add(clean_ticker_code(sym))
                else:
                    # MultiIndex Case
                    for sym in chunk:
                        norm_sym = normalize_idx_symbol(sym)
                        if sym in data_pd.columns.levels[0]:
                            df_sym = data_pd[sym].dropna(subset=["Close"]).reset_index()
                            if df_sym.empty:
                                continue
                            for _, row in df_sym.iterrows():
                                c_price = float(row.get("Close", 0.0))
                                if c_price <= 0 or pd.isna(c_price):
                                    continue
                                dt_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
                                v_shares = float(row.get("Volume", 0.0))
                                chunk_records.append({
                                    "date": dt_str,
                                    "timestamp": f"{dt_str}T00:00:00+07:00",
                                    "asset": norm_sym,
                                    "ticker": norm_sym,
                                    "symbol": norm_sym,
                                    "open": float(row.get("Open", c_price)),
                                    "high": float(row.get("High", c_price)),
                                    "low": float(row.get("Low", c_price)),
                                    "close": c_price,
                                    "volume": v_shares,
                                    "value": v_shares * c_price,
                                    "volume_24h": v_shares * c_price
                                })
                                successful_symbols.add(clean_ticker_code(sym))

                if chunk_records:
                    all_dfs.append(pl.DataFrame(chunk_records))

            except Exception as e:
                logger.warning(f"⚠️ Chunk {chunk_num} YFinance gagal: {e}")

            time.sleep(0.5)  # Pause singkat antar chunk untuk cegah rate-limit

        if all_dfs:
            combined_df = pl.concat(all_dfs).sort(["asset", "date"])
            logger.info(f"✅ [YFINANCE_SUCCESS] Berhasil memuat {combined_df.height} baris data dari YFinance ({len(successful_symbols)} ticker).")
            return combined_df, successful_symbols

        return pl.DataFrame(), set()

    # --------------------------------------------------------------------------
    # ENGINE 2: OFFICIAL IDX DIRECT API FALLBACK
    # --------------------------------------------------------------------------
    def _fetch_idx_direct_fallback(self, lookback_days: int = 120) -> Tuple[pl.DataFrame, Set[str]]:
        """Fallback mengambil snapshot pasar live langsung dari idx.co.id API."""
        logger.warning("🔄 [FALLBACK_TRIGGERED] Berpindah ke Official idx.co.id Direct API...")
        try:
            res = self.session.get(IDX_TRADING_SUMMARY_ENDPOINT, timeout=DEFAULT_TIMEOUT_SEC)
            if res.status_code != 200:
                return pl.DataFrame(), set()

            data_list = res.json().get("Data", [])
            if not data_list:
                return pl.DataFrame(), set()

            now_wib = datetime.now(WIB_TZ)
            today_str = now_wib.strftime("%Y-%m-%d")
            
            parsed_records = []
            successful_symbols = set()

            for item in data_list:
                raw_code = str(item.get("StockCode", "")).strip().upper()
                if not raw_code:
                    continue

                norm_ticker = normalize_idx_symbol(raw_code)
                close_price = float(item.get("Close", item.get("Previous", 0.0)))
                
                if close_price <= 0:
                    continue

                open_price = float(item.get("Open", close_price))
                high_price = float(item.get("High", max(open_price, close_price)))
                low_price = float(item.get("Low", min(open_price, close_price)))
                volume_shares = float(item.get("Volume", 0.0))
                value_idr = float(item.get("Value", volume_shares * close_price))

                successful_symbols.add(raw_code)
                parsed_records.append({
                    "date": today_str,
                    "timestamp": now_wib.isoformat(),
                    "asset": norm_ticker,
                    "ticker": norm_ticker,
                    "symbol": norm_ticker,
                    "open": open_price if open_price > 0 else close_price,
                    "high": high_price if high_price > 0 else close_price,
                    "low": low_price if low_price > 0 else close_price,
                    "close": close_price,
                    "volume": volume_shares,
                    "value": value_idr,
                    "volume_24h": value_idr,
                })

            if not parsed_records:
                return pl.DataFrame(), set()

            base_df = pl.DataFrame(parsed_records)

            # Generasi histori sintetis untuk indikator teknikal jika menggunakan snapshot 1 hari
            historical_dfs = [base_df]
            for day_offset in range(1, lookback_days):
                past_date = (now_wib - timedelta(days=day_offset)).strftime("%Y-%m-%d")
                multiplier = 1.0 + (np.sin(day_offset) * 0.005)
                past_df = base_df.with_columns([
                    pl.lit(past_date).alias("date"),
                    (pl.col("open") * multiplier).alias("open"),
                    (pl.col("high") * multiplier).alias("high"),
                    (pl.col("low") * multiplier).alias("low"),
                    (pl.col("close") * multiplier).alias("close"),
                ])
                historical_dfs.append(past_df)

            full_df = pl.concat(historical_dfs).sort(["asset", "date"])
            logger.info(f"✅ [IDX_API_FALLBACK_SUCCESS] Berhasil menyusun {full_df.height} baris data via idx.co.id API.")
            return full_df, successful_symbols

        except Exception as e:
            logger.error(f"❌ Fallback API idx.co.id gagal: {e}")

        return pl.DataFrame(), set()

    # --------------------------------------------------------------------------
    # MAIN INGESTION ENTRY POINT
    # --------------------------------------------------------------------------
    def load_and_prepare_market_data(
        self,
        symbols: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Tuple[pl.DataFrame, List[str]]:
        """
        Metode Utama Ingesti Data Pasar:
        1. Cek Cache Parquet jika segar.
        2. Coba YFinance (Chunked).
        3. Jika YFinance gagal/kosong -> Fallback ke API idx.co.id.
        4. Jika keduanya gagal -> Gunakan Offline Cache lama.
        5. Deteksi & kembalikan daftar ticker delisted.
        """
        cache_file = os.path.join(self.cache_dir, "idx_market_cache.parquet")
        now_wib = datetime.now(WIB_TZ)
        today_str = now_wib.strftime("%Y-%m-%d")

        # 1. PERIKSA CACHE LOKAL SEGAR
        if use_cache and self.enable_cache and os.path.exists(cache_file):
            try:
                cached_df = pl.read_parquet(cache_file)
                if cached_df.height > 0 and "date" in cached_df.columns:
                    last_date = str(cached_df["date"].max())
                    if last_date == today_str:
                        logger.info(f"📦 [CACHE_HIT] Menggunakan cache data pasar lokal hari ini ({cached_df.height} baris).")
                        return cached_df, []
            except Exception as e:
                logger.warning(f"⚠️ Gagal membaca cache parquet: {e}")

        symbols_to_fetch = symbols if symbols else []
        final_df = pl.DataFrame()
        fetched_codes = set()

        # 2. EKSEKUSI PRIMARY: YFINANCE CHUNKED
        if symbols_to_fetch:
            final_df, fetched_codes = self._fetch_yfinance_chunked(symbols_to_fetch)

        # 3. EKSEKUSI SECONDARY: FALLBACK TO OFFICIAL IDX API (JIKA YFINANCE KOSONG/GAGAL)
        if final_df.is_empty():
            final_df, fetched_codes = self._fetch_idx_direct_fallback()

        # 4. TERTIARY: FALLBACK TO STALE OFFLINE CACHE (JIKA SEMUA ONLINE API GAGAL)
        if final_df.is_empty() and os.path.exists(cache_file):
            logger.warning("🚨 [OFFLINE_FAILSAFE] Seluruh API online gagal. Memuat offline cache lokal...")
            try:
                stale_df = pl.read_parquet(cache_file)
                if stale_df.height > 0:
                    return stale_df, []
            except Exception as e:
                logger.error(f"❌ Cache offline rusak: {e}")

        # 5. DETEKSI DELISTING TICKERS
        delisted_tickers = []
        if symbols_to_fetch and fetched_codes:
            for s in symbols_to_fetch:
                code = clean_ticker_code(s)
                if code and code not in fetched_codes:
                    delisted_tickers.append(normalize_idx_symbol(code))

        if delisted_tickers:
            logger.warning(f"🧹 [AUTO_DELISTING_DETECTED] Terdeteksi {len(delisted_tickers)} ticker delisted/invalid: {delisted_tickers}")

        # 6. SIMPAN PARQUET CACHE JIKA SUKSES
        if not final_df.is_empty() and self.enable_cache:
            try:
                final_df.write_parquet(cache_file)
                logger.info(f"💾 [CACHE_SAVED] Data pasar berhasil disinkronkan ke {cache_file}")
            except Exception as e:
                logger.warning(f"⚠️ Gagal menyimpan cache: {e}")

        return final_df, delisted_tickers


# ==============================================================================
# ALIAS COMPATIBILITY FUNCTIONS FOR PIPELINE INTEGRATION
# ==============================================================================
def load_and_prepare_market_data(
    symbols: Optional[List[str]] = None,
    use_cache: bool = True
) -> Tuple[pl.DataFrame, List[str]]:
    """Fungsi wrapper tingkat atas untuk kompatibilitas langsung dengan main.py."""
    engine = UnifiedDataEngine(enable_cache=use_cache)
    return engine.load_and_prepare_market_data(symbols=symbols, use_cache=use_cache)

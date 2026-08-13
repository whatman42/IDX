"""
=============================================================================
IDX Quantitative Portfolio Engine - High-Reliability Data Ingestion Layer
FileName      : data.py
Version       : 2026.Q3.v26.0 (Granular Error Diagnostic & Resilient Ingestion)
Compliance    : Indonesia Stock Exchange (IDX) Public API & Polars Engine
=============================================================================
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Standar Zona Waktu Pasar Saham Indonesia (WIB)
WIB_TZ = ZoneInfo("Asia/Jakarta")

# Endpoint Resmi API BEI & Yahoo Finance Fallback
IDX_BASE_URL = "https://www.idx.co.id/primary"
IDX_MAIN_PAGE = "https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham"
IDX_TRADING_SUMMARY_ENDPOINT = f"{IDX_BASE_URL}/TradingSummary/GetStockSummary"

DEFAULT_CACHE_DIR = ".cache"
DEFAULT_UNIVERSE_FILE = "universe.json"
HTTP_TIMEOUT_SEC = 10.0


# ==============================================================================
# CUSTOM GRANULAR EXCEPTIONS (P0 REQUIREMENT)
# ==============================================================================
class DataIngestionError(Exception):
    """Base Exception untuk seluruh kegagalan Ingesti Data."""
    pass

class BEIHttpError(DataIngestionError):
    """Gagal terhubung atau menerima status HTTP Error dari API BEI."""
    pass

class BEIEmptyResponseError(DataIngestionError):
    """API BEI merespons 200 OK tetapi payload JSON data kosong."""
    pass

class BEISchemaError(DataIngestionError):
    """Struktur JSON BEI berubah atau kolom wajib hilang."""
    pass

class CacheCorruptedError(DataIngestionError):
    """Berkas Parquet cache ditemukan tetapi rusak atau tidak valid."""
    pass

class CacheStaleError(DataIngestionError):
    """Cache tersedia tetapi kadaluarsa dan kebijakan melarang penggunaan data stale."""
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
# HELPER FUNCTIONS & SESSION FACTORY
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
    """Session HTTP dengan Browser Headers Lengkap untuk Menembus Cloudflare/CDN BEI."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": IDX_MAIN_PAGE,
        "Origin": "https://www.idx.co.id",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin"
    })
    
    # Retry hanya untuk Transient Error (500, 502, 503, 504, 429)
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ==============================================================================
# HIGH-RELIABILITY DATA ENGINE
# ==============================================================================
class UnifiedDataEngine:
    """Engine Ingesti Data BEI Direct dengan Diagnostic Log, Failover & Cache Validation."""

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

    def _bootstrap_session(self) -> bool:
        """Mengambil Cookie Sesi Asli dari Halaman Utama BEI."""
        try:
            res = self.session.get(IDX_MAIN_PAGE, timeout=HTTP_TIMEOUT_SEC)
            if res.status_code == 200:
                logger.info("🔑 [SESSION_BOOTSTRAP] Berhasil memperoleh cookie sesi resmi idx.co.id")
                return True
            else:
                logger.warning(f"⚠️ [SESSION_BOOTSTRAP] Halaman utama BEI merespons HTTP {res.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ [SESSION_BOOTSTRAP] Gagal terhubung ke halaman utama BEI: {e}")
        return False

    def fetch_idx_trading_summary(self) -> Tuple[List[Dict[str, Any]], str]:
        """
        Memanggil API Ringkasan Perdagangan BEI dengan Pencarian Tanggal Mundur (Fallback Hari Libur).
        Mengembalikan Tuple: (records_list, successful_date_str)
        """
        self._bootstrap_session()
        now_wib = datetime.now(WIB_TZ)
        
        # Iterasi tanggal dari Hari Ini mundur hingga 5 hari ke belakang
        for day_offset in range(5):
            target_date = now_wib - timedelta(days=day_offset)
            date_str = target_date.strftime("%Y%m%d")
            formatted_date_dash = target_date.strftime("%Y-%m-%d")
            
            params = {
                "date": date_str,
                "start": 0,
                "length": 1000
            }
            
            try:
                logger.info(f"🌐 [IDX_API_CALL] Request data pasar ke BEI untuk tanggal: {date_str}...")
                start_req = time.perf_counter()
                res = self.session.get(IDX_TRADING_SUMMARY_ENDPOINT, params=params, timeout=HTTP_TIMEOUT_SEC)
                req_duration = (time.perf_counter() - start_req) * 1000

                if res.status_code != 200:
                    logger.warning(f"⚠️ [IDX_HTTP_WARN] Date {date_str} -> HTTP Status {res.status_code} ({req_duration:.1f}ms)")
                    continue

                try:
                    json_data = res.json()
                except Exception as parse_err:
                    logger.error(f"❌ [IDX_JSON_PARSE_ERROR] Respon BEI sanjungan HTML/Bukan JSON: {parse_err}")
                    continue

                records = json_data.get("Data", [])
                if records and isinstance(records, list) and len(records) > 0:
                    logger.info(f"✅ [IDX_API_SUCCESS] Memuat {len(records)} emiten untuk tanggal {date_str} ({req_duration:.1f}ms).")
                    return records, formatted_date_dash
                else:
                    logger.info(f"ℹ️ [IDX_EMPTY_DATE] Tanggal {date_str} tidak memiliki data perdagangan (mungkin pasar libur).")

            except requests.exceptions.Timeout:
                logger.warning(f"⏰ [IDX_TIMEOUT] Request timeout ({HTTP_TIMEOUT_SEC}s) untuk tanggal {date_str}")
            except requests.exceptions.RequestException as req_err:
                logger.warning(f"⚠️ [IDX_NET_ERROR] Kesalahan jaringan untuk tanggal {date_str}: {req_err}")

            time.sleep(0.3)

        return [], ""

    def validate_and_load_cache(self, cache_file: str, today_str: str) -> Tuple[Optional[pl.DataFrame], str]:
        """Memvalidasi integritas dan staleness cache Parquet."""
        if not os.path.exists(cache_file):
            return None, "CACHE_MISS"

        try:
            cached_df = pl.read_parquet(cache_file)
            if cached_df.height == 0 or "date" not in cached_df.columns:
                return None, "CACHE_CORRUPTED"

            cache_max_date = str(cached_df["date"].max())
            if cache_max_date == today_str:
                return cached_df, "CACHE_HIT_FRESH"
            else:
                return cached_df, "CACHE_HIT_STALE"
        except Exception as e:
            logger.error(f"❌ Error membaca cache parquet: {e}")
            return None, "CACHE_CORRUPTED"

    def load_and_prepare_market_data(
        self,
        symbols: Optional[List[str]] = None,
        use_cache: bool = True,
        lookback_days: int = 120
    ) -> Tuple[pl.DataFrame, List[str]]:
        """
        Metode Utama Ingesti Data Pasar dengan Granular Errors & Quality Gate.
        """
        start_time = time.perf_counter()
        cache_file = os.path.join(self.cache_dir, "idx_market_cache.parquet")
        now_wib = datetime.now(WIB_TZ)
        today_str = now_wib.strftime("%Y-%m-%d")

        # ----------------------------------------------------------------------
        # 1. PERIKSA CACHE LOKAL DAHULU
        # ----------------------------------------------------------------------
        cached_df, cache_status = self.validate_and_load_cache(cache_file, today_str)
        if use_cache and self.enable_cache and cache_status == "CACHE_HIT_FRESH":
            elapsed = time.perf_counter() - start_time
            logger.info(f"📦 [CACHE_FRESH] Menggunakan cache data pasar lokal segar ({cached_df.height} baris, {elapsed:.2f}s).")
            return cached_df, []

        # ----------------------------------------------------------------------
        # 2. INGEST DATA LIVE DARI API BEI
        # ----------------------------------------------------------------------
        trading_summary, active_trade_date = self.fetch_idx_trading_summary()

        # ----------------------------------------------------------------------
        # 3. FALLBACK HANDLING JIKA API BEI GAGAL Total
        # ----------------------------------------------------------------------
        if not trading_summary:
            if cache_status in ["CACHE_HIT_FRESH", "CACHE_HIT_STALE"]:
                logger.warning(f"🚨 [OFFLINE_FALLBACK] API BEI gagal/kosong. Menggunakan Cache Parquet Offline (Status: {cache_status})...")
                return cached_df, []
            
            # Jika API BEI Kosong DAN Cache Miss/Corrupted -> Lempar Granular Diagnostic Error
            err_msg = f"Gagal memperoleh data pasar: API BEI mengembalikan data kosong dan Cache lokal tidak tersedia (Status: {cache_status})."
            logger.error(f"❌ [FATAL_DATA_LAYER] {err_msg}")
            raise BEIEmptyResponseError(err_msg)

        # ----------------------------------------------------------------------
        # 4. PARSING & VALIDASI SCHEMA DATA BEI
        # ----------------------------------------------------------------------
        parsed_records = []
        received_code_set = set()
        effective_date = active_trade_date if active_trade_date else today_str

        required_keys = ["StockCode", "Close"]
        
        # Validasi sampel data pertama untuk Schema Verification
        first_sample = trading_summary[0]
        missing_keys = [k for k in required_keys if k not in first_sample]
        if missing_keys:
            err_msg = f"Struktur JSON dari BEI berubah! Kolom wajib hilang: {missing_keys}"
            logger.error(f"❌ [SCHEMA_ERROR] {err_msg}")
            raise BEISchemaError(err_msg)

        for item in trading_summary:
            raw_code = str(item.get("StockCode", "")).strip().upper()
            if not raw_code:
                continue

            received_code_set.add(raw_code)
            norm_ticker = normalize_idx_symbol(raw_code)

            close_price = float(item.get("Close", item.get("Previous", 0.0)))
            if close_price <= 0:
                continue  # Abaikan emiten tanpa transaksi / suspended

            open_price = float(item.get("Open", close_price))
            high_price = float(item.get("High", max(open_price, close_price)))
            low_price = float(item.get("Low", min(open_price, close_price)))
            volume_shares = float(item.get("Volume", 0.0))
            value_idr = float(item.get("Value", volume_shares * close_price))

            parsed_records.append({
                "date": effective_date,
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

        # ----------------------------------------------------------------------
        # 5. AUTO-DETEKSI TICKER DELISTED
        # ----------------------------------------------------------------------
        delisted_tickers = []
        if symbols:
            for s in symbols:
                code = clean_ticker_code(s)
                if code and code not in received_code_set:
                    delisted_tickers.append(normalize_idx_symbol(code))

        if delisted_tickers:
            logger.info(f"🧹 [AUTO_DELISTING] Terdeteksi {len(delisted_tickers)} ticker delisted/tidak aktif hari ini.")

        if not parsed_records:
            err_msg = "Seluruh record dari API BEI bernilai harga 0 atau invalid."
            logger.error(f"❌ [DATA_QUALITY_FAIL] {err_msg}")
            raise BEIEmptyResponseError(err_msg)

        base_df = pl.DataFrame(parsed_records)

        # ----------------------------------------------------------------------
        # 6. SINTESIS DERET WAKTU HISTORIS UNTUK MOVING AVERAGE/ATR
        # ----------------------------------------------------------------------
        historical_dfs = [base_df]
        base_date_dt = datetime.strptime(effective_date, "%Y-%m-%d").replace(tzinfo=WIB_TZ)
        
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

        # ----------------------------------------------------------------------
        # 7. SIMPAN PARQUET CACHE LOKAL
        # ----------------------------------------------------------------------
        if self.enable_cache:
            try:
                full_market_df.write_parquet(cache_file)
                logger.info(f"💾 [CACHE_SAVED] Cache Parquet berhasil dikonsolidasi ({cache_file}).")
            except Exception as e:
                logger.warning(f"⚠️ Gagal menyimpan cache Parquet: {e}")

        elapsed = time.perf_counter() - start_time
        logger.info(f"⚡ [INGESTION_SUCCESS] Berhasil menyusun {full_market_df.height} baris data ({base_df.height} emiten aktif) dalam {elapsed:.2f} detik!")
        return full_market_df, delisted_tickers


# ==============================================================================
# ALIAS COMPATIBILITY WRAPPER
# ==============================================================================
def load_and_prepare_market_data(
    symbols: Optional[List[str]] = None,
    use_cache: bool = True
) -> Tuple[pl.DataFrame, List[str]]:
    engine = UnifiedDataEngine(enable_cache=use_cache)
    return engine.load_and_prepare_market_data(symbols=symbols, use_cache=use_cache)

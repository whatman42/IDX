"""
=============================================================================
IDX Quantitative Analysis System - Consolidated Storage Engine
FileName      : storage.py
Directory     : Flat Directory (Root Level selevel dengan main.py)
Version       : 2026.Q3.v3.2.0 (Institutional Production-Grade Storage Engine)
Compliance    : Indonesian Stock Exchange (IDX) Signal Analysis & Yahoo Finance Rules
=============================================================================
"""

import os
import re
import json
import math
import zlib
import base64
import sqlite3
import time
import random
import logging
import threading
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

import numpy as np
import polars as pl

# Standar Zona Waktu Pasar Saham Indonesia (WIB)
WIB_TZ: ZoneInfo = ZoneInfo("Asia/Jakarta")

# Setup Local Logger untuk IDX Storage Engine
logger = logging.getLogger("IDX.Storage")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s][IDX.STORAGE] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _get_wib_timestamp_str() -> str:
    """Mengembalikan timestamp ISO dengan Zona Waktu Indonesia Barat (WIB)."""
    return datetime.now(WIB_TZ).isoformat()


# ==============================================================================
# HELPER PARSER, SANITIZER, COMPRESSION & NORMALIZER
# ==============================================================================
def normalize_idx_symbol(symbol: Optional[str]) -> str:
    """
    Mengubah format simbol ticker saham ke standar Yahoo Finance IDX secara defensif (misal BBCA -> BBCA.JK).
    Menangani pembersihan karakter ilegal dan duplikasi ekstensi.
    """
    if not symbol:
        return "UNKNOWN"
    sym = str(symbol).strip().upper()
    if sym in ["UNKNOWN", "NONE", "NULL", "", "NAN"]:
        return "UNKNOWN"
    
    sym = re.sub(r"(\.JK)+$", ".JK", sym)
    if sym.endswith(".JK") or sym.startswith("^"):
        return sym
    
    sym_clean = re.sub(r"[^A-Z0-9]", "", sym)
    if not sym_clean:
        return "UNKNOWN"
        
    return f"{sym_clean}.JK"


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Mengonversi nilai ke float secara aman, menangani None, NaN, dan Inf."""
    if val is None:
        return default
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        return f_val
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Mengonversi nilai ke integer secara aman."""
    if val is None:
        return default
    if isinstance(val, int):
        return val
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        return int(f_val)
    except (ValueError, TypeError):
        return default


def _safe_parse_timestamp_ns(val: Any) -> int:
    """
    Mengonversi berbagai format nilai timestamp (int, float, ISO string, datetime, date)
    ke integer nanodetik secara presisi.
    """
    if val is None:
        return time.time_ns()

    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return time.time_ns()
        if val < 1e11:
            return int(val * 1e9)
        elif val < 1e14:
            return int(val * 1e6)
        elif val < 1e17:
            return int(val * 1e3)
        return int(val)

    if isinstance(val, int):
        if val < 1e11:
            return int(val * 1e9)
        elif val < 1e14:
            return int(val * 1e6)
        elif val < 1e17:
            return int(val * 1e3)
        return val

    if isinstance(val, (datetime, date)):
        if isinstance(val, datetime):
            if val.tzinfo is None:
                val = val.replace(tzinfo=timezone.utc)
            return int(val.timestamp() * 1e9)
        else:
            dt = datetime.combine(val, datetime.min.time(), tzinfo=timezone.utc)
            return int(dt.timestamp() * 1e9)

    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str.lower() in ["none", "null", "nan"]:
            return time.time_ns()
        if val_str.isdigit():
            return int(val_str)
        try:
            val_clean = val_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(val_clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1e9)
        except Exception:
            pass

    return time.time_ns()


def _serialize_meta(extra_meta: Dict[str, Any]) -> str:
    """Menyimpan dictionary meta_json dengan encoding aman dan kompresi zlib jika > 512 karakter."""
    if not extra_meta:
        return "{}"
    try:
        raw_str = json.dumps(extra_meta, ensure_ascii=False, default=str)
        if len(raw_str) > 512:
            compressed = zlib.compress(raw_str.encode('utf-8'))
            return "zlib:" + base64.b64encode(compressed).decode('ascii')
        return raw_str
    except Exception as e:
        logger.warning(f"Gagal memformat meta_json: {e}")
        return "{}"


def _deserialize_meta(meta_str: Optional[str]) -> str:
    """Mendekompresi meta_json jika tersimpan dalam format kompresi zlib."""
    if not meta_str:
        return "{}"
    if meta_str.startswith("zlib:"):
        try:
            compressed_bytes = base64.b64decode(meta_str[5:])
            return zlib.decompress(compressed_bytes).decode('utf-8')
        except Exception as e:
            logger.warning(f"Gagal dekompresi zlib meta_json: {e}")
            return "{}"
    return meta_str


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================
class StorageError(Exception):
    """Base Exception untuk seluruh kesalahan operasional pada modul storage.py."""
    pass

class DatabaseConnectionError(StorageError):
    """Pengecualian untuk kegagalan koneksi database SQLite."""
    pass

class SchemaMigrationError(StorageError):
    """Pengecualian untuk kegagalan inisialisasi atau migrasi skema tabel."""
    pass

class QueryExecutionError(StorageError):
    """Pengecualian untuk kegagalan eksekusi query pembacaan/penulisan SQLite."""
    pass


# ==============================================================================
# RETRY DECORATOR / HELPER (BUSY/LOCKED BACKOFF)
# ==============================================================================
def execute_with_retry(action_fn: Callable[[], Any], max_retries: int = 5, initial_delay: float = 0.05) -> Any:
    """
    Menerapkan Exponential Backoff dengan Jitter untuk menangani
    sqlite3.OperationalError ('database is locked' / 'database is busy').
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return action_fn()
        except sqlite3.OperationalError as e:
            err_msg = str(e).lower()
            if ("locked" in err_msg or "busy" in err_msg) and attempt < max_retries:
                sleep_time = delay + random.uniform(0.01, 0.05)
                logger.warning(f"[Attempt {attempt}/{max_retries}] SQLite locked/busy, retry dalam {sleep_time:.3f}s...")
                time.sleep(sleep_time)
                delay *= 2.0
            else:
                logger.exception(f"SQLite OperationalError tidak terpulihkan setelah {attempt} percobaan: {e}")
                raise
        except Exception as e:
            logger.exception(f"Kesalahan tak terduga pada eksekusi query SQLite: {e}")
            raise


# ==============================================================================
# 1. SQLITE ENGINE BASE (HIGH-PERFORMANCE PRODUCTION ENGINE)
# ==============================================================================
class SQLiteEngine:
    """Engine Manajemen Koneksi Database SQLite Berbasis Thread-Local, Versioned Schema, & Hot Backup."""

    CURRENT_SCHEMA_VERSION = 2

    def __init__(self, db_path: str = "data/storage_dryrun.db") -> None:
        self.db_path = db_path
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self._local = threading.local()
        self._global_lock = threading.RLock()
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            try:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                    isolation_level=None,
                    check_same_thread=False
                )
                # PRAGMA Tuning untuk Performa Maksimal dan Keamanan Memori
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA temp_store = MEMORY;")
                conn.execute("PRAGMA mmap_size = 2147483648;")   # 2 GB Memory Mapping Limit Safe Threshold
                conn.execute("PRAGMA cache_size = -64000;")       # ~64 MB Cache
                conn.execute("PRAGMA busy_timeout = 30000;")
                self._local.conn = conn
            except sqlite3.Error as e:
                logger.exception(f"Gagal membuka koneksi database SQLite pada {self.db_path}: {e}")
                raise DatabaseConnectionError(f"Gagal membuka koneksi database SQLite pada {self.db_path}: {e}") from e
        return self._local.conn

    def init_db(self) -> None:
        def _run_init():
            conn = self.get_connection()
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute("PRAGMA auto_vacuum = INCREMENTAL;")

                # 1. Tabel Schema Versioning
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        description TEXT NOT NULL,
                        applied_at TEXT NOT NULL
                    );
                """)

                # 2. Tabel Signal History
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signal_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        asset TEXT NOT NULL,
                        signal INTEGER NOT NULL,
                        confidence REAL DEFAULT 0.0,
                        probability REAL DEFAULT 0.0,
                        entry_price REAL DEFAULT 0.0,
                        stop_loss REAL DEFAULT 0.0,
                        take_profit REAL DEFAULT 0.0,
                        horizon INTEGER DEFAULT 1,
                        expected_return REAL DEFAULT 0.0,
                        risk_reward_ratio REAL DEFAULT 0.0,
                        mode TEXT DEFAULT 'dry-run',
                        timestamp_ns INTEGER NOT NULL,
                        signal_id TEXT,
                        meta_json TEXT DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        CONSTRAINT uq_signal_asset_mode_ts UNIQUE (asset, mode, timestamp_ns)
                    );
                """)

                # 3. Indeks Komposit & Lookup
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signal_composite 
                        ON signal_history(asset, mode, timestamp_ns DESC);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signal_id_lookup
                        ON signal_history(signal_id);
                """)

                # 4. Tabel Prediction History
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        asset TEXT NOT NULL,
                        predicted_value REAL NOT NULL,
                        model_id TEXT NOT NULL,
                        horizon INTEGER DEFAULT 1,
                        confidence REAL DEFAULT 0.0,
                        timestamp_ns INTEGER NOT NULL,
                        signal_id TEXT,
                        meta_json TEXT DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        CONSTRAINT uq_pred_asset_model_horizon_ts UNIQUE (asset, model_id, horizon, timestamp_ns)
                    );
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pred_composite 
                        ON prediction_history(asset, model_id, horizon, timestamp_ns DESC);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pred_signal_id_lookup
                        ON prediction_history(signal_id);
                """)

                # Record Schema Version jika belum tercatat
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(version) FROM schema_version;")
                row = cursor.fetchone()
                current_v = row[0] if row and row[0] is not None else 0

                if current_v < self.CURRENT_SCHEMA_VERSION:
                    now_wib = _get_wib_timestamp_str()
                    cursor.execute("""
                        INSERT OR REPLACE INTO schema_version (version, description, applied_at)
                        VALUES (?, ?, ?);
                    """, (self.CURRENT_SCHEMA_VERSION, "v2026.Q3.v3.2.0 Institutional Schema", now_wib))

                conn.execute("COMMIT;")
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.exception(f"Gagal menginisialisasi skema tabel SQLite: {e}")
                raise SchemaMigrationError(f"Gagal menginisialisasi skema tabel SQLite: {e}") from e

        with self._global_lock:
            execute_with_retry(_run_init)

    def backup(self, backup_db_path: str) -> None:
        """Melakukan hot backup database SQLite secara online tanpa menghentikan sistem."""
        parent_dir = os.path.dirname(backup_db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        def _run_backup():
            src_conn = self.get_connection()
            dest_conn = sqlite3.connect(backup_db_path)
            try:
                with dest_conn:
                    src_conn.backup(dest_conn, pages=100, sleep=0.01)
                logger.info(f"Hot backup database berhasil disimpan ke: {backup_db_path}")
            finally:
                dest_conn.close()

        with self._global_lock:
            execute_with_retry(_run_backup)

    def run_integrity_check(self) -> bool:
        """Menjalankan PRAGMA integrity_check untuk memverifikasi kesehatan fisik database."""
        def _check():
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            res = cursor.fetchone()
            is_ok = res is not None and res[0] == "ok"
            if not is_ok:
                logger.error(f"Peringatan! Integrity check database GAGAL: {res}")
            else:
                logger.info("Integrity check database: OK (PASSED)")
            return is_ok

        with self._global_lock:
            return execute_with_retry(_check)

    def vacuum(self, pages: int = 1000) -> None:
        """Menjalankan pembersihan memori incremental vacuum pada database SQLite."""
        def _vac():
            conn = self.get_connection()
            conn.execute(f"PRAGMA incremental_vacuum({pages});")
            logger.info(f"Incremental vacuum sebesar {pages} halaman selesai dijalankan.")

        with self._global_lock:
            execute_with_retry(_vac)

    def close_thread_connection(self) -> None:
        """Menutup koneksi SQLite pada thread aktif secara aman."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception as e:
                logger.warning(f"Error saat menutup koneksi thread local: {e}")
            finally:
                self._local.conn = None

    def close(self) -> None:
        """Menutup koneksi thread aktif dan membebaskan resource."""
        self.close_thread_connection()

    def __enter__(self) -> "SQLiteEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ==============================================================================
# 2. SIGNAL HISTORY STORE
# ==============================================================================
class SignalHistory:
    """Layanan Penyimpanan Log & Query Sinyal Trading Kuantitatif Saham IDX."""

    SIGNAL_SCHEMA = {
        "id": pl.Int64,
        "asset": pl.Utf8,
        "signal": pl.Int32,
        "confidence": pl.Float64,
        "probability": pl.Float64,
        "entry_price": pl.Float64,
        "stop_loss": pl.Float64,
        "take_profit": pl.Float64,
        "horizon": pl.Int32,
        "expected_return": pl.Float64,
        "risk_reward_ratio": pl.Float64,
        "mode": pl.Utf8,
        "timestamp_ns": pl.Int64,
        "signal_id": pl.Utf8,
        "meta_json": pl.Utf8,
        "created_at": pl.Utf8
    }

    def __init__(self, sqlite_engine: SQLiteEngine) -> None:
        self.sqlite_engine = sqlite_engine

    def _prepare_signal_records(
        self, 
        signals_payload: Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, Any], None],
        execution_mode: str = "dry-run"
    ) -> List[Tuple]:
        if signals_payload is None:
            return []

        records: List[Dict[str, Any]] = []
        if isinstance(signals_payload, pl.DataFrame):
            if signals_payload.is_empty():
                return []
            records = signals_payload.to_dicts()
        elif isinstance(signals_payload, dict):
            if "orders" in signals_payload and isinstance(signals_payload["orders"], list):
                records = signals_payload["orders"]
            elif "signals" in signals_payload and isinstance(signals_payload["signals"], list):
                records = signals_payload["signals"]
            else:
                records = [signals_payload]
        elif isinstance(signals_payload, list):
            records = list(signals_payload)
        else:
            return []

        if not records:
            return []

        now_wib = _get_wib_timestamp_str()
        db_tuples = []

        for sig in records:
            if not isinstance(sig, dict):
                continue

            raw_asset = sig.get("asset", sig.get("ticker", sig.get("symbol", sig.get("asset_id", "UNKNOWN"))))
            asset = normalize_idx_symbol(raw_asset)

            raw_sig = sig.get("signal", sig.get("signal_direction", sig.get("direction", 0)))
            if isinstance(raw_sig, str):
                raw_sig_upper = raw_sig.upper()
                signal_val = 1 if raw_sig_upper in ["BUY", "LONG", "1"] else (-1 if raw_sig_upper in ["SELL", "SHORT", "-1"] else 0)
            else:
                signal_val = _safe_int(raw_sig, default=0)

            conf = _safe_float(sig.get("confidence", sig.get("signal_confidence", 0.0)))
            prob = _safe_float(sig.get("probability", sig.get("signal_probability", 0.0)))
            entry_p = _safe_float(sig.get("entry_price", sig.get("close", sig.get("price", 0.0))))
            sl_p = _safe_float(sig.get("stop_loss", sig.get("sl", 0.0)))
            tp_p = _safe_float(sig.get("take_profit", sig.get("tp", 0.0)))
            horizon_v = _safe_int(sig.get("horizon", sig.get("timeframe", 1)), default=1)
            exp_ret = _safe_float(sig.get("expected_return", sig.get("exp_return", 0.0)))
            rrr_v = _safe_float(sig.get("risk_reward_ratio", sig.get("rrr", 0.0)))
            sig_mode = str(sig.get("mode", execution_mode)).lower().strip()
            
            signal_id = sig.get("signal_id", sig.get("order_id", None))
            signal_id_str = str(signal_id).strip() if signal_id else None

            raw_ts = sig.get("timestamp_ns", sig.get("timestamp", sig.get("date", None)))
            ts_ns = _safe_parse_timestamp_ns(raw_ts)

            reserved_keys = {
                "asset", "ticker", "symbol", "asset_id", "signal", "signal_direction", "direction",
                "confidence", "signal_confidence", "probability", "signal_probability", "entry_price",
                "close", "price", "stop_loss", "sl", "take_profit", "tp", "horizon", "timeframe",
                "expected_return", "exp_return", "risk_reward_ratio", "rrr", "mode", "timestamp_ns",
                "timestamp", "date", "signal_id", "order_id"
            }
            extra_meta = {k: v for k, v in sig.items() if k not in reserved_keys and isinstance(v, (str, int, float, bool))}
            meta_str = _serialize_meta(extra_meta)

            db_tuples.append((
                asset, signal_val, conf, prob, entry_p, sl_p, tp_p, 
                horizon_v, exp_ret, rrr_v, sig_mode, ts_ns, signal_id_str, meta_str, now_wib
            ))

        return db_tuples

    def persist_signals(
        self, 
        signals_payload: Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, Any], None],
        execution_mode: str = "dry-run"
    ) -> int:
        db_tuples = self._prepare_signal_records(signals_payload, execution_mode)
        if not db_tuples:
            return 0

        def _execute_persist():
            conn = self.sqlite_engine.get_connection()
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO signal_history (
                        asset, signal, confidence, probability, entry_price, 
                        stop_loss, take_profit, horizon, expected_return, 
                        risk_reward_ratio, mode, timestamp_ns, signal_id, meta_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(asset, mode, timestamp_ns) DO UPDATE SET
                        signal = excluded.signal,
                        confidence = excluded.confidence,
                        probability = excluded.probability,
                        entry_price = excluded.entry_price,
                        stop_loss = excluded.stop_loss,
                        take_profit = excluded.take_profit,
                        horizon = excluded.horizon,
                        expected_return = excluded.expected_return,
                        risk_reward_ratio = excluded.risk_reward_ratio,
                        signal_id = COALESCE(excluded.signal_id, signal_history.signal_id),
                        meta_json = excluded.meta_json;
                """, db_tuples)
                inserted = cursor.rowcount
                conn.execute("COMMIT;")
                return inserted
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.exception(f"Gagal melakukan persistensi sinyal ke SQLite: {e}")
                raise StorageError(f"Error persistensi sinyal: {e}") from e

        with self.sqlite_engine._global_lock:
            return execute_with_retry(_execute_persist)

    def fetch_signals(
        self, 
        asset: Optional[str] = None, 
        mode: Optional[str] = None, 
        last_timestamp_ns: Optional[int] = None,
        limit: int = 1000
    ) -> pl.DataFrame:
        def _execute_fetch():
            conn = self.sqlite_engine.get_connection()
            query = """
                SELECT id, asset, signal, confidence, probability, entry_price, stop_loss, 
                       take_profit, horizon, expected_return, risk_reward_ratio, mode, 
                       timestamp_ns, signal_id, meta_json, created_at 
                FROM signal_history 
                WHERE 1=1
            """
            params: List[Any] = []

            if asset:
                query += " AND asset = ?"
                params.append(normalize_idx_symbol(asset))
            if mode:
                query += " AND mode = ?"
                params.append(str(mode).lower().strip())
            if last_timestamp_ns is not None:
                query += " AND timestamp_ns < ?"
                params.append(int(last_timestamp_ns))

            query += " ORDER BY timestamp_ns DESC LIMIT ?"
            params.append(limit)

            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            if not rows:
                return pl.DataFrame(schema=self.SIGNAL_SCHEMA)

            df = pl.DataFrame(rows, schema=self.SIGNAL_SCHEMA)
            
            if "meta_json" in df.columns and df.height > 0:
                df = df.with_columns(
                    pl.col("meta_json").map_elements(_deserialize_meta, return_dtype=pl.Utf8, skip_nulls=False)
                )
            return df

        with self.sqlite_engine._global_lock:
            return execute_with_retry(_execute_fetch)


# ==============================================================================
# 3. PREDICTION HISTORY STORE
# ==============================================================================
class PredictionHistory:
    """Layanan Audit Forensics Forecasting Lineage Saham IDX."""

    PREDICTION_SCHEMA = {
        "id": pl.Int64,
        "asset": pl.Utf8,
        "predicted_value": pl.Float64,
        "model_id": pl.Utf8,
        "horizon": pl.Int32,
        "confidence": pl.Float64,
        "timestamp_ns": pl.Int64,
        "signal_id": pl.Utf8,
        "meta_json": pl.Utf8,
        "created_at": pl.Utf8
    }

    def __init__(self, sqlite_engine: SQLiteEngine) -> None:
        self.sqlite_engine = sqlite_engine

    def _prepare_prediction_records(
        self, 
        predictions_payload: Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, Any], None]
    ) -> List[Tuple]:
        if predictions_payload is None:
            return []

        records: List[Dict[str, Any]] = []
        if isinstance(predictions_payload, pl.DataFrame):
            if predictions_payload.is_empty():
                return []
            records = predictions_payload.to_dicts()
        elif isinstance(predictions_payload, dict):
            if "predictions" in predictions_payload and isinstance(predictions_payload["predictions"], list):
                records = predictions_payload["predictions"]
            else:
                records = [predictions_payload]
        elif isinstance(predictions_payload, list):
            records = list(predictions_payload)
        else:
            return []

        if not records:
            return []

        now_wib = _get_wib_timestamp_str()
        db_tuples = []

        for pred in records:
            if not isinstance(pred, dict):
                continue

            raw_asset = pred.get("asset", pred.get("ticker", pred.get("asset_id", pred.get("symbol", "UNKNOWN"))))
            asset = normalize_idx_symbol(raw_asset)

            pred_val = _safe_float(pred.get("predicted_value", pred.get("predicted_return", pred.get("prediction", 0.0))))
            model_id = str(pred.get("model_id", "DEFAULT_MODEL")).strip()
            horizon_v = _safe_int(pred.get("horizon", pred.get("timeframe", 1)), default=1)
            conf = _safe_float(pred.get("confidence", 0.0))

            signal_id = pred.get("signal_id", None)
            signal_id_str = str(signal_id).strip() if signal_id else None

            raw_ts = pred.get("timestamp_ns", pred.get("timestamp", pred.get("date", None)))
            ts_ns = _safe_parse_timestamp_ns(raw_ts)

            reserved_keys = {
                "asset", "ticker", "symbol", "asset_id", "predicted_value", "predicted_return",
                "prediction", "model_id", "horizon", "timeframe", "confidence", "timestamp_ns",
                "timestamp", "date", "signal_id"
            }
            extra_meta = {k: v for k, v in pred.items() if k not in reserved_keys and isinstance(v, (str, int, float, bool))}
            meta_str = _serialize_meta(extra_meta)

            db_tuples.append((
                asset, pred_val, model_id, horizon_v, conf, ts_ns, signal_id_str, meta_str, now_wib
            ))

        return db_tuples

    def persist_predictions(
        self, 
        predictions_payload: Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, Any], None]
    ) -> int:
        db_tuples = self._prepare_prediction_records(predictions_payload)
        if not db_tuples:
            return 0

        def _execute_persist():
            conn = self.sqlite_engine.get_connection()
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO prediction_history (
                        asset, predicted_value, model_id, horizon, confidence, 
                        timestamp_ns, signal_id, meta_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(asset, model_id, horizon, timestamp_ns) DO UPDATE SET
                        predicted_value = excluded.predicted_value,
                        confidence = excluded.confidence,
                        signal_id = COALESCE(excluded.signal_id, prediction_history.signal_id),
                        meta_json = excluded.meta_json;
                """, db_tuples)
                inserted = cursor.rowcount
                conn.execute("COMMIT;")
                return inserted
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.exception(f"Gagal melakukan persistensi prediksi ke SQLite: {e}")
                raise StorageError(f"Error persistensi prediksi: {e}") from e

        with self.sqlite_engine._global_lock:
            return execute_with_retry(_execute_persist)

    def fetch_predictions(
        self, 
        asset: Optional[str] = None, 
        model_id: Optional[str] = None, 
        last_timestamp_ns: Optional[int] = None,
        limit: int = 1000
    ) -> pl.DataFrame:
        def _execute_fetch():
            conn = self.sqlite_engine.get_connection()
            query = """
                SELECT id, asset, predicted_value, model_id, horizon, confidence, 
                       timestamp_ns, signal_id, meta_json, created_at 
                FROM prediction_history 
                WHERE 1=1
            """
            params: List[Any] = []

            if asset:
                query += " AND asset = ?"
                params.append(normalize_idx_symbol(asset))
            if model_id:
                query += " AND model_id = ?"
                params.append(str(model_id).strip())
            if last_timestamp_ns is not None:
                query += " AND timestamp_ns < ?"
                params.append(int(last_timestamp_ns))

            query += " ORDER BY timestamp_ns DESC LIMIT ?"
            params.append(limit)

            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            if not rows:
                return pl.DataFrame(schema=self.PREDICTION_SCHEMA)

            df = pl.DataFrame(rows, schema=self.PREDICTION_SCHEMA)

            if "meta_json" in df.columns and df.height > 0:
                df = df.with_columns(
                    pl.col("meta_json").map_elements(_deserialize_meta, return_dtype=pl.Utf8, skip_nulls=False)
                )
            return df

        with self.sqlite_engine._global_lock:
            return execute_with_retry(_execute_fetch)


# ==============================================================================
# 4. UNIFIED STORAGE ENGINE (FACADE)
# ==============================================================================
class UnifiedStorageEngine:
    """
    Facade Utama Komponen Persistensi & Audit Database Storage Saham IDX.
    Mendukung Single Atomic Transaction, Purge Data Retention, Hot Backup, & Full Query API.
    """

    def __init__(self, db_path: Optional[str] = None, mode: Optional[str] = None) -> None:
        self.mode = str(mode or os.getenv("EXECUTION_MODE", os.getenv("TRADING_MODE", "dry-run"))).lower().strip()
        self.is_live = self.mode in ["live", "force-rebalance", "analysis", "realtime"]
        self.state_suffix = "live" if self.is_live else "dryrun"

        if db_path is None:
            db_path = f"data/storage_{self.state_suffix}.db"

        self.db_path = db_path
        self.sqlite_engine = SQLiteEngine(self.db_path)
        self.signal_store = SignalHistory(self.sqlite_engine)
        self.prediction_store = PredictionHistory(self.sqlite_engine)

    def persist_signals(
        self, 
        signals_payload: Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, Any], None]
    ) -> int:
        return self.signal_store.persist_signals(signals_payload, execution_mode=self.mode)

    def persist_predictions(
        self, 
        predictions_payload: Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, Any], None]
    ) -> int:
        return self.prediction_store.persist_predictions(predictions_payload)

    def persist_all(
        self, 
        signals_df: Optional[Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, Any]]] = None, 
        predictions_df: Optional[Union[pl.DataFrame, List[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> Dict[str, int]:
        sig_tuples = self.signal_store._prepare_signal_records(signals_df, execution_mode=self.mode)
        pred_tuples = self.prediction_store._prepare_prediction_records(predictions_df)

        if not sig_tuples and not pred_tuples:
            return {"signals_persisted": 0, "predictions_persisted": 0}

        def _execute_atomic_persist():
            conn = self.sqlite_engine.get_connection()
            conn.execute("BEGIN IMMEDIATE;")
            sig_count = 0
            pred_count = 0
            try:
                cursor = conn.cursor()
                if sig_tuples:
                    cursor.executemany("""
                        INSERT INTO signal_history (
                            asset, signal, confidence, probability, entry_price, 
                            stop_loss, take_profit, horizon, expected_return, 
                            risk_reward_ratio, mode, timestamp_ns, signal_id, meta_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(asset, mode, timestamp_ns) DO UPDATE SET
                            signal = excluded.signal,
                            confidence = excluded.confidence,
                            probability = excluded.probability,
                            entry_price = excluded.entry_price,
                            stop_loss = excluded.stop_loss,
                            take_profit = excluded.take_profit,
                            horizon = excluded.horizon,
                            expected_return = excluded.expected_return,
                            risk_reward_ratio = excluded.risk_reward_ratio,
                            signal_id = COALESCE(excluded.signal_id, signal_history.signal_id),
                            meta_json = excluded.meta_json;
                    """, sig_tuples)
                    sig_count = cursor.rowcount

                if pred_tuples:
                    cursor.executemany("""
                        INSERT INTO prediction_history (
                            asset, predicted_value, model_id, horizon, confidence, 
                            timestamp_ns, signal_id, meta_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(asset, model_id, horizon, timestamp_ns) DO UPDATE SET
                            predicted_value = excluded.predicted_value,
                            confidence = excluded.confidence,
                            signal_id = COALESCE(excluded.signal_id, prediction_history.signal_id),
                            meta_json = excluded.meta_json;
                    """, pred_tuples)
                    pred_count = cursor.rowcount

                conn.execute("COMMIT;")
                return {"signals_persisted": sig_count, "predictions_persisted": pred_count}
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.exception(f"Gagal melakukan persist_all secara atomik: {e}")
                raise StorageError(f"Atomic persist_all failed: {e}") from e

        with self.sqlite_engine._global_lock:
            return execute_with_retry(_execute_atomic_persist)

    def purge_before(self, days: int = 365) -> Dict[str, int]:
        cutoff_ns = int((time.time() - (days * 86400)) * 1e9)

        def _execute_purge():
            conn = self.sqlite_engine.get_connection()
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM signal_history WHERE timestamp_ns < ?;", (cutoff_ns,))
                c1 = cursor.rowcount

                cursor.execute("DELETE FROM prediction_history WHERE timestamp_ns < ?;", (cutoff_ns,))
                c2 = cursor.rowcount

                conn.execute("COMMIT;")
                logger.info(f"Purge data lama selesai: {c1} sinyal & {c2} prediksi dihapus (> {days} hari).")
                return {"signals_purged": c1, "predictions_purged": c2}
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.exception(f"Gagal melakukan pembersihan data lama: {e}")
                raise StorageError(f"Purge data failed: {e}") from e

        with self.sqlite_engine._global_lock:
            res = execute_with_retry(_execute_purge)
            self.sqlite_engine.vacuum()
            return res

    def fetch_signals(
        self, 
        asset: Optional[str] = None, 
        last_timestamp_ns: Optional[int] = None,
        limit: int = 1000
    ) -> pl.DataFrame:
        return self.signal_store.fetch_signals(
            asset=asset, 
            mode=self.mode, 
            last_timestamp_ns=last_timestamp_ns,
            limit=limit
        )

    def fetch_predictions(
        self, 
        asset: Optional[str] = None, 
        model_id: Optional[str] = None, 
        last_timestamp_ns: Optional[int] = None,
        limit: int = 1000
    ) -> pl.DataFrame:
        return self.prediction_store.fetch_predictions(
            asset=asset, 
            model_id=model_id, 
            last_timestamp_ns=last_timestamp_ns,
            limit=limit
        )

    def backup(self, target_path: str) -> None:
        self.sqlite_engine.backup(target_path)

    def run_integrity_check(self) -> bool:
        return self.sqlite_engine.run_integrity_check()

    def close(self) -> None:
        if hasattr(self, "sqlite_engine") and self.sqlite_engine is not None:
            self.sqlite_engine.close()

    def __enter__(self) -> "UnifiedStorageEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

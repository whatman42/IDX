"""
=============================================================================
IDX Stock Signal Engine - Unified Telemetry & System Monitoring Engine
Module           : monitoring.py
Directory Context: Flat Root Directory (selevel dengan main.py)
Version          : 2026.Q3.v1.5.4 (CLI Health Check Entry Point & Production Grade)

Aturan & Kepatuhan:
1. Mematuhi spesifikasi API & batas waktu staleness candlestick saham IDX (<= 48 jam).
2. Memeriksa konektivitas egress secara dinamis via Proxy URL (IDX_PROXY_URL / TOKOCRYPTO_PROXY_URL / BASE_URL_SITE) 
   atau langsung ke Yahoo Finance / IDX API (query1.finance.yahoo.com, idx.co.id) secara paralel.
3. Waktu dilaporkan dalam standar WIB (Asia/Jakarta) sesuai aturan Bursa Efek Indonesia.
4. Struktur flat directory tanpa subfolder import.
=============================================================================
"""

import os
import sys
import socket
import sqlite3
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import polars as pl
import psutil
from scipy import stats

# Standar Zona Waktu Pasar Saham Indonesia (WIB)
WIB_TZ = ZoneInfo("Asia/Jakarta")

# ============================================================================
# ADAPTIVE IMPORTS & FALLBACK MECHANISMS (FLAT DIRECTORY COMPLIANCE)
# ============================================================================
try:
    from logger import get_logger
except ImportError:
    import logging
    def get_logger(name: str):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

try:
    from exceptions import (
        DataMonitorError,
        DriftDashboardError,
        HealthCheckError,
        ModelMonitorError,
        RuntimeMonitorError,
        MonitoringEngineError,
    )
except ImportError:
    class DataMonitorError(Exception): pass
    class DriftDashboardError(Exception): pass
    class HealthCheckError(Exception): pass
    class ModelMonitorError(Exception): pass
    class RuntimeMonitorError(Exception): pass
    class MonitoringEngineError(Exception): pass

logger = get_logger("IDX.Monitoring")

# ============================================================================
# DEFENSIVE SANITIZATION HELPERS
# ============================================================================
def _ensure_polars_df_monitoring(data: Any, default_cols: Optional[List[str]] = None) -> pl.DataFrame:
    """Sanitasi input internal untuk menjamin objek bertipe pl.DataFrame."""
    if data is None:
        cols = default_cols or ["timestamp", "symbol", "close"]
        return pl.DataFrame(schema={col: pl.String for col in cols})
    if isinstance(data, list):
        if not data:
            cols = default_cols or ["timestamp", "symbol", "close"]
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

def _get_wib_timestamp_str(epoch_sec: Optional[float] = None) -> str:
    """Mengembalikan string waktu berformat WIB (Asia/Jakarta)."""
    if epoch_sec is not None:
        dt = datetime.fromtimestamp(epoch_sec, tz=WIB_TZ)
    else:
        dt = datetime.now(WIB_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S WIB")

# ============================================================================
# IDX LOCKED CONSTANTS & BOUNDARIES
# ============================================================================
IDX_MAX_STALENESS_SEC: float = 172800.0  # Batas maksimal usia data candle harian saham (48 Jam)
TOKOCRYPTO_MAX_STALENESS_SEC: float = IDX_MAX_STALENESS_SEC  # Compatibility Alias

def _resolve_default_network_targets() -> List[Tuple[str, int]]:
    """Resolves network egress check targets dynamically for IDX market data."""
    targets: List[Tuple[str, int]] = []
    proxy_url = (
        os.getenv("IDX_PROXY_URL", "").strip() or 
        os.getenv("TOKOCRYPTO_PROXY_URL", "").strip() or 
        os.getenv("BASE_URL_SITE", "").strip()
    )
    
    if proxy_url:
        try:
            parsed = urlparse(proxy_url if "://" in proxy_url else f"https://{proxy_url}")
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else (80 if parsed.scheme == "http" else 443))
            if host:
                targets.append((host, port))
        except Exception as err:
            logger.debug(f"Failed to parse proxy URL '{proxy_url}' for network check: {err}")
    
    if not targets:
        targets.extend([
            ("query1.finance.yahoo.com", 443),
            ("finance.yahoo.com", 443),
            ("www.idx.co.id", 443)
        ])
        
    fallbacks = [("1.1.1.1", 53), ("8.8.8.8", 53)]
    for h, p in fallbacks:
        if not any(t[0] == h for t in targets):
            targets.append((h, p))
            
    return targets

EXCLUDED_NON_NUMERIC_COLS = {
    "date", "timestamp", "time", "asset", "ticker", "symbol", "created_at",
    "portfolio_asset_id", "allocation_reason", "sector", "industry", "country"
}

# ============================================================================
# 1. DATA MONITOR ENGINE
# ============================================================================
class DataMonitor:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.iqr_factor: float = float(self.config.get("iqr_factor", 1.5))
        self.max_staleness_sec: float = float(
            self.config.get("max_staleness_sec", IDX_MAX_STALENESS_SEC)
        )
        self.missing_rate_threshold: float = float(self.config.get("missing_rate_threshold", 0.05))

        logger.info(
            f"DataMonitor initialized | IQR Factor: {self.iqr_factor} | "
            f"Max Staleness: {self.max_staleness_sec}s | Missing Threshold: {self.missing_rate_threshold * 100}%"
        )

    def validate_schema(self, df: pl.DataFrame, expected_schema: Dict[str, pl.DataType]) -> Dict[str, Any]:
        df = _ensure_polars_df_monitoring(df)
        if df.is_empty():
            return {"status": "FAILED", "error": "Target DataFrame contains zero rows structural allocation."}

        current_schema = df.schema
        missing_fields: List[str] = []
        type_mismatches: Dict[str, Dict[str, str]] = {}

        for field, expected_type in expected_schema.items():
            if field not in current_schema:
                missing_fields.append(field)
            elif current_schema[field] != expected_type:
                type_mismatches[field] = {
                    "expected": str(expected_type),
                    "actual": str(current_schema[field])
                }

        is_healthy = len(missing_fields) == 0 and len(type_mismatches) == 0
        status = "HEALTHY" if is_healthy else "CRITICAL"

        metrics = {
            "status": status,
            "metrics": {
                "total_columns": len(current_schema),
                "missing_columns_count": len(missing_fields),
                "type_mismatches_count": len(type_mismatches)
            },
            "anomalies": {
                "missing_fields": missing_fields,
                "type_mismatches": type_mismatches
            }
        }

        if not is_healthy:
            logger.error(f"Schema non-conformity detected. Status: {status}. Anomalies: {metrics['anomalies']}")

        return metrics

    def check_missing_and_invalid(self, df: pl.DataFrame, columns: List[str]) -> Dict[str, Any]:
        df = _ensure_polars_df_monitoring(df)
        total_rows = df.height
        if total_rows == 0:
            return {"status": "EMPTY", "metrics": {}}

        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns:
            return {"status": "HEALTHY", "metrics": {"message": "Zero target validation columns found matching schema."}}

        select_expressions: List[pl.Expr] = []
        for col in valid_columns:
            dtype = df.schema[col]
            select_expressions.append(pl.col(col).is_null().sum().alias(f"{col}_nulls"))

            if dtype in (pl.Float32, pl.Float64):
                select_expressions.append(pl.col(col).is_nan().sum().alias(f"{col}_nans"))
                select_expressions.append(
                    ((pl.col(col) == float("inf")) | (pl.col(col) == float("-inf"))).sum().alias(f"{col}_infs")
                )
            else:
                select_expressions.append(pl.lit(0, dtype=pl.UInt32).alias(f"{col}_nans"))
                select_expressions.append(pl.lit(0, dtype=pl.UInt32).alias(f"{col}_infs"))

        computed_res = df.select(select_expressions).row(0)
        res_map = dict(zip([expr.meta.output_name() for expr in select_expressions], computed_res))

        column_reports: Dict[str, Any] = {}
        global_breach = False

        for col in valid_columns:
            nulls = res_map[f"{col}_nulls"] or 0
            nans = res_map[f"{col}_nans"] or 0
            infs = res_map[f"{col}_infs"] or 0
            total_invalid = nulls + nans + infs

            missing_rate = total_invalid / total_rows
            breached = missing_rate > self.missing_rate_threshold
            if breached:
                global_breach = True

            column_reports[col] = {
                "null_count": nulls,
                "nan_count": nans,
                "inf_count": infs,
                "total_invalid": total_invalid,
                "missing_rate": round(missing_rate, 6),
                "threshold_breached": breached
            }

        status = "CRITICAL" if global_breach else "HEALTHY"
        return {
            "status": status,
            "metrics": {
                "total_records_evaluated": total_rows,
                "columns": column_reports
            }
        }

    def check_chronology_and_freshness(self, df: pl.DataFrame, timestamp_col: str) -> Dict[str, Any]:
        df = _ensure_polars_df_monitoring(df)
        total_rows = df.height
        if total_rows == 0:
            return {"status": "EMPTY", "metrics": {}}

        if timestamp_col not in df.columns:
            return {"status": "HEALTHY", "metrics": {"message": f"Timestamp col '{timestamp_col}' absent from schema."}}

        ts_dtype = df.schema[timestamp_col]

        try:
            if ts_dtype == pl.Datetime:
                time_unit = getattr(ts_dtype, "time_unit", "us")
                divisor = 1_000_000_000.0 if time_unit == "ns" else (1_000_000.0 if time_unit == "us" else 1000.0)
                ts_vector = df.select(pl.col(timestamp_col).to_physical().cast(pl.Float64) / divisor)
            elif ts_dtype == pl.Date:
                ts_vector = df.select(pl.col(timestamp_col).to_physical().cast(pl.Float64) * 86400.0)
            elif ts_dtype in (pl.String, pl.Categorical):
                parsed = pl.col(timestamp_col).str.to_datetime(strict=False)
                ts_vector = df.select(parsed.to_physical().cast(pl.Float64) / 1_000_000.0)
            elif ts_dtype.is_numeric():
                max_val = df.select(pl.col(timestamp_col).max()).row(0)[0]
                if max_val is not None and max_val > 0:
                    divisor = 1_000_000_000.0 if max_val > 1e16 else (1_000_000.0 if max_val > 1e13 else (1000.0 if max_val > 1e10 else 1.0))
                    ts_vector = df.select(pl.col(timestamp_col).cast(pl.Float64) / divisor)
                else:
                    ts_vector = df.select(pl.col(timestamp_col).cast(pl.Float64))
            else:
                return {"status": "HEALTHY", "metrics": {"message": "Unsupported timestamp column type."}}

            v_col = ts_vector.columns[0]
            chronology_expr = ts_vector.select([
                pl.col(v_col).max().alias("max_ts"),
                (pl.col(v_col).diff() < 0).cast(pl.UInt32).sum().alias("inversion_count")
            ]).row(0)

            max_timestamp = chronology_expr[0]
            inversion_count = chronology_expr[1] or 0

            if max_timestamp is None:
                return {"status": "DEGRADED", "error": "Chronological evaluation vector yielded null boundaries."}

            current_epoch = time.time()
            staleness_sec = max(0.0, current_epoch - max_timestamp)
            freshness_score = max(0.0, 1.0 - (staleness_sec / self.max_staleness_sec))
            
            is_stale = staleness_sec > self.max_staleness_sec
            is_monotonic = inversion_count == 0

            if is_stale or not is_monotonic:
                status = "CRITICAL"
                logger.warning(
                    f"Chronological degradation. Stale: {is_stale} ({round(staleness_sec, 2)}s), "
                    f"Inversions: {inversion_count}"
                )
            else:
                status = "HEALTHY"

            return {
                "status": status,
                "metrics": {
                    "max_timestamp_epoch": round(max_timestamp, 4),
                    "staleness_seconds": round(staleness_sec, 4),
                    "freshness_score": round(freshness_score, 4),
                    "chronological_inversions": inversion_count,
                    "is_monotonic": is_monotonic
                }
            }
        except Exception as err:
            logger.error(f"Error checking chronology and freshness: {err}")
            return {"status": "DEGRADED", "error": str(err)}

    def check_outliers_iqr(self, df: pl.DataFrame, columns: List[str]) -> Dict[str, Any]:
        df = _ensure_polars_df_monitoring(df)
        total_rows = df.height
        if total_rows == 0:
            return {"status": "EMPTY", "metrics": {}}

        numeric_cols = [
            c for c in columns 
            if c in df.columns 
            and df.schema[c].is_numeric() 
            and c.lower() not in EXCLUDED_NON_NUMERIC_COLS
        ]

        if not numeric_cols:
            return {"status": "HEALTHY", "metrics": {"message": "No numerical features discovered for IQR profiling."}}

        iqr_expressions: List[pl.Expr] = []
        for col in numeric_cols:
            q25 = pl.col(col).quantile(0.25)
            q75 = pl.col(col).quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - (self.iqr_factor * iqr)
            upper_bound = q75 + (self.iqr_factor * iqr)

            iqr_expressions.append(
                ((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).cast(pl.UInt32).sum().alias(f"{col}_outliers")
            )

        computed_outliers = df.select(iqr_expressions).row(0)
        outlier_map = dict(zip(numeric_cols, computed_outliers))

        outlier_reports: Dict[str, Any] = {}
        for col in numeric_cols:
            count = outlier_map[col] or 0
            rate = count / total_rows
            outlier_reports[col] = {
                "outlier_count": count,
                "outlier_rate": round(rate, 6)
            }

        return {
            "status": "HEALTHY",
            "metrics": {
                "total_records_evaluated": total_rows,
                "columns": outlier_reports
            }
        }

    def run_all(
        self,
        df: pl.DataFrame,
        target_columns: List[str],
        timestamp_col: Optional[str] = None,
        expected_schema: Optional[Dict[str, pl.DataType]] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        df = _ensure_polars_df_monitoring(df)

        report: Dict[str, Any] = {
            "schema_validation": {"status": "SKIPPED"},
            "missing_and_invalid": {"status": "SKIPPED"},
            "chronology_and_freshness": {"status": "SKIPPED"},
            "outliers_iqr": {"status": "SKIPPED"}
        }

        if expected_schema:
            report["schema_validation"] = self.validate_schema(df, expected_schema)

        if target_columns:
            report["missing_and_invalid"] = self.check_missing_and_invalid(df, target_columns)
            report["outliers_iqr"] = self.check_outliers_iqr(df, target_columns)

        if timestamp_col:
            report["chronology_and_freshness"] = self.check_chronology_and_freshness(df, timestamp_col)

        statuses = [
            section["status"] for section in report.values() 
            if isinstance(section, dict) and "status" in section and section["status"] != "SKIPPED"
        ]

        if "CRITICAL" in statuses or "FAILED" in statuses:
            aggregate_status = "UNHEALTHY"
        elif "DEGRADED" in statuses:
            aggregate_status = "DEGRADED"
        else:
            aggregate_status = "HEALTHY"

        return {
            "framework": "IDX_Stock_Analysis_Engine",
            "metrics_domain": "DATA_QUALITY",
            "timestamp": _get_wib_timestamp_str(),
            "aggregate_status": aggregate_status,
            "execution_duration_sec": round(time.perf_counter() - start_time, 6),
            "details": report
        }

# ============================================================================
# 2. HEALTH CHECK ENGINE
# ============================================================================
class HealthCheckEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        
        self.memory_threshold_pct: float = float(self.config.get("memory_threshold_pct", 85.0))
        self.disk_threshold_pct: float = float(self.config.get("disk_threshold_pct", 90.0))
        self.network_timeout_sec: float = float(self.config.get("network_timeout_sec", 2.0))
        
        self.workspace_dir: Path = Path(self.config.get("workspace_dir", Path.cwd()))
        self.db_path: Optional[Path] = Path(self.config["db_path"]) if "db_path" in self.config else None
        
        self.network_targets: List[Tuple[str, int]] = (
            self.config.get("network_targets") or _resolve_default_network_targets()
        )

    def check_memory(self) -> Dict[str, Any]:
        try:
            vm = psutil.virtual_memory()
            used_pct = vm.percent
            is_healthy = used_pct < self.memory_threshold_pct
            return {
                "status": "HEALTHY" if is_healthy else "CRITICAL",
                "metrics": {
                    "total_memory_mb": round(vm.total / (1024 * 1024), 2),
                    "available_memory_mb": round(vm.available / (1024 * 1024), 2),
                    "used_percentage": used_pct
                },
                "remediation": None if is_healthy else "Trigger GC or scale down execution workers."
            }
        except Exception as e:
            logger.error(f"Memory diagnostics failure: {str(e)}")
            return {"status": "FAILED", "metrics": {}, "error": str(e)}

    def check_disk(self) -> Dict[str, Any]:
        try:
            target_path = self.workspace_dir.resolve()
            usage = psutil.disk_usage(str(target_path))
            used_pct = usage.percent
            is_healthy = used_pct < self.disk_threshold_pct
            return {
                "status": "HEALTHY" if is_healthy else "CRITICAL",
                "metrics": {
                    "target_path": str(target_path),
                    "total_disk_gb": round(usage.total / (1024**3), 2),
                    "free_disk_gb": round(usage.free / (1024**3), 2),
                    "used_percentage": used_pct
                },
                "remediation": None if is_healthy else "Purge obsolete checkpoints and log files."
            }
        except Exception as e:
            logger.error(f"Disk diagnostics failure: {str(e)}")
            return {"status": "FAILED", "metrics": {}, "error": str(e)}

    def _probe_socket(self, host: str, port: int) -> Tuple[str, int, bool, float, str]:
        start_time = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=self.network_timeout_sec):
                latency_ms = (time.perf_counter() - start_time) * 1000
                return (host, port, True, round(latency_ms, 2), "")
        except Exception as err:
            return (host, port, False, 0.0, str(err))

    def check_network(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"targets_evaluated": [], "successful_connections": 0}
        if not self.network_targets:
            return {"status": "HEALTHY", "metrics": metrics, "remediation": None}

        max_workers = min(8, len(self.network_targets))
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._probe_socket, host, port) for host, port in self.network_targets]
            for future in as_completed(futures):
                results.append(future.result())

        all_failed = True
        for host, port, reachable, latency_ms, err_msg in results:
            target_str = f"{host}:{port}"
            if reachable:
                metrics["targets_evaluated"].append({"target": target_str, "reachable": True, "latency_ms": latency_ms})
                metrics["successful_connections"] += 1
                all_failed = False
            else:
                metrics["targets_evaluated"].append({"target": target_str, "reachable": False, "error": err_msg})

        if all_failed:
            status, remediation = "CRITICAL", "Investigate local routing, outbound firewall, or DNS config."
        elif metrics["successful_connections"] < len(self.network_targets):
            status, remediation = "DEGRADED", "Intermittent external dependency failure. Verify Proxy or ISP status."
        else:
            status, remediation = "HEALTHY", None

        return {"status": status, "metrics": metrics, "remediation": remediation}

    def check_file_io(self) -> Dict[str, Any]:
        can_read = can_write = can_delete = False
        test_file = self.workspace_dir / ".health_io_canary.tmp"
        canary_payload = b"IDX_STOCK_BOT_HEALTH_CHECK_PAYLOAD_2026"
        
        try:
            with open(test_file, "wb") as f:
                f.write(canary_payload)
            can_write = True
            
            with open(test_file, "rb") as f:
                if f.read() == canary_payload:
                    can_read = True
                
            os.remove(test_file)
            if not test_file.exists():
                can_delete = True
                
            is_healthy = can_read and can_write and can_delete
            return {
                "status": "HEALTHY" if is_healthy else "CRITICAL",
                "metrics": {"write_privilege": can_write, "read_privilege": can_read, "delete_privilege": can_delete},
                "remediation": None if is_healthy else "Modify file system ACL settings."
            }
        except Exception as e:
            logger.error(f"File system I/O operation failure: {str(e)}")
            if test_file.exists():
                try: os.remove(test_file)
                except Exception: pass
            return {
                "status": "CRITICAL",
                "metrics": {"write_privilege": can_write, "read_privilege": can_read, "delete_privilege": can_delete},
                "error": str(e)
            }

    def check_sqlite(self) -> Dict[str, Any]:
        if not self.db_path:
            return {"status": "HEALTHY", "metrics": {"database_configured": False}, "remediation": None}
            
        start_time = time.perf_counter()
        conn: Optional[sqlite3.Connection] = None
        
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=self.network_timeout_sec)
            cursor = conn.cursor()
            cursor.execute("PRAGMA busy_timeout = 2000;")
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()
            
            cursor.execute("PRAGMA quick_check(1);")
            check_res = cursor.fetchone()
            db_ok = check_res and str(check_res[0]).lower() == "ok"

            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if result and result[0] == 1 and db_ok:
                return {
                    "status": "HEALTHY",
                    "metrics": {
                        "database_configured": True,
                        "responsive": True,
                        "integrity": "OK",
                        "query_latency_ms": round(latency_ms, 2)
                    },
                    "remediation": None
                }
            else:
                raise HealthCheckError("SQLite quick check returned non-ok response.")
        except Exception as e:
            logger.error(f"Database structural health failure at {self.db_path}: {str(e)}")
            return {"status": "CRITICAL", "metrics": {"database_configured": True, "responsive": False}, "error": str(e)}
        finally:
            if conn: conn.close()

    def run_all(self) -> Dict[str, Any]:
        start_timestamp = time.time()
        diagnostics = {
            "memory": self.check_memory(),
            "disk": self.check_disk(),
            "network": self.check_network(),
            "file_io": self.check_file_io(),
            "database": self.check_sqlite()
        }
        
        statuses = [res["status"] for res in diagnostics.values()]
        aggregate_status = "UNHEALTHY" if ("CRITICAL" in statuses or "FAILED" in statuses) else ("DEGRADED" if "DEGRADED" in statuses else "HEALTHY")
        
        return {
            "framework": "IDX_Stock_Analysis_Engine",
            "timestamp": _get_wib_timestamp_str(start_timestamp),
            "aggregate_status": aggregate_status,
            "execution_duration_sec": round(time.time() - start_timestamp, 4),
            "diagnostics": diagnostics
        }

# ============================================================================
# 3. MODEL MONITOR ENGINE
# ============================================================================
class ModelMonitor:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.max_history_size: int = int(self.config.get("max_history_size", 10000))
        self.drift_alpha: float = float(self.config.get("drift_alpha", 0.05))

        self._lock = threading.Lock()
        self._inference_history: List[Dict[str, Any]] = []

    def record_inference_metrics(
        self,
        model_id: str,
        latency_sec: float,
        predictions: Union[List[float], np.ndarray, pl.Series],
        actuals: Optional[Union[List[float], np.ndarray, pl.Series]] = None
    ) -> None:
        preds_array = np.asarray(predictions, dtype=np.float64).ravel()
        acts_array = np.asarray(actuals, dtype=np.float64).ravel() if actuals is not None else None

        if preds_array.size == 0:
            return

        payload = {
            "timestamp": time.time(),
            "model_id": model_id,
            "latency_sec": float(latency_sec),
            "predictions": preds_array,
            "actuals": acts_array
        }

        with self._lock:
            if len(self._inference_history) >= self.max_history_size:
                self._inference_history.pop(0)
            self._inference_history.append(payload)

    def compute_latency_percentiles(self, model_id: str) -> Dict[str, Any]:
        with self._lock:
            latencies = [
                entry["latency_sec"] for entry in self._inference_history 
                if entry["model_id"] == model_id
            ]

        if not latencies:
            return {"status": "INSUFFICIENT_DATA", "metrics": {}}

        lat_array = np.array(latencies, dtype=np.float64)
        return {
            "status": "HEALTHY",
            "metrics": {
                "sample_count": len(lat_array),
                "mean_sec": round(float(np.mean(lat_array)), 6),
                "min_sec": round(float(np.min(lat_array)), 6),
                "max_sec": round(float(np.max(lat_array)), 6),
                "p50_sec": round(float(np.percentile(lat_array, 50)), 6),
                "p95_sec": round(float(np.percentile(lat_array, 95)), 6),
                "p99_sec": round(float(np.percentile(lat_array, 99)), 6)
            }
        }

    def evaluate_prediction_drift(
        self, 
        model_id: str, 
        baseline_predictions: Union[List[float], np.ndarray, pl.Series]
    ) -> Dict[str, Any]:
        baseline_arr = np.asarray(baseline_predictions, dtype=np.float64).ravel()
        if baseline_arr.size == 0:
            raise ModelMonitorError("Baseline prediction sequence cannot be empty.")

        with self._lock:
            current_chunks = [
                entry["predictions"] for entry in self._inference_history 
                if entry["model_id"] == model_id
            ]

        if not current_chunks:
            return {"status": "INSUFFICIENT_DATA", "drift_detected": False}

        current_arr = np.concatenate(current_chunks)
        baseline_clean = baseline_arr[np.isfinite(baseline_arr)]
        current_clean = current_arr[np.isfinite(current_arr)]

        if baseline_clean.size == 0 or current_clean.size == 0:
            return {"status": "INSUFFICIENT_DATA", "drift_detected": False}

        try:
            ks_stat, p_value = stats.ks_2samp(baseline_clean, current_clean)
            drift_detected = bool(p_value < self.drift_alpha)
            return {
                "status": "CRITICAL" if drift_detected else "HEALTHY",
                "drift_detected": drift_detected,
                "metrics": {
                    "ks_statistic": round(float(ks_stat), 6),
                    "p_value": float(p_value),
                    "baseline_samples": len(baseline_clean),
                    "current_samples": len(current_clean)
                }
            }
        except Exception as e:
            return {"status": "ERROR", "error": str(e), "drift_detected": False}

    def generate_comprehensive_report(
        self, 
        model_id: str, 
        task_type: str, 
        baseline_predictions: Optional[Union[List[float], np.ndarray, pl.Series]] = None
    ) -> Dict[str, Any]:
        latency_report = self.compute_latency_percentiles(model_id)
        drift_report = {"status": "SKIPPED", "drift_detected": False}
        if baseline_predictions is not None:
            drift_report = self.evaluate_prediction_drift(model_id, baseline_predictions)

        statuses = [latency_report.get("status"), drift_report.get("status")]
        if "CRITICAL" in statuses or "ERROR" in statuses or drift_report.get("drift_detected"):
            aggregate_status = "UNHEALTHY"
        elif "DEGRADED" in statuses or "INSUFFICIENT_DATA" in statuses:
            aggregate_status = "DEGRADED"
        else:
            aggregate_status = "HEALTHY"

        return {
            "framework": "IDX_Stock_Analysis_Engine",
            "metrics_domain": "MODEL_PERFORMANCE",
            "model_id": model_id,
            "timestamp": _get_wib_timestamp_str(),
            "aggregate_status": aggregate_status,
            "latency_profile": latency_report,
            "drift_profile": drift_report
        }

# ============================================================================
# 4. RUNTIME MONITOR ENGINE & COMPONENT CONTEXT
# ============================================================================
class ComponentContext:
    def __init__(self, component_name: str, monitor: "RuntimeMonitor") -> None:
        self.component_name = component_name
        self.monitor = monitor
        self._process = psutil.Process(os.getpid())
        self.start_wall_time: float = 0.0
        self.start_cpu_times: Optional[Any] = None
        self.start_memory_rss: int = 0

    def __enter__(self) -> "ComponentContext":
        try:
            self.start_memory_rss = self._process.memory_info().rss
            self.start_cpu_times = self._process.cpu_times()
            self.start_wall_time = time.perf_counter()
        except Exception as e:
            logger.error(f"Failed to initialize instrumentation for {self.component_name}: {str(e)}")
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> bool:
        end_wall_time = time.perf_counter()
        if not self.start_cpu_times:
            return False

        try:
            end_cpu_times = self._process.cpu_times()
            end_memory_rss = self._process.memory_info().rss

            wall_duration = end_wall_time - self.start_wall_time
            cpu_user_delta = end_cpu_times.user - self.start_cpu_times.user
            cpu_system_delta = end_cpu_times.system - self.start_cpu_times.system
            rss_delta_mb = (end_memory_rss - self.start_memory_rss) / (1024 * 1024)

            self.monitor._record_metrics({
                "timestamp": time.time(),
                "component_name": self.component_name,
                "wall_duration_sec": round(wall_duration, 6),
                "cpu_total_sec": round(cpu_user_delta + cpu_system_delta, 6),
                "rss_delta_mb": round(rss_delta_mb, 4),
                "status": "SUCCESS" if exc_type is None else "FAILED",
                "error_message": str(exc_val) if exc_val else None
            })
        except Exception as e:
            logger.error(f"Failed to record metrics for {self.component_name}: {str(e)}")
        return False

class RuntimeMonitor:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.max_history_size: int = int(self.config.get("max_history_size", 10000))
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []

    def track(self, component_name: str) -> ComponentContext:
        return ComponentContext(component_name=component_name, monitor=self)

    def _record_metrics(self, metrics: Dict[str, Any]) -> None:
        with self._lock:
            if len(self._history) >= self.max_history_size:
                self._history.pop(0)
            self._history.append(metrics)

    def compute_summary_statistics(self) -> Dict[str, Any]:
        with self._lock:
            if not self._history:
                return {"status": "EMPTY", "components": {}}
            df = pl.DataFrame(self._history)

        try:
            summary_df = df.group_by("component_name").agg([
                pl.len().alias("execution_count"),
                pl.col("wall_duration_sec").mean().alias("avg_wall_duration"),
                pl.col("wall_duration_sec").max().alias("max_wall_duration"),
                pl.col("cpu_total_sec").sum().alias("cumulative_cpu_sec"),
                pl.col("rss_delta_mb").mean().alias("avg_rss_delta_mb"),
                (pl.col("status") == "FAILED").cast(pl.UInt32).sum().alias("failure_count")
            ])

            summary_dict: Dict[str, Any] = {"status": "ACTIVE", "components": {}}
            for row in summary_df.iter_rows(named=True):
                comp_name = row["component_name"]
                exec_count = row["execution_count"]
                fail_count = row["failure_count"]

                summary_dict["components"][comp_name] = {
                    "execution_count": exec_count,
                    "failure_rate": round((fail_count / exec_count) if exec_count > 0 else 0.0, 4),
                    "avg_wall_duration_sec": round(row["avg_wall_duration"], 4),
                    "max_wall_duration_sec": round(row["max_wall_duration"], 4),
                    "cumulative_cpu_time_sec": round(row["cumulative_cpu_sec"], 4),
                    "avg_memory_growth_mb": round(row["avg_rss_delta_mb"], 4),
                }

            return summary_dict
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

# ============================================================================
# 5. DRIFT DASHBOARD
# ============================================================================
class DriftDashboard:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def generate_summary_dict(self, data_report: Dict[str, Any], model_report: Dict[str, Any]) -> Dict[str, Any]:
        data_status = data_report.get("aggregate_status", "SKIPPED")
        model_status = model_report.get("aggregate_status", "SKIPPED")
        
        statuses = [data_status, model_status]
        if "UNHEALTHY" in statuses or "CRITICAL" in statuses:
            aggregate_status = "UNHEALTHY"
        elif "DEGRADED" in statuses:
            aggregate_status = "DEGRADED"
        else:
            aggregate_status = "HEALTHY"

        return {
            "framework": "IDX_Stock_Analysis_Engine",
            "timestamp": _get_wib_timestamp_str(),
            "aggregate_status": aggregate_status,
            "data_subsystem": {"status": data_status},
            "model_subsystem": {"status": model_status}
        }

    def render_markdown(self, data_report: Dict[str, Any], model_report: Dict[str, Any]) -> str:
        summary = self.generate_summary_dict(data_report, model_report)
        return f"# IDX Stock Signal Engine Diagnostic Report\n**Status**: `{summary['aggregate_status']}`\n**Time**: {summary['timestamp']}"

    def render_html(self, data_report: Dict[str, Any], model_report: Dict[str, Any]) -> str:
        summary = self.generate_summary_dict(data_report, model_report)
        return f"<html><body><h1>IDX Health Check: {summary['aggregate_status']}</h1></body></html>"

# ============================================================================
# 6. FACADE CLASS & CLI ENTRY POINT
# ============================================================================
class UnifiedMonitoringEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.data_monitor = DataMonitor(self.config.get("data_monitor"))
        self.health_engine = HealthCheckEngine(self.config.get("health_check"))
        self.model_monitor = ModelMonitor(self.config.get("model_monitor"))
        self.runtime_monitor = RuntimeMonitor(self.config.get("runtime_monitor"))
        self.drift_dashboard = DriftDashboard(self.config.get("drift_dashboard"))

    def execute_full_audit(self) -> Dict[str, Any]:
        health_report = self.health_engine.run_all()
        return {
            "framework": "IDX_Stock_Analysis_Engine",
            "audit_timestamp": _get_wib_timestamp_str(),
            "global_status": health_report.get("aggregate_status", "UNKNOWN"),
            "health_diagnostics": health_report
        }

if __name__ == "__main__":
    logger.info("==================================================")
    logger.info("🩺 EXECUTING SYSTEM HEALTH & NETWORK EGRESS DIAGNOSTICS")
    logger.info("==================================================")
    
    engine = HealthCheckEngine()
    report = engine.run_all()
    
    status = report.get("aggregate_status", "UNKNOWN")
    logger.info(f"📊 Diagnostic Status Result: [{status}]")
    
    for diag_name, diag_data in report.get("diagnostics", {}).items():
        sub_status = diag_data.get("status", "UNKNOWN")
        logger.info(f"  └─ Subsystem '{diag_name}': {sub_status}")
        if sub_status in ("CRITICAL", "FAILED"):
            logger.error(f"     Details: {diag_data}")

    if status in ("UNHEALTHY", "CRITICAL", "FAILED"):
        logger.error("❌ System Diagnostics Failed! Halting Workflow execution.")
        sys.exit(1)
    else:
        logger.info("✅ All System Diagnostics Passed.")
        sys.exit(0)

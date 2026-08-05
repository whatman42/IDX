"""
=============================================================================
IDX Stock Signal Engine - Unified Telemetry & System Monitoring Engine
Module           : monitoring.py
Directory Context: Flat Root Directory (selevel dengan main.py)
Version          : 2026.Q3.v1.5.1 (IDX & Proxy-Aware Egress Health Check - Production Grade)

Aturan & Kepatuhan:
1. Mematuhi spesifikasi API & batas waktu staleness candlestick saham IDX (<= 12 jam).
2. Memeriksa konektivitas egress secara dinamis via Proxy URL (IDX_PROXY_URL / TOKOCRYPTO_PROXY_URL / BASE_URL_SITE) 
   atau langsung ke Yahoo Finance / IDX API (query1.finance.yahoo.com, idx.co.id) secara paralel.
3. Struktur flat directory tanpa subfolder import (src.).
=============================================================================
"""

import os
import socket
import sqlite3
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from urllib.parse import urlparse

import numpy as np
import polars as pl
import psutil
from scipy import stats

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
        return pl.DataFrame(schema={col: pl.Utf8 for col in cols})
    if isinstance(data, list):
        if not data:
            cols = default_cols or ["timestamp", "symbol", "close"]
            return pl.DataFrame(schema={col: pl.Utf8 for col in cols})
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

# ============================================================================
# IDX LOCKED CONSTANTS & BOUNDARIES
# ============================================================================
IDX_MAX_STALENESS_SEC: float = 43200.0  # Batas maksimal usia data candle harian saham (12 Jam)
TOKOCRYPTO_MAX_STALENESS_SEC: float = IDX_MAX_STALENESS_SEC  # Compatibility Alias

def _resolve_default_network_targets() -> List[Tuple[str, int]]:
    """
    Resolves network egress check targets dynamically for IDX market data.
    Prioritizes IDX_PROXY_URL, TOKOCRYPTO_PROXY_URL, or BASE_URL_SITE if defined to prevent timeouts.
    """
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
    
    # Fallback to default Yahoo Finance / IDX endpoints if no proxy URL is specified
    if not targets:
        targets.append(("query1.finance.yahoo.com", 443))
        targets.append(("finance.yahoo.com", 443))
        targets.append(("www.idx.co.id", 443))
        
    fallbacks = [
        ("1.1.1.1", 53),
        ("8.8.8.8", 53)
    ]
    
    for h, p in fallbacks:
        if not any(t[0] == h for t in targets):
            targets.append((h, p))
            
    return targets

# Kolom-kolom metadata yang wajib dikecualikan dari kalkulasi statistik numerik murni
EXCLUDED_NON_NUMERIC_COLS = {
    "date", "timestamp", "time", "asset", "ticker", "symbol", "created_at",
    "portfolio_asset_id", "allocation_reason", "sector", "industry", "country"
}


# ============================================================================
# 1. DATA MONITOR ENGINE
# ============================================================================
class DataMonitor:
    """
    Engine for high-throughput, vectorized data validation and health assessment.
    Executes structural schema checking, missing/invalid data identification,
    chronological integrity verification, and statistical outlier isolation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.iqr_factor: float = float(self.config.get("iqr_factor", 1.5))
        self.max_staleness_sec: float = float(
            self.config.get("max_staleness_sec", IDX_MAX_STALENESS_SEC)
        )
        self.missing_rate_threshold: float = float(self.config.get("missing_rate_threshold", 0.05))

        logger.info(
            f"DataMonitor instantiated. Config -> IQR Factor: {self.iqr_factor}, "
            f"Max Staleness: {self.max_staleness_sec}s, Missing Threshold: {self.missing_rate_threshold * 100}%"
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
            logger.error(f"Schema non-conformity detected. Status: {status}. Anomalies tracked: {metrics['anomalies']}")

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
                    if max_val > 1e16:
                        divisor = 1_000_000_000.0  # Nanoseconds
                    elif max_val > 1e13:
                        divisor = 1_000_000.0      # Microseconds
                    elif max_val > 1e10:
                        divisor = 1000.0           # Milliseconds
                    else:
                        divisor = 1.0              # Seconds
                    ts_vector = df.select(pl.col(timestamp_col).cast(pl.Float64) / divisor)
                else:
                    ts_vector = df.select(pl.col(timestamp_col).cast(pl.Float64))
            else:
                return {"status": "HEALTHY", "metrics": {"message": "Unsupported timestamp column type."}}

            v_col = ts_vector.columns[0]
            chronology_expr = ts_vector.select([
                pl.col(v_col).max().alias("max_ts"),
                (pl.col(v_col).diff() < 0).sum().alias("inversion_count")
            ]).row(0)

            max_timestamp = chronology_expr[0]
            inversion_count = chronology_expr[1] or 0

            if max_timestamp is None:
                return {"status": "DEGRADED", "error": "Chronological evaluation vector yielded null data boundaries."}

            current_epoch = time.time()
            staleness_sec = max(0.0, current_epoch - max_timestamp)
            freshness_score = max(0.0, 1.0 - (staleness_sec / self.max_staleness_sec))
            
            is_stale = staleness_sec > self.max_staleness_sec
            is_monotonic = inversion_count == 0

            if is_stale or not is_monotonic:
                status = "CRITICAL"
                logger.warning(
                    f"Chronological degradation detected. Stale State: {is_stale} ({round(staleness_sec, 2)}s), "
                    f"Monotonic Violations: {inversion_count}"
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
                ((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).sum().alias(f"{col}_outliers")
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

        execution_duration = time.perf_counter() - start_time

        return {
            "framework": "IDX_Stock_Analysis_Engine",
            "metrics_domain": "DATA_QUALITY",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "aggregate_status": aggregate_status,
            "execution_duration_sec": round(execution_duration, 6),
            "details": report
        }


# ============================================================================
# 2. HEALTH CHECK ENGINE
# ============================================================================
class HealthCheckEngine:
    """
    Engine for executing deterministic diagnostic validation across critical system resources.
    Evaluates memory utilization, disk space, network egress (Yahoo Finance / IDX API Proxy),
    storage permissions, and database health.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        
        self.memory_threshold_pct: float = float(self.config.get("memory_threshold_pct", 85.0))
        self.disk_threshold_pct: float = float(self.config.get("disk_threshold_pct", 90.0))
        self.network_timeout_sec: float = float(self.config.get("network_timeout_sec", 2.0))
        
        self.workspace_dir: Path = Path(self.config.get("workspace_dir", Path.cwd()))
        self.db_path: Optional[Path] = Path(self.config["db_path"]) if "db_path" in self.config else None
        
        # Dynamically resolve network targets based on proxy settings
        self.network_targets: List[Tuple[str, int]] = (
            self.config.get("network_targets") or _resolve_default_network_targets()
        )

        logger.info(
            f"HealthCheckEngine initialized. Memory Max: {self.memory_threshold_pct}%, "
            f"Disk Max: {self.disk_threshold_pct}%. Workspace: {self.workspace_dir}"
        )

    def check_memory(self) -> Dict[str, Any]:
        try:
            vm = psutil.virtual_memory()
            used_pct = vm.percent
            available_mb = vm.available / (1024 * 1024)
            total_mb = vm.total / (1024 * 1024)
            
            is_healthy = used_pct < self.memory_threshold_pct
            status = "HEALTHY" if is_healthy else "CRITICAL"
            
            metrics = {
                "status": status,
                "metrics": {
                    "total_memory_mb": round(total_mb, 2),
                    "available_memory_mb": round(available_mb, 2),
                    "used_percentage": used_pct
                },
                "remediation": None if is_healthy else "Trigger process garaging, scale down execution workers, or invoke GC."
            }
            
            if not is_healthy:
                logger.error(f"Memory threshold breached: {used_pct}% utilized against max {self.memory_threshold_pct}%")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Memory diagnostics failure: {str(e)}")
            return {"status": "FAILED", "metrics": {}, "error": str(e), "remediation": "Verify OS-level telemetry permissions."}

    def check_disk(self) -> Dict[str, Any]:
        try:
            target_path = self.workspace_dir.resolve()
            usage = psutil.disk_usage(str(target_path))
            used_pct = usage.percent
            free_gb = usage.free / (1024 * 1024 * 1024)
            total_gb = usage.total / (1024 * 1024 * 1024)
            
            is_healthy = used_pct < self.disk_threshold_pct
            status = "HEALTHY" if is_healthy else "CRITICAL"
            
            metrics = {
                "status": status,
                "metrics": {
                    "target_path": str(target_path),
                    "total_disk_gb": round(total_gb, 2),
                    "free_disk_gb": round(free_gb, 2),
                    "used_percentage": used_pct
                },
                "remediation": None if is_healthy else "Purge obsolete checkpoints, historical logs, or archival metrics data."
            }
            
            if not is_healthy:
                logger.error(f"Disk allocation threshold breached at {target_path}: {used_pct}% utilized against max {self.disk_threshold_pct}%")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Disk diagnostics failure: {str(e)}")
            return {"status": "FAILED", "metrics": {}, "error": str(e), "remediation": "Confirm accessibility of the workspace path."}

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

        # Parallel Socket Connectivity Inspection
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
                metrics["targets_evaluated"].append({
                    "target": target_str,
                    "reachable": True,
                    "latency_ms": latency_ms
                })
                metrics["successful_connections"] += 1
                all_failed = False
            else:
                metrics["targets_evaluated"].append({
                    "target": target_str,
                    "reachable": False,
                    "error": err_msg
                })
                logger.warning(f"Egress target unreachable: {target_str} -> Reason: {err_msg}")

        if all_failed:
            status = "CRITICAL"
            remediation = "Investigate local routing table, outbound firewall rules, or DNS configuration."
            logger.error("Complete network egress paralysis detected across all diagnostic targets.")
        elif metrics["successful_connections"] < len(self.network_targets):
            status = "DEGRADED"
            remediation = "Intermittent external dependency failure. Verify ISP reliability or IDX Proxy status."
        else:
            status = "HEALTHY"
            remediation = None

        return {
            "status": status,
            "metrics": metrics,
            "remediation": remediation
        }

    def check_file_io(self) -> Dict[str, Any]:
        can_read = False
        can_write = False
        can_delete = False
        
        test_file = self.workspace_dir / ".health_io_canary.tmp"
        canary_payload = b"IDX_STOCK_BOT_HEALTH_CHECK_PAYLOAD_2026"
        
        try:
            with open(test_file, "wb") as f:
                f.write(canary_payload)
            can_write = True
            
            with open(test_file, "rb") as f:
                content = f.read()
            if content == canary_payload:
                can_read = True
                
            os.remove(test_file)
            if not test_file.exists():
                can_delete = True
                
            is_healthy = can_read and can_write and can_delete
            status = "HEALTHY" if is_healthy else "CRITICAL"
            
            return {
                "status": status,
                "metrics": {
                    "write_privilege": can_write,
                    "read_privilege": can_read,
                    "delete_privilege": can_delete
                },
                "remediation": None if is_healthy else "Modify file system ACL settings or correct execution privileges."
            }
            
        except Exception as e:
            logger.error(f"File system I/O operation failure: {str(e)}")
            if test_file.exists():
                try:
                    os.remove(test_file)
                except Exception:
                    pass
            return {
                "status": "CRITICAL",
                "metrics": {"write_privilege": can_write, "read_privilege": can_read, "delete_privilege": can_delete},
                "error": str(e),
                "remediation": "Resolve structural disk lock constraints or system level root permission degradation."
            }

    def check_sqlite(self) -> Dict[str, Any]:
        if not self.db_path:
            return {
                "status": "HEALTHY",
                "metrics": {"database_configured": False},
                "remediation": None
            }
            
        start_time = time.perf_counter()
        conn: Optional[sqlite3.Connection] = None
        
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=self.network_timeout_sec)
            cursor = conn.cursor()
            
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()
            
            # Additional DB Integrity Check
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
                raise HealthCheckError("SQLite diagnostic probe returned non-deterministic or corrupted response stream.")
                
        except Exception as e:
            logger.error(f"Database structural health failure at {self.db_path}: {str(e)}")
            return {
                "status": "CRITICAL",
                "metrics": {
                    "database_configured": True,
                    "responsive": False
                },
                "error": str(e),
                "remediation": "Clear structural database journal file (.wal/.journal) corruption or fix file locks."
            }
        finally:
            if conn:
                conn.close()

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
        
        if "CRITICAL" in statuses or "FAILED" in statuses:
            aggregate_status = "UNHEALTHY"
        elif "DEGRADED" in statuses:
            aggregate_status = "DEGRADED"
        else:
            aggregate_status = "HEALTHY"
            
        execution_duration = time.time() - start_timestamp
        
        report = {
            "framework": "IDX_Stock_Analysis_Engine",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_timestamp)),
            "aggregate_status": aggregate_status,
            "execution_duration_sec": round(execution_duration, 4),
            "diagnostics": diagnostics
        }
        
        if aggregate_status != "HEALTHY":
            logger.warning(f"System performance anomaly detected. Aggregate Framework State: {aggregate_status}")
            
        return report


# ============================================================================
# 3. MODEL MONITOR ENGINE
# ============================================================================
class ModelMonitor:
    """
    Enterprise-grade engine for analytical model monitoring.
    Calculates statistical data drift using non-parametric tests (Kolmogorov-Smirnov),
    tracks precise execution latency distributions (p50, p95, p99), and captures 
    performance metric degradation across machine learning tasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.max_history_size: int = int(self.config.get("max_history_size", 10000))
        self.drift_alpha: float = float(self.config.get("drift_alpha", 0.05))
        self.epsilon: float = 1e-9

        self._lock = threading.Lock()
        self._inference_history: List[Dict[str, Any]] = []

        logger.info(
            f"ModelMonitor initialized. Max Capacity: {self.max_history_size} entries. "
            f"Drift Alpha Level: {self.drift_alpha}"
        )

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
            logger.warning(f"Empty prediction sequence rejected for model_id: {model_id}")
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

    def clear_metrics(self) -> None:
        with self._lock:
            self._inference_history.clear()
            logger.info("Internal inference telemetry state metrics cleared successfully.")

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
            raise ModelMonitorError("Baseline target validation sequence cannot be empty for drift profile metrics.")

        with self._lock:
            current_chunks = [
                entry["predictions"] for entry in self._inference_history 
                if entry["model_id"] == model_id
            ]

        if not current_chunks:
            return {"status": "INSUFFICIENT_DATA", "drift_detected": False}

        current_arr = np.concatenate([c.ravel() for c in current_chunks])
        
        try:
            ks_stat, p_value = stats.ks_2samp(baseline_arr, current_arr)
            drift_detected = bool(p_value < self.drift_alpha)
            status = "CRITICAL" if drift_detected else "HEALTHY"
            
            if drift_detected:
                logger.warning(
                    f"Statistical prediction drift isolated on model: {model_id}. "
                    f"KS-Stat: {round(ks_stat, 4)}, P-Value: {p_value}"
                )

            return {
                "status": status,
                "drift_detected": drift_detected,
                "metrics": {
                    "ks_statistic": round(float(ks_stat), 6),
                    "p_value": float(p_value),
                    "baseline_samples": len(baseline_arr),
                    "current_samples": len(current_arr)
                }
            }
        except Exception as e:
            logger.error(f"Algorithmic exception during model drift calculation: {str(e)}")
            return {"status": "ERROR", "error": str(e), "drift_detected": False}

    def evaluate_performance_degradation(self, model_id: str, task_type: str) -> Dict[str, Any]:
        with self._lock:
            valid_pairs = [
                (entry["predictions"], entry["actuals"]) 
                for entry in self._inference_history 
                if entry["model_id"] == model_id and entry["actuals"] is not None
            ]

        if not valid_pairs:
            return {"status": "NO_GROUND_TRUTH", "metrics": {}}

        all_preds = np.concatenate([pair[0].ravel() for pair in valid_pairs])
        all_acts = np.concatenate([pair[1].ravel() for pair in valid_pairs])

        if all_preds.size == 0 or all_acts.size == 0:
            return {"status": "INSUFFICIENT_DATA", "metrics": {}}

        try:
            if task_type.lower() == "regression":
                return self._compute_regression_metrics(all_preds, all_acts)
            elif task_type.lower() == "classification":
                return self._compute_classification_metrics(all_preds, all_acts)
            else:
                raise ModelMonitorError(f"Unsupported predictive operational task argument: {task_type}")
        except Exception as e:
            logger.error(f"Performance analysis metrics compilation failure: {str(e)}")
            return {"status": "ERROR", "error": str(e)}

    def _compute_regression_metrics(self, preds: np.ndarray, acts: np.ndarray) -> Dict[str, Any]:
        errors = preds - acts
        mae = float(np.mean(np.abs(errors)))
        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))
        
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((acts - np.mean(acts)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + self.epsilon)))

        return {
            "status": "HEALTHY",
            "task": "regression",
            "metrics": {
                "sample_count": len(preds),
                "mae": round(mae, 6),
                "mse": round(mse, 6),
                "rmse": round(rmse, 6),
                "r_squared": round(r2, 6)
            }
        }

    def _compute_classification_metrics(self, preds: np.ndarray, acts: np.ndarray) -> Dict[str, Any]:
        y_pred = np.where(preds >= 0.5, 1, 0) if np.issubdtype(preds.dtype, np.floating) else preds.astype(np.int64)
        y_true = acts.astype(np.int64)

        total = len(y_true)
        accuracy = float(np.sum(y_pred == y_true) / total)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = float(tp / (tp + fp + self.epsilon))
        recall = float(tp / (tp + fn + self.epsilon))
        f1_score = float(2 * (precision * recall) / (precision + recall + self.epsilon))

        return {
            "status": "HEALTHY",
            "task": "classification",
            "metrics": {
                "sample_count": total,
                "accuracy": round(accuracy, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1_score": round(f1_score, 6)
            }
        }

    def generate_comprehensive_report(
        self, 
        model_id: str, 
        task_type: str, 
        baseline_predictions: Optional[Union[List[float], np.ndarray, pl.Series]] = None
    ) -> Dict[str, Any]:
        latency_report = self.compute_latency_percentiles(model_id)
        performance_report = self.evaluate_performance_degradation(model_id, task_type)
        
        drift_report = {"status": "SKIPPED", "drift_detected": False}
        if baseline_predictions is not None:
            drift_report = self.evaluate_prediction_drift(model_id, baseline_predictions)

        statuses = [
            latency_report.get("status"), 
            performance_report.get("status"), 
            drift_report.get("status")
        ]
        
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
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "aggregate_status": aggregate_status,
            "latency_profile": latency_report,
            "drift_profile": drift_report,
            "performance_profile": performance_report
        }


# ============================================================================
# 4. RUNTIME MONITOR ENGINE & COMPONENT CONTEXT
# ============================================================================
class ComponentContext:
    """
    Context manager for micro-profiling isolated execution blocks.
    Captures high-resolution wall-clock time, CPU times, and RSS/VMS memory deltas.
    """

    def __init__(self, component_name: str, monitor: "RuntimeMonitor") -> None:
        self.component_name = component_name
        self.monitor = monitor
        self._process = psutil.Process(os.getpid())
        
        self.start_wall_time: float = 0.0
        self.start_cpu_times: Optional[Any] = None
        self.start_memory_rss: int = 0
        self.start_memory_vms: int = 0

    def __enter__(self) -> "ComponentContext":
        try:
            self.start_memory_rss = self._process.memory_info().rss
            self.start_memory_vms = self._process.memory_info().vms
            self.start_cpu_times = self._process.cpu_times()
            self.start_wall_time = time.perf_counter()
        except Exception as e:
            logger.error(f"Failed to initialize instrumentation baselines for {self.component_name}: {str(e)}")
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> bool:
        end_wall_time = time.perf_counter()
        
        if not self.start_cpu_times:
            return False

        try:
            end_cpu_times = self._process.cpu_times()
            end_memory_rss = self._process.memory_info().rss
            end_memory_vms = self._process.memory_info().vms

            wall_duration = end_wall_time - self.start_wall_time
            cpu_user_delta = end_cpu_times.user - self.start_cpu_times.user
            cpu_system_delta = end_cpu_times.system - self.start_cpu_times.system
            total_cpu_time = cpu_user_delta + cpu_system_delta

            rss_delta_mb = (end_memory_rss - self.start_memory_rss) / (1024 * 1024)
            vms_delta_mb = (end_memory_vms - self.start_memory_vms) / (1024 * 1024)

            status = "SUCCESS" if exc_type is None else "FAILED"
            error_msg = str(exc_val) if exc_val else None

            metrics = {
                "timestamp": time.time(),
                "component_name": self.component_name,
                "wall_duration_sec": round(wall_duration, 6),
                "cpu_user_sec": round(cpu_user_delta, 6),
                "cpu_system_sec": round(cpu_system_delta, 6),
                "cpu_total_sec": round(total_cpu_time, 6),
                "rss_delta_mb": round(rss_delta_mb, 4),
                "vms_delta_mb": round(vms_delta_mb, 4),
                "status": status,
                "error_message": error_msg
            }

            self.monitor._record_metrics(metrics)

        except Exception as e:
            logger.error(f"Failed to record runtime telemetry metrics for {self.component_name}: {str(e)}")
        
        return False


class RuntimeMonitor:
    """
    Thread-safe, high-cohesion orchestrator for managing, scaling, and exporting 
    historical performance profiles across arbitrary execution components.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.max_history_size: int = int(self.config.get("max_history_size", 10000))
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []

        logger.info(f"RuntimeMonitor initialized with dynamic history ceiling set to {self.max_history_size} entries.")

    def track(self, component_name: str) -> ComponentContext:
        return ComponentContext(component_name=component_name, monitor=self)

    def profile(self, component_name: str) -> Callable[..., Any]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.track(component_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def _record_metrics(self, metrics: Dict[str, Any]) -> None:
        with self._lock:
            if len(self._history) >= self.max_history_size:
                self._history.pop(0)
            self._history.append(metrics)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            logger.info("Internal runtime monitoring history cleared successfully.")

    def export_metrics_df(self) -> pl.DataFrame:
        with self._lock:
            data_snapshot = list(self._history)

        if not data_snapshot:
            return pl.DataFrame(schema={
                "timestamp": pl.Float64,
                "component_name": pl.String,
                "wall_duration_sec": pl.Float64,
                "cpu_user_sec": pl.Float64,
                "cpu_system_sec": pl.Float64,
                "cpu_total_sec": pl.Float64,
                "rss_delta_mb": pl.Float64,
                "vms_delta_mb": pl.Float64,
                "status": pl.String,
                "error_message": pl.String
            })

        try:
            return pl.DataFrame(data_snapshot)
        except Exception as e:
            logger.error(f"Failed to structure Polars telemetry DataFrame: {str(e)}")
            raise RuntimeMonitorError(f"Polars structural conversion error: {str(e)}")

    def compute_summary_statistics(self) -> Dict[str, Any]:
        df = self.export_metrics_df()
        
        if df.is_empty():
            return {"status": "EMPTY", "components": {}}

        try:
            summary_df = df.group_by("component_name").agg([
                pl.len().alias("execution_count"),
                pl.col("wall_duration_sec").mean().alias("avg_wall_duration"),
                pl.col("wall_duration_sec").max().alias("max_wall_duration"),
                pl.col("cpu_total_sec").sum().alias("cumulative_cpu_sec"),
                pl.col("rss_delta_mb").mean().alias("avg_rss_delta_mb"),
                pl.col("rss_delta_mb").max().alias("max_rss_delta_mb"),
                (pl.col("status") == "FAILED").sum().alias("failure_count")
            ])

            summary_dict: Dict[str, Any] = {"status": "ACTIVE", "components": {}}
            
            for row in summary_df.iter_rows(named=True):
                comp_name = row["component_name"]
                exec_count = row["execution_count"]
                fail_count = row["failure_count"]
                
                failure_rate = (fail_count / exec_count) if exec_count > 0 else 0.0

                summary_dict["components"][comp_name] = {
                    "execution_count": exec_count,
                    "failure_rate": round(failure_rate, 4),
                    "avg_wall_duration_sec": round(row["avg_wall_duration"], 4),
                    "max_wall_duration_sec": round(row["max_wall_duration"], 4),
                    "cumulative_cpu_time_sec": round(row["cumulative_cpu_sec"], 4),
                    "avg_memory_growth_mb": round(row["avg_rss_delta_mb"], 4),
                    "max_memory_growth_mb": round(row["max_rss_delta_mb"], 4),
                }

            return summary_dict

        except Exception as e:
            logger.error(f"Algorithmic failure processing summary statistics: {str(e)}")
            return {"status": "ERROR", "error": str(e)}


# ============================================================================
# 5. DRIFT DASHBOARD
# ============================================================================
class DriftDashboard:
    """
    Orchestrator for text-based drift visualization and telemetry reporting.
    Transforms raw dictionaries from data and model monitoring subsystems into 
    highly structured Markdown and deployment-ready HTML fragments.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.critical_css_theme = self.config.get(
            "html_theme",
            """
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
                .idx-header, .tokocrypto-header { border-bottom: 2px solid #111; padding-bottom: 10px; margin-bottom: 20px; }
                .status-card { padding: 15px; border-radius: 4px; margin-bottom: 20px; font-weight: bold; text-align: center; }
                .status-HEALTHY { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
                .status-DEGRADED { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
                .status-UNHEALTHY { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
                table { width: 100%; border-collapse: collapse; margin-bottom: 25px; }
                th, td { border: 1px solid #dee2e6; padding: 10px; text-align: left; }
                th { background-color: #f8f9fa; }
                .metric-alert { color: #dc3545; font-weight: bold; }
            </style>
            """
        )
        logger.info("DriftDashboard engine successfully initialized.")

    def generate_summary_dict(
        self, 
        data_report: Dict[str, Any], 
        model_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            data_status = data_report.get("aggregate_status", "UNKNOWN")
            model_status = model_report.get("aggregate_status", "UNKNOWN")
            
            drift_profile = model_report.get("drift_profile", {})
            concept_drift_detected = drift_profile.get("drift_detected", False)
            ks_stat = drift_profile.get("metrics", {}).get("ks_statistic", 0.0)
            
            data_details = data_report.get("details", {})
            missing_info = data_details.get("missing_and_invalid", {}).get("metrics", {}).get("columns", {})
            outlier_info = data_details.get("outliers_iqr", {}).get("metrics", {}).get("columns", {})
            
            breached_features: List[str] = []
            total_outliers = 0
            
            for feat, meta in missing_info.items():
                if meta.get("threshold_breached", False):
                    breached_features.append(feat)
                    
            for feat, meta in outlier_info.items():
                total_outliers += meta.get("outlier_count", 0)

            statuses = [data_status, model_status]
            if "UNHEALTHY" in statuses or concept_drift_detected:
                aggregate_status = "UNHEALTHY"
            elif "DEGRADED" in statuses:
                aggregate_status = "DEGRADED"
            else:
                aggregate_status = "HEALTHY"

            return {
                "framework": "IDX_Stock_Analysis_Engine",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "aggregate_status": aggregate_status,
                "data_subsystem": {
                    "status": data_status,
                    "breached_features_count": len(breached_features),
                    "breached_features_list": breached_features,
                    "total_isolated_outliers": total_outliers
                },
                "model_subsystem": {
                    "status": model_status,
                    "concept_drift_detected": concept_drift_detected,
                    "ks_test_statistic": ks_stat
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate structured unified summary map: {str(e)}")
            raise DriftDashboardError(f"Summary dictionary generation exception: {str(e)}")

    def render_markdown(
        self, 
        data_report: Dict[str, Any], 
        model_report: Dict[str, Any]
    ) -> str:
        summary = self.generate_summary_dict(data_report, model_report)
        
        md = []
        md.append("# IDX Stock Signal Engine — Telemetry & Drift Dashboard")
        md.append(f"**Generated UTC Time:** {summary['timestamp']}  ")
        md.append(f"**System Aggregated Status:** `[{summary['aggregate_status']}]` \n")
        md.append("---")
        
        md.append("## 1. Executive Subsystem Overview\n")
        md.append("| Subsystem Domain | Status Metric | Primary Analytical Indicator |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| **Data Quality Engine** | `{summary['data_subsystem']['status']}` | Outliers: {summary['data_subsystem']['total_isolated_outliers']} units |")
        md.append(f"| **Model Performance Engine** | `{summary['model_subsystem']['status']}` | Concept Drift: {summary['model_subsystem']['concept_drift_detected']} (KS-Stat: {summary['model_subsystem']['ks_test_statistic']}) | \n")
        
        md.append("## 2. Structured Data Quality Telemetry")
        data_details = data_report.get("details", {})
        missing_columns = data_details.get("missing_and_invalid", {}).get("metrics", {}).get("columns", {})
        
        if missing_columns:
            md.append("### Feature Validation Matrix")
            md.append("| Feature Name | Null Count | NaN Count | Missing Rate | Limit Status |")
            md.append("| :--- | :--- | :--- | :--- | :--- |")
            for col, metrics in missing_columns.items():
                status_str = "BREACHED" if metrics['threshold_breached'] else "OK"
                md.append(f"| {col} | {metrics['null_count']} | {metrics['nan_count']} | {metrics['missing_rate'] * 100:.2f}% | `{status_str}` |")
            md.append("\n")
        else:
            md.append("*No detailed data monitoring field matrices recorded explicitly.*\n")

        md.append("## 3. Inference Latency & Prediction Drift Profiles")
        latency_profile = model_report.get("latency_profile", {}).get("metrics", {})
        
        if latency_profile:
            md.append("### High-Resolution Latency Percentiles")
            md.append("| Metric Reference | Value (Seconds) |")
            md.append("| :--- | :--- |")
            md.append(f"| Sample Evaluation Volume | {latency_profile.get('sample_count', 0)} |")
            md.append(f"| Mean Latency Time | {latency_profile.get('mean_sec', 0.0):.6f}s |")
            md.append(f"| Median Latency (p50) | {latency_profile.get('p50_sec', 0.0):.6f}s |")
            md.append(f"| Tail Latency (p95) | {latency_profile.get('p95_sec', 0.0):.6f}s |")
            md.append(f"| Mission-Critical Tail (p99) | {latency_profile.get('p99_sec', 0.0):.6f}s |\n")
        else:
            md.append("*Model latency metrics segment unallocated or processing insufficient historical datasets.*\n")
            
        return "\n".join(md)

    def render_html(
        self, 
        data_report: Dict[str, Any], 
        model_report: Dict[str, Any]
    ) -> str:
        summary = self.generate_summary_dict(data_report, model_report)
        
        html = []
        html.append(f"<!DOCTYPE html>\n<html>\n<head>\n<title>IDX Stock Signal Engine Telemetry Dashboard</title>\n{self.critical_css_theme}\n</head>\n<body>")
        
        html.append("<div class='idx-header'>")
        html.append("  <h1>IDX Stock Signal Engine — Structural Telemetry Dashboard</h1>")
        html.append(f"  <p><strong>Generated UTC Timestamp:</strong> {summary['timestamp']}</p>")
        html.append("</div>")
        
        agg_status = summary['aggregate_status']
        html.append(f"<div class='status-card status-{agg_status}'>GLOBAL SYSTEM STATUS: {agg_status}</div>")
        
        # PERBAIKAN BUG HTML SYNTAX LINE ERROR
        html.append("<h2>System Subsystem Operational Vectors</h2>")
        html.append("<table>")
        html.append("  <thead><tr><th>Subsystem Component Target</th><th>Operational Status</th><th>Key Diagnostic Metadata</th></tr></thead>")
        html.append("  <tbody>")
        html.append(f"    <tr><td><strong>Data Quality Subsystem</strong></td><td>{summary['data_subsystem']['status']}</td><td>Isolated Outliers: {summary['data_subsystem']['total_isolated_outliers']} columns</td></tr>")
        
        drift_flag_str = "<span class='metric-alert'>TRUE</span>" if summary['model_subsystem']['concept_drift_detected'] else "FALSE"
        html.append(f"    <tr><td><strong>Model Performance Subsystem</strong></td><td>{summary['model_subsystem']['status']}</td><td>Concept Drift Isolated: {drift_flag_str} (KS Test Stat: {summary['model_subsystem']['ks_test_statistic']:.6f})</td></tr>")
        html.append("  </tbody>")
        html.append("</table>")
        
        html.append("<h2>Detailed Data Integrity Registry</h2>")
        data_details = data_report.get("details", {})
        missing_columns = data_details.get("missing_and_invalid", {}).get("metrics", {}).get("columns", {})
        
        if missing_columns:
            html.append("<table>")
            html.append("  <thead><tr><th>Feature Field Target</th><th>Null Records</th><th>NaN Records</th><th>Missing Rate Ratio</th><th>Threshold Limit Check</th></tr></thead>")
            html.append("  <tbody>")
            for col, m in missing_columns.items():
                td_class = "class='metric-alert'" if m['threshold_breached'] else ""
                status_label = "BREACHED" if m['threshold_breached'] else "OK"
                html.append(f"    <tr><td>{col}</td><td>{m['null_count']}</td><td>{m['nan_count']}</td><td>{m['missing_rate'] * 100:.2f}%</td><td {td_class}>{status_label}</td></tr>")
            html.append("  </tbody>")
            html.append("</table>")
        else:
            html.append("<p><em>No active structural features analyzed or present in dataset inputs.</em></p>")

        html.append("</body>\n</html>")
        return "\n".join(html)


# ============================================================================
# 6. FACADE CLASS: UNIFIED MONITORING ENGINE
# ============================================================================
class UnifiedMonitoringEngine:
    """
    Unified Facade class providing a single point of entry for executing 
    full system health checks, data quality audits, model monitoring, 
    runtime performance tracking, and drift reporting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        
        self.data_monitor = DataMonitor(self.config.get("data_monitor"))
        self.health_engine = HealthCheckEngine(self.config.get("health_check"))
        self.model_monitor = ModelMonitor(self.config.get("model_monitor"))
        self.runtime_monitor = RuntimeMonitor(self.config.get("runtime_monitor"))
        self.drift_dashboard = DriftDashboard(self.config.get("drift_dashboard"))

        logger.info("UnifiedMonitoringEngine (Facade) successfully initialized.")

    def execute_full_audit(
        self,
        df: Optional[pl.DataFrame] = None,
        target_columns: Optional[List[str]] = None,
        timestamp_col: Optional[str] = None,
        expected_schema: Optional[Dict[str, pl.DataType]] = None,
        model_id: Optional[str] = None,
        task_type: str = "regression",
        baseline_predictions: Optional[Union[List[float], np.ndarray, pl.Series]] = None
    ) -> Dict[str, Any]:
        """
        Executes a comprehensive system audit across hardware health, data quality, 
        model performance, and runtime metrics, producing Markdown and HTML reports.
        """
        start_time = time.perf_counter()

        # Step 1: Health Diagnostics Check
        health_report = self.health_engine.run_all()

        # Step 2: Data Quality Audit
        data_report = {"aggregate_status": "SKIPPED", "details": {}}
        if df is not None:
            df = _ensure_polars_df_monitoring(df)
            if not df.is_empty():
                data_report = self.data_monitor.run_all(
                    df=df,
                    target_columns=target_columns or [],
                    timestamp_col=timestamp_col,
                    expected_schema=expected_schema
                )

        # Step 3: Model Telemetry & Drift Evaluation
        model_report = {"aggregate_status": "SKIPPED"}
        if model_id:
            model_report = self.model_monitor.generate_comprehensive_report(
                model_id=model_id,
                task_type=task_type,
                baseline_predictions=baseline_predictions
            )

        # Step 4: Runtime Performance Summary
        runtime_summary = self.runtime_monitor.compute_summary_statistics()

        # Step 5: Dashboard Rendering
        dashboard_markdown = self.drift_dashboard.render_markdown(data_report, model_report)
        dashboard_html = self.drift_dashboard.render_html(data_report, model_report)

        # Compute Global System Aggregate Status
        statuses = [
            health_report.get("aggregate_status"),
            data_report.get("aggregate_status"),
            model_report.get("aggregate_status")
        ]
        statuses = [s for s in statuses if s != "SKIPPED" and s is not None]

        if "UNHEALTHY" in statuses or "CRITICAL" in statuses:
            global_status = "UNHEALTHY"
        elif "DEGRADED" in statuses:
            global_status = "DEGRADED"
        else:
            global_status = "HEALTHY"

        execution_duration = time.perf_counter() - start_time

        return {
            "framework": "IDX_Stock_Analysis_Engine",
            "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "global_status": global_status,
            "execution_duration_sec": round(execution_duration, 4),
            "reports": {
                "health_diagnostics": health_report,
                "data_quality": data_report,
                "model_performance": model_report,
                "runtime_statistics": runtime_summary
            },
            "dashboards": {
                "markdown": dashboard_markdown,
                "html": dashboard_html
            }
        }

"""
=============================================================================
IDX Quantitative Stock Analysis Engine - Reporting Module (reporting.py)
Directory     : Flat Directory (Root Level with main.py)
Version       : 2026.Q3.v16.31 (Institutional Production-Grade & Gemini AI Integrated)
Compliance    : Indonesia Stock Exchange (IDX) Signal & Analytics Standard
=============================================================================
"""

import os
import re
import time
import json
import math
import hashlib
import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict

import polars as pl
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Import Google GenAI Client
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# ==============================================================================
# INTEGRASI LOGGING & EXCEPTION HANDLER
# ==============================================================================
try:
    from exceptions import (
        ReportingError,
        ValidationError,
        TelegramPayloadError,
        ReportingNetworkError,
        SignalGeometryError
    )
except ImportError:
    class ReportingError(Exception): pass
    class ValidationError(ReportingError): pass
    class TelegramPayloadError(ReportingError): pass
    class ReportingNetworkError(ReportingError): pass
    class SignalGeometryError(ReportingError): pass

try:
    from logger import get_logger
    logger = get_logger("IDX.Reporting")
except ImportError:
    logger = logging.getLogger("IDX.Reporting")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# ==============================================================================
# TYPE DEFINITIONS FOR STRICT TYPING COMPLIANCE
# ==============================================================================
class PositionDict(TypedDict, total=False):
    asset: str
    symbol: str
    ticker: str
    lots: int
    avg_price: float
    tp_price: float
    sl_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    market_value: float
    buy_date: str

class PortfolioStateDict(TypedDict, total=False):
    equity: float
    total_equity: float
    cash: float
    cash_balance: float
    invested_amount: float
    exposure_pct: float
    return_pct: float
    active_positions_count: int
    positions: Dict[str, PositionDict]
    top_pick: str
    reset_event: bool
    ai_insight: str

class SignalDict(TypedDict, total=False):
    signal_uuid: str
    asset: str
    timestamp: str
    direction: str
    entry_price: float
    tp_price: float
    sl_price: float
    probability: float
    confidence: float
    ranking_score: float
    market_regime: str
    primary_reason: str
    prediction_horizon: str

class ReportMetadataDict(TypedDict):
    evaluation_timestamp: str
    report_date: str
    framework_version: str
    model_version: str
    symbol_count: int
    payload_hash: str

class SummaryMetricsDict(TypedDict):
    total_received: int
    total_approved: int
    total_transmitted: int
    total_rejected: int

class ProcessedSummaryPayload(TypedDict):
    metadata: ReportMetadataDict
    summary_metrics: SummaryMetricsDict
    signals: List[SignalDict]

# ==============================================================================
# KONSTANTA LOKAL PASAR SAHAM INDONESIA (IDX)
# ==============================================================================
INDONESIAN_MONTHS: List[str] = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

WIB_OFFSET: timedelta = timedelta(hours=7)
TZ_WIB: timezone = timezone(WIB_OFFSET, name="WIB")

IDX_ROUNDTRIP_FEE_PCT: float = 0.003  # 0.3% Roundtrip Trading Fee
MIN_REQUIRED_RRR: float = 1.2         # Minimum Risk-to-Reward Ratio

# ==============================================================================
# HELPER FUNCTIONS (FORMATTING, ESCAPING, POLARS & SANITIZATION)
# ==============================================================================
def normalize_stock_symbol(symbol: Any) -> str:
    """Memastikan format ticker saham Indonesia selalu menggunakan akhiran .JK (misal: ASII -> ASII.JK)."""
    if symbol is None:
        raise ValidationError("Symbol tidak boleh None")
    
    cleaned = str(symbol).upper().strip()
    if not cleaned or cleaned == "-":
        raise ValidationError("Symbol kosong atau invalid")

    if re.search(r"(USDT|BIDR|BUSD|USDC)$", cleaned):
        raise ValidationError(f"Symbol {cleaned} merupakan instrumen Kripto, bukan Saham BEI/IDX.")

    cleaned = re.sub(r"[^A-Z0-9\.]", "", cleaned)
    cleaned = cleaned.strip(".")
    if not cleaned:
        raise ValidationError("Symbol tidak memiliki karakter alfanumerik valid")

    if not cleaned.endswith(".JK") and not cleaned.startswith("^"):
        cleaned = f"{cleaned}.JK"
        
    return cleaned


def format_stock_price(price: Any) -> str:
    """Format nominal mata uang Rupiah Indonesia (IDR) secara rapi."""
    if price is None:
        raise ValidationError("Harga bernilai None")
    try:
        val = float(price)
    except (ValueError, TypeError):
        raise ValidationError(f"Harga bernilai non-numeric: {price}")

    if math.isnan(val) or math.isinf(val):
        raise ValidationError("Harga mengandung NaN atau Infinity")
    if val < 0:
        raise ValidationError(f"Harga tidak boleh negatif: {val}")

    if val == 0:
        return "Rp 0"

    if abs(val - round(val)) < 1e-4:
        return f"Rp {int(round(val)):,}".replace(",", ".")
    return f"Rp {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def escape_markdown_v2_text(text: Any) -> str:
    """Sanitasi lengkap seluruh karakter spesial MarkdownV2 Telegram."""
    if text is None:
        return ""
    s = str(text)
    special_chars = r"_*[]()~`>#+-=|{}.!"
    for char in special_chars:
        s = s.replace(char, f"\\{char}")
    return s


def clamp_probability(val: Any, default: float = 0.5) -> float:
    """Sanitasi & Clamping ketat probabilitas/konfidensi ke rentang [0.0, 1.0]."""
    try:
        if val is None:
            return default
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        if f_val > 1.0 and f_val <= 100.0:
            f_val = f_val / 100.0
        return max(0.0, min(1.0, f_val))
    except (ValueError, TypeError):
        return default


def normalize_polars_datetime(
    df: pl.DataFrame,
    column_name: str,
    target_tz: str = "Asia/Jakarta"
) -> pl.DataFrame:
    if column_name not in df.columns or df.is_empty():
        return df

    dtype = df.schema[column_name]

    if isinstance(dtype, pl.Date):
        df = df.with_columns(pl.col(column_name).cast(pl.Datetime))
        dtype = df.schema[column_name]

    if not isinstance(dtype, pl.Datetime):
        return df

    current_tz: Optional[str] = getattr(dtype, "time_zone", None)

    if current_tz is None:
        df = df.with_columns(
            pl.col(column_name).dt.replace_time_zone(target_tz)
        )
    elif current_tz != target_tz:
        df = df.with_columns(
            pl.col(column_name).dt.convert_time_zone(target_tz)
        )

    return df

# ==============================================================================
# HELPER: GOOGLE GEMINI NARRATIVE GENERATOR
# ==============================================================================
class GeminiNarrativeEngine:
    """
    Engine integrasi Google Gemini untuk menghasilkan analisis naratif
    kualitatif dan ringkasan eksekutif pasar saham IDX.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None

        if not HAS_GEMINI_SDK:
            logger.warning("⚠️ Package 'google-genai' belum terpasang. Gemini AI Insight dinonaktifkan.")
            return

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("🤖 [GEMINI_INIT_SUCCESS] Google Gemini Client berhasil diinisialisasi pada reporting module.")
            except Exception as e:
                logger.warning(f"⚠️ Gagal inisialisasi Gemini Client: {e}")
        else:
            logger.info("ℹ️ GEMINI_API_KEY tidak ditemukan. Narasi AI dilewati.")

    def generate_market_insight(self, portfolio_data: PortfolioStateDict, top_signals: List[Dict[str, Any]]) -> str:
        """Menghasilkan narasi analisis kualitatif berbasis data kuantitatif dengan model fallback."""
        if not self.client:
            return ""

        prompt = f"""
        Anda adalah Analis Kuantitatif & Strategis Utama Bursa Efek Indonesia (BEI/IDX).
        Berikan ringkasan analisis naratif profesional (maksimal 2 paragraf singkat, bahasa Indonesia lugas & profesional)
        berdasarkan kondisi sistem perdagangan harian berikut:

        Ringkasan Portofolio:
        - Total Ekuitas: Rp {portfolio_data.get('equity', 0):,.0f} IDR
        - Saldo Kas: Rp {portfolio_data.get('cash', 0):,.0f} IDR
        - Exposure: {portfolio_data.get('exposure_pct', 0.0):.1f}%
        - Return Kumulatif: {portfolio_data.get('return_pct', 0.0):+.2f}%
        - Top Pick Aktif: {portfolio_data.get('top_pick', '-')}

        Top Sinyal Saham Terpilih:
        {top_signals}

        Panduan Narasi:
        1. Jelaskan secara singkat alasan rasional di balik alokasi portofolio atau pemilihan sinyal top pick.
        2. Berikan saran manajemen risiko singkat untuk perdagangan sesi berikutnya.
        3. Hindari penggunaan format markdown kompleks seperti bold/italic yang tidak standar.
        """

        candidate_models = [
            "models/gemini-2.5-flash",
            "gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "gemini-2.0-flash",
            "models/gemini-1.5-flash",
            "gemini-1.5-flash"
        ]

        for model_name in candidate_models:
            try:
                logger.info(f"🧠 Memanggil Gemini API ({model_name}) untuk membentuk narasi AI Market Insight...")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"⚠️ Gagal memperoleh respon Gemini API pada model '{model_name}': {e}")

        return ""

# ==============================================================================
# SIGNAL GEOMETRY VALIDATOR
# ==============================================================================
class SignalGeometryValidator:
    """Validator Ketat Geometri Sinyal Saham & Risk-Reward Ratio (RRR)."""
    
    @staticmethod
    def validate_buy_geometry(entry: float, tp: float, sl: float, min_rrr: float = MIN_REQUIRED_RRR) -> Tuple[bool, str]:
        if entry <= 0:
            return False, "Harga Entry harus > 0"
        if tp <= entry:
            return False, f"Target TP ({tp}) harus lebih besar dari Harga Entry ({entry})"
        if sl >= entry or sl <= 0:
            return False, f"Stop Loss ({sl}) harus berada di bawah Harga Entry ({entry}) dan > 0"
        
        risk = entry - sl
        reward = tp - entry
        rrr = reward / risk if risk > 0 else 0.0

        if rrr < min_rrr:
            return False, f"Risk-Reward Ratio ({rrr:.2f}) di bawah ambang batas minimum ({min_rrr:.2f})"

        return True, "VALID"

# ==============================================================================
# IN-MEMORY EVENT-AWARE CACHE UNTUK STATE PORTOFOLIO (MTIME SENSITIVE)
# ==============================================================================
class PortfolioStateCache:
    """Cache In-Memory pintar dengan TTL & Deteksi Modifikasi File (mtime)."""
    
    def __init__(self, ttl_seconds: float = 5.0) -> None:
        self.ttl = ttl_seconds
        self._cached_state: Optional[PortfolioStateDict] = None
        self._last_update: float = 0.0
        self._last_mtime: float = 0.0
        self.lock = threading.RLock()

    def validate_schema(self, data: Dict[str, Any], default_capital: float) -> PortfolioStateDict:
        sanitized: PortfolioStateDict = {}
        try:
            sanitized["equity"] = float(data.get("equity", data.get("total_equity", default_capital)))
            sanitized["total_equity"] = sanitized["equity"]
        except (ValueError, TypeError):
            sanitized["equity"] = default_capital
            sanitized["total_equity"] = default_capital

        positions_raw = data.get("positions", {})
        cleaned_positions: Dict[str, PositionDict] = {}
        
        if isinstance(positions_raw, dict):
            for k, v in positions_raw.items():
                if isinstance(v, dict):
                    try:
                        raw_sym = v.get("asset") or v.get("symbol") or v.get("ticker") or k
                        asset_key = normalize_stock_symbol(raw_sym)
                        cleaned_positions[asset_key] = v
                    except ValidationError:
                        continue
        elif isinstance(positions_raw, list):
            for p in positions_raw:
                if isinstance(p, dict):
                    raw_asset = p.get("symbol") or p.get("asset") or p.get("ticker")
                    if raw_asset:
                        try:
                            asset_key = normalize_stock_symbol(raw_asset)
                            cleaned_positions[asset_key] = p
                        except ValidationError:
                            continue

        sanitized["positions"] = cleaned_positions
        sanitized["active_positions_count"] = len(cleaned_positions)

        if len(cleaned_positions) == 0:
            sanitized["cash"] = sanitized["equity"]
            sanitized["cash_balance"] = sanitized["equity"]
            sanitized["invested_amount"] = 0.0
            sanitized["exposure_pct"] = 0.0
            sanitized["top_pick"] = "-"
        else:
            try:
                sanitized["cash"] = float(data.get("cash", data.get("cash_balance", default_capital)))
                sanitized["cash_balance"] = sanitized["cash"]
            except (ValueError, TypeError):
                sanitized["cash"] = default_capital
                sanitized["cash_balance"] = sanitized["cash"]

            try:
                inv_amt = float(data.get("invested_amount", 0.0))
                if inv_amt <= 0:
                    for p in cleaned_positions.values():
                        avg_p = float(p.get("avg_price", p.get("buy_price", 0.0)))
                        lots = int(p.get("lots", 0))
                        inv_amt += avg_p * lots * 100.0
                sanitized["invested_amount"] = inv_amt
            except (ValueError, TypeError):
                sanitized["invested_amount"] = 0.0

            top_pick = data.get("top_pick", "-")
            if top_pick == "-":
                top_pick = list(cleaned_positions.keys())[0]

            try:
                sanitized["top_pick"] = normalize_stock_symbol(top_pick)
            except ValidationError:
                sanitized["top_pick"] = "-"

        if "ai_insight" in data:
            sanitized["ai_insight"] = str(data["ai_insight"])

        return sanitized

    def get_state(self, file_path: str, default_capital: float) -> PortfolioStateDict:
        with self.lock:
            now = time.time()
            
            candidate_files = [
                file_path,
                "portfolio_state.json",
                "portfolio_simulation_state.json",
                "portfolio_live_state.json"
            ]

            target_file = None
            for cf in candidate_files:
                if cf and os.path.exists(cf) and os.path.getsize(cf) > 0:
                    target_file = cf
                    break

            current_mtime = os.path.getmtime(target_file) if target_file else 0.0

            if (
                self._cached_state is not None and
                (now - self._last_update) < self.ttl and
                current_mtime == self._last_mtime
            ):
                return dict(self._cached_state)

            default_state: PortfolioStateDict = {
                "equity": default_capital,
                "total_equity": default_capital,
                "cash": default_capital,
                "cash_balance": default_capital,
                "invested_amount": 0.0,
                "exposure_pct": 0.0,
                "return_pct": 0.0,
                "active_positions_count": 0,
                "positions": {},
                "top_pick": "-"
            }

            if target_file:
                backoff_schedule = [0.05, 0.1, 0.2, 0.4]
                for attempt in range(4):
                    try:
                        with open(target_file, "r", encoding="utf-8") as f:
                            raw_data = json.load(f)
                            validated_data = self.validate_schema(raw_data, default_capital)
                            default_state.update(validated_data)

                            eq = default_state["equity"]
                            inv = default_state.get("invested_amount", 0.0)
                            csh = default_state["cash"]
                            
                            if len(default_state["positions"]) == 0:
                                default_state["cash"] = eq
                                default_state["cash_balance"] = eq
                                default_state["exposure_pct"] = 0.0
                            elif eq > 0:
                                if inv > 0:
                                    default_state["exposure_pct"] = min(100.0, max(0.0, (inv / eq) * 100.0))
                                else:
                                    default_state["exposure_pct"] = min(100.0, max(0.0, ((eq - csh) / eq) * 100.0))

                            if default_capital > 0:
                                default_state["return_pct"] = ((eq - default_capital) / default_capital) * 100.0
                            break
                    except (json.JSONDecodeError, OSError) as e:
                        time.sleep(backoff_schedule[attempt])
                        if attempt == 3:
                            logger.warning(f"⚠️ Gagal membaca state portofolio {target_file}: {e}", exc_info=True)

            self._cached_state = dict(default_state)
            self._last_update = now
            self._last_mtime = current_mtime
            return default_state

    def invalidate(self) -> None:
        with self.lock:
            self._cached_state = None
            self._last_update = 0.0
            self._last_mtime = 0.0

# ==============================================================================
# TELEGRAM RATE LIMITER
# ==============================================================================
class TelegramRateLimiter:
    def __init__(self, min_interval_seconds: float = 0.05) -> None:
        self.min_interval = min_interval_seconds
        self.last_call = 0.0
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

GLOBAL_RATE_LIMITER = TelegramRateLimiter(min_interval_seconds=0.05)
GLOBAL_PORTFOLIO_CACHE = PortfolioStateCache(ttl_seconds=5.0)

# ==============================================================================
# SIGNAL SUMMARY GENERATOR ENGINE
# ==============================================================================
class SignalSummaryGenerator:
    """Pemproses & Pembuat Laporan Terpadu Telegram (MarkdownV2 Compliant & AI Integrated)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, mode: str = "dry_run") -> None:
        self.config = config or {}
        self.mode = str(mode).lower().strip()
        self.lock = threading.RLock()
        
        self.min_confidence: float = float(self.config.get("REPORTING_MIN_CONFIDENCE", 0.60))
        self.min_probability: float = float(self.config.get("REPORTING_MIN_PROBABILITY", 0.50))
        self.max_signals: int = int(self.config.get("REPORTING_MAX_SIGNALS", 3))
        self.framework_version: str = str(self.config.get("FRAMEWORK_VERSION", "2026.Q3.v16.31"))
        self.model_version: str = str(self.config.get("MODEL_VERSION", "IDX-PROD-V2026"))
        self.gemini_engine = GeminiNarrativeEngine()

    def _get_mode_header(self) -> str:
        if self.mode in ["live", "force-rebalance"]:
            return "🚨 *[LIVE SIGNAL ANALYSIS \\- IDX]*"
        elif self.mode in ["reset-dryrun", "reset_dryrun"]:
            return "🔄 *[SIMULATION CAPITAL RESET]*"
        else:
            return "🧪 *[PAPER TRADING / SIMULASI IDX]*"

    def _calculate_payload_hash(self, raw_signals: List[Dict[str, Any]]) -> str:
        try:
            canonical_list = []
            for item in raw_signals:
                if not isinstance(item, dict):
                    continue

                def _to_float(v: Any) -> float:
                    try:
                        val = float(v)
                        return 0.0 if (math.isnan(val) or math.isinf(val)) else val
                    except (ValueError, TypeError):
                        return 0.0

                c_item = {
                    "asset": str(item.get("asset", "")),
                    "entry_price": round(_to_float(item.get("entry_price")), 4),
                    "tp_price": round(_to_float(item.get("tp_price")), 4),
                    "sl_price": round(_to_float(item.get("sl_price")), 4),
                    "direction": str(item.get("direction", "")).upper(),
                    "probability": round(clamp_probability(item.get("probability", 0.0)), 4),
                    "confidence": round(clamp_probability(item.get("confidence", 0.0)), 4),
                    "ranking_score": round(_to_float(item.get("ranking_score")), 4)
                }
                canonical_list.append(c_item)
            
            canonical_list.sort(key=lambda x: (x["asset"], x["ranking_score"]), reverse=True)
            serialized = json.dumps(canonical_list, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        except Exception as err:
            logger.warning(f"⚠️ Fallback payload hash generator akibat error: {err}")
            return hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()

    def load_portfolio_state(self) -> PortfolioStateDict:
        with self.lock:
            is_live = self.mode in ["live", "force-rebalance"]
            suffix = "live" if is_live else "simulation"
            file_path = f"portfolio_{suffix}_state.json"
            default_capital = float(self.config.get("INITIAL_CAPITAL_IDR", 10_000_000.0))

            return GLOBAL_PORTFOLIO_CACHE.get_state(file_path, default_capital)

    def _validate_and_clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        required_cols = [
            "signal_uuid", "asset", "timestamp", "direction", "entry_price",
            "tp_price", "sl_price", "probability", "confidence", "ranking_score",
            "market_regime", "primary_reason", "prediction_horizon"
        ]
        
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            exprs = []
            for col in missing_cols:
                if col in ["entry_price", "tp_price", "sl_price", "probability", "confidence", "ranking_score"]:
                    exprs.append(pl.lit(0.0).alias(col))
                elif col in ["direction"]:
                    exprs.append(pl.lit("BUY").alias(col))
                else:
                    exprs.append(pl.lit("-").alias(col))
            df = df.with_columns(exprs)

        if df.is_empty():
            return df

        df = df.with_columns([
            pl.col("probability").map_elements(clamp_probability, return_dtype=pl.Float64).alias("probability"),
            pl.col("confidence").map_elements(clamp_probability, return_dtype=pl.Float64).alias("confidence")
        ])

        return df.with_columns([
            pl.col("direction").cast(pl.String).str.to_uppercase().str.strip_chars().alias("direction"),
            pl.col("entry_price").cast(pl.Float64, strict=False).fill_null(0.0).alias("entry_price"),
            pl.col("tp_price").cast(pl.Float64, strict=False).fill_null(0.0).alias("tp_price"),
            pl.col("sl_price").cast(pl.Float64, strict=False).fill_null(0.0).alias("sl_price"),
            pl.col("ranking_score").cast(pl.Float64, strict=False).fill_null(0.0).alias("ranking_score")
        ])

    def process_signals(self, raw_signals: List[Dict[str, Any]], target_date: Optional[date] = None) -> ProcessedSummaryPayload:
        with self.lock:
            now_wib = datetime.now(timezone.utc) + WIB_OFFSET
            analysis_date = target_date or now_wib.date()
            payload_hash = self._calculate_payload_hash(raw_signals)
            iso_eval_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            if not raw_signals:
                return self._create_empty_payload(analysis_date, payload_hash, iso_eval_time)

            try:
                df = pl.DataFrame(raw_signals)
                df = self._validate_and_clean_data(df)
                if df.is_empty():
                    return self._create_empty_payload(analysis_date, payload_hash, iso_eval_time)

                if "timestamp" in df.columns:
                    df = df.with_columns(
                        pl.col("timestamp").cast(pl.Datetime, strict=False).alias("_ts_dt")
                    )
                    df = normalize_polars_datetime(df, "_ts_dt", target_tz="Asia/Jakarta")
                    df = df.with_columns(
                        pl.col("_ts_dt").dt.date().alias("signal_date")
                    ).drop("_ts_dt")
                else:
                    df = df.with_columns(pl.lit(analysis_date).alias("signal_date"))

                df_day = df.filter(pl.col("signal_date") == analysis_date)
                working_df = df_day if df_day.shape[0] > 0 else df
                total_received = working_df.shape[0]

                df_filtered = working_df.filter(
                    (pl.col("direction").is_in(["BUY", "BELI", "STRONG_BUY"])) &
                    (pl.col("confidence") >= self.min_confidence) &
                    (pl.col("probability") >= self.min_probability)
                )

                if df_filtered.is_empty():
                    return self._create_empty_payload(analysis_date, payload_hash, iso_eval_time, total_received=total_received)

                valid_rows = []
                for row in df_filtered.iter_rows(named=True):
                    e = float(row.get("entry_price", 0.0))
                    tp = float(row.get("tp_price", 0.0))
                    sl = float(row.get("sl_price", 0.0))
                    
                    is_valid, reason = SignalGeometryValidator.validate_buy_geometry(e, tp, sl)
                    if is_valid:
                        valid_rows.append(row)
                    else:
                        logger.warning(f"⚠️ Sinyal {row.get('asset')} ditolak akibat Geometri/RRR Cacat: {reason}")

                if not valid_rows:
                    return self._create_empty_payload(analysis_date, payload_hash, iso_eval_time, total_received=total_received)

                df_valid = pl.DataFrame(valid_rows)

                df_unique = (
                    df_valid
                    .sort(["ranking_score", "timestamp"], descending=[True, True])
                    .unique(subset=["asset"], keep="first")
                )

                df_sorted = df_unique.sort("ranking_score", descending=True)
                df_limited = df_sorted.head(self.max_signals)

                signals_list: List[SignalDict] = list(df_limited.iter_rows(named=True))

                return {
                    "metadata": {
                        "evaluation_timestamp": iso_eval_time,
                        "report_date": analysis_date.isoformat(),
                        "framework_version": self.framework_version,
                        "model_version": self.model_version,
                        "symbol_count": int(df_limited["asset"].n_unique()),
                        "payload_hash": payload_hash
                    },
                    "summary_metrics": {
                        "total_received": total_received,
                        "total_approved": int(df_sorted.shape[0]),
                        "total_transmitted": int(df_limited.shape[0]),
                        "total_rejected": int(total_received - df_sorted.shape[0])
                    },
                    "signals": signals_list
                }
            except Exception as e:
                logger.error(f"Gagal memproses data sinyal: {str(e)}", exc_info=True)
                return self._create_empty_payload(analysis_date, payload_hash, iso_eval_time)

    def build_telegram_message(self, summary_payload: ProcessedSummaryPayload, portfolio_data: Optional[PortfolioStateDict] = None) -> str:
        with self.lock:
            p_state = portfolio_data or self.load_portfolio_state()
            
            now_wib = datetime.now(timezone.utc) + WIB_OFFSET
            month_str = INDONESIAN_MONTHS[now_wib.month - 1]
            header_time = escape_markdown_v2_text(f"{now_wib.day} {month_str} {now_wib.year} pukul {now_wib.strftime('%H:%M')} WIB")

            default_capital = float(self.config.get("INITIAL_CAPITAL_IDR", 10_000_000.0))

            equity = float(p_state.get("equity", default_capital))
            positions_dict = p_state.get("positions", {})
            active_pos_count = len(positions_dict) if isinstance(positions_dict, dict) else int(p_state.get("active_positions_count", 0))

            if active_pos_count == 0:
                cash = equity
                cash_pct = 100.0
                exposure_pct = 0.0
                top_pick = "-"
            else:
                cash = float(p_state.get("cash", default_capital))
                cash_pct = (cash / equity * 100.0) if equity > 0 else 100.0
                exposure_pct = float(p_state.get("exposure_pct", 0.0))
                top_pick = list(positions_dict.keys())[0] if isinstance(positions_dict, dict) else "-"

            return_pct = float(p_state.get("return_pct", 0.0))

            top_pick_clean = escape_markdown_v2_text(top_pick)
            equity_str = escape_markdown_v2_text(format_stock_price(equity))
            cash_str = escape_markdown_v2_text(format_stock_price(cash))

            cash_pct_str = escape_markdown_v2_text(f"{cash_pct:.1f}%")
            exposure_pct_str = escape_markdown_v2_text(f"{exposure_pct:.1f}%")
            return_pct_str = escape_markdown_v2_text(f"{return_pct:+.2f}%")

            md = [
                "📊 *DASHBOARD PORTOFOLIO SAHAM IDX*",
                f"📍 Mode: {self._get_mode_header()}",
                f"🗓️ {header_time}",
                "══════════════════════════════",
                "📌 *RINGKASAN PORTOFOLIO SAHAM*",
                f"💰 Total Ekuitas : `{equity_str}`",
                f"💵 Saldo Kas     : `{cash_str}` \\(`{cash_pct_str}`\\)",
                f"📊 Exposure      : `{exposure_pct_str}`",
                f"📈 Total Return  : `{return_pct_str}`",
                f"💼 Posisi Aktif  : `{active_pos_count} Saham`",
                f"🏆 Top Pick      : `{top_pick_clean}`",
                "══════════════════════════════",
                "📋 *DETAIL POSISI AKTIF \\(TP / SL / BUY DATE\\)*"
            ]

            if isinstance(positions_dict, dict) and len(positions_dict) > 0:
                for idx, (ticker, pos_info) in enumerate(positions_dict.items(), 1):
                    t_sym = escape_markdown_v2_text(ticker)
                    lots = pos_info.get("lots", 0)
                    
                    avg_p_num = float(pos_info.get("avg_price", pos_info.get("buy_price", 0.0)))
                    avg_p = escape_markdown_v2_text(format_stock_price(avg_p_num))
                    
                    tp_val = float(pos_info.get("tp_price", pos_info.get("take_profit", avg_p_num * 1.05)))
                    sl_val = float(pos_info.get("sl_price", pos_info.get("stop_loss", avg_p_num * 0.95)))
                    
                    tp_str = escape_markdown_v2_text(format_stock_price(tp_val))
                    sl_str = escape_markdown_v2_text(format_stock_price(sl_val))
                    
                    unrealized_pnl_pct = pos_info.get("unrealized_pnl_pct", pos_info.get("pnl_pct", 0.0))
                    pnl_pct_str = escape_markdown_v2_text(f"{unrealized_pnl_pct:+.2f}%")
                    
                    buy_date = str(pos_info.get("buy_date", pos_info.get("entry_date", pos_info.get("timestamp", "Sesi Sebelumnya"))))
                    if "T" in buy_date:
                        buy_date = buy_date.split("T")[0]
                    buy_date_clean = escape_markdown_v2_text(buy_date)

                    md.append(f"{idx}\\. *{t_sym}* : `{lots} Lot` @ `{avg_p}`")
                    md.append(f"   🎯 TP: `{tp_str}` \\| 🛑 SL: `{sl_str}`")
                    md.append(f"   📅 Beli: `{buy_date_clean}` \\| PnL: `{pnl_pct_str}`")
            else:
                md.append("⚠️ *Tidak ada posisi saham aktif saat ini\\.*")

            md.extend([
                "══════════════════════════════",
                "🔹 *SINYAL PERDAGANGAN SAHAM \\(TOP 3 BELI\\)*"
            ])

            signals = summary_payload.get("signals", []) if summary_payload else []
            if not signals:
                md.append("\n⚠️ *Tidak ada sinyal BELI yang memenuhi kualifikasi audit pada periode ini \\(Mode Kas 100% Proteksi\\)\\.*")
            else:
                for sig in signals:
                    asset_raw = escape_markdown_v2_text(sig.get("asset", ""))
                    entry_val = float(sig.get("entry_price", 0.0))
                    tp_val = float(sig.get("tp_price", 0.0))
                    sl_val = float(sig.get("sl_price", 0.0))

                    gross_profit_pct = ((tp_val - entry_val) / entry_val) * 100.0 if entry_val > 0 else 0.0
                    net_profit_pct = gross_profit_pct - (IDX_ROUNDTRIP_FEE_PCT * 100.0)

                    prob = clamp_probability(sig.get("probability", 0.0)) * 100.0
                    conf = clamp_probability(sig.get("confidence", 0.0)) * 100.0
                    
                    exp_date = escape_markdown_v2_text(str(sig.get("prediction_horizon", sig.get("expected_date", "3-5 Hari"))))

                    entry_str = escape_markdown_v2_text(format_stock_price(entry_val))
                    tp_str = escape_markdown_v2_text(format_stock_price(tp_val))
                    sl_str = escape_markdown_v2_text(format_stock_price(sl_val))

                    gross_str = escape_markdown_v2_text(f"+{gross_profit_pct:.2f}% Gross")
                    net_str = escape_markdown_v2_text(f"+{net_profit_pct:.2f}% Net")
                    prob_str = escape_markdown_v2_text(f"{prob:.1f}%")
                    conf_str = escape_markdown_v2_text(f"{conf:.1f}%")

                    next_lines = [
                        f"🔸 *{asset_raw}*",
                        f"   💰 Harga Entry : `{entry_str}`",
                        f"   🎯 Target TP   : `{tp_str}`",
                        f"   🛑 Stop Loss   : `{sl_str}`",
                        f"   📈 Est\\. Profit : `{gross_str}` \\| `{net_str}`",
                        f"   ✅ Probabilitas: `{prob_str}` \\| ❇️ Conf: `{conf_str}`",
                        f"   🗓️ Horizon     : `{exp_date}`",
                        f"   📌 Rekomendasi : *BELI*",
                        ""
                    ]
                    md.extend(next_lines)

            # 🟢 GENERASI DAN INJEKSI GEMINI AI INSIGHT NARRATIVE
            ai_insight = p_state.get("ai_insight")
            if not ai_insight and self.gemini_engine.client:
                ai_insight = self.gemini_engine.generate_market_insight(p_state, signals)

            if ai_insight:
                clean_ai_narrative = escape_markdown_v2_text(ai_insight)
                md.extend([
                    "══════════════════════════════",
                    "🤖 *AI MARKET INSIGHT \\(GEMINI\\)*",
                    f"{clean_ai_narrative}"
                ])

            metrics = summary_payload.get("summary_metrics", {}) if summary_payload else {}
            tot_rec = metrics.get("total_received", 0)
            tot_app = metrics.get("total_approved", 0)
            f_ver = escape_markdown_v2_text(self.framework_version)

            md.extend([
                "══════════════════════════════",
                "🧠 *STATISTICAL & SYSTEM METRICS*",
                f"• Total Sinyal Diterima : `{tot_rec}`",
                f"• Sinyal Lolos Audit  : `{tot_app}`",
                "• Circuit Breaker Status: `🟢 NORMAL`",
                f"• Engine Version        : `v{f_ver}`",
                "══════════════════════════════"
            ])
            return "\n".join(md)

    def build_portfolio_state_message(self, portfolio_data: PortfolioStateDict) -> str:
        with self.lock:
            now_wib = datetime.now(timezone.utc) + WIB_OFFSET
            month_str = INDONESIAN_MONTHS[now_wib.month - 1]
            header_time = escape_markdown_v2_text(f"{now_wib.day} {month_str} {now_wib.year} pukul {now_wib.strftime('%H:%M')} WIB")

            default_capital = float(self.config.get("INITIAL_CAPITAL_IDR", 10_000_000.0))
            equity = float(portfolio_data.get("equity", default_capital))
            cash = float(portfolio_data.get("cash", default_capital))
            active_pos = portfolio_data.get("active_positions_count", 0)
            top_pick = escape_markdown_v2_text(portfolio_data.get("top_pick", "-"))

            lines = [
                "🔄 *[RESTORE & RESET PORTFOLIO SIMULATION]*",
                f"🗓️ {header_time}",
                "",
                "💵 *Virtual Balance Restored:*",
                f"   💰 Total Equity : `{escape_markdown_v2_text(format_stock_price(equity))}`",
                f"   🏦 Cash Balance : `{escape_markdown_v2_text(format_stock_price(cash))}`",
                f"   📊 Active Positions : `{active_pos}`",
                f"   🏆 Top Pick Asset : `{top_pick}`",
                "",
                "⚙️ *System Status:*",
                "   • Dry\\-Run State Checkpoint: `CLEARED & REINITIALIZED`",
                f"   • Paper Capital Allocation: `{escape_markdown_v2_text(format_stock_price(default_capital))} Baseline`",
                "   • Next Pipeline Execution: `READY`"
            ]
            return "\n".join(lines)

    def _create_empty_payload(self, analysis_date: date, payload_hash: str, iso_eval_time: str, total_received: int = 0) -> ProcessedSummaryPayload:
        return {
            "metadata": {
                "evaluation_timestamp": iso_eval_time,
                "report_date": analysis_date.isoformat(),
                "framework_version": self.framework_version,
                "model_version": self.model_version,
                "symbol_count": 0,
                "payload_hash": payload_hash
            },
            "summary_metrics": {
                "total_received": total_received,
                "total_approved": 0,
                "total_transmitted": 0,
                "total_rejected": total_received
            },
            "signals": []
        }

# ==============================================================================
# TELEGRAM REPORTER ENGINE
# ==============================================================================
class TelegramReporter:
    def __init__(self, config: Any = None, mode: str = "dry_run") -> None:
        self.lock = threading.RLock()
        self.mode = str(mode or os.getenv("EXECUTION_MODE", "dry_run")).lower().strip()
        self.config_dict = config if isinstance(config, dict) else {}
        
        self.timeout = float(
            self.config_dict.get("TELEGRAM_TIMEOUT") or
            self.config_dict.get("REPORTING_TELEGRAM_TIMEOUT") or 15.0
        )
        
        self.token = (
            self.config_dict.get("TELEGRAM_BOT_TOKEN") or
            self.config_dict.get("TELEGRAM_TOKEN") or
            os.environ.get("TELEGRAM_BOT_TOKEN") or
            os.environ.get("TELEGRAM_TOKEN")
        )
        self.chat_id = (
            self.config_dict.get("TELEGRAM_CHAT_ID") or
            self.config_dict.get("CHAT_ID") or
            os.environ.get("TELEGRAM_CHAT_ID") or
            os.environ.get("CHAT_ID")
        )
        
        self.session = requests.Session()
        retries = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=[408, 409, 500, 502, 503, 504, 520],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get_config_dict(self) -> Dict[str, Any]:
        return self.config_dict

    def _split_message(self, message_text: str, max_chars: int = 3900) -> List[str]:
        if len(message_text) <= max_chars:
            return [message_text]

        chunks = []
        lines = message_text.split("\n")
        current_chunk: List[str] = []
        current_length = 0

        for line in lines:
            line_len = len(line) + 1
            if line_len > max_chars:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                for i in range(0, len(line), max_chars):
                    chunks.append(line[i:i + max_chars])
                continue

            if current_length + line_len > max_chars:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_len
            else:
                current_chunk.append(line)
                current_length += line_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def send_message(self, message_text: str, parse_mode: str = "MarkdownV2") -> bool:
        with self.lock:
            if not self.token or not self.chat_id:
                logger.warning("⚠️ Kredensial Telegram (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) belum terkonfigurasi.")
                return False

            chunks = self._split_message(message_text, max_chars=3900)
            all_success = True

            for chunk in chunks:
                GLOBAL_RATE_LIMITER.acquire()
                url = f"https://api.telegram.org/bot{self.token}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": chunk,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True
                }

                for attempt in range(3):
                    try:
                        response = self.session.post(url, json=payload, timeout=self.timeout)
                        if response.status_code == 200:
                            res_data = response.json()
                            if res_data.get("ok") is True:
                                msg_id = res_data.get("result", {}).get("message_id", "N/A")
                                logger.info(f"✅ Bagian laporan berhasil dikirim ke Telegram (Msg ID: {msg_id}).")
                                break
                            else:
                                error_desc = res_data.get("description", "Unknown Telegram Error")
                                logger.error(f"❌ Telegram API Logical Error: {error_desc}")
                                all_success = False
                                break
                        elif response.status_code == 429:
                            try:
                                res_data = response.json()
                                retry_after = int(res_data.get("parameters", {}).get("retry_after", 3))
                            except Exception:
                                retry_after = 3
                            logger.warning(f"⚠️ Telegram Rate Limit (HTTP 429). Waiting {retry_after}s...")
                            time.sleep(retry_after)
                        else:
                            logger.error(f"❌ Telegram API HTTP Error [{response.status_code}]: {response.text}")
                            all_success = False
                            break
                    except Exception as err:
                        logger.error(f"❌ Kegagalan jaringan saat mengirim pesan Telegram: {err}", exc_info=True)
                        all_success = False
                        break

            return all_success

    def _adapt_orders_for_generator(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        adapted = []
        now_wib = datetime.now(timezone.utc) + WIB_OFFSET
        now_date_str = now_wib.date().isoformat()
        now_iso = datetime.now(timezone.utc).isoformat()

        for idx, item in enumerate(orders):
            if not isinstance(item, dict):
                continue

            ticker_raw = item.get("ticker") or item.get("symbol") or item.get("asset") or f"UNKNOWN_{idx}"
            try:
                ticker = normalize_stock_symbol(ticker_raw)
            except ValidationError as e:
                logger.warning(f"⚠️ Melewati sinyal tidak valid [{ticker_raw}]: {e}")
                continue
            
            price = float(item.get("entry_price", item.get("price", item.get("close", item.get("current_price", 0.0)))))
            tp = float(item.get("target_price", item.get("take_profit", item.get("tp_price", item.get("optimized_take_profit", price * 1.05)))))
            sl = float(item.get("stop_loss", item.get("sl_price", item.get("optimized_stop_loss", price * 0.95))))
            
            prob = clamp_probability(item.get("prediction_probability") or item.get("probability") or item.get("calibrated_prob") or 0.50)
            conf = clamp_probability(item.get("prediction_confidence") or item.get("confidence") or item.get("signal_confidence") or (prob * 0.9))

            rank = float(item.get("signal_rank_score", item.get("ranking_score", prob * conf)))
            
            direction_raw = item.get("direction") or item.get("candidate_signal") or item.get("recommendation") or item.get("signal") or item.get("action") or "BUY"
            direction_clean = str(direction_raw).upper().strip()
            
            if direction_clean in ["BUY", "BELI", "STRONG_BUY", "STRONG BUY", "LONG", "1", "1.0"]:
                direction_clean = "BUY"
            else:
                direction_clean = "SELL"

            horizon = str(item.get("prediction_horizon") or item.get("expected_holding_days") or item.get("horizon") or "3-5 Hari")
            primary_reason = str(item.get("primary_reason", item.get("signal_reason", "MODEL_INFERENCE")))

            adapted.append({
                "signal_uuid": str(item.get("signal_uuid", f"SIG-{now_date_str}-{ticker}-{idx}")),
                "asset": ticker,
                "timestamp": item.get("timestamp", now_iso),
                "direction": direction_clean,
                "entry_price": price,
                "tp_price": tp,
                "sl_price": sl,
                "probability": prob,
                "confidence": conf,
                "ranking_score": rank,
                "market_regime": str(item.get("market_regime", "BULLISH")),
                "primary_reason": primary_reason,
                "prediction_horizon": horizon
            })
        return adapted

    def broadcast_signals(self, orders: Optional[List[Dict[str, Any]]] = None, portfolio_data: Optional[PortfolioStateDict] = None) -> bool:
        with self.lock:
            config_dict = self._get_config_dict()
            generator = SignalSummaryGenerator(config=config_dict, mode=self.mode)
            
            if portfolio_data is not None and portfolio_data.get("reset_event") is True:
                msg_text = generator.build_portfolio_state_message(portfolio_data)
                return self.send_message(msg_text, parse_mode="MarkdownV2")

            adapted_orders = self._adapt_orders_for_generator(orders) if orders else []
            summary_payload = generator.process_signals(adapted_orders)
            dashboard_message_text = generator.build_telegram_message(summary_payload, portfolio_data=portfolio_data)
            return self.send_message(dashboard_message_text, parse_mode="MarkdownV2")

# ==============================================================================
# UNIFIED REPORTING ENGINE (FACADE PATTERN)
# ==============================================================================
class UnifiedReportingEngine:
    """Facade Class Terpusat sebagai Single Entry Point Pelaporan & Notifikasi Telegram."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, mode: Optional[str] = None) -> None:
        self.config = config or {}
        self.mode = str(mode or os.getenv("EXECUTION_MODE", "dry_run")).lower().strip()
        self.reporter = TelegramReporter(config=self.config, mode=self.mode)

    def is_configured(self) -> bool:
        return bool(self.reporter.token and self.reporter.chat_id)

    def send_telegram_broadcast(self, orders: Optional[List[Dict[str, Any]]] = None, portfolio_data: Optional[PortfolioStateDict] = None) -> bool:
        return self.reporter.broadcast_signals(orders=orders, portfolio_data=portfolio_data)

    def broadcast_signals(self, orders: Optional[List[Dict[str, Any]]] = None, portfolio_data: Optional[PortfolioStateDict] = None) -> bool:
        return self.reporter.broadcast_signals(orders=orders, portfolio_data=portfolio_data)

    def send_portfolio_reset_notification(self, portfolio_data: PortfolioStateDict) -> bool:
        data = dict(portfolio_data) if portfolio_data else {}
        data["reset_event"] = True
        return self.reporter.broadcast_signals(orders=None, portfolio_data=data)

    def send_alert(self, title: str, message: str, level: str = "INFO") -> bool:
        icon = "ℹ️"
        if level.upper() == "WARNING": icon = "⚠️"
        elif level.upper() in ["ERROR", "CRITICAL"]: icon = "🚨"
        
        title_clean = escape_markdown_v2_text(title)
        message_clean = escape_markdown_v2_text(message)

        formatted_msg = (
            f"{icon} *[SYSTEM ALERT \\- {escape_markdown_v2_text(level.upper())}]*\n"
            f"📌 *{title_clean}*\n"
            "══════════════════════════════\n"
            f"{message_clean}\n"
            "══════════════════════════════"
        )
        return self.reporter.send_message(formatted_msg, parse_mode="MarkdownV2")

    def generate_report_payload(self, orders: Optional[List[Dict[str, Any]]] = None, target_date: Optional[date] = None) -> ProcessedSummaryPayload:
        generator = SignalSummaryGenerator(config=self.config, mode=self.mode)
        adapted_orders = self.reporter._adapt_orders_for_generator(orders) if orders else []
        return generator.process_signals(adapted_orders, target_date=target_date)

    def invalidate_state_cache(self) -> None:
        GLOBAL_PORTFOLIO_CACHE.invalidate()


def broadcast_signals(orders: Optional[List[Dict[str, Any]]] = None, portfolio_data: Optional[PortfolioStateDict] = None, config: Optional[Dict[str, Any]] = None) -> bool:
    engine = UnifiedReportingEngine(config=config)
    return engine.broadcast_signals(orders=orders, portfolio_data=portfolio_data)

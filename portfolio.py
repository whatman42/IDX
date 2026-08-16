"""
=============================================================================
IDX Quantitative Portfolio & Execution Engine - Consolidated Module
FileName      : portfolio.py
Directory     : Flat Directory (Root Level selevel dengan main.py)
Version       : 2026.Q3.v23.2 (Single Unified Mode & Gemini AI Advisory Layer Patched)
Compliance    : Indonesia Stock Exchange (IDX) Trading Rules & Yahoo Finance (.JK)
=============================================================================
"""

import os
import json
import math
import logging
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Tuple, Optional, Union, Set

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import polars as pl

# Standar Zona Waktu Pasar Saham Indonesia (WIB)
WIB_TZ = ZoneInfo("Asia/Jakarta")

# ==============================================================================
# MODEL BASELINE GEMINI SDK (SERAGAM DENGAN MAIN.PY & SYSTEM-WIDE)
# ==============================================================================
PRIMARY_MODEL: str = "gemini-3.6-flash"
FALLBACK_MODEL: str = "gemini-3.5-flash-lite"


# Import Google GenAI Client
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# ==============================================================================
# LOGGER CONFIGURATION & COMPATIBILITY FALLBACKS
# ==============================================================================
try:
    from logger import get_logger
    logger = get_logger("IDX.Portfolio")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("IDX.Portfolio")

# ==============================================================================
# KONSTANTA PASAR SAHAM INDONESIA & INSTITUTIONAL OVERLAYS (SINGLE UNIFIED MODE)
# ==============================================================================
IDX_LOT_SIZE: int = 100                             # 1 Lot = 100 Lembar Saham
IDX_BUY_FEE_PCT: float = 0.0015                     # Fee Transaksi Beli Broker Default (0.15%)
IDX_SELL_FEE_PCT: float = 0.0025                    # Fee Transaksi Jual Broker (0.25% termasuk PPh Final 0.1%)
IDX_MAX_SINGLE_STOCK_WEIGHT: float = 0.20           # Maksimal Alokasi per Saham (20% dari Total Equity)
IDX_MAX_SECTOR_WEIGHT: float = 0.35                 # Maksimal Eksposure per Sektor (35% dari Total Equity)
IDX_MIN_PRICE_IDR: float = 50.0                     # Batas Harga Minimal Papan Reguler BEI (Rp 50)
IDX_MIN_ADTV_IDR: float = 100_000_000.0             # ADTV Minimal Tunggal (Rp 100 Juta)
MAX_ADTV_PARTICIPATION_PCT: float = 0.05             # Maksimal Batas Partisipasi Order (5% dari ADTV 24 Jam)
MAX_PORTFOLIO_LEVERAGE: float = 1.0                 # Long-Only Equity Spot (Batas Atas Leverage 1.0x)
DEFAULT_SLIPPAGE_BPS: float = 5.0                   # Half-Spread / Slippage Base (5 Basis Points / 0.05%)

# Koefisien Model Almgren-Chriss Optimal Execution
ALMGREN_CHRISS_GAMMA: float = 0.10                  # Permanent Impact Coefficient
ALMGREN_CHRISS_ETA: float = 0.15                    # Temporary Impact Coefficient
EXECUTION_HORIZON_TAU: float = 0.25                 # Base Execution Horizon (Fraction of Trading Day)

CASH_BUFFER_BASE_PCT: float = 0.02                  # Base Cash Buffer Guardrail (2% dari Total Ekuitas)
MIN_REBALANCE_BENEFIT_RATIO: float = 1.5            # Rasio Manfaat Rebalance terhadap Total Biaya Transaksi
MIN_REBALANCE_THRESHOLD_PCT: float = 0.03           # Ambang Batas Minimum Rebalance (3%)
MAX_ANNUAL_TURNOVER_PCT: float = 1.50               # Batas Maksimal Annual Portfolio Turnover (150%)

# Risk Switch, Volatility & Black-Litterman Constants
MAX_DRAWDOWN_KILL_SWITCH_PCT: float = 0.15          # Drawdown Kill Switch (15% Peak-to-Trough)
RECOVERY_DRAWDOWN_THRESHOLD_PCT: float = 0.07       # Recovery Threshold (7% Drawdown untuk reset Kill Switch)
RISK_AVERSION_GAMMA: float = 2.5                    # Risk Aversion Parameter Mean-Variance
BLACK_LITTERMAN_TAU: float = 0.05                   # Black-Litterman Tau Parameter
QUADRATIC_TURNOVER_LAMBDA: float = 0.01             # Penalti Turnover Kuadratik

DEFAULT_INITIAL_CAPITAL_IDR: float = 10_000_000.0   # Modal Baseline Tunggal Default (Rp 10 Juta)
DEFAULT_STATE_FILENAME: str = "portfolio_state.json" # File Persistensi State Tunggal


class PortfolioError(Exception): pass
class InsufficientCashError(PortfolioError): pass
class InvalidOrderError(PortfolioError): pass
class CovarianceError(PortfolioError): pass
class CorrelationError(PortfolioError): pass
class HRPError(PortfolioError): pass
class AllocationOptimizerError(PortfolioError): pass
class DiversificationError(PortfolioError): pass
class ExposureControllerError(PortfolioError): pass
class SectorConstraintsError(PortfolioError): pass


# ==============================================================================
# HELPER & UTILITY FUNCTIONS
# ==============================================================================
def _ensure_polars_df(data: Any) -> pl.DataFrame:
    """Sanitasi input internal untuk menjamin objek bertipe pl.DataFrame."""
    if data is None:
        return pl.DataFrame()
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if isinstance(data, list):
        if not data:
            return pl.DataFrame()
        return pl.DataFrame(data)
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
    except ImportError:
        pass
    return pl.DataFrame(data)


def normalize_idx_symbol(symbol: str) -> str:
    """Memastikan format ticker saham Indonesia selalu menggunakan akhiran .JK (misal: ASII -> ASII.JK)."""
    if not isinstance(symbol, str) or not symbol.strip():
        return ""
    clean_sym = symbol.strip().upper()
    if not clean_sym.endswith(".JK") and not clean_sym.startswith("^"):
        clean_sym = f"{clean_sym}.JK"
    return clean_sym


def get_idx_tick_size(price: float) -> float:
    """Menentukan fraksi harga (tick size) resmi Bursa Efek Indonesia (BEI)."""
    if price < 200.0:
        return 1.0
    elif price < 500.0:
        return 2.0
    elif price < 2000.0:
        return 5.0
    elif price < 5000.0:
        return 10.0
    else:
        return 25.0


def round_to_idx_tick(price: float, round_mode: str = "nearest") -> float:
    """Membulatkan harga sesuai dengan aturan tick size BEI dan mode pembulatan."""
    tick = get_idx_tick_size(price)
    if tick <= 0:
        return price
    if round_mode == "up":
        return math.ceil(price / tick) * tick
    elif round_mode == "down":
        return math.floor(price / tick) * tick
    return round(price / tick) * tick


def make_positive_semi_definite(cov: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Proyeksi matriks kovarians ke Nearest Positive Semi-Definite (PSD) menggunakan Eigenvalue Clipping."""
    cov = np.nan_to_num(cov, nan=epsilon)
    cov_sym = (cov + cov.T) / 2.0
    evals, evecs = np.linalg.eigh(cov_sym)
    evals = np.maximum(evals, epsilon)
    psd_cov = evecs @ np.diag(evals) @ evecs.T
    return (psd_cov + psd_cov.T) / 2.0


def robust_clean_returns(X: np.ndarray, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> np.ndarray:
    """Robust Winsorization untuk memangkas outlier ekstrem dari data return saham."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[0] < 2:
        return X
    q_low = np.quantile(X, lower_quantile, axis=0)
    q_high = np.quantile(X, upper_quantile, axis=0)
    return np.clip(X, q_low, q_high)


def denoise_covariance_marcenko_pastur(cov: np.ndarray, T: int, N: int) -> np.ndarray:
    """Random Matrix Theory (RMT) Eigenvalue Denoising menggunakan Batas Teoritis Marcenko-Pastur."""
    if T <= N or N <= 1:
        return make_positive_semi_definite(cov)

    std_devs = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    inv_std = 1.0 / std_devs
    corr = cov * np.outer(inv_std, inv_std)
    np.clip(corr, -1.0, 1.0, out=corr)

    evals, evecs = np.linalg.eigh(corr)
    
    q = float(T) / float(N)
    sigma_sq = 1.0
    lambda_plus = sigma_sq * ((1.0 + np.sqrt(1.0 / q)) ** 2)

    noise_evals = evals[evals <= lambda_plus]
    if len(noise_evals) > 0:
        avg_noise = np.mean(noise_evals)
        evals[evals <= lambda_plus] = avg_noise

    denoised_corr = evecs @ np.diag(evals) @ evecs.T
    np.fill_diagonal(denoised_corr, 1.0)
    denoised_corr = np.clip(denoised_corr, -1.0, 1.0)

    denoised_cov = denoised_corr * np.outer(std_devs, std_devs)
    return make_positive_semi_definite(denoised_cov)


def _normalize_volume_idr(volume_24h: float, price: float, min_adtv: float = IDX_MIN_ADTV_IDR) -> float:
    """
    Memastikan volume 24 jam selalu direpresentasikan dalam estimasi nominal Rupiah (turnover IDR).
    Mengonversi unit lembar saham menjadi Rupiah jika volume yang diberikan < min_adtv.
    """
    if volume_24h <= 0 or math.isinf(volume_24h) or math.isnan(volume_24h):
        return float('inf')
    if volume_24h < min_adtv and price > 0:
        recalculated_idr = volume_24h * price
        if recalculated_idr >= min_adtv:
            return recalculated_idr
    return volume_24h


def _extract_action(row: Dict[str, Any]) -> str:
    """Ekstraksi aksi sinyal (BUY/SELL) dari berbagai variasi nama kolom."""
    for k in ["recommendation", "rekomendasi", "direction", "signal", "action", "trade_action", "side"]:
        val = row.get(k)
        if val is not None and str(val).strip():
            v_str = str(val).strip().upper()
            if v_str in ["BUY", "BELI", "STRONG_BUY", "STRONG BUY", "BUY_SIGNAL", "1", "1.0"]:
                return "BUY"
            if v_str in ["SELL", "JUAL", "STRONG_SELL", "STRONG SELL", "EXIT", "STOP_LOSS", "TAKE_PROFIT", "-1", "-1.0"]:
                return "SELL"
            return v_str
    return ""


def _extract_symbol(row: Dict[str, Any]) -> str:
    """Ekstraksi ticker saham dari berbagai variasi nama kolom."""
    for k in ["symbol", "ticker", "asset", "asset_id"]:
        val = row.get(k)
        if val is not None and str(val).strip():
            return normalize_idx_symbol(str(val).strip())
    return ""


def _extract_float(row: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    """Ekstraksi nilai float opsional secara defensif."""
    for k in keys:
        val = row.get(k)
        if val is not None:
            try:
                v = float(val)
                if not math.isnan(v) and not math.isinf(v):
                    return v
            except (ValueError, TypeError):
                pass
    return default


# ==============================================================================
# GEMINI PORTFOLIO DIAGNOSTIC ENGINE
# ==============================================================================
class GeminiPortfolioDiagnosticEngine:
    """
    Sub-Engine Google Gemini AI sebagai Meta-Diagnostic & Advisory Layer untuk Portofolio.
    Mengevaluasi alokasi portofolio, risiko konsentrasi, serta skenario stress test pasar BEI.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None

        if HAS_GEMINI_SDK and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("🤖 [GEMINI_PORTFOLIO_INIT] Gemini Client terhubung untuk Diagnostik Portofolio.")
            except Exception as err:
                logger.warning(f"⚠️ [GEMINI_PORTFOLIO_INIT_FAILED] Gagal inisialisasi Gemini Client: {err}")

    def run_portfolio_advisory(
        self, 
        portfolio_summary: Dict[str, Any], 
        stress_test_results: Dict[str, Any],
        market_regime: str = "SIDEWAYS"
    ) -> Dict[str, Any]:
        """Melakukan analisis kualitatif dan advisory atas portofolio menggunakan Google Gemini AI."""
        if not self.client:
            return {
                "ai_portfolio_diagnostic": "Gemini AI Client tidak aktif.",
                "ai_advisory_recommendation": "Pertahankan guardrail alokasi standar (Max Stock 20%, Max Sector 35%).",
                "suggested_cash_buffer_pct": CASH_BUFFER_BASE_PCT * 100.0
            }

        prompt = f"""
        Sebagai Chief Risk Officer & Portfolio Manager BEI (IDX), lakukan evaluasi diagnostik terhadap status portofolio saham berikut:

        === RINGKASAN PORTOFOLIO SAAT INI ===
        - Total Ekuitas: Rp {portfolio_summary.get('total_equity', 0.0):,.0f}
        - Saldo Kas: Rp {portfolio_summary.get('cash_balance', 0.0):,.0f}
        - Drawdown Saat Ini: {portfolio_summary.get('drawdown_pct', 0.0):.2f}%
        - Jumlah Posisi Aktif: {portfolio_summary.get('active_positions_count', 0)}
        - Regime Pasar: {market_regime}
        - Kill Switch Status: {portfolio_summary.get('kill_switch_active', False)}

        === HASIL STRESS TEST SEKTORAL & KRISIS ===
        {json.dumps(stress_test_results, indent=2, default=str)}

        Tugas:
        1. Berikan opini diagnostik ringkas (maksimal 2-3 kalimat) tentang kesehatan alokasi dan risiko konsentrasi portofolio.
        2. Berikan rekomendasi operasional (misal: Pertahankan Alokasi, Tingkatkan Cash Buffer, atau Rebalancing Sektoral).
        3. Sarankan persentase Cash Buffer ideal (antara 2.0% hingga 20.0%).

        Format Jawaban (JSON murni):
        {{
            "diagnostic": "...",
            "recommendation": "...",
            "suggested_cash_buffer_pct": 5.0
        }}
        """

        for model_target in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                response = self.client.models.generate_content(
                    model=model_target,
                    contents=prompt,
                )
                raw_text = response.text.strip()
                
                if raw_text.startswith("```json"):
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()
                    
                parsed = json.loads(raw_text)
                return {
                    "ai_portfolio_diagnostic": parsed.get("diagnostic", "Analisis portofolio berhasil."),
                    "ai_advisory_recommendation": parsed.get("recommendation", "Lanjutkan alokasi sesuai parameter optimizer."),
                    "suggested_cash_buffer_pct": float(parsed.get("suggested_cash_buffer_pct", CASH_BUFFER_BASE_PCT * 100.0))
                }
            except Exception as err:
                logger.warning(f"⚠️ [GEMINI_PORTFOLIO_ADVISORY_FAILED] Error pada model {model_target}: {err}")

        return {
            "ai_portfolio_diagnostic": "Gagal memperoleh respon AI Gemini.",
            "ai_advisory_recommendation": "Gunakan guardrail risiko standar.",
            "suggested_cash_buffer_pct": CASH_BUFFER_BASE_PCT * 100.0
        }


# ==============================================================================
# IDX POSITION, EXECUTION & MTM ENGINE
# ==============================================================================
class IDXExecutionEngine:
    def __init__(
        self, 
        state_file: str = DEFAULT_STATE_FILENAME, 
        initial_capital: float = DEFAULT_INITIAL_CAPITAL_IDR,
        max_stock_weight: float = IDX_MAX_SINGLE_STOCK_WEIGHT,
        buy_fee_pct: float = IDX_BUY_FEE_PCT, 
        sell_fee_pct: float = IDX_SELL_FEE_PCT
    ) -> None:
        self._lock = threading.RLock()
        self.state_file = state_file
        
        self.initial_capital = max(float(initial_capital), 1_000_000.0)
        self.max_stock_weight = float(max_stock_weight)
        self.buy_fee_pct = float(buy_fee_pct)
        self.sell_fee_pct = float(sell_fee_pct)
        self.min_adtv_idr = IDX_MIN_ADTV_IDR

        self._logged_adtv_warnings: Set[str] = set()
        
        self.cash_balance: float = self.initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.transaction_history: List[Dict[str, Any]] = []
        self.total_equity: float = self.initial_capital
        self.peak_equity: float = self.initial_capital
        self.realized_pnl_accumulated: float = 0.0
        self.cumulative_turnover_idr: float = 0.0
        self.kill_switch_active: bool = False
        self.start_timestamp_utc: str = datetime.now(timezone.utc).isoformat()
        
        self.load_state()

    def reset_warning_cache(self) -> None:
        with self._lock:
            self._logged_adtv_warnings.clear()

    def load_state(self) -> Dict[str, Any]:
        with self._lock:
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.cash_balance = max(float(data.get("cash_balance", data.get("cash", self.initial_capital))), 0.0)
                        self.positions = data.get("positions", {})
                        self.transaction_history = data.get("transaction_history", [])
                        self.realized_pnl_accumulated = float(data.get("realized_pnl_accumulated", 0.0))
                        self.total_equity = max(float(data.get("total_equity", data.get("equity", self.cash_balance))), self.cash_balance)
                        self.peak_equity = float(data.get("peak_equity", max(self.total_equity, self.initial_capital)))
                        self.cumulative_turnover_idr = float(data.get("cumulative_turnover_idr", 0.0))
                        self.kill_switch_active = bool(data.get("kill_switch_active", False))
                        self.start_timestamp_utc = data.get("start_timestamp_utc", datetime.now(timezone.utc).isoformat())
                        
                        # REKONSILIASI OTOMATIS SAAT STATE DIMUAT
                        if not self.positions:
                            self.cash_balance = self.total_equity
                            
                        logger.info(f"State portofolio dimuat dari {self.state_file}. Kas: Rp {self.cash_balance:,.2f}")
                except Exception as e:
                    logger.error(f"Gagal memuat state dari {self.state_file}: {e}. Reset ke state aman.")
            else:
                self.save_state()
            return self.get_state_summary()

    def get_elapsed_years(self) -> float:
        try:
            clean_ts = self.start_timestamp_utc.replace("Z", "+00:00")
            start_dt = datetime.fromisoformat(clean_ts)
            now_dt = datetime.now(timezone.utc)
            elapsed_sec = max((now_dt - start_dt).total_seconds(), 86400.0)
            return elapsed_sec / 31536000.0
        except Exception:
            return 1.0

    def save_state(self) -> bool:
        with self._lock:
            try:
                # REKONSILIASI KAS & EXPOSURE SAAT POSISI KOSONG (0 SAHAM)
                if not self.positions:
                    self.cash_balance = self.total_equity
                    invested_amount = 0.0
                    exposure_pct = 0.0
                    top_pick = "-"
                else:
                    invested_amount = sum(
                        float(p.get("market_value", p.get("shares", 0) * p.get("current_price", 0)))
                        for p in self.positions.values()
                    )
                    exposure_pct = min(100.0, max(0.0, (invested_amount / self.total_equity * 100.0))) if self.total_equity > 0 else 0.0
                    top_pick = list(self.positions.keys())[0]

                elapsed_years = self.get_elapsed_years()
                avg_equity = (self.initial_capital + self.total_equity) / 2.0
                
                annualized_turnover = (self.cumulative_turnover_idr / avg_equity / elapsed_years) if avg_equity > 0 else 0.0
                drawdown = ((self.peak_equity - self.total_equity) / self.peak_equity) if self.peak_equity > 0 else 0.0

                state_data = {
                    "equity": self.total_equity,
                    "cash": self.cash_balance,
                    "cash_balance": self.cash_balance,
                    "total_equity": self.total_equity,
                    "peak_equity": self.peak_equity,
                    "invested_amount": invested_amount,
                    "drawdown_pct": drawdown * 100.0,
                    "kill_switch_active": self.kill_switch_active,
                    "exposure_pct": exposure_pct,
                    "return_pct": ((self.total_equity - self.initial_capital) / self.initial_capital * 100.0) if self.initial_capital > 0 else 0.0,
                    "active_positions_count": len(self.positions),
                    "positions_count": len(self.positions),
                    "active_positions": list(self.positions.values()),
                    "positions": self.positions,
                    "top_pick": top_pick,
                    "realized_pnl_accumulated": self.realized_pnl_accumulated,
                    "cumulative_turnover_idr": self.cumulative_turnover_idr,
                    "annualized_turnover_pct": annualized_turnover * 100.0,
                    "start_timestamp_utc": self.start_timestamp_utc,
                    "transaction_history": self.transaction_history[-500:],
                    "last_updated_wib": datetime.now(WIB_TZ).strftime("%Y-%m-%d %H:%M:%S WIB")
                }

                tmp_file = f"{self.state_file}.tmp"
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(state_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                
                os.replace(tmp_file, self.state_file)
                return True
            except Exception as e:
                logger.error(f"Gagal menyimpan state secara atomik ke {self.state_file}: {e}")
                return False

    def get_state_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cash_balance": self.cash_balance,
                "total_equity": self.total_equity,
                "peak_equity": self.peak_equity,
                "kill_switch_active": self.kill_switch_active,
                "realized_pnl_accumulated": self.realized_pnl_accumulated,
                "cumulative_turnover_idr": self.cumulative_turnover_idr,
                "active_positions_count": len(self.positions),
                "positions": self.positions,
                "state_file": self.state_file
            }

    def compute_almgren_chriss_impact_price(self, base_price: float, trade_nominal: float, volume_24h_idr: float, volatility_daily: float = 0.02, is_buy: bool = True) -> float:
        """Model Eksekusi Almgren-Chriss dengan Intraday U-Shape Liquidity Factor & Tick-size Awareness."""
        vol_idr = _normalize_volume_idr(volume_24h_idr, base_price, self.min_adtv_idr)
        
        if vol_idr <= 0 or trade_nominal <= 0:
            return round_to_idx_tick(base_price, round_mode="up" if is_buy else "down")
        
        part_ratio = min(trade_nominal / vol_idr, MAX_ADTV_PARTICIPATION_PCT)
        intraday_u_shape_factor = 1.15
        
        optimal_tau = math.sqrt(max(ALMGREN_CHRISS_ETA / (ALMGREN_CHRISS_GAMMA * RISK_AVERSION_GAMMA * (volatility_daily**2) + 1e-12), 0.05))
        tau = min(optimal_tau, EXECUTION_HORIZON_TAU)

        perm_impact = ALMGREN_CHRISS_GAMMA * part_ratio
        temp_impact = ALMGREN_CHRISS_ETA * volatility_daily * math.sqrt(part_ratio / tau) * intraday_u_shape_factor
        
        half_spread = (get_idx_tick_size(base_price) / max(base_price, 1.0)) / 2.0
        total_impact_pct = max(half_spread, DEFAULT_SLIPPAGE_BPS / 10000.0) + perm_impact + temp_impact

        if is_buy:
            impacted_price = base_price * (1.0 + total_impact_pct)
            return round_to_idx_tick(impacted_price, round_mode="up")
        else:
            impacted_price = base_price * (1.0 - total_impact_pct)
            return round_to_idx_tick(impacted_price, round_mode="down")

    def calculate_max_buyable_lots(self, ticker: str, target_nominal: float, execution_price: float, volume_24h_idr: float = float('inf')) -> Tuple[int, float, float]:
        if execution_price < IDX_MIN_PRICE_IDR:
            return 0, 0.0, 0.0

        vol_idr = _normalize_volume_idr(volume_24h_idr, execution_price, self.min_adtv_idr)

        if vol_idr < self.min_adtv_idr:
            if ticker not in self._logged_adtv_warnings:
                logger.warning(
                    f"⚠️ [ADTV REJECT] {ticker}: ADTV 24j (Rp {vol_idr:,.0f}) "
                    f"di bawah minimum Rp {self.min_adtv_idr:,.0f}"
                )
                self._logged_adtv_warnings.add(ticker)
            return 0, 0.0, 0.0

        required_cash_buffer = self.total_equity * CASH_BUFFER_BASE_PCT
        available_trading_cash = max(0.0, self.cash_balance - required_cash_buffer)
        
        if available_trading_cash <= 0.0:
            return 0, 0.0, 0.0

        max_adtv_nominal = vol_idr * MAX_ADTV_PARTICIPATION_PCT
        max_allowed_single_stock = self.total_equity * self.max_stock_weight
        
        effective_target_nominal = min(target_nominal, max_adtv_nominal, max_allowed_single_stock, available_trading_cash)

        cost_per_lot_with_fee = IDX_LOT_SIZE * execution_price * (1.0 + self.buy_fee_pct)
        if cost_per_lot_with_fee <= 0.0 or effective_target_nominal < cost_per_lot_with_fee:
            return 0, 0.0, 0.0

        max_lots = int(math.floor(effective_target_nominal / cost_per_lot_with_fee))
        if max_lots <= 0:
            return 0, 0.0, 0.0

        total_shares = max_lots * IDX_LOT_SIZE
        gross_cost = total_shares * execution_price
        buy_fee = gross_cost * self.buy_fee_pct
        total_required_cash = gross_cost + buy_fee

        if total_required_cash > available_trading_cash:
            max_lots = int(math.floor(available_trading_cash / cost_per_lot_with_fee))
            total_shares = max_lots * IDX_LOT_SIZE
            gross_cost = total_shares * execution_price
            buy_fee = gross_cost * self.buy_fee_pct
            total_required_cash = gross_cost + buy_fee

        return max_lots, total_shares, total_required_cash

    def execute_buy(self, ticker: str, target_nominal: float, current_price: float, volume_24h_idr: float = float('inf'), volatility_daily: float = 0.02) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self.kill_switch_active:
                logger.error(f"🛑 [BUY BLOCKED] Drawdown Kill Switch Aktif! Pembelian {ticker} ditolak.")
                return None

            ticker = normalize_idx_symbol(ticker)
            vol_idr = _normalize_volume_idr(volume_24h_idr, current_price, self.min_adtv_idr)
            execution_price = self.compute_almgren_chriss_impact_price(current_price, target_nominal, vol_idr, volatility_daily, is_buy=True)
            lots, shares, total_cost = self.calculate_max_buyable_lots(ticker, target_nominal, execution_price, vol_idr)

            if lots <= 0:
                return None

            gross_amount = shares * execution_price
            fee = gross_amount * self.buy_fee_pct
            self.cash_balance -= total_cost
            self.cumulative_turnover_idr += gross_amount

            if ticker in self.positions:
                pos = self.positions[ticker]
                old_shares = pos["shares"]
                old_avg_price = pos["avg_price"]
                old_buy_fee = float(pos.get("buy_fee_total", 0.0) or 0.0)
                new_shares = old_shares + shares
                new_avg_price = ((old_shares * old_avg_price) + gross_amount) / new_shares
                
                pos["lots"] += lots
                pos["shares"] = new_shares
                pos["avg_price"] = new_avg_price
                pos["current_price"] = execution_price
                pos["market_value"] = new_shares * execution_price
                # Accumulate buy fees for correct round-trip realized PnL on close
                pos["buy_fee_total"] = old_buy_fee + fee
            else:
                self.positions[ticker] = {
                    "asset": ticker,
                    "ticker": ticker,
                    "symbol": ticker,
                    "lots": lots,
                    "shares": shares,
                    "avg_price": execution_price,
                    "buy_price": execution_price,
                    "current_price": execution_price,
                    "market_value": shares * execution_price,
                    "buy_fee_total": fee,  # store entry fee for economic RT PnL
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                    "buy_date": datetime.now(WIB_TZ).strftime("%Y-%m-%d")
                }

            trade_log = {
                "timestamp": datetime.now(WIB_TZ).strftime("%Y-%m-%d %H:%M:%S WIB"),
                "action": "BUY",
                "ticker": ticker,
                "symbol": ticker,
                "lots": lots,
                "shares": shares,
                "price": execution_price,
                "gross_amount": gross_amount,
                "fee": fee,
                "buy_fee": fee,
                "total_cost": total_cost,
                "realized_pnl": 0.0,
                "cash_after": self.cash_balance
            }
            self.transaction_history.append(trade_log)
            self.save_state()
            logger.info(f"🟢 [BUY EXECUTED] {ticker}: {lots} Lot (@ Rp {execution_price:,.0f}) | Sisa Kas: Rp {self.cash_balance:,.2f}")
            return trade_log

    def execute_sell(self, ticker: str, lots_to_sell: Optional[int], current_price: float, reason: str = "SIGNAL", volume_24h_idr: float = float('inf'), volatility_daily: float = 0.02) -> Optional[Dict[str, Any]]:
        with self._lock:
            ticker = normalize_idx_symbol(ticker)
            if ticker not in self.positions:
                return None

            pos = self.positions[ticker]
            available_lots = pos["lots"]
            sell_lots = available_lots if (lots_to_sell is None or lots_to_sell >= available_lots) else lots_to_sell
            
            if sell_lots <= 0:
                return None

            vol_idr = _normalize_volume_idr(volume_24h_idr, current_price, self.min_adtv_idr)
            trade_nominal = sell_lots * IDX_LOT_SIZE * current_price
            execution_price = self.compute_almgren_chriss_impact_price(current_price, trade_nominal, vol_idr, volatility_daily, is_buy=False)

            sell_shares = sell_lots * IDX_LOT_SIZE
            gross_amount = sell_shares * execution_price
            sell_fee = gross_amount * self.sell_fee_pct
            net_proceeds = gross_amount - sell_fee

            avg_buy_price = float(pos.get("avg_price", pos.get("buy_price", execution_price)) or execution_price)
            cost_basis = sell_shares * avg_buy_price
            # Allocate entry (buy) fee pro-rata to shares sold for true economic round-trip PnL
            pos_shares = float(pos.get("shares", sell_shares) or sell_shares)
            buy_fee_total = float(pos.get("buy_fee_total", 0.0) or 0.0)
            if buy_fee_total <= 0.0 and avg_buy_price > 0:
                # Legacy positions without stored buy_fee: reconstruct from avg price × qty × buy_fee_pct
                buy_fee_total = pos_shares * avg_buy_price * self.buy_fee_pct
            buy_fee_allocated = buy_fee_total * (sell_shares / pos_shares) if pos_shares > 0 else 0.0
            # Economic RT: net_proceeds - cost_basis - allocated_buy_fee
            realized_pnl = net_proceeds - cost_basis - buy_fee_allocated

            self.cash_balance += net_proceeds
            self.realized_pnl_accumulated += realized_pnl
            self.cumulative_turnover_idr += gross_amount

            remaining_shares = pos["shares"] - sell_shares
            if remaining_shares <= 0:
                del self.positions[ticker]
            else:
                pos["lots"] -= sell_lots
                pos["shares"] = remaining_shares
                pos["market_value"] = remaining_shares * execution_price
                pos["buy_fee_total"] = max(0.0, buy_fee_total - buy_fee_allocated)

            # AUTO-RECONCILIATION KETIKA SELURUH POSISI DIJUAL (POSISI 0 SAHAM)
            if not self.positions:
                self.total_equity = self.cash_balance
                logger.info(f"🧹 [PORTFOLIO CLEANSED] Seluruh posisi ditutup. Kas direkonsiliasi: Rp {self.cash_balance:,.2f}")

            trade_log = {
                "timestamp": datetime.now(WIB_TZ).strftime("%Y-%m-%d %H:%M:%S WIB"),
                "action": f"SELL_{reason.upper()}",
                "ticker": ticker,
                "symbol": ticker,
                "lots": sell_lots,
                "shares": sell_shares,
                "price": execution_price,
                "gross_amount": gross_amount,
                "fee": sell_fee,
                "sell_fee": sell_fee,
                "buy_fee_allocated": buy_fee_allocated,
                "net_proceeds": net_proceeds,
                "realized_pnl": realized_pnl,
                "cash_after": self.cash_balance
            }
            self.transaction_history.append(trade_log)
            self.save_state()
            logger.info(f"🔴 [SELL EXECUTED] {ticker}: {sell_lots} Lot (@ Rp {execution_price:,.0f}) | PnL: Rp {realized_pnl:,.2f}")
            return trade_log

    def update_mark_to_market(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        with self._lock:
            total_market_value = 0.0
            for ticker, pos in list(self.positions.items()):
                norm_ticker = normalize_idx_symbol(ticker)
                close_price = current_prices.get(norm_ticker, current_prices.get(ticker, pos["current_price"]))
                if close_price > 0.0:
                    pos["current_price"] = round_to_idx_tick(close_price, round_mode="nearest")
                
                shares = pos["shares"]
                avg_price = pos.get("avg_price", pos.get("buy_price", pos["current_price"]))
                market_val = shares * pos["current_price"]
                unrealized_pnl = (pos["current_price"] - avg_price) * shares
                unrealized_pnl_pct = ((pos["current_price"] - avg_price) / avg_price) * 100.0 if avg_price > 0 else 0.0
                
                pos["market_value"] = market_val
                pos["unrealized_pnl"] = unrealized_pnl
                pos["unrealized_pnl_pct"] = unrealized_pnl_pct
                total_market_value += market_val

            if not self.positions:
                self.total_equity = self.cash_balance
            else:
                self.total_equity = self.cash_balance + total_market_value
            
            if self.total_equity > self.peak_equity:
                self.peak_equity = self.total_equity
            
            current_drawdown = ((self.peak_equity - self.total_equity) / self.peak_equity) if self.peak_equity > 0 else 0.0
            
            if current_drawdown >= MAX_DRAWDOWN_KILL_SWITCH_PCT and not self.kill_switch_active:
                self.kill_switch_active = True
                logger.critical(f"🚨 [KILL SWITCH TRIGGERED] Drawdown ({current_drawdown*100:.2f}%) melampaui batas {MAX_DRAWDOWN_KILL_SWITCH_PCT*100:.1f}%.")
            elif self.kill_switch_active and current_drawdown < RECOVERY_DRAWDOWN_THRESHOLD_PCT:
                self.kill_switch_active = False
                logger.info(f"✅ [KILL SWITCH RECOVERED] Drawdown pulih ke ({current_drawdown*100:.2f}%). Eksekusi dibuka kembali.")

            self.save_state()

            return {
                "cash_balance": self.cash_balance,
                "portfolio_market_value": total_market_value,
                "total_equity": self.total_equity,
                "peak_equity": self.peak_equity,
                "drawdown_pct": current_drawdown * 100.0,
                "kill_switch_active": self.kill_switch_active,
                "positions_detail": self.positions
            }


# ==============================================================================
# EXACT LEDOIT-WOLF (2004) & RMT MARCENKO-PASTUR DENOISING ENGINE
# ==============================================================================
class CovarianceEngine:
    def __init__(self, config=None):
        self._is_active = False
        self.config = config or {}

    def activate(self): self._is_active = True
    def deactivate(self): self._is_active = False

    def compute_covariance(self, returns_df: pl.DataFrame) -> pl.DataFrame:
        returns_df = _ensure_polars_df(returns_df)
        asset_cols = [c for c in returns_df.columns if c not in ["date", "timestamp", "time", "asset", "ticker"]]
        if not asset_cols:
            raise CovarianceError("Tidak ada kolom aset ditemukan dalam DataFrame returns untuk perhitungan kovarians.")

        X_raw = returns_df.select(asset_cols).to_numpy()
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X = robust_clean_returns(X_raw)
        t, n = X.shape

        if t <= 2 or n == 1:
            cov = np.cov(X, rowvar=False) if n > 1 else np.array([[np.var(X)]])
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            cov = make_positive_semi_definite(cov)
            return pl.DataFrame(cov, schema=asset_cols).insert_column(0, pl.Series("asset", asset_cols))

        X_centered = X - np.mean(X, axis=0)
        S = (X_centered.T @ X_centered) / t

        sample_var = np.diag(S)
        sqrt_var = np.sqrt(np.clip(sample_var, 1e-12, None))
        r_bar = (np.sum(S / np.outer(sqrt_var, sqrt_var)) - n) / max(n * (n - 1), 1)
        F = r_bar * np.outer(sqrt_var, sqrt_var)
        np.fill_diagonal(F, sample_var)

        y = X_centered ** 2
        pi_mat = (y.T @ y) / t - S ** 2
        pi = np.sum(pi_mat)

        r_mat = np.outer(sqrt_var, 1.0 / sqrt_var)
        term1 = (X_centered ** 3).T @ X_centered / t
        term2 = np.outer(sample_var, np.ones(n)) * S
        theta_mat = term1 - term2
        rho = np.sum(np.diag(pi_mat)) + r_bar * np.sum((1.0 / np.maximum(r_mat, 1e-12)) * theta_mat)

        gamma = np.sum((S - F) ** 2)

        kappa = (pi - rho) / t
        delta = max(0.0, min(1.0, kappa / max(gamma, 1e-12)))

        shrunk_cov = (1.0 - delta) * S + delta * F
        denoised_cov = denoise_covariance_marcenko_pastur(shrunk_cov, T=t, N=n)

        return pl.DataFrame(denoised_cov, schema=asset_cols).insert_column(0, pl.Series("asset", asset_cols))


class CorrelationEngine:
    def __init__(self, config=None): self._is_active = False
    def activate(self): self._is_active = True
    def deactivate(self): self._is_active = False

    def compute_correlation(self, cov_df: pl.DataFrame):
        asset_names = [c for c in cov_df.columns if c != "asset"]
        cov = cov_df.select(asset_names).to_numpy()
        
        std_devs = np.sqrt(np.clip(np.diag(cov), 1.0e-12, None))
        inv_std = 1.0 / std_devs
        corr = cov * np.outer(inv_std, inv_std)
        np.clip(corr, -1.0, 1.0, out=corr)
        
        dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
        return pl.DataFrame(corr, schema=asset_names).insert_column(0, pl.Series("asset", asset_names)), \
               pl.DataFrame(dist, schema=asset_names).insert_column(0, pl.Series("asset", asset_names))


# ==============================================================================
# ML EXPECTED RETURN & CORRELATED OMEGA BLACK-LITTERMAN ENGINE
# ==============================================================================
class BlackLittermanEngine:
    @classmethod
    def compute_posterior_returns(
        cls,
        cov: np.ndarray,
        market_weights: np.ndarray,
        views_Q: np.ndarray,
        P_pick: np.ndarray,
        confidence_vector: Optional[np.ndarray] = None,
        tau: float = BLACK_LITTERMAN_TAU
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = cov.shape[0]
        if market_weights is None or len(market_weights) != n:
            market_weights = np.ones(n) / n

        pi_equil = RISK_AVERSION_GAMMA * (cov @ market_weights)

        if views_Q is None or len(views_Q) == 0 or P_pick is None or P_pick.shape[0] == 0:
            return pi_equil, cov

        k = P_pick.shape[0]
        if confidence_vector is None or len(confidence_vector) != k:
            confidence_vector = np.ones(k) * 0.5

        tau_cov = tau * cov
        p_tau_cov_p = P_pick @ tau_cov @ P_pick.T
        
        conf_scale = np.diag((1.0 - confidence_vector) / np.maximum(confidence_vector, 1e-4))
        omega = p_tau_cov_p @ conf_scale
        omega_safe = make_positive_semi_definite(omega, epsilon=1e-8)
        omega_inv = np.linalg.pinv(omega_safe)

        tau_cov_safe = tau_cov + (1e-8 * np.eye(n))
        tau_cov_inv = np.linalg.pinv(tau_cov_safe)

        post_cov_part = np.linalg.pinv(tau_cov_inv + P_pick.T @ omega_inv @ P_pick)
        mu_bl = post_cov_part @ (tau_cov_inv @ pi_equil + P_pick.T @ omega_inv @ views_Q)
        sigma_bl = cov + post_cov_part

        return mu_bl, make_positive_semi_definite(sigma_bl)


# ==============================================================================
# HRP WITH LÓPEZ DE PRADO QUASI-DIAGONALIZATION REORDERING
# ==============================================================================
class HierarchicalRiskParity:
    def __init__(self, config=None): self._is_active = False
    def activate(self): self._is_active = True
    def deactivate(self): self._is_active = False

    def _get_cluster_var(self, cov: np.ndarray, cluster: List[int]) -> float:
        sub_cov = cov[np.ix_(cluster, cluster)]
        if sub_cov.size == 0:
            return 1.0
        inv_diag = 1.0 / np.clip(np.diag(sub_cov), 1e-12, None)
        weights = inv_diag / np.sum(inv_diag)
        return float(weights.T @ sub_cov @ weights)

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        link = link.astype(int)
        sort_ix = [link[-1, 0], link[-1, 1]]
        num_items = link[-1, 3]

        while max(sort_ix) >= num_items:
            id_list = []
            for item in sort_ix:
                if item >= num_items:
                    idx = item - num_items
                    id_list.append(link[idx, 0])
                    id_list.append(link[idx, 1])
                else:
                    id_list.append(item)
            sort_ix = id_list
        return sort_ix

    def _rec_bisection(self, cov_reordered: np.ndarray, sort_ix: List[int]) -> np.ndarray:
        w = np.ones(len(sort_ix), dtype=float)
        clusters = [list(range(len(sort_ix)))]

        while len(clusters) > 0:
            next_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    half = len(cluster) // 2
                    c1 = cluster[:half]
                    c2 = cluster[half:]

                    v1 = self._get_cluster_var(cov_reordered, c1)
                    v2 = self._get_cluster_var(cov_reordered, c2)

                    alpha = 1.0 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
                    w[c1] *= alpha
                    w[c2] *= (1.0 - alpha)

                    if len(c1) > 1:
                        next_clusters.append(c1)
                    if len(c2) > 1:
                        next_clusters.append(c2)
            clusters = next_clusters
        return w

    def allocate(self, cov_df: pl.DataFrame, corr_df: pl.DataFrame, dist_df: pl.DataFrame, signals_df: Union[pl.DataFrame, List[Dict[str, Any]]]) -> pl.DataFrame:
        signals_df = _ensure_polars_df(signals_df)
        asset_names = [c for c in cov_df.columns if c != "asset"]
        cov = cov_df.select(asset_names).to_numpy()
        dist = dist_df.select(asset_names).to_numpy()

        n_cov = len(asset_names)
        n_sig = signals_df.height

        if n_cov <= 1 or n_sig <= 1:
            w = np.ones(n_sig, dtype=np.float64) / max(n_sig, 1)
            return signals_df.with_columns([
                pl.Series("hrp_weight", w, dtype=pl.Float64),
                pl.Series("portfolio_weight", w, dtype=pl.Float64)
            ])

        try:
            dist_sym = (dist + dist.T) / 2.0
            np.fill_diagonal(dist_sym, 0.0)

            cond_dist = squareform(dist_sym, checks=False)
            link = sch.linkage(cond_dist, method="single")
            
            sort_ix = self._get_quasi_diag(link)
            cov_reordered = cov[np.ix_(sort_ix, sort_ix)]

            raw_weights_reordered = self._rec_bisection(cov_reordered, sort_ix)
            
            weight_map = {asset_names[sort_ix[i]]: raw_weights_reordered[i] for i in range(n_cov)}
            w_list = []
            for r in signals_df.to_dicts():
                sym = _extract_symbol(r)
                w_list.append(weight_map.get(sym, 1.0 / max(n_cov, 1)))
            
            w_arr = np.array(w_list, dtype=np.float64)
            w_arr = np.clip(w_arr, 0.0, IDX_MAX_SINGLE_STOCK_WEIGHT)
            sum_w = np.sum(w_arr)
            if sum_w > 0:
                w_arr /= sum_w
            else:
                w_arr = np.ones(n_sig, dtype=np.float64) / n_sig

            return signals_df.with_columns([
                pl.Series("hrp_weight", w_arr, dtype=pl.Float64),
                pl.Series("portfolio_weight", w_arr, dtype=pl.Float64)
            ])
        except Exception as e:
            logger.warning(f"⚠️ HRP Fallback ke equal weight karena error clustering: {e}")
            w = np.ones(n_sig, dtype=np.float64) / max(n_sig, 1)
            return signals_df.with_columns([
                pl.Series("hrp_weight", w, dtype=pl.Float64),
                pl.Series("portfolio_weight", w, dtype=pl.Float64)
            ])


# ==============================================================================
# MEAN-VARIANCE SLSQP OPTIMIZER WITH ALMGREN-CHRISS IMPACT & LOT ROUNDING
# ==============================================================================
class PortfolioOptimizer:
    def __init__(self, config=None, default_capital=DEFAULT_INITIAL_CAPITAL_IDR):
        self._is_active = False
        self._capital = default_capital
    def activate(self): self._is_active = True
    def deactivate(self): self._is_active = False

    def round_weights_to_idx_lots(self, weights: np.ndarray, prices: np.ndarray, total_equity: float) -> Tuple[np.ndarray, np.ndarray]:
        n = len(weights)
        lots = np.zeros(n, dtype=int)
        
        for i in range(n):
            if weights[i] <= 0 or prices[i] < IDX_MIN_PRICE_IDR:
                continue
            target_nominal = weights[i] * total_equity
            cost_per_lot = IDX_LOT_SIZE * prices[i] * (1.0 + IDX_BUY_FEE_PCT)
            lots[i] = int(math.floor(target_nominal / cost_per_lot))
        
        actual_nominals = lots * IDX_LOT_SIZE * prices
        actual_weights = actual_nominals / max(total_equity, 1.0)
        return actual_weights, lots

    def optimize_allocation(
        self,
        hrp_df: pl.DataFrame,
        cov_clean_df: pl.DataFrame,
        current_weights: Optional[np.ndarray] = None,
        sector_series: Optional[List[str]] = None,
        market_regime: str = "SIDEWAYS"
    ) -> pl.DataFrame:
        hrp_df = _ensure_polars_df(hrp_df)
        asset_cols = [c for c in cov_clean_df.columns if c != "asset"]
        cov_clean = cov_clean_df.select(asset_cols).to_numpy()
        cov_clean = np.nan_to_num(cov_clean, nan=1e-6, posinf=1e-6, neginf=1e-6)
        n = cov_clean.shape[0]
        n_sig = hrp_df.height

        w0 = hrp_df["hrp_weight"].to_numpy() if "hrp_weight" in hrp_df.columns else np.ones(n, dtype=np.float64) / max(n, 1)

        if n <= 1 or n_sig <= 1:
            w_opt = np.ones(n_sig, dtype=np.float64)
            return hrp_df.with_columns([
                pl.Series("optimized_weight", w_opt, dtype=pl.Float64),
                pl.Series("capital_allocation", w_opt * self._capital, dtype=pl.Float64)
            ])

        w_prev = current_weights if (current_weights is not None and len(current_weights) == n) else w0
        w_prev = np.nan_to_num(w_prev, nan=0.0)

        raw_views_list = []
        conf_list = []
        prices_list = []

        for r in hrp_df.to_dicts():
            exp_ret = _extract_float(r, ["expected_return", "predicted_return", "alpha_forecast"], default=0.0)
            if exp_ret == 0.0:
                prob = _extract_float(r, ["probability", "prediction_probability", "confidence"], default=0.50)
                rrr = _extract_float(r, ["risk_reward_ratio", "rrr"], default=1.5)
                exp_ret = (prob - 0.50) * 0.05 * rrr
            
            raw_views_list.append(exp_ret)
            conf_list.append(_extract_float(r, ["prediction_confidence", "confidence"], default=0.50))
            prices_list.append(_extract_float(r, ["entry_price", "close", "price", "last_price"], default=1000.0))

        mu_views = np.array(raw_views_list, dtype=np.float64)
        conf_vec = np.array(conf_list, dtype=np.float64)
        prices_vec = np.array(prices_list, dtype=np.float64)

        mu_bl, _ = BlackLittermanEngine.compute_posterior_returns(cov_clean, w_prev, mu_views, np.eye(n), confidence_vector=conf_vec)
        mu_bl = np.nan_to_num(mu_bl, nan=0.0)

        def objective(w):
            port_var = float(w.T @ cov_clean @ w)
            port_return = float(w.T @ mu_bl)
            
            delta_w = w - w_prev
            linear_fee_cost = IDX_BUY_FEE_PCT * np.sum(np.abs(delta_w))
            almgren_chriss_impact = ALMGREN_CHRISS_GAMMA * np.sum(delta_w ** 2)
            quadratic_turnover = QUADRATIC_TURNOVER_LAMBDA * np.sum(delta_w ** 2)

            return (0.5 * RISK_AVERSION_GAMMA * port_var) - port_return + linear_fee_cost + almgren_chriss_impact + quadratic_turnover

        bounds = [(0.0, IDX_MAX_SINGLE_STOCK_WEIGHT) for _ in range(n)]
        
        if market_regime == "BULL":
            target_annual_vol = 0.18
        elif market_regime in ["CRASH", "BEAR"]:
            target_annual_vol = 0.07
        else:
            target_annual_vol = 0.12

        max_daily_var = (target_annual_vol ** 2) / 252.0
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: max_daily_var - float(w.T @ cov_clean @ w)}
        ]

        if sector_series and len(sector_series) == n:
            unique_sectors = set(sector_series)
            for sec in unique_sectors:
                sec_indices = [i for i, s in enumerate(sector_series) if s == sec]
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, idxs=sec_indices: IDX_MAX_SECTOR_WEIGHT - np.sum(w[idxs])
                })

        try:
            res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 250, 'ftol': 1e-8})
            w_opt = res.x if res.success else w0
        except Exception as e:
            logger.warning(f"⚠️ SLSQP Optimizer error, menggunakan bobot HRP: {e}")
            w_opt = w0

        w_opt = np.clip(w_opt, 0.0, IDX_MAX_SINGLE_STOCK_WEIGHT)
        if np.sum(w_opt) > 0:
            w_opt /= np.sum(w_opt)

        w_feasible, lot_allocations = self.round_weights_to_idx_lots(w_opt, prices_vec, self._capital)

        return hrp_df.with_columns([
            pl.Series("optimized_weight", w_feasible, dtype=pl.Float64),
            pl.Series("capital_allocation", w_feasible * self._capital, dtype=pl.Float64),
            pl.Series("allocated_lots", lot_allocations, dtype=pl.Int64)
        ])


class DiversificationMeasurer:
    def __init__(self, config=None): self._is_active = False
    def activate(self): self._is_active = True
    def deactivate(self): self._is_active = False

    def measure_diversification(self, exp_df: pl.DataFrame, cov_matrix: pl.DataFrame) -> pl.DataFrame:
        exp_df = _ensure_polars_df(exp_df)
        try:
            asset_cols = [c for c in cov_matrix.columns if c != "asset"]
            cov = cov_matrix.select(asset_cols).to_numpy()
            w = exp_df["optimized_weight"].to_numpy() if "optimized_weight" in exp_df.columns else np.ones(len(asset_cols)) / len(asset_cols)
            
            port_var = float(w.T @ cov @ w)
            port_vol = math.sqrt(max(port_var, 1e-12))

            mrc = (cov @ w) / port_vol
            trc = w * mrc
            prc = trc / port_vol

            evals = np.linalg.eigvalsh(cov)
            evals = np.clip(evals, 1.0e-12, None)
            p_evals = evals / np.sum(evals)
            enb = float(np.exp(-np.sum(p_evals * np.log(p_evals))))
        except Exception:
            enb = float(exp_df.height)
            trc = np.ones(exp_df.height, dtype=np.float64) / max(exp_df.height, 1)
            prc = trc

        return exp_df.with_columns([
            pl.lit(enb, dtype=pl.Float64).alias("effective_assets"),
            pl.Series("total_risk_contribution", trc, dtype=pl.Float64),
            pl.Series("percent_risk_contribution", prc, dtype=pl.Float64)
        ])


class ExposureController:
    def __init__(self, config=None, default_capital=DEFAULT_INITIAL_CAPITAL_IDR): self._is_active = False
    def activate(self): self._is_active = True
    def deactivate(self): self._is_active = False

    def control_exposure(self, opt_df: pl.DataFrame) -> pl.DataFrame:
        opt_df = _ensure_polars_df(opt_df)
        w = opt_df["optimized_weight"].to_numpy() if "optimized_weight" in opt_df.columns else np.zeros(opt_df.height)
        gross_exp = float(np.sum(np.abs(w)))
        net_exp = float(np.sum(w))
        max_asset_exp = float(np.max(w)) if len(w) > 0 else 0.0

        exposure_pass = (gross_exp <= MAX_PORTFOLIO_LEVERAGE) and (max_asset_exp <= IDX_MAX_SINGLE_STOCK_WEIGHT + 1e-4)

        return opt_df.with_columns([
            pl.lit(gross_exp, dtype=pl.Float64).alias("gross_exposure"),
            pl.lit(net_exp, dtype=pl.Float64).alias("net_exposure"),
            pl.lit(max_asset_exp, dtype=pl.Float64).alias("max_single_exposure"),
            pl.lit(exposure_pass, dtype=pl.Boolean).alias("exposure_pass")
        ])


class SectorConstraintsAuditor:
    def __init__(self, config=None): self._is_active = False
    def activate(self): self._is_active = True
    def deactivate(self): self._is_active = False

    def audit_sector_limits(self, metrics_df: pl.DataFrame, asset_metadata_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        metrics_df = _ensure_polars_df(metrics_df)
        sector_map = {}
        if asset_metadata_df is not None and "asset" in asset_metadata_df.columns and "sector" in asset_metadata_df.columns:
            for row in asset_metadata_df.to_dicts():
                sector_map[normalize_idx_symbol(row["asset"])] = str(row["sector"]).upper()

        records = metrics_df.to_dicts()
        mapped_sectors = []
        for r in records:
            existing_sec = r.get("sector")
            if existing_sec and str(existing_sec).strip():
                mapped_sectors.append(str(existing_sec).upper())
            else:
                sym = _extract_symbol(r)
                mapped_sectors.append(sector_map.get(sym, "FINANCIALS"))

        return metrics_df.with_columns([
            pl.Series("sector", mapped_sectors),
            pl.lit(True, dtype=pl.Boolean).alias("sector_pass")
        ])


# ==============================================================================
# ADVANCED INSTITUTIONAL RISK METRICS ENGINE
# ==============================================================================
class InstitutionalRiskMetricsEngine:
    @classmethod
    def compute_risk_metrics(
        cls,
        weights: np.ndarray,
        cov: np.ndarray,
        returns_history: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        n = len(weights)
        if n == 0 or cov.shape[0] != n:
            return {}

        port_var = float(weights.T @ cov @ weights)
        port_vol_daily = math.sqrt(max(port_var, 1e-12))
        port_vol_annual = port_vol_daily * math.sqrt(252.0)

        mvar = (cov @ weights) / port_vol_daily
        cvar = weights * mvar
        cvar_pct = cvar / port_vol_daily

        if returns_history is not None and returns_history.shape[0] > 10:
            port_historical_returns = returns_history @ weights
            var_95 = np.percentile(port_historical_returns, 5)
            es_95 = float(np.mean(port_historical_returns[port_historical_returns <= var_95]))
            
            cum_returns = np.cumsum(port_historical_returns)
            running_peak = np.maximum.accumulate(cum_returns)
            drawdowns = cum_returns - running_peak
            cdar_95 = float(np.mean(drawdowns[drawdowns <= np.percentile(drawdowns, 5)]))
        else:
            es_95 = -1.645 * port_vol_daily
            cdar_95 = -2.0 * port_vol_daily

        tracking_error = 0.05
        info_ratio = 0.0
        portfolio_beta = 1.0

        if benchmark_returns is not None and len(benchmark_returns) == len(returns_history):
            excess_returns = (returns_history @ weights) - benchmark_returns
            tracking_error = float(np.std(excess_returns) * math.sqrt(252.0))
            mean_excess = float(np.mean(excess_returns) * 252.0)
            info_ratio = mean_excess / max(tracking_error, 1e-6)

            bench_var = float(np.var(benchmark_returns))
            if bench_var > 0:
                cov_bench = float(np.cov(returns_history @ weights, benchmark_returns)[0, 1])
                portfolio_beta = cov_bench / bench_var

        return {
            "portfolio_volatility_annual": port_vol_annual,
            "expected_shortfall_es95": es_95,
            "conditional_drawdown_cdar95": cdar_95,
            "tracking_error_annual": tracking_error,
            "information_ratio": info_ratio,
            "portfolio_beta_ihsg": portfolio_beta,
            "max_component_var_pct": float(np.max(cvar_pct)) if len(cvar_pct) > 0 else 0.0
        }


# ==============================================================================
# SECTORAL & MACRO STRESS TESTING ENGINE
# ==============================================================================
class StressTestEngine:
    @classmethod
    def run_stress_test(cls, portfolio_equity: float, weights: np.ndarray, cov: np.ndarray, sectors: Optional[List[str]] = None) -> Dict[str, Any]:
        n = len(weights)
        if n == 0 or cov.shape[0] != n:
            return {}

        results = {}
        port_var = float(weights.T @ cov @ weights)
        port_vol_annual = math.sqrt(max(port_var, 1e-12)) * math.sqrt(252.0)

        std_devs = np.sqrt(np.diag(cov))
        rho_panic = 0.85
        panic_corr = (1.0 - rho_panic) * np.eye(n) + rho_panic * np.ones((n, n))
        panic_cov = np.outer(std_devs, std_devs) * panic_corr
        
        panic_var = float(weights.T @ panic_cov @ weights)
        panic_vol_annual = math.sqrt(max(panic_var, 1e-12)) * math.sqrt(252.0)

        sector_shocks = {
            "2008_GLOBAL_FINANCIAL_CRISIS": {"FINANCIALS": -0.35, "CONSUMER": -0.15, "MINING": -0.40, "OTHER": -0.25},
            "2020_COVID19_CRASH": {"FINANCIALS": -0.28, "CONSUMER": -0.10, "MINING": -0.20, "OTHER": -0.22},
            "IDX_SECTORAL_COMMODITY_CRASH": {"FINANCIALS": -0.05, "CONSUMER": -0.02, "MINING": -0.30, "OTHER": -0.10}
        }

        sec_list = sectors if (sectors and len(sectors) == n) else ["FINANCIALS"] * n

        for scenario_name, shocks in sector_shocks.items():
            portfolio_shock = 0.0
            for i in range(n):
                sec = sec_list[i].upper()
                s_factor = shocks.get(sec, shocks.get("OTHER", -0.20))
                portfolio_shock += weights[i] * s_factor

            simulated_loss_idr = portfolio_equity * abs(portfolio_shock)
            post_shock_equity = portfolio_equity * (1.0 + portfolio_shock)

            results[scenario_name] = {
                "asymmetric_equity_shock_pct": portfolio_shock * 100.0,
                "projected_loss_idr": simulated_loss_idr,
                "post_shock_equity_idr": post_shock_equity,
                "normal_annual_volatility_pct": port_vol_annual * 100.0,
                "panic_covariance_volatility_pct": panic_vol_annual * 100.0
            }

        return results


# ==============================================================================
# UNIFIED PORTFOLIO ENGINE (FACADE CLASS WITH GEMINI AI ADVISORY LAYER)
# ==============================================================================
class UnifiedPortfolioEngine:
    FACADE_VERSION: str = "2026.Q3.v23.2"

    def __init__(self, config: Optional[Dict[str, Any]] = None, state_file: Optional[str] = None, gemini_api_key: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._raw_config = dict(config) if config is not None else {}
        self.state_file = state_file or DEFAULT_STATE_FILENAME
        self.default_capital = DEFAULT_INITIAL_CAPITAL_IDR

        self.cov_engine = CovarianceEngine(self._raw_config.get("covariance", {}))
        self.corr_engine = CorrelationEngine(self._raw_config.get("correlation", {}))
        self.hrp_engine = HierarchicalRiskParity(self._raw_config.get("hrp", {}))
        self.optimizer_engine = PortfolioOptimizer(config=self._raw_config.get("optimizer", {}), default_capital=self.default_capital)
        self.diversification_engine = DiversificationMeasurer(self._raw_config.get("diversification", {}))
        self.exposure_engine = ExposureController(config=self._raw_config.get("exposure", {}), default_capital=self.default_capital)
        self.sector_engine = SectorConstraintsAuditor(self._raw_config.get("sector", {}))
        self.gemini_engine = GeminiPortfolioDiagnosticEngine(api_key=gemini_api_key)

        self.execution_engine = IDXExecutionEngine(
            state_file=self.state_file,
            initial_capital=self.default_capital,
            max_stock_weight=IDX_MAX_SINGLE_STOCK_WEIGHT
        )

        self.activate()

    def activate(self) -> None:
        with self._lock:
            self.cov_engine.activate()
            self.corr_engine.activate()
            self.hrp_engine.activate()
            self.optimizer_engine.activate()
            self.diversification_engine.activate()
            self.exposure_engine.activate()
            self.sector_engine.activate()

    def deactivate(self) -> None:
        with self._lock:
            self.cov_engine.deactivate()
            self.corr_engine.deactivate()
            self.hrp_engine.deactivate()
            self.optimizer_engine.deactivate()
            self.diversification_engine.deactivate()
            self.exposure_engine.deactivate()
            self.sector_engine.deactivate()

    def load_portfolio_state(self) -> Dict[str, Any]:
        return self.execution_engine.load_state()

    def save_portfolio_state(self, portfolio_data: Optional[Dict[str, Any]] = None) -> bool:
        return self.execution_engine.save_state()

    def update_market_valuation(self, latest_prices: Dict[str, float]) -> Dict[str, Any]:
        return self.execution_engine.update_mark_to_market(latest_prices)

    @property
    def total_equity(self) -> float:
        return self.execution_engine.total_equity

    @property
    def available_cash(self) -> float:
        return self.execution_engine.cash_balance

    @property
    def positions(self) -> Dict[str, Dict[str, Any]]:
        return self.execution_engine.positions

    @property
    def max_position_pct(self) -> float:
        return IDX_MAX_SINGLE_STOCK_WEIGHT

    def process_trading_signals(
        self,
        signals_df: Union[pl.DataFrame, List[Dict[str, Any]]],
        latest_prices: Dict[str, float],
        top_n: int = 1
    ) -> Union[pl.DataFrame, Dict[str, Any]]:
        """
        Menerima sinyal kuantitatif dan membatasi eksekusi pembelian HANYA pada 
        Top-1 Saham Terbaik (Restricted Top-1 Execution Mode).
        """
        with self._lock:
            self.execution_engine.reset_warning_cache()
            self.update_market_valuation(latest_prices)

            is_polars = isinstance(signals_df, pl.DataFrame)
            records = signals_df.to_dicts() if is_polars else (list(signals_df) if isinstance(signals_df, list) else [])

            if not records:
                return signals_df if is_polars else {"executed_buys": 0, "executed_sells": 0}

            executed_buys = 0
            executed_sells = 0

            # ------------------------------------------------------------------
            # STEP 1: PISAHKAN SINYAL SELL DAN BUY
            # ------------------------------------------------------------------
            sell_records = []
            buy_records = []

            for r in records:
                action = _extract_action(r)
                if action == "SELL":
                    sell_records.append(r)
                else:
                    buy_records.append(r)

            # ------------------------------------------------------------------
            # STEP 2: PROSES SELURUH SINYAL SELL (PELEPASAN KAS)
            # ------------------------------------------------------------------
            for r in sell_records:
                sym = _extract_symbol(r)
                if not sym or sym not in self.positions:
                    continue
                
                norm_ticker = normalize_idx_symbol(sym)
                current_price = latest_prices.get(norm_ticker, latest_prices.get(sym, self.positions[sym]["current_price"]))
                vol_24h = _extract_float(r, ["volume_24h", "volume", "adtv", "volume_idr"], default=float('inf'))
                
                sell_res = self.execution_engine.execute_sell(
                    norm_ticker, 
                    lots_to_sell=None, 
                    current_price=current_price, 
                    reason="SIGNAL_SELL", 
                    volume_24h_idr=vol_24h
                )
                if sell_res:
                    executed_sells += 1

            # ------------------------------------------------------------------
            # STEP 3: URUTKAN SINYAL BUY DAN BATASI KETAT HANYA UNTUK TOP-1
            # ------------------------------------------------------------------
            buy_records.sort(
                key=lambda x: _extract_float(
                    x, 
                    ["ranking_score", "score", "signal_rank_score"], 
                    default=_extract_float(x, ["probability"], 0.5) * _extract_float(x, ["confidence"], 0.5)
                ),
                reverse=True
            )

            top_buy_records = buy_records[:top_n]
            
            volume_map: Dict[str, float] = {}
            target_weights_map: Dict[str, float] = {}

            for r in top_buy_records:
                sym = _extract_symbol(r)
                raw_price = _extract_float(r, ["entry_price", "close", "price", "last_price"], default=1.0)
                raw_vol = _extract_float(r, ["volume_24h", "volume", "adtv", "volume_idr"], default=float('inf'))
                vol_idr = _normalize_volume_idr(raw_vol, raw_price, IDX_MIN_ADTV_IDR)

                if sym:
                    volume_map[sym] = vol_idr
                    weight_raw = _extract_float(r, ["weight", "optimized_weight", "target_weight"], default=self.max_position_pct)
                    target_weight = weight_raw / 100.0 if weight_raw > 1.0 else weight_raw
                    target_weights_map[sym] = min(max(target_weight, 0.0), self.max_position_pct)

            # ------------------------------------------------------------------
            # STEP 4: EKSEKUSI PEMBELIAN TOP-1 SAHAM
            # ------------------------------------------------------------------
            processed_records = []
            for r in records:
                norm_sym = _extract_symbol(r)
                action = _extract_action(r)

                if not norm_sym or action == "SELL":
                    r["optimized_weight"] = 0.0
                    r["capital_allocation"] = 0.0
                    processed_records.append(r)
                    continue

                if norm_sym in target_weights_map:
                    target_weight = target_weights_map[norm_sym]
                    current_price = _extract_float(r, ["entry_price", "close", "price", "last_price"], default=0.0)
                    if norm_sym in latest_prices and latest_prices[norm_sym] > 0:
                        current_price = float(latest_prices[norm_sym])

                    vol_24h = volume_map.get(norm_sym, float('inf'))
                    target_nominal = self.total_equity * target_weight

                    if current_price > 0:
                        # Horizon-aware position key compatible with normalize_idx_symbol:
                        # e.g. BBCA_SHORT.JK / BBCA_MEDIUM.JK / BBCA_LONG.JK
                        horizon_new = str(r.get("horizon") or r.get("expected_holding_days") or "SHORT").upper()
                        if horizon_new not in ("SHORT", "MEDIUM", "LONG"):
                            try:
                                hd = float(horizon_new)
                                horizon_new = "SHORT" if hd <= 2 else ("MEDIUM" if hd <= 10 else "LONG")
                            except Exception:
                                horizon_new = "SHORT"
                        base_sym = normalize_idx_symbol(norm_sym)
                        bare = base_sym[:-3] if base_sym.endswith(".JK") else base_sym
                        pos_key = f"{bare}_{horizon_new}.JK"

                        # Duplicate protection: same symbol + same horizon only
                        existing = self.positions.get(pos_key)
                        if existing is None:
                            legacy = self.positions.get(base_sym) or self.positions.get(norm_sym)
                            if legacy and str(legacy.get("horizon", "SHORT")).upper() == horizon_new and str(legacy.get("status", "ACTIVE")).upper() in ("ACTIVE", "OPEN"):
                                existing = legacy
                        if existing and str(existing.get("status", "ACTIVE")).upper() in ("ACTIVE", "OPEN"):
                            logger.info(f"🛡️ [DUP_SKIP] {pos_key} already ACTIVE — ignore duplicate BUY")
                            r["optimized_weight"] = 0.0
                            r["capital_allocation"] = 0.0
                            r["duplicate_rejected"] = True
                            processed_records.append(r)
                            continue

                        # ---- Signal-contract gate BEFORE any simulated BUY (fail-closed) ----
                        tp = _extract_float(r, ["tp_price", "take_profit", "optimized_take_profit"], default=0.0)
                        sl = _extract_float(r, ["sl_price", "stop_loss", "optimized_stop_loss"], default=0.0)
                        conf = _extract_float(
                            r,
                            ["confidence", "signal_confidence", "prediction_probability", "probability"],
                            default=float("nan"),
                        )
                        rrr = _extract_float(
                            r,
                            ["risk_reward_ratio", "optimized_risk_reward", "calculated_risk_reward", "realized_risk_reward_ratio"],
                            default=0.0,
                        )
                        reason_txt = str(
                            r.get("reason")
                            or r.get("signal_explanation")
                            or r.get("signal_explanation_text")
                            or r.get("final_validator_reason")
                            or ""
                        ).strip()
                        reject_reason = None
                        if conf != conf or conf <= 0.0 or conf > 1.0:
                            reject_reason = f"invalid confidence={conf}"
                        elif tp <= 0 or sl <= 0 or current_price <= 0:
                            reject_reason = f"invalid geometry entry={current_price} tp={tp} sl={sl}"
                        elif not (sl < current_price < tp):
                            # BUY geometry: SL < entry < TP (conservative)
                            reject_reason = f"invalid TP/SL order entry={current_price} tp={tp} sl={sl}"
                        elif rrr != rrr or rrr < 0.5:
                            reject_reason = f"invalid RRR={rrr}"
                        elif not reason_txt:
                            reject_reason = "missing reason"
                        elif horizon_new not in ("SHORT", "MEDIUM", "LONG"):
                            reject_reason = f"missing/invalid horizon={horizon_new}"

                        if reject_reason is not None:
                            logger.warning(f"🛡️ [PORTFOLIO_REJECT] {pos_key} {reject_reason}")
                            r["optimized_weight"] = 0.0
                            r["capital_allocation"] = 0.0
                            r["portfolio_rejected"] = True
                            r["reject_reason"] = reject_reason
                            processed_records.append(r)
                            continue

                        success = self.execution_engine.execute_buy(
                            pos_key,
                            target_nominal=target_nominal,
                            current_price=current_price,
                            volume_24h_idr=vol_24h
                        )
                        if success:
                            executed_buys += 1
                            pos = self.positions.get(pos_key)
                            if pos is not None:
                                pos["status"] = "ACTIVE"
                                pos["side"] = "BUY"
                                pos["base_symbol"] = base_sym
                                pos["tp_price"] = tp
                                pos["take_profit"] = tp
                                pos["sl_price"] = sl
                                pos["stop_loss"] = sl
                                pos["confidence"] = conf
                                pos["risk_reward_ratio"] = rrr
                                pos["reason"] = reason_txt or "SIMULATED_BUY"
                                pos["horizon"] = horizon_new
                                pos["entry_timestamp"] = str(r.get("timestamp") or r.get("date") or pos.get("buy_date") or "")
                                pos["expected_return"] = _extract_float(r, ["expected_return", "calculated_expected_value_pct"], default=0.0)
                            logger.info(f"🏆 [SIMULATED BUY ACTIVE] {pos_key} | positions={len(self.positions)}")

                    r["optimized_weight"] = target_weight
                    r["capital_allocation"] = float(self.total_equity * target_weight)
                else:
                    r["optimized_weight"] = 0.0
                    r["capital_allocation"] = 0.0

                processed_records.append(r)

            self.update_market_valuation(latest_prices)
            self.save_portfolio_state()

            if is_polars:
                return pl.DataFrame(processed_records)
            
            return {
                "executed_buys": executed_buys,
                "executed_sells": executed_sells,
                "available_cash": self.available_cash,
                "total_equity": self.total_equity,
                "open_positions": len(self.positions),
            }

    def execute_pipeline(
        self,
        returns_df: pl.DataFrame,
        signals_df: Union[pl.DataFrame, List[Dict[str, Any]]],
        asset_metadata_df: Optional[pl.DataFrame] = None,
        market_regime: str = "SIDEWAYS",
        **kwargs
    ) -> pl.DataFrame:
        with self._lock:
            signals_df = _ensure_polars_df(signals_df)
            returns_df = _ensure_polars_df(returns_df)

            if signals_df.is_empty():
                logger.warning("⚠️ [PORTFOLIO PIPELINE] signals_df kosong. Mengembalikan DataFrame kosong.")
                return signals_df

            signal_records = signals_df.to_dicts()
            active_signal_tickers = [_extract_symbol(r) for r in signal_records if _extract_symbol(r)]
            
            available_return_cols = [c for c in returns_df.columns if c not in ["date", "timestamp", "time", "asset", "ticker"]]
            matching_tickers = [t for t in active_signal_tickers if t in available_return_cols]

            if matching_tickers:
                returns_subset = returns_df.select(["date"] + matching_tickers) if "date" in returns_df.columns else returns_df.select(matching_tickers)
            else:
                returns_subset = returns_df

            cov_df = self.cov_engine.compute_covariance(returns_subset)
            corr_df, dist_df = self.corr_engine.compute_correlation(cov_df)
            hrp_df = self.hrp_engine.allocate(cov_df, corr_df, dist_df, signals_df)
            
            sectors_list = None
            if asset_metadata_df is not None and "sector" in asset_metadata_df.columns:
                sectors_list = asset_metadata_df["sector"].to_list()

            opt_df = self.optimizer_engine.optimize_allocation(
                hrp_df, cov_df, sector_series=sectors_list, market_regime=market_regime
            )
            div_df = self.diversification_engine.measure_diversification(opt_df, cov_df)
            exp_df = self.exposure_engine.control_exposure(div_df)

            if asset_metadata_df is None:
                assets = signals_df["asset"].to_list() if "asset" in signals_df.columns else (signals_df["ticker"].to_list() if "ticker" in signals_df.columns else active_signal_tickers)
                asset_metadata_df = pl.DataFrame({"asset": assets, "sector": ["FINANCIALS"] * len(assets), "industry": ["BANKS"] * len(assets), "country": ["INDONESIA"] * len(assets)})

            return self.sector_engine.audit_sector_limits(exp_df, asset_metadata_df)

    def run_portfolio_stress_test(self, returns_df: pl.DataFrame, asset_metadata_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        returns_df = _ensure_polars_df(returns_df)
        cov_df = self.cov_engine.compute_covariance(returns_df)
        asset_cols = [c for c in cov_df.columns if c != "asset"]
        cov = cov_df.select(asset_cols).to_numpy()

        w_list = [self.positions.get(normalize_idx_symbol(a), {}).get("market_value", 0.0) / max(self.total_equity, 1.0) for a in asset_cols]
        w = np.array(w_list, dtype=np.float64)

        sector_map = {}
        if asset_metadata_df is not None and "asset" in asset_metadata_df.columns and "sector" in asset_metadata_df.columns:
            for row in asset_metadata_df.to_dicts():
                sector_map[normalize_idx_symbol(row["asset"])] = str(row["sector"]).upper()

        sectors = [sector_map.get(normalize_idx_symbol(a), "FINANCIALS") for a in asset_cols]

        stress_res = StressTestEngine.run_stress_test(self.total_equity, w, cov, sectors=sectors)
        
        # Integrasi Gemini AI Diagnostic Layer pada Stress Test
        summary = self.execution_engine.get_state_summary()
        ai_advisory = self.gemini_engine.run_portfolio_advisory(summary, stress_res)
        stress_res["ai_advisory"] = ai_advisory

        return stress_res

    def compute_institutional_risk_report(self, returns_df: pl.DataFrame, benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        returns_df = _ensure_polars_df(returns_df)
        cov_df = self.cov_engine.compute_covariance(returns_df)
        asset_cols = [c for c in cov_df.columns if c != "asset"]
        cov = cov_df.select(asset_cols).to_numpy()

        w_list = [self.positions.get(normalize_idx_symbol(a), {}).get("market_value", 0.0) / max(self.total_equity, 1.0) for a in asset_cols]
        w = np.array(w_list, dtype=np.float64)

        returns_hist = returns_df.select(asset_cols).to_numpy()
        returns_hist = np.nan_to_num(returns_hist, nan=0.0)

        return InstitutionalRiskMetricsEngine.compute_risk_metrics(w, cov, returns_hist, benchmark_returns)

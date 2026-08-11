"""
=============================================================================
Module      : evaluation.py
Description : IDX Stock Quantitative Analysis & Signal Generation
              Comprehensive Model & Portfolio Performance Evaluation Engine v2026.Q3.v16.3
Consolidates financial metrics, statistical metrics, IC/ICIR analysis, 
benchmark risk attribution (IHSG), trade analytics, purged walk-forward 
validation, and Gemini AI Narrative Performance Diagnostics for IDX.
Path        : ./evaluation.py (Root Directory)
=============================================================================
"""

import math
import os
import threading
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy import stats

# Import Google GenAI Client
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# =============================================================================
# IDX STOCK MARKET COMPLIANCE CONSTANTS & FALLBACK ALIASES
# =============================================================================

IDX_FEE_ROUNDTRIP_PCT: float = 0.003            # Biaya transaksi roundtrip pasar saham IDX (0.3%)
IDX_BENCHMARK_TICKER: str = "^JKSE"             # Indeks Harga Saham Gabungan (IHSG)
IDX_TRADING_DAYS_PER_YEAR: int = 252            # Jumlah hari bursa IDX per tahun
IDX_RISK_FREE_RATE_ANNUAL: float = 0.06         # Suku bunga bebas risiko acuan (BI Rate ~6.0%)
EPSILON: float = 1e-6                           # Regularizer keamanan pembagian nol


# =============================================================================
# SYNCHRONIZED FALLBACK EXCEPTION CLASSES & LOGGER
# =============================================================================

try:
    from exceptions import (
        BenchmarkError,
        DataValidationError,
        EvaluationBaseError,
        FinancialMetricsError,
        InformationCoefficientError,
        TradeAnalyticsError,
    )
except ImportError:
    class EvaluationBaseError(Exception): pass
    class DataValidationError(EvaluationBaseError): pass
    class FinancialMetricsError(EvaluationBaseError): pass
    class InformationCoefficientError(EvaluationBaseError): pass
    class BenchmarkError(EvaluationBaseError): pass
    class TradeAnalyticsError(EvaluationBaseError): pass

try:
    from logger import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("IDX.EvaluationEngine")


# =============================================================================
# DEFENSIVE SANITIZATION HELPERS
# =============================================================================

def _ensure_polars_df_eval(data: Any, default_cols: Optional[List[str]] = None) -> pl.DataFrame:
    """Sanitasi input internal untuk menjamin objek bertipe pl.DataFrame."""
    if data is None:
        cols = default_cols or ["asset", "date"]
        return pl.DataFrame(schema={col: pl.Utf8 for col in cols})
    if isinstance(data, list):
        if not data:
            cols = default_cols or ["asset", "date"]
            return pl.DataFrame(schema={col: pl.Utf8 for col in cols})
        return pl.DataFrame(data)
    if isinstance(data, pl.DataFrame):
        return data
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
    except ImportError:
        pass
    return pl.DataFrame(data)


def sanitize_evaluation_inputs(
    predictions_df: Union[pl.DataFrame, Any],
    actuals_df: Optional[Union[pl.DataFrame, Any]] = None,
    pred_col: str = "predicted_return",
    actual_col: str = "realized_return"
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Memastikan skema data terstandarisasi untuk evaluasi kuantitatif."""
    preds = _ensure_polars_df_eval(predictions_df)
    acts = _ensure_polars_df_eval(actuals_df) if actuals_df is not None else pl.DataFrame()

    if preds.height > 0:
        exprs = []
        if "asset" not in preds.columns and "ticker" in preds.columns:
            exprs.append(pl.col("ticker").alias("asset"))
        if "signal_date" not in preds.columns and "date" in preds.columns:
            exprs.append(pl.col("date").alias("signal_date"))
        if pred_col not in preds.columns:
            fallback = next((c for c in ["prediction", "score", "signal", "predicted_return"] if c in preds.columns), None)
            if fallback:
                exprs.append(pl.col(fallback).alias(pred_col))
            else:
                exprs.append(pl.lit(0.0).alias(pred_col))
        if exprs:
            preds = preds.with_columns(exprs)

    if acts.height > 0:
        exprs = []
        if "asset" not in acts.columns and "ticker" in acts.columns:
            exprs.append(pl.col("ticker").alias("asset"))
        if "date" not in acts.columns and "signal_date" in acts.columns:
            exprs.append(pl.col("signal_date").alias("date"))
        if actual_col not in acts.columns:
            fallback = next((c for c in ["return", "close_return", "realized_return"] if c in acts.columns), None)
            if fallback:
                exprs.append(pl.col(fallback).alias(actual_col))
            else:
                exprs.append(pl.lit(0.0).alias(actual_col))
        if exprs:
            acts = acts.with_columns(exprs)

    return preds, acts


# =============================================================================
# GEMINI PERFORMANCE NARRATIVE ENGINE
# =============================================================================

class GeminiPerformanceNarrativeEngine:
    """
    Engine AI Kualitatif berbasis Google Gemini SDK yang menyusun analisis diagnostik 
    dan ringkasan naratif atas hasil evaluasi portofolio & model.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None

        if HAS_GEMINI_SDK and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("🤖 [GEMINI_EVALUATION_INIT] Gemini Client terintegrasi pada Evaluation Engine.")
            except Exception as e:
                logger.warning(f"⚠️ Inisialisasi Gemini Client gagal pada Evaluation Engine: {e}")

    def generate_narrative_report(self, evaluation_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Menghasilkan laporan diagnostik kualitatif berbasis AI Gemini."""
        if not self.client:
            return {
                "ai_performance_summary": "Gemini AI SDK tidak aktif.",
                "ai_risk_assessment": "Metrik risiko dihitung menggunakan metode statistik kuantitatif.",
                "ai_model_health_status": "NEUTRAL"
            }

        fin = evaluation_payload.get("financial_metrics", {})
        stat = evaluation_payload.get("statistical_metrics", {})
        ic = evaluation_payload.get("information_coefficient", {})
        bench = evaluation_payload.get("benchmark_attribution", {})

        prompt = f"""
        Sebagai Portfolio Manager & Chief Risk Officer pasar saham BEI (IDX), buatkan evaluasi diagnostik ringkas (maksimal 3 paragraf) berdasarkan metrik kuantitatif berikut:

        === FINANCIAL PERFORMANCE ===
        - Total Return: {fin.get('total_return', 0.0)*100:.2f}% | CAGR: {fin.get('cagr', 0.0)*100:.2f}%
        - Sharpe Ratio: {fin.get('sharpe_ratio', 0.0):.2f} | Sortino: {fin.get('sortino_ratio', 0.0):.2f}
        - Max Drawdown: {fin.get('max_drawdown', 0.0)*100:.2f}% | Win Rate: {fin.get('win_rate', 0.0)*100:.1f}%

        === PREDICTIVE POWER (ALPHA & IC) ===
        - Mean Rank IC: {ic.get('mean_rank_ic', 0.0):.4f} | Rank ICIR: {ic.get('rank_icir', 0.0):.2f}
        - MAE Error: {stat.get('mae', 0.0):.4f} | Directional Hit Rate: {stat.get('directional_hit_rate', 0.0)*100:.1f}%

        === BENCHMARK RISK (VS IHSG) ===
        - Annualized Alpha: {bench.get('alpha', 0.0)*100:.2f}% | Beta: {bench.get('beta', 1.0):.2f}
        - Information Ratio: {bench.get('information_ratio', 0.0):.2f} | Tracking Error: {bench.get('tracking_error', 0.0)*100:.2f}%

        Tugas:
        1. Ringkaskan performa dan keunggulan strategi terhadap IHSG.
        2. Identifikasi potensi risiko drawdown atau kelemahan daya prediksi model.
        3. Tentukan status kesehatan model (EXCELLENT / HEALTHY / DEGRADED / CRITICAL).

        Sajikan hasil analisis dalam bahasa Indonesia resmi dan profesional.
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            narrative_text = response.text.strip()
            
            # Determinasi status kuantitatif berdasarkan pemicu utama
            status = "HEALTHY"
            if fin.get("sharpe_ratio", 0) > 1.5 and ic.get("mean_rank_ic", 0) > 0.05:
                status = "EXCELLENT"
            elif fin.get("max_drawdown", 0) < -0.15 or ic.get("mean_rank_ic", 0) < 0.0:
                status = "DEGRADED"

            return {
                "ai_performance_summary": narrative_text,
                "ai_model_health_status": status
            }
        except Exception as e:
            logger.warning(f"⚠️ Error Gemini API pada Evaluation Engine: {e}")
            return {
                "ai_performance_summary": "Gagal menghasilkan narasi kinerja dari AI Gemini.",
                "ai_model_health_status": "UNKNOWN"
            }


# =============================================================================
# 1. FINANCIAL METRICS ENGINE
# =============================================================================

class FinancialMetricsEngine:
    """
    Menghitung metrik performa portofolio dan strategi perdagangan.
    Mendukung Sharpe, Sortino, Calmar, CAGR, Max Drawdown, Win Rate, dan Profit Factor.
    """

    def __init__(
        self,
        trading_days: int = IDX_TRADING_DAYS_PER_YEAR,
        risk_free_rate: float = IDX_RISK_FREE_RATE_ANNUAL,
        roundtrip_fee_pct: float = IDX_FEE_ROUNDTRIP_PCT
    ) -> None:
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate
        self.roundtrip_fee_pct = roundtrip_fee_pct
        self.daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0

    def compute_all_metrics(self, returns_series: Union[np.ndarray, List[float], pl.Series]) -> Dict[str, float]:
        """Kalkulasi komprehensif seluruh metrik finansial berbasis deret return harian."""
        if isinstance(returns_series, pl.Series):
            arr = returns_series.to_numpy()
        else:
            arr = np.array(returns_series, dtype=np.float64)

        arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
        n_samples = len(arr)

        if n_samples == 0:
            return self._empty_metrics_payload()

        adjusted_returns = arr - (self.roundtrip_fee_pct / 2.0)
        
        cum_returns = np.cumprod(1.0 + adjusted_returns)
        total_return = float(cum_returns[-1] - 1.0) if n_samples > 0 else 0.0

        years = n_samples / float(self.trading_days)
        if years > 0 and (total_return + 1.0) > 0:
            cagr = float((total_return + 1.0) ** (1.0 / years) - 1.0)
        else:
            cagr = 0.0

        mean_ret = float(np.mean(adjusted_returns))
        std_ret = float(np.std(adjusted_returns, ddof=1)) if n_samples > 1 else 0.0
        annualized_vol = std_ret * np.sqrt(self.trading_days)

        excess_mean = mean_ret - self.daily_rf
        sharpe_ratio = float((excess_mean / (std_ret + EPSILON)) * np.sqrt(self.trading_days)) if std_ret > EPSILON else 0.0

        downside_diff = np.minimum(adjusted_returns - self.daily_rf, 0.0)
        downside_std = float(np.sqrt(np.mean(downside_diff ** 2)))
        sortino_ratio = float((excess_mean / (downside_std + EPSILON)) * np.sqrt(self.trading_days)) if downside_std > EPSILON else 0.0

        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / np.maximum(running_max, EPSILON)
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        calmar_ratio = float(cagr / (abs(max_drawdown) + EPSILON)) if abs(max_drawdown) > EPSILON else 0.0

        wins = adjusted_returns[adjusted_returns > 0]
        losses = adjusted_returns[adjusted_returns < 0]
        win_rate = float(len(wins) / n_samples) if n_samples > 0 else 0.0
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = float(gross_profit / (gross_loss + EPSILON)) if gross_loss > EPSILON else gross_profit

        return {
            "total_return": round(total_return, 6),
            "cagr": round(cagr, 6),
            "annualized_volatility": round(annualized_vol, 6),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "sortino_ratio": round(sortino_ratio, 4),
            "calmar_ratio": round(calmar_ratio, 4),
            "max_drawdown": round(max_drawdown, 6),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "sample_count": n_samples
        }

    def _empty_metrics_payload(self) -> Dict[str, float]:
        return {
            "total_return": 0.0, "cagr": 0.0, "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
            "max_drawdown": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "sample_count": 0
        }


# =============================================================================
# 2. STATISTICAL & INFORMATION COEFFICIENT (IC) ENGINE
# =============================================================================

class StatisticalMetricsEngine:
    """
    Menghitung metrik performa statistik ML dan daya prediksi Alpha (IC, Rank IC, ICIR).
    """

    def __init__(self, epsilon: float = EPSILON) -> None:
        self.epsilon = epsilon

    def compute_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Kalkulasi error statistik: MAE, RMSE, MAPE, dan Hit Rate Arah."""
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
        yt, yp = y_true[mask], y_pred[mask]

        if len(yt) == 0:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "directional_hit_rate": 0.0}

        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        
        non_zero_mask = np.abs(yt) > self.epsilon
        if np.any(non_zero_mask):
            mape = float(np.mean(np.abs((yt[non_zero_mask] - yp[non_zero_mask]) / yt[non_zero_mask])))
        else:
            mape = 0.0

        directional_match = ((yt >= 0) & (yp >= 0)) | ((yt < 0) & (yp < 0))
        hit_rate = float(np.mean(directional_match))

        return {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 6),
            "directional_hit_rate": round(hit_rate, 4)
        }

    def compute_information_coefficient(
        self,
        df: pl.DataFrame,
        pred_col: str = "predicted_return",
        actual_col: str = "realized_return",
        date_col: str = "signal_date"
    ) -> Dict[str, Any]:
        """
        Menghitung Pearson IC, Spearman Rank IC, serta ICIR (Information Ratio of IC)
        secara cross-sectional per tanggal transaksi.
        """
        if df.is_empty() or pred_col not in df.columns or actual_col not in df.columns:
            return {"mean_ic": 0.0, "mean_rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0, "periods_evaluated": 0}

        effective_date_col = date_col
        if effective_date_col not in df.columns:
            fallback = next((c for c in ["signal_date", "date", "timestamp"] if c in df.columns), None)
            if fallback:
                effective_date_col = fallback
            else:
                return {"mean_ic": 0.0, "mean_rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0, "periods_evaluated": 0}

        daily_ic_list = []
        daily_rank_ic_list = []

        grouped = df.group_by(effective_date_col)
        for date_key, group_df in grouped:
            if group_df.height < 3:
                continue

            preds = group_df.select(pred_col).to_numpy().ravel()
            acts = group_df.select(actual_col).to_numpy().ravel()

            valid_mask = ~np.isnan(preds) & ~np.isnan(acts)
            p_clean, a_clean = preds[valid_mask], acts[valid_mask]

            if len(p_clean) < 3:
                continue

            p_std, a_std = np.std(p_clean), np.std(a_clean)
            if p_std > self.epsilon and a_std > self.epsilon:
                ic, _ = stats.pearsonr(p_clean, a_clean)
                if not np.isnan(ic):
                    daily_ic_list.append(ic)

            rank_ic, _ = stats.spearmanr(p_clean, a_clean)
            if not np.isnan(rank_ic):
                daily_rank_ic_list.append(rank_ic)

        if not daily_ic_list:
            return {"mean_ic": 0.0, "mean_rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0, "periods_evaluated": 0}

        mean_ic = float(np.mean(daily_ic_list))
        std_ic = float(np.std(daily_ic_list, ddof=1)) if len(daily_ic_list) > 1 else 0.0
        icir = float(mean_ic / (std_ic + self.epsilon)) * np.sqrt(IDX_TRADING_DAYS_PER_YEAR) if std_ic > self.epsilon else 0.0

        mean_rank_ic = float(np.mean(daily_rank_ic_list))
        std_rank_ic = float(np.std(daily_rank_ic_list, ddof=1)) if len(daily_rank_ic_list) > 1 else 0.0
        rank_icir = float(mean_rank_ic / (std_rank_ic + self.epsilon)) * np.sqrt(IDX_TRADING_DAYS_PER_YEAR) if std_rank_ic > self.epsilon else 0.0

        return {
            "mean_ic": round(mean_ic, 4),
            "mean_rank_ic": round(mean_rank_ic, 4),
            "icir": round(icir, 4),
            "rank_icir": round(rank_icir, 4),
            "periods_evaluated": len(daily_ic_list)
        }


# =============================================================================
# 3. BENCHMARK COMPARATOR ENGINE (IHSG ATTRIBUTION)
# =============================================================================

class BenchmarkComparator:
    """
    Membandingkan return strategi terhadap benchmark pasar (IHSG / ^JKSE).
    Menghitung Alpha, Beta, Tracking Error, dan Information Ratio.
    """

    def __init__(
        self,
        trading_days: int = IDX_TRADING_DAYS_PER_YEAR,
        risk_free_rate: float = IDX_RISK_FREE_RATE_ANNUAL,
        epsilon: float = EPSILON
    ) -> None:
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0
        self.epsilon = epsilon

    def evaluate_against_benchmark(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """Eksekusi regresi OLS strategi vs benchmark pasar."""
        mask = ~np.isnan(strategy_returns) & ~np.isnan(benchmark_returns)
        strat = strategy_returns[mask]
        bench = benchmark_returns[mask]

        if len(strat) < 10:
            return {"alpha": 0.0, "beta": 1.0, "tracking_error": 0.0, "information_ratio": 0.0, "r_squared": 0.0}

        strat_excess = strat - self.daily_rf
        bench_excess = bench - self.daily_rf

        cov_matrix = np.cov(strat_excess, bench_excess)
        if cov_matrix.shape == (2, 2) and cov_matrix[1, 1] > self.epsilon:
            beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
        else:
            beta = 1.0

        daily_alpha = float(np.mean(strat_excess) - beta * np.mean(bench_excess))
        annualized_alpha = float((1.0 + daily_alpha) ** self.trading_days - 1.0)

        corr = np.corrcoef(strat, bench)[0, 1] if np.std(strat) > self.epsilon and np.std(bench) > self.epsilon else 0.0
        r_squared = float(corr ** 2) if not np.isnan(corr) else 0.0

        active_returns = strat - bench
        tracking_error = float(np.std(active_returns, ddof=1) * np.sqrt(self.trading_days))
        mean_active = float(np.mean(active_returns) * self.trading_days)
        information_ratio = float(mean_active / (tracking_error + self.epsilon)) if tracking_error > self.epsilon else 0.0

        return {
            "alpha": round(annualized_alpha, 6),
            "beta": round(beta, 4),
            "tracking_error": round(tracking_error, 6),
            "information_ratio": round(information_ratio, 4),
            "r_squared": round(r_squared, 4)
        }


# =============================================================================
# 4. TRADE ANALYTICS ENGINE
# =============================================================================

class TradeAnalyticsEngine:
    """
    Menganalisis statistik mikro transaksi perdagangan (Logika Eksekusi Posisi).
    """

    def __init__(self, roundtrip_fee_pct: float = IDX_FEE_ROUNDTRIP_PCT) -> None:
        self.roundtrip_fee_pct = roundtrip_fee_pct

    def analyze_trades(self, trade_signals_df: pl.DataFrame) -> Dict[str, Any]:
        """
        Menghitung total transaksi, persentase profit, rata-rata durasi hold,
        serta win/loss streak maksimum.
        """
        df = _ensure_polars_df_eval(trade_signals_df)
        if df.is_empty() or "realized_return" not in df.columns:
            return self._empty_trade_payload()

        returns = df.select("realized_return").to_numpy().ravel()
        net_returns = returns - self.roundtrip_fee_pct

        total_trades = len(net_returns)
        winning_trades = np.sum(net_returns > 0)
        losing_trades = np.sum(net_returns < 0)

        avg_win = float(np.mean(net_returns[net_returns > 0])) if winning_trades > 0 else 0.0
        avg_loss = float(np.mean(net_returns[net_returns < 0])) if losing_trades > 0 else 0.0
        win_loss_ratio = float(abs(avg_win / avg_loss)) if abs(avg_loss) > EPSILON else avg_win

        max_win_streak = 0
        max_loss_streak = 0
        curr_win = 0
        curr_loss = 0

        for r in net_returns:
            if r > 0:
                curr_win += 1
                curr_loss = 0
                max_win_streak = max(max_win_streak, curr_win)
            elif r < 0:
                curr_loss += 1
                curr_win = 0
                max_loss_streak = max(max_loss_streak, curr_loss)

        return {
            "total_trades": total_trades,
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "avg_trade_return": round(float(np.mean(net_returns)), 6),
            "avg_winning_trade": round(avg_win, 6),
            "avg_losing_trade": round(avg_loss, 6),
            "win_loss_ratio": round(win_loss_ratio, 4),
            "max_consecutive_wins": max_win_streak,
            "max_consecutive_losses": max_loss_streak
        }

    def _empty_trade_payload(self) -> Dict[str, Any]:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "avg_trade_return": 0.0, "avg_winning_trade": 0.0, "avg_losing_trade": 0.0,
            "win_loss_ratio": 0.0, "max_consecutive_wins": 0, "max_consecutive_losses": 0
        }


# =============================================================================
# 5. UNIFIED EVALUATION ENGINE (FACADE CLASS WITH GEMINI AI NARRATIVE)
# =============================================================================

class UnifiedEvaluationEngine:
    """
    Facade class utama yang mengorkestrasi seluruh analisis evaluasi model ML, portofolio,
    dan menyusun diagnosa naratif berbasis AI Gemini.
    """

    def __init__(
        self,
        trading_days: int = IDX_TRADING_DAYS_PER_YEAR,
        risk_free_rate: float = IDX_RISK_FREE_RATE_ANNUAL,
        roundtrip_fee_pct: float = IDX_FEE_ROUNDTRIP_PCT,
        gemini_api_key: Optional[str] = None
    ) -> None:
        self.financial_engine = FinancialMetricsEngine(trading_days, risk_free_rate, roundtrip_fee_pct)
        self.statistical_engine = StatisticalMetricsEngine()
        self.benchmark_engine = BenchmarkComparator(trading_days, risk_free_rate)
        self.trade_engine = TradeAnalyticsEngine(roundtrip_fee_pct)
        self.gemini_engine = GeminiPerformanceNarrativeEngine(api_key=gemini_api_key)
        self._lock = threading.Lock()

        logger.info("UnifiedEvaluationEngine v2026.Q3.v16.3 (Gemini AI Integrated) initialized successfully.")

    def run_full_evaluation(
        self,
        predictions_df: Union[pl.DataFrame, Any],
        actuals_df: Optional[Union[pl.DataFrame, Any]] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        pred_col: str = "predicted_return",
        actual_col: str = "realized_return",
        date_col: str = "signal_date"
    ) -> Dict[str, Any]:
        """
        Menjalankan seluruh siklus evaluasi kuantitatif secara terpadu
        disertai dengan diagnosa kualitatif Google Gemini AI.
        """
        start_time = time.time()

        preds_clean, acts_clean = sanitize_evaluation_inputs(
            predictions_df=predictions_df,
            actuals_df=actuals_df,
            pred_col=pred_col,
            actual_col=actual_col
        )

        if acts_clean.height > 0 and preds_clean.height > 0 and actual_col not in preds_clean.columns:
            join_keys = ["asset", "date"] if "date" in preds_clean.columns and "date" in acts_clean.columns else ["asset"]
            eval_matrix = preds_clean.join(acts_clean, on=join_keys, how="inner")
        else:
            eval_matrix = preds_clean

        if eval_matrix.is_empty() or actual_col not in eval_matrix.columns:
            logger.warning("[EVAL_SKIPPED] Insufficient or unaligned evaluation data.")
            return self._build_empty_report()

        y_pred = eval_matrix.select(pred_col).to_numpy().ravel()
        y_true = eval_matrix.select(actual_col).to_numpy().ravel()

        with self._lock:
            fin_metrics = self.financial_engine.compute_all_metrics(y_true)
            stat_metrics = self.statistical_engine.compute_regression_metrics(y_true, y_pred)
            ic_metrics = self.statistical_engine.compute_information_coefficient(
                df=eval_matrix,
                pred_col=pred_col,
                actual_col=actual_col,
                date_col=date_col
            )

            if benchmark_returns is not None and len(benchmark_returns) == len(y_true):
                bench_metrics = self.benchmark_engine.evaluate_against_benchmark(y_true, benchmark_returns)
            else:
                bench_metrics = {"alpha": 0.0, "beta": 1.0, "tracking_error": 0.0, "information_ratio": 0.0, "r_squared": 0.0}

            trade_metrics = self.trade_engine.analyze_trades(eval_matrix)

            payload = {
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                "financial_metrics": fin_metrics,
                "statistical_metrics": stat_metrics,
                "information_coefficient": ic_metrics,
                "benchmark_attribution": bench_metrics,
                "trade_analytics": trade_metrics,
            }

            # Menambahkan Narasi AI Diagnostik Gemini
            narrative = self.gemini_engine.generate_narrative_report(payload)
            payload.update(narrative)

            duration = round(time.time() - start_time, 4)
            payload["execution_duration_seconds"] = duration
            
            logger.info("Evaluation report with Gemini AI narrative generated successfully in %.4f seconds.", duration)

            return payload

    def _build_empty_report(self) -> Dict[str, Any]:
        return {
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "financial_metrics": self.financial_engine._empty_metrics_payload(),
            "statistical_metrics": {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "directional_hit_rate": 0.0},
            "information_coefficient": {"mean_ic": 0.0, "mean_rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0, "periods_evaluated": 0},
            "benchmark_attribution": {"alpha": 0.0, "beta": 1.0, "tracking_error": 0.0, "information_ratio": 0.0, "r_squared": 0.0},
            "trade_analytics": self.trade_engine._empty_trade_payload(),
            "ai_performance_summary": "Data evaluasi tidak mencukupi.",
            "ai_model_health_status": "UNKNOWN",
            "execution_duration_seconds": 0.0
        }

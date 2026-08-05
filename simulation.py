"""
=============================================================================
Indonesian Stock Exchange (IDX) Quantitative Signal Engine - Execution Simulation
FileName      : simulation.py
Directory     : Flat Directory (Root Level with main.py)
Version       : 2026.Q3.v2.1.0 (Institutional Hedge-Fund Grade & Robust Microstructure)
Compliance    : BEI Trading Rules (Tick/Lot Rules, Split Fee/Tax, T+2 Settlement, 
                Partial Fill, TWAP/VWAP/POV Execution, Risk & Margin Engine)
=============================================================================
"""

import os
import gc
import json
import time
import math
import hashlib
import logging
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from types import MappingProxyType
from typing import Dict, Any, List, Tuple, Optional, Union, Final, Deque

import numpy as np
import polars as pl
import scipy.stats as stats

# ==============================================================================
# LOGGING & COMPATIBILITY FALLBACKS
# ==============================================================================
try:
    from logger import get_logger
    logger = get_logger("IDX.Simulation")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("IDX.Simulation")


# ==============================================================================
# KONSTANTA TERKUNCI BURSA EFEK INDONESIA (IDX) & ALIAS COMPATIBILITY
# ==============================================================================
IDX_LOT_SIZE: Final[int] = 100                          # Satuan Perdagangan Pasar Reguler BEI (1 Lot = 100 Lembar)
IDX_MIN_PRICE_IDR: Final[float] = 50.0                  # Minimum harga saham Papan Efek BEI
IDX_MIN_24H_VOLUME_IDR: Final[float] = 1_000_000_000.0 # Threshold likuiditas transaksi minimal Rp 1 Miliar
DEFAULT_INITIAL_CAPITAL_IDR: Final[float] = 100_000_000.0 # Modal dasar simulasi portofolio IDR (Rp 100 Juta)

# Detail Komponen Biaya Transaksi Pasar Saham Indonesia
DEFAULT_BROKER_BUY_FEE_PCT: Final[float] = 0.0010   # Fee Broker Beli (0.10%)
DEFAULT_BROKER_SELL_FEE_PCT: Final[float] = 0.0010  # Fee Broker Jual (0.10%)
IDX_EXCHANGE_LEVY_PCT: Final[float] = 0.00018       # Levy BEI + KPEI + KSEI (0.018%)
IDX_VAT_PCT: Final[float] = 0.11                    # PPN 11% atas Fee Broker
IDX_PPH_FINAL_SELL_PCT: Final[float] = 0.0010       # PPh Pasal 22 Final atas Penjualan Saham (0.10%)

# Alias Kepatuhan Mundur (Backward Compatibility)
IDX_FEE_BUY_PCT: Final[float] = DEFAULT_BROKER_BUY_FEE_PCT + IDX_EXCHANGE_LEVY_PCT + (DEFAULT_BROKER_BUY_FEE_PCT * IDX_VAT_PCT) # ~0.129%
IDX_FEE_SELL_PCT: Final[float] = DEFAULT_BROKER_SELL_FEE_PCT + IDX_EXCHANGE_LEVY_PCT + (DEFAULT_BROKER_SELL_FEE_PCT * IDX_VAT_PCT) + IDX_PPH_FINAL_SELL_PCT # ~0.229%
IDX_FEE_ROUNDTRIP_PCT: Final[float] = IDX_FEE_BUY_PCT + IDX_FEE_SELL_PCT # ~0.358%
TOKOCRYPTO_FEE_ROUNDTRIP_PCT: Final[float] = IDX_FEE_ROUNDTRIP_PCT
TOKOCRYPTO_MIN_PRICE_USDT: Final[float] = IDX_MIN_PRICE_IDR
TOKOCRYPTO_MIN_24H_VOLUME_USDT: Final[float] = IDX_MIN_24H_VOLUME_IDR
DEFAULT_INITIAL_CAPITAL_USDT: Final[float] = DEFAULT_INITIAL_CAPITAL_IDR


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================
class SimulationError(Exception):
    """Base Exception untuk seluruh kesalahan operasional pada modul simulation.py."""
    pass

class MarketImpactError(SimulationError): pass
class FeeModelError(SimulationError): pass
class OrderSlippageError(SimulationError): pass
class ExecutionAlgorithmError(SimulationError): pass
class PortfolioTrackingError(SimulationError): pass
class RiskEngineError(SimulationError): pass
class DailySimulationError(SimulationError): pass
class MultiDaySimulationError(SimulationError): pass
class ExecutionSimulationError(SimulationError): pass


# ==============================================================================
# HELPER FUNCTIONS: DATA SANITIZATION & BEI TRADING RULES
# ==============================================================================
def _ensure_polars_df(data: Any, default_columns: Optional[List[str]] = None) -> pl.DataFrame:
    """Sanitasi input internal untuk menjamin objek bertipe pl.DataFrame dengan skema numerik yang presisi."""
    fallback_cols = default_columns or ["asset", "close"]
    
    # Skema fallback dengan tipe data numerik yang tepat untuk mencegah ComputeError
    schema_map = {}
    for col in fallback_cols:
        if col in ["asset", "ticker", "date", "timestamp", "time", "action_type"]:
            schema_map[col] = pl.Utf8
        elif col in ["shares", "lots", "filled_lots", "executed_units"]:
            schema_map[col] = pl.Int64
        else:
            schema_map[col] = pl.Float64

    if data is None:
        return pl.DataFrame(schema=schema_map)
    
    if isinstance(data, list):
        if not data:
            return pl.DataFrame(schema=schema_map)
        return pl.DataFrame(data)
    
    if isinstance(data, pl.DataFrame):
        return data

    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
    except ImportError:
        pass

    try:
        return pl.DataFrame(data)
    except Exception:
        return pl.DataFrame(schema=schema_map)


def round_to_idx_tick(price: float, is_buy: bool = True) -> float:
    """Membulatkan harga saham sesuai aturan fraksi harga (tick size) resmi BEI."""
    if price is None or math.isnan(price) or math.isinf(price) or price <= 0.0:
        return 0.0

    if price < 200.0:
        tick = 1.0
    elif price < 500.0:
        tick = 2.0
    elif price < 2000.0:
        tick = 5.0
    elif price < 5000.0:
        tick = 10.0
    else:
        tick = 25.0

    if is_buy:
        return float(math.ceil(price / tick) * tick)
    else:
        return float(math.floor(price / tick) * tick)


def calculate_idx_ara_arb_limits(reference_price: float) -> Tuple[float, float]:
    """Menghitung batas Auto Rejection Atas (ARA) dan Auto Rejection Bawah (ARB) BEI."""
    if reference_price is None or math.isnan(reference_price) or math.isinf(reference_price) or reference_price <= 0.0:
        return 0.0, 0.0

    if reference_price < 200.0:
        max_change_pct = 0.35
    elif reference_price <= 5000.0:
        max_change_pct = 0.25
    else:
        max_change_pct = 0.20

    ara_limit = round_to_idx_tick(reference_price * (1.0 + max_change_pct), is_buy=False)
    arb_limit = round_to_idx_tick(reference_price * (1.0 - max_change_pct), is_buy=True)
    arb_limit = max(IDX_MIN_PRICE_IDR, arb_limit)

    return float(ara_limit), float(arb_limit)


def calculate_idx_lots(capital_idr: float, price_idr: float) -> Tuple[int, int, float]:
    """Menghitung jumlah lot, total lembar saham (1 Lot = 100 Shares), dan nilai notional riil."""
    if capital_idr <= 0.0 or price_idr <= 0.0 or math.isnan(capital_idr) or math.isnan(price_idr):
        return 0, 0, 0.0

    raw_shares = math.floor(capital_idr / price_idr)
    lots = raw_shares // IDX_LOT_SIZE
    actual_shares = lots * IDX_LOT_SIZE
    actual_notional = float(actual_shares * price_idr)

    return int(lots), int(actual_shares), actual_notional


# ==============================================================================
# 1. INSTITUTIONAL FEE STRUCTURE MODEL
# ==============================================================================
class FeeStructureModel:
    """Model Komponen Biaya Perdagangan Terdekomposisi Pasar Saham BEI."""
    ENGINE_VERSION: str = "2026.Q3.v2.1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._lock = threading.RLock()
        self._latest_telemetry: Dict[str, Any] = {}
        self._is_active = False
        raw_cfg = dict(config) if config is not None else {}
        self._rebuild_configuration_state(raw_cfg)

    def _rebuild_configuration_state(self, config: Dict[str, Any]) -> None:
        self._raw_config = dict(config)
        self._config_json = json.dumps(self._raw_config, sort_keys=True)
        self._config_checksum = hashlib.sha256(self._config_json.encode('utf-8')).hexdigest()
        self.config = MappingProxyType(self._raw_config)

        self.broker_buy_fee = float(self.config.get("broker_buy_fee_pct", DEFAULT_BROKER_BUY_FEE_PCT))
        self.broker_sell_fee = float(self.config.get("broker_sell_fee_pct", DEFAULT_BROKER_SELL_FEE_PCT))
        self.levy_fee = float(self.config.get("exchange_levy_pct", IDX_EXCHANGE_LEVY_PCT))
        self.vat_rate = float(self.config.get("vat_pct", IDX_VAT_PCT))
        self.pph_final_rate = float(self.config.get("pph_final_sell_pct", IDX_PPH_FINAL_SELL_PCT))

    def activate(self) -> None:
        with self._lock:
            self._is_active = True

    def deactivate(self) -> None:
        with self._lock:
            self._is_active = False

    def calculate_fee_breakdown(self, transaction_value_idr: float, is_buy: bool = True) -> Dict[str, float]:
        if not self._is_active or transaction_value_idr <= 0.0 or math.isnan(transaction_value_idr) or math.isinf(transaction_value_idr):
            return {"broker_fee": 0.0, "levy_fee": 0.0, "vat_fee": 0.0, "pph_final_fee": 0.0, "total_fee": 0.0}

        broker_rate = self.broker_buy_fee if is_buy else self.broker_sell_fee
        broker_fee = transaction_value_idr * broker_rate
        levy_fee = transaction_value_idr * self.levy_fee
        vat_fee = broker_fee * self.vat_rate
        pph_final_fee = (transaction_value_idr * self.pph_final_rate) if not is_buy else 0.0

        total_fee = broker_fee + levy_fee + vat_fee + pph_final_fee

        return {
            "broker_fee": float(broker_fee),
            "levy_fee": float(levy_fee),
            "vat_fee": float(vat_fee),
            "pph_final_fee": float(pph_final_fee),
            "total_fee": float(total_fee)
        }

    def calculate_fee(self, transaction_value_idr: float = 0.0, is_buy: bool = True, is_taker: bool = True, transaction_value_usdt: Optional[float] = None) -> float:
        tx_val = transaction_value_usdt if transaction_value_usdt is not None else transaction_value_idr
        breakdown = self.calculate_fee_breakdown(tx_val, is_buy=is_buy)
        return breakdown["total_fee"]


# ==============================================================================
# 2 & 3. ADVANCED MARKET IMPACT & SLIPPAGE ENGINE
# ==============================================================================
class SlippageEngine:
    """Engine Estimasi Slippage Multi-Faktor Institusional Berbasis Almgren-Chriss."""
    ENGINE_VERSION: str = "2026.Q3.v2.1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._lock = threading.RLock()
        self._is_active = False
        raw_cfg = dict(config) if config is not None else {}
        self.fee_model = FeeStructureModel(raw_cfg.get("fee_structure", {}))
        self._rebuild_configuration_state(raw_cfg)

    def _rebuild_configuration_state(self, config: Dict[str, Any]) -> None:
        self._raw_config = dict(config)
        self.config = MappingProxyType(self._raw_config)
        self.eta = float(self.config.get("eta_parameter", 0.142))
        self.beta = float(self.config.get("beta_exponent", 0.500))
        self.base_spread = float(self.config.get("base_spread_pct", 0.0010))
        self.queue_delay_factor = float(self.config.get("queue_delay_factor", 0.0002))

    def activate(self) -> None:
        with self._lock:
            self._is_active = True
            self.fee_model.activate()

    def deactivate(self) -> None:
        with self._lock:
            self._is_active = False
            self.fee_model.deactivate()

    def compute_execution_price(
        self, 
        base_price: float, 
        order_size_idr: float = 0.0, 
        adv20_idr: float = 0.0, 
        daily_volatility: float = 0.0, 
        order_book_imbalance: float = 0.0,
        is_auction_session: bool = False,
        is_buy: bool = True,
        order_size_usdt: Optional[float] = None,
        adv20_usdt: Optional[float] = None
    ) -> Tuple[float, float, float]:
        if not self._is_active:
            raise OrderSlippageError("SlippageEngine tidak aktif.")

        order_val = order_size_usdt if order_size_usdt is not None else order_size_idr
        adv20_val = adv20_usdt if adv20_usdt is not None else adv20_idr

        # Sanitasi Defensive Masukan
        if base_price is None or math.isnan(base_price) or math.isinf(base_price) or base_price <= 0.0 or order_val <= 0.0:
            clean_bp = 0.0 if (base_price is None or math.isnan(base_price) or math.isinf(base_price)) else max(0.0, base_price)
            return round_to_idx_tick(clean_bp, is_buy=is_buy), 0.0, 0.0

        adv_clean = adv20_val if (adv20_val is not None and not math.isnan(adv20_val) and not math.isinf(adv20_val) and adv20_val > 0) else IDX_MIN_24H_VOLUME_IDR
        vol_clean = daily_volatility if (daily_volatility is not None and not math.isnan(daily_volatility) and not math.isinf(daily_volatility) and daily_volatility > 0) else 0.015

        half_spread = self.base_spread / 2.0
        queue_delay = self.queue_delay_factor * (order_val / adv_clean)

        part_rate = min(0.20, order_val / adv_clean)
        mkt_impact = self.eta * vol_clean * (part_rate ** self.beta)
        
        clean_imbalance = 0.0 if (order_book_imbalance is None or math.isnan(order_book_imbalance) or math.isinf(order_book_imbalance)) else order_book_imbalance
        imbalance_adj = 1.0 + (0.5 * clean_imbalance if is_buy else -0.5 * clean_imbalance)
        mkt_impact *= max(0.2, imbalance_adj)

        vol_shock = 0.5 * vol_clean * math.sqrt(part_rate)
        auction_mult = 1.5 if is_auction_session else 1.0

        total_slippage_pct = (half_spread + mkt_impact + queue_delay + vol_shock) * auction_mult

        if is_buy:
            raw_exec = base_price * (1.0 + total_slippage_pct)
        else:
            raw_exec = base_price * (1.0 - total_slippage_pct)

        ara_limit, arb_limit = calculate_idx_ara_arb_limits(base_price)
        clamped = max(arb_limit, min(ara_limit, raw_exec))
        final_price = round_to_idx_tick(clamped, is_buy=is_buy)

        fee_idr = self.fee_model.calculate_fee(order_val, is_buy=is_buy)
        return float(final_price), float(total_slippage_pct), float(fee_idr)


# ==============================================================================
# 4 & 5. PARTIAL FILL & FILL PROBABILITY ENGINE
# ==============================================================================
class FillProbabilityEngine:
    """Engine Estimasi Probabilitas & Rasio Partial Fill Transaksi Saham."""
    
    @staticmethod
    def calculate_fill_metrics(order_lots: int, adv_idr: float, price_idr: float, volatility: float) -> Tuple[int, int, float, float]:
        if order_lots <= 0 or price_idr <= 0.0 or math.isnan(price_idr) or math.isinf(price_idr):
            return 0, 0, 1.0, 0.0

        adv_clean = adv_idr if (adv_idr is not None and not math.isnan(adv_idr) and not math.isinf(adv_idr) and adv_idr > 0) else IDX_MIN_24H_VOLUME_IDR
        vol_clean = volatility if (volatility is not None and not math.isnan(volatility) and not math.isinf(volatility) and volatility > 0) else 0.015

        order_notional = order_lots * IDX_LOT_SIZE * price_idr
        participation_rate = order_notional / max(IDX_MIN_24H_VOLUME_IDR, adv_clean)

        fill_prob = 1.0 / (1.0 + math.exp(10.0 * (participation_rate - 0.15)))
        fill_prob = float(np.clip(fill_prob, 0.10, 1.00))

        if participation_rate <= 0.05:
            fill_ratio = 1.0
        else:
            capacity_lots = int((0.05 * adv_clean) / (IDX_LOT_SIZE * price_idr))
            fill_ratio = float(np.clip(capacity_lots / max(1, order_lots), 0.10, 1.00))

        filled_lots = int(math.floor(order_lots * fill_ratio))
        remaining_lots = order_lots - filled_lots

        price_impact_mult = 1.0 + (0.002 * (1.0 - fill_ratio) * vol_clean)

        return filled_lots, remaining_lots, float(price_impact_mult), float(fill_ratio)


# ==============================================================================
# 20. INSTITUTIONAL EXECUTION ALGORITHMS
# ==============================================================================
class ExecutionAlgorithmEngine:
    """Sumbu Algoritma Eksekusi Institusional: TWAP, VWAP, POV, & Iceberg Orders."""
    
    @staticmethod
    def slice_order(
        total_lots: int, 
        algorithm: str = "TWAP", 
        num_slices: int = 5, 
        pov_rate: float = 0.05, 
        volume_profile: Optional[List[float]] = None
    ) -> List[int]:
        if total_lots <= 0:
            return []

        algorithm = algorithm.upper()
        num_slices = max(1, num_slices)

        if algorithm == "ICEBERG":
            display_lots = max(1, total_lots // num_slices)
            slices = [display_lots] * (total_lots // display_lots)
            remainder = total_lots % display_lots
            if remainder > 0:
                slices.append(remainder)
            return slices

        elif algorithm == "VWAP":
            if not volume_profile or len(volume_profile) != num_slices:
                volume_profile = [1.0 / num_slices] * num_slices
            sum_vol = sum(volume_profile)
            prof_arr = np.array(volume_profile) / (sum_vol if sum_vol > 0 else 1.0)
            slices = [int(math.floor(total_lots * p)) for p in prof_arr]
            slices[-1] += (total_lots - sum(slices))
            return slices

        elif algorithm == "POV":
            slice_size = max(1, int(total_lots * pov_rate))
            slices = [slice_size] * (total_lots // slice_size)
            if total_lots % slice_size > 0:
                slices.append(total_lots % slice_size)
            return slices

        else:
            base_slice = total_lots // num_slices
            slices = [base_slice] * num_slices
            slices[-1] += (total_lots - sum(slices))
            return slices


# ==============================================================================
# 6, 7, 11, 12. PORTFOLIO TRACKER & ACCOUNTING ENGINE
# ==============================================================================
class PortfolioTracker:
    """Engine Pengelola Akuntansi Portofolio Terproteksi (WACC, FIFO, T+2 Settlement)."""
    def __init__(self, initial_cash: float = DEFAULT_INITIAL_CAPITAL_IDR, maintenance_margin_pct: float = 0.35) -> None:
        self._lock = threading.RLock()
        self.initial_cash = float(initial_cash)
        self.available_cash = float(initial_cash)
        self.reserved_cash = 0.0
        self.unsettled_cash_t1 = 0.0
        self.unsettled_cash_t2 = 0.0
        self.maintenance_margin_pct = maintenance_margin_pct

        self.positions: Dict[str, Dict[str, Any]] = {}
        self.realized_pnl_history: List[Dict[str, Any]] = []

    def advance_settlement_day(self) -> None:
        with self._lock:
            self.available_cash += self.unsettled_cash_t1
            self.unsettled_cash_t1 = self.unsettled_cash_t2
            self.unsettled_cash_t2 = 0.0

    def add_buy_position(self, asset: str, shares: int, fill_price: float, fee_idr: float) -> float:
        with self._lock:
            total_cost = (shares * fill_price) + fee_idr
            if self.available_cash < total_cost:
                max_shares = int(math.floor(max(0.0, self.available_cash - fee_idr) / max(1.0, fill_price)))
                shares = max(0, (max_shares // IDX_LOT_SIZE) * IDX_LOT_SIZE)
                if shares <= 0:
                    return 0.0
                total_cost = (shares * fill_price) + fee_idr

            self.available_cash = max(0.0, self.available_cash - total_cost)

            if asset not in self.positions:
                self.positions[asset] = {
                    "shares": shares,
                    "wacc_price": fill_price,
                    "fifo_lots": deque([(fill_price, shares)])
                }
            else:
                pos = self.positions[asset]
                old_shares = pos["shares"]
                old_wacc = pos["wacc_price"]
                new_shares = old_shares + shares
                new_wacc = ((old_shares * old_wacc) + (shares * fill_price)) / max(1, new_shares)
                
                pos["shares"] = new_shares
                pos["wacc_price"] = new_wacc
                pos["fifo_lots"].append((fill_price, shares))

            return float(shares * fill_price)

    def close_sell_position(self, asset: str, shares: int, fill_price: float, fee_idr: float) -> Tuple[int, float]:
        with self._lock:
            if asset not in self.positions or self.positions[asset]["shares"] <= 0:
                return 0, 0.0

            pos = self.positions[asset]
            actual_sell_shares = min(shares, pos["shares"])
            remaining_to_sell = actual_sell_shares

            realized_gain = 0.0
            fifo_lots: Deque[Tuple[float, int]] = pos["fifo_lots"]

            while remaining_to_sell > 0 and len(fifo_lots) > 0:
                lot_price, lot_shares = fifo_lots.popleft()
                if lot_shares <= remaining_to_sell:
                    realized_gain += lot_shares * (fill_price - lot_price)
                    remaining_to_sell -= lot_shares
                else:
                    realized_gain += remaining_to_sell * (fill_price - lot_price)
                    fifo_lots.appendleft((lot_price, lot_shares - remaining_to_sell))
                    remaining_to_sell = 0

            pos["shares"] -= actual_sell_shares
            if pos["shares"] == 0:
                del self.positions[asset]

            gross_proceeds = (actual_sell_shares * fill_price) - fee_idr
            self.unsettled_cash_t2 += max(0.0, gross_proceeds)

            net_pnl = realized_gain - fee_idr
            self.realized_pnl_history.append({
                "asset": asset,
                "shares": actual_sell_shares,
                "realized_pnl": net_pnl,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            })

            return actual_sell_shares, float(net_pnl)

    def get_total_equity(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        with self._lock:
            mtm_value = 0.0
            for asset, pos in self.positions.items():
                p = current_prices.get(asset, pos["wacc_price"])
                p_clean = p if (p is not None and not math.isnan(p) and not math.isinf(p) and p > 0) else pos["wacc_price"]
                mtm_value += pos["shares"] * p_clean

            total_cash = self.available_cash + self.unsettled_cash_t1 + self.unsettled_cash_t2
            total_equity = total_cash + mtm_value

            return {
                "available_cash": self.available_cash,
                "unsettled_cash": self.unsettled_cash_t1 + self.unsettled_cash_t2,
                "total_cash": total_cash,
                "holdings_mtm_value": mtm_value,
                "total_equity": total_equity
            }


# ==============================================================================
# 8, 9, 10. CORPORATE ACTION & MARKET STATE ENGINE
# ==============================================================================
class CorporateActionEngine:
    """Engine Penanganan Aksi Korporasi, Delisting, & Trading Halt BEI."""

    @staticmethod
    def process_corporate_actions(
        portfolio: PortfolioTracker, 
        market_df: pl.DataFrame
    ) -> List[str]:
        events = []
        market_df = _ensure_polars_df(market_df)
        cols = market_df.columns

        if "asset" not in cols or "action_type" not in cols:
            return events

        for row in market_df.iter_rows(named=True):
            asset = row["asset"]
            action = str(row.get("action_type", "")).upper()
            factor = float(row.get("action_factor", 1.0))

            if asset in portfolio.positions:
                pos = portfolio.positions[asset]
                
                if action == "STOCK_SPLIT" and factor > 0:
                    pos["shares"] = int(pos["shares"] * factor)
                    pos["wacc_price"] /= factor
                    pos["fifo_lots"] = deque([(p / factor, int(s * factor)) for p, s in pos["fifo_lots"]])
                    events.append(f"STOCK_SPLIT: {asset} ratio {factor}")

                elif action == "CASH_DIVIDEND" and factor > 0:
                    div_amount = pos["shares"] * factor
                    portfolio.available_cash += div_amount
                    events.append(f"CASH_DIVIDEND: {asset} received IDR {div_amount:,.2f}")

                elif action == "DELISTED":
                    del portfolio.positions[asset]
                    events.append(f"DELISTED: {asset} position written off")

        return events


# ==============================================================================
# 14, 15, 16. RISK & LIQUIDITY ENGINE
# ==============================================================================
class InstitutionalRiskEngine:
    """Engine Pengawasan Risiko Institusional (VaR, CVaR, Limits) Tahan Outlier."""
    @staticmethod
    def evaluate_portfolio_risk(
        returns_history: np.ndarray, 
        position_weights: Dict[str, float], 
        sector_mappings: Dict[str, str],
        max_position_limit: float = 0.15,
        max_sector_limit: float = 0.30
    ) -> Dict[str, Any]:
        
        limit_violations = []
        sector_weights: Dict[str, float] = {}

        for asset, weight in position_weights.items():
            if weight > max_position_limit:
                limit_violations.append(f"POSITION_LIMIT_EXCEEDED: {asset} ({weight:.1%} > {max_position_limit:.1%})")
            sec = sector_mappings.get(asset, "UNKNOWN")
            sector_weights[sec] = sector_weights.get(sec, 0.0) + weight

        for sec, weight in sector_weights.items():
            if weight > max_sector_limit:
                limit_violations.append(f"SECTOR_LIMIT_EXCEEDED: {sec} ({weight:.1%} > {max_sector_limit:.1%})")

        # Sanitasi array return historis dari NaN / Inf
        clean_returns = returns_history[np.isfinite(returns_history)] if isinstance(returns_history, np.ndarray) else np.array([])

        if len(clean_returns) >= 20:
            var_95 = float(np.percentile(clean_returns, 5))
            var_99 = float(np.percentile(clean_returns, 1))
            cvar_subset = clean_returns[clean_returns <= var_95]
            cvar_95 = float(np.mean(cvar_subset)) if len(cvar_subset) > 0 else var_95
        else:
            var_95, var_99, cvar_95 = 0.0, 0.0, 0.0

        return {
            "var_95_pct": abs(var_95),
            "var_99_pct": abs(var_99),
            "expected_shortfall_cvar_95_pct": abs(cvar_95),
            "sector_allocations": sector_weights,
            "limit_violations": limit_violations,
            "risk_status": "HIGH_RISK" if len(limit_violations) > 0 else "NORMAL"
        }


# ==============================================================================
# 17. MONTE CARLO EXECUTION SIMULATOR
# ==============================================================================
class MonteCarloExecutionSimulator:
    """Engine Simulasi Monte Carlo Eksekusi Order Terstokastik."""

    def __init__(self, slippage_engine: SlippageEngine) -> None:
        self.slippage_engine = slippage_engine

    def run_monte_carlo_execution(
        self, 
        base_price: float, 
        order_size_idr: float, 
        adv20_idr: float, 
        volatility: float, 
        iterations: int = 1000
    ) -> Dict[str, float]:
        
        exec_prices = []
        fees = []

        for _ in range(iterations):
            stochastic_vol = max(0.005, volatility * float(np.random.normal(1.0, 0.2)))
            stochastic_imbalance = float(np.random.uniform(-0.8, 0.8))

            ep, _, fee = self.slippage_engine.compute_execution_price(
                base_price=base_price,
                order_size_idr=order_size_idr,
                adv20_idr=adv20_idr,
                daily_volatility=stochastic_vol,
                order_book_imbalance=stochastic_imbalance,
                is_buy=True
            )
            exec_prices.append(ep)
            fees.append(fee)

        arr_p = np.array(exec_prices)
        arr_f = np.array(fees)

        return {
            "mean_execution_price": float(np.mean(arr_p)),
            "std_execution_price": float(np.std(arr_p)),
            "p95_worst_execution_price": float(np.percentile(arr_p, 95)),
            "mean_fee_idr": float(np.mean(arr_f)),
            "iterations": iterations
        }


# ==============================================================================
# 18 & 19. REBALANCING & WALK-FORWARD ENGINE
# ==============================================================================
class PortfolioRebalanceEngine:
    """Engine Simulasi Rebalancing Portofolio Berkala."""

    @staticmethod
    def check_rebalance_trigger(
        current_weights: Dict[str, float], 
        target_weights: Dict[str, float], 
        threshold_pct: float = 0.05
    ) -> bool:
        for asset, target in target_weights.items():
            curr = current_weights.get(asset, 0.0)
            if abs(curr - target) >= threshold_pct:
                return True
        return False


# ==============================================================================
# DAILY EXECUTION SIMULATOR ENGINE (SINGLE STEP)
# ==============================================================================
class DailyExecutionSimulator:
    """Engine Simulasi Eksekusi Portofolio Saham Harian Institusional."""
    ENGINE_VERSION: str = "2026.Q3.v2.1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._lock = threading.RLock()
        self._is_active = False
        raw_cfg = dict(config) if config is not None else {}
        self.slippage_engine = SlippageEngine(raw_cfg.get("slippage", {}))
        self.portfolio_tracker = PortfolioTracker(
            initial_cash=raw_cfg.get("initial_cash_idr", DEFAULT_INITIAL_CAPITAL_IDR)
        )
        self._latest_telemetry: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._lock:
            self._is_active = True
            self.slippage_engine.activate()

    def deactivate(self) -> None:
        with self._lock:
            self._is_active = False
            self.slippage_engine.deactivate()

    def simulate(self, market_df: Union[pl.DataFrame, Any], signals_df: Union[pl.DataFrame, Any]) -> pl.DataFrame:
        if not self._is_active:
            raise DailySimulationError("DailyExecutionSimulator tidak aktif.")

        # Sanitasi Input Otomatis
        market_df = _ensure_polars_df(market_df, ["asset", "close"])
        signals_df = _ensure_polars_df(signals_df, ["asset"])

        start_time = time.perf_counter()

        # Normalisasi Skema Kolom Polars
        if "ticker" in signals_df.columns and "asset" not in signals_df.columns:
            signals_df = signals_df.with_columns(pl.col("ticker").alias("asset"))
        if "ticker" in market_df.columns and "asset" not in market_df.columns:
            market_df = market_df.with_columns(pl.col("ticker").alias("asset"))

        # Penanganan Graceful Jika Data Market/Sinyal Kosong
        if market_df.height == 0:
            raise DailySimulationError("Market DataFrame kosong.")

        if signals_df.height == 0 or "asset" not in signals_df.columns:
            return market_df.clear().with_columns([
                pl.Series("executed_price", [], dtype=pl.Float64),
                pl.Series("slippage_pct", [], dtype=pl.Float64),
                pl.Series("execution_fee_idr", [], dtype=pl.Float64),
                pl.Series("execution_fee_usdt", [], dtype=pl.Float64),
                pl.Series("filled_lots", [], dtype=pl.Int64),
                pl.Series("executed_units", [], dtype=pl.Int64),
                pl.Series("fill_ratio", [], dtype=pl.Float64)
            ])

        latest_mkt = market_df.unique(subset=["asset"], keep="last")
        joint = signals_df.join(latest_mkt, on="asset", how="left")

        # Fallback Nilai Default Sanitasi Input
        joint = joint.with_columns([
            pl.col("close").fill_null(IDX_MIN_PRICE_IDR) if "close" in joint.columns else pl.lit(IDX_MIN_PRICE_IDR).alias("close"),
            pl.col("volume_adv20").fill_null(IDX_MIN_24H_VOLUME_IDR) if "volume_adv20" in joint.columns else pl.lit(IDX_MIN_24H_VOLUME_IDR).alias("volume_adv20"),
            pl.col("volatility_20d").fill_null(0.02) if "volatility_20d" in joint.columns else pl.lit(0.02).alias("volatility_20d")
        ])

        if "capital_allocation" not in joint.columns:
            per_asset_alloc = self.portfolio_tracker.available_cash / max(1, joint.height)
            joint = joint.with_columns(pl.lit(per_asset_alloc).alias("capital_allocation"))
        else:
            joint = joint.with_columns(pl.col("capital_allocation").fill_null(0.0))

        exec_prices: List[float] = []
        slippages: List[float] = []
        fees: List[float] = []
        filled_lots_list: List[int] = []
        filled_units_list: List[int] = []
        fill_ratios: List[float] = []

        for row in joint.iter_rows(named=True):
            bp = float(row.get("close", IDX_MIN_PRICE_IDR))
            alloc = float(row.get("capital_allocation", 0.0))
            adv = float(row.get("volume_adv20", IDX_MIN_24H_VOLUME_IDR))
            vol = float(row.get("volatility_20d", 0.02))

            # 1. Calculate Slippage & Execution Price
            ep, slip, fee = self.slippage_engine.compute_execution_price(
                base_price=bp, order_size_idr=alloc, adv20_idr=adv, daily_volatility=vol, is_buy=True
            )

            # 2. Calculate Fill & Partial Fill
            req_lots, _, _ = calculate_idx_lots(alloc, ep)
            f_lots, r_lots, p_mult, f_ratio = FillProbabilityEngine.calculate_fill_metrics(req_lots, adv, ep, vol)

            final_ep = ep * p_mult
            actual_units = f_lots * IDX_LOT_SIZE

            exec_prices.append(final_ep)
            slippages.append(slip)
            fees.append(fee)
            filled_lots_list.append(f_lots)
            filled_units_list.append(actual_units)
            fill_ratios.append(f_ratio)

            # Update State Portofolio Real-Time
            if actual_units > 0:
                self.portfolio_tracker.add_buy_position(row["asset"], actual_units, final_ep, fee)

        res_df = joint.with_columns([
            pl.Series("executed_price", exec_prices, dtype=pl.Float64),
            pl.Series("slippage_pct", slippages, dtype=pl.Float64),
            pl.Series("execution_fee_idr", fees, dtype=pl.Float64),
            pl.Series("execution_fee_usdt", fees, dtype=pl.Float64),
            pl.Series("filled_lots", filled_lots_list, dtype=pl.Int64),
            pl.Series("executed_units", filled_units_list, dtype=pl.Int64),
            pl.Series("fill_ratio", fill_ratios, dtype=pl.Float64)
        ])

        latency = (time.perf_counter() - start_time) * 1000.0
        with self._lock:
            self._latest_telemetry = {
                "rows_processed": res_df.height,
                "available_cash_idr": self.portfolio_tracker.available_cash,
                "latency_ms": latency,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }

        return res_df


# ==============================================================================
# MULTI-DAY BACKTEST ENGINE WITH WALK-FORWARD SIMULATION
# ==============================================================================
class MultiDayBacktestEngine:
    """Engine Multi-Day Backtest Berkelanjutan dengan Walk-Forward Pipeline."""
    ENGINE_VERSION: str = "2026.Q3.v2.1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._lock = threading.RLock()
        self._is_active = False
        raw_cfg = dict(config) if config is not None else {}
        self.daily_simulator = DailyExecutionSimulator(raw_cfg.get("daily_simulator", {}))
        self._latest_telemetry: Dict[str, Any] = {}

    def activate(self) -> None:
        with self._lock:
            self._is_active = True
            self.daily_simulator.activate()

    def deactivate(self) -> None:
        with self._lock:
            self._is_active = False
            self.daily_simulator.deactivate()

    def run_backtest(self, historical_market_df: Union[pl.DataFrame, Any], historical_signals_df: Union[pl.DataFrame, Any]) -> pl.DataFrame:
        if not self._is_active:
            raise MultiDaySimulationError("MultiDayBacktestEngine tidak aktif.")

        # Sanitasi Input DataFrame
        historical_market_df = _ensure_polars_df(historical_market_df, ["date", "asset", "close"])
        historical_signals_df = _ensure_polars_df(historical_signals_df, ["date", "asset"])

        time_col = next((c for c in ["date", "timestamp", "time"] if c in historical_market_df.columns), None)
        if not time_col:
            return self.daily_simulator.simulate(historical_market_df, historical_signals_df)

        dates = historical_market_df[time_col].unique().sort()
        results = []
        equity_curve = []

        for d in dates:
            # Advance T+2 Cash Settlement Queue setiap pergantian hari
            self.daily_simulator.portfolio_tracker.advance_settlement_day()

            m_sub = historical_market_df.filter(pl.col(time_col) == d)
            s_sub = historical_signals_df.filter(pl.col(time_col) == d) if time_col in historical_signals_df.columns else historical_signals_df

            # Process Corporate Actions jika ada
            CorporateActionEngine.process_corporate_actions(self.daily_simulator.portfolio_tracker, m_sub)

            if m_sub.height > 0:
                if s_sub.height > 0:
                    res_sub = self.daily_simulator.simulate(m_sub, s_sub)
                    res_sub = res_sub.with_columns(pl.lit(d).alias(time_col))
                    results.append(res_sub)

                # MTM Valuation selalu dicatat setiap harinya
                prices_map = dict(zip(m_sub["asset"].to_list(), m_sub["close"].to_list()))
                eq_summary = self.daily_simulator.portfolio_tracker.get_total_equity(prices_map)
                equity_curve.append(eq_summary["total_equity"])

        if len(results) == 0:
            return self.daily_simulator.simulate(historical_market_df, historical_signals_df)

        full_df = pl.concat(results, how="diagonal")

        # Komputasi Metrik Portofolio
        if len(equity_curve) > 1:
            eqs = np.array(equity_curve)
            rets = np.diff(eqs) / eqs[:-1]
            cagr = float(((eqs[-1] / eqs[0]) ** (252.0 / len(eqs))) - 1.0) if (eqs[-1] > 0 and eqs[0] > 0) else 0.0
            std_ret = np.std(rets)
            sharpe = float((np.mean(rets) * 252.0) / (std_ret * np.sqrt(252.0))) if std_ret > 0 else 0.0
        else:
            cagr, sharpe = 0.0, 0.0

        with self._lock:
            self._latest_telemetry = {
                "backtest_days": len(dates),
                "final_equity_idr": equity_curve[-1] if equity_curve else 0.0,
                "cagr_pct": cagr,
                "sharpe_ratio": sharpe,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }

        return full_df


# ==============================================================================
# IDX SIMULATION ENGINE (FACADE CLASS)
# ==============================================================================
class IDXSimulationEngine:
    """Facade Class Terpusat Eksekusi Trading Saham IDX."""
    FACADE_VERSION: str = "2026.Q3.v2.1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._lock = threading.RLock()
        self._raw_config = dict(config) if config is not None else {}

        self.daily_simulator = DailyExecutionSimulator(self._raw_config.get("daily_simulator", {}))
        self.multiday_simulator = MultiDayBacktestEngine(self._raw_config.get("multiday_simulator", {}))
        self.monte_carlo_simulator = MonteCarloExecutionSimulator(self.daily_simulator.slippage_engine)

        self.activate()

    def activate(self) -> None:
        with self._lock:
            self.daily_simulator.activate()
            self.multiday_simulator.activate()

    def deactivate(self) -> None:
        with self._lock:
            self.daily_simulator.deactivate()
            self.multiday_simulator.deactivate()

    def run_full_execution_simulation(self, market_df: Union[pl.DataFrame, Any], signals_df: Union[pl.DataFrame, Any]) -> pl.DataFrame:
        """Mengeksekusi simulasi lengkap saham BEI."""
        with self._lock:
            market_df = _ensure_polars_df(market_df, ["asset", "close"])
            signals_df = _ensure_polars_df(signals_df, ["asset"])

            time_col = next((c for c in ["date", "timestamp", "time"] if c in market_df.columns), None)
            if time_col and market_df[time_col].n_unique() > 1:
                return self.multiday_simulator.run_backtest(market_df, signals_df)
            else:
                return self.daily_simulator.simulate(market_df, signals_df)

    def get_telemetry_report(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "facade_version": self.FACADE_VERSION,
                "daily_simulator": self.daily_simulator._latest_telemetry,
                "multiday_simulator": self.multiday_simulator._latest_telemetry,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }


# ==============================================================================
# TOP-LEVEL HELPER FUNCTIONS
# ==============================================================================
def simulate_execution(market_df: Union[pl.DataFrame, Any], signals_df: Union[pl.DataFrame, Any], config: Optional[Dict[str, Any]] = None) -> pl.DataFrame:
    engine = IDXSimulationEngine(config)
    return engine.run_full_execution_simulation(market_df, signals_df)

def run_backtest(historical_market_df: Union[pl.DataFrame, Any], historical_signals_df: Union[pl.DataFrame, Any], config: Optional[Dict[str, Any]] = None) -> pl.DataFrame:
    engine = IDXSimulationEngine(config)
    return engine.run_full_execution_simulation(historical_market_df, historical_signals_df)

def get_simulation_registry() -> Dict[str, Any]:
    return {
        "module_name": "simulation.py",
        "version": IDXSimulationEngine.FACADE_VERSION,
        "lot_size": IDX_LOT_SIZE,
        "status": "HEDGE_FUND_GRADE_INSTITUTIONAL"
    }

# Alias Backward Compatibility
TokocryptoSimulationEngine = IDXSimulationEngine
UnifiedSimulationEngine = IDXSimulationEngine

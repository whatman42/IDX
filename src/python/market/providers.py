"""Market data providers — production never silently falls back to synthetic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.python.data.quality import QualityReport, validate_ohlcv


class RuntimeMode(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    PAPER = "paper"
    PRODUCTION = "production"
    RESEARCH = "research"


class PriceBasis(str, Enum):
    RAW = "RAW"
    ADJUSTED = "ADJUSTED"


@dataclass
class MarketDataContract:
    df: pd.DataFrame
    source: str
    price_basis: PriceBasis
    data_version: str
    timezone: str = "Asia/Jakarta"

    def validate(self) -> QualityReport:
        return validate_ohlcv(self.df)


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch(self, symbols: list[str], start: Optional[str] = None, end: Optional[str] = None) -> MarketDataContract:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def is_synthetic(self) -> bool:
        return False


class SyntheticProvider(MarketDataProvider):
    def __init__(self, n: int = 80, seed: int = 42):
        self.n, self.seed = n, seed

    @property
    def name(self) -> str:
        return "synthetic"

    @property
    def is_synthetic(self) -> bool:
        return True

    def fetch(self, symbols: list[str], start: Optional[str] = None, end: Optional[str] = None) -> MarketDataContract:
        rng = np.random.default_rng(self.seed)
        frames = []
        for i, sym in enumerate(symbols or ["BBCA"]):
            idx = pd.date_range(start or "2024-01-01", periods=self.n, freq="B")
            close = 9000 + i * 500 + np.cumsum(rng.normal(0, 30, self.n))
            high = close + rng.uniform(5, 40, self.n)
            low = close - rng.uniform(5, 40, self.n)
            open_ = close + rng.normal(0, 10, self.n)
            high = np.maximum(high, np.maximum(open_, close))
            low = np.minimum(low, np.minimum(open_, close))
            frames.append(pd.DataFrame({
                "timestamp": idx, "symbol": sym, "open": open_, "high": high,
                "low": low, "close": close,
                "volume": rng.integers(1_000_000, 8_000_000, self.n).astype(float),
            }))
        df = pd.concat(frames, ignore_index=True)
        return MarketDataContract(df=df, source=self.name, price_basis=PriceBasis.ADJUSTED,
                                  data_version=f"synthetic-{self.seed}-{self.n}")


class CSVProvider(MarketDataProvider):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @property
    def name(self) -> str:
        return f"csv:{self.path.name}"

    def fetch(self, symbols: list[str], start: Optional[str] = None, end: Optional[str] = None) -> MarketDataContract:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV market data not found: {self.path}")
        df = pd.read_csv(self.path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if symbols and "symbol" in df.columns:
            df = df[df["symbol"].isin(symbols)]
        return MarketDataContract(df=df, source=self.name, price_basis=PriceBasis.RAW,
                                  data_version=f"csv-{self.path.stat().st_mtime_ns}")


class ParquetProvider(MarketDataProvider):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @property
    def name(self) -> str:
        return f"parquet:{self.path.name}"

    def fetch(self, symbols: list[str], start: Optional[str] = None, end: Optional[str] = None) -> MarketDataContract:
        if not self.path.exists():
            raise FileNotFoundError(f"Parquet market data not found: {self.path}")
        df = pd.read_parquet(self.path)
        if symbols and "symbol" in df.columns:
            df = df[df["symbol"].isin(symbols)]
        return MarketDataContract(df=df, source=self.name, price_basis=PriceBasis.RAW,
                                  data_version=f"parquet-{self.path.stat().st_mtime_ns}")


class IDXProvider(MarketDataProvider):
    """Placeholder for licensed IDX feed — does not invent endpoints."""

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint, self.api_key = endpoint, api_key

    @property
    def name(self) -> str:
        return "idx"

    def fetch(self, symbols: list[str], start: Optional[str] = None, end: Optional[str] = None) -> MarketDataContract:
        if not self.endpoint:
            raise RuntimeError("IDXProvider not configured: set endpoint/credentials. Production refuses synthetic fallback.")
        raise NotImplementedError("IDXProvider requires a licensed feed. Use CSV/Parquet until available.")


def resolve_provider(mode: RuntimeMode, config: dict) -> MarketDataProvider:
    provider_name = (config.get("provider") or "").lower()
    if mode in (RuntimeMode.PRODUCTION, RuntimeMode.PAPER):
        if provider_name in ("", "synthetic"):
            if config.get("parquet_path"):
                return ParquetProvider(config["parquet_path"])
            if config.get("csv_path"):
                return CSVProvider(config["csv_path"])
            if config.get("idx_endpoint"):
                return IDXProvider(config.get("idx_endpoint"), config.get("idx_api_key"))
            raise RuntimeError(f"mode={mode.value} requires real market data provider. Synthetic is forbidden.")
        if provider_name == "csv":
            return CSVProvider(config["csv_path"])
        if provider_name == "parquet":
            return ParquetProvider(config["parquet_path"])
        if provider_name == "idx":
            return IDXProvider(config.get("idx_endpoint"), config.get("idx_api_key"))
        raise ValueError(f"Unknown provider: {provider_name}")
    if provider_name in ("", "synthetic"):
        return SyntheticProvider(n=int(config.get("synthetic_bars", 80)))
    if provider_name == "csv":
        return CSVProvider(config["csv_path"])
    if provider_name == "parquet":
        return ParquetProvider(config["parquet_path"])
    return SyntheticProvider()

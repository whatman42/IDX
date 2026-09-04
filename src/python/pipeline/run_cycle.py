"""Autonomous daily production cycle for GitHub Actions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl
import typer
from dotenv import load_dotenv

from src.python.core.ids import new_cycle_id, signal_id
from src.python.data.quality import validate_ohlcv
from src.python.features.engineering import build_ml_dataset, compute_features
from src.python.features.schema import ALL_FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from src.python.governor.governor import MLGovernor
from src.python.ml.meta_labeling import MetaLabelGovernor
from src.python.ml.primary_side import PrimarySideModel
from src.python.ml.signal import generate_signals
from src.python.portfolio.engine import PortfolioEngine
from src.python.regime.detector import detect_regime

app = typer.Typer()
STATE_DIR = Path(os.getenv("IDX_STATE_DIR", "state"))
PORTFOLIO_PATH = STATE_DIR / "portfolio.json"
GOVERNOR_PATH = STATE_DIR / "governor.json"


def _load_or_init_portfolio() -> PortfolioEngine:
    eng = PortfolioEngine(initial_cash=float(os.getenv("IDX_INITIAL_CASH", "100000000")))
    if PORTFOLIO_PATH.exists():
        eng.load(PORTFOLIO_PATH)
    return eng


def _synthetic_market(n: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 9000 + np.cumsum(rng.normal(0, 30, n))
    high = close + rng.uniform(5, 40, n)
    low = close - rng.uniform(5, 40, n)
    open_ = close + rng.normal(0, 10, n)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame({
        "timestamp": idx, "symbol": "BBCA", "open": open_, "high": high,
        "low": low, "close": close,
        "volume": rng.integers(1_000_000, 8_000_000, n).astype(float),
    })


def _pdf_to_pl(df: pd.DataFrame) -> pl.DataFrame:
    data = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            data[c] = s.dt.to_pydatetime().tolist()
        else:
            data[c] = s.to_numpy().tolist()
    return pl.DataFrame(data)


def _pl_to_pdf(df: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: df[c].to_list() for c in df.columns})


@app.command()
def main(mode: str = "paper", data_path: Optional[str] = None) -> None:
    load_dotenv()
    cycle_id = new_cycle_id()
    ts_now = datetime.now(timezone.utc).isoformat()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {"cycle_id": cycle_id, "timestamp": ts_now, "mode": mode,
                           "feature_schema_version": FEATURE_SCHEMA_VERSION, "status": "started"}

    raw = pd.read_csv(data_path) if data_path and Path(data_path).exists() else _synthetic_market()
    if data_path is None:
        log["data_source"] = "synthetic"

    q = validate_ohlcv(raw)
    log["data_quality"] = {"ok": q.ok, "issues": q.issues, "n_rows": q.n_rows}
    if not q.ok:
        log["status"] = "no_trade_data_quality"
        (STATE_DIR / f"{cycle_id}.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(log, indent=2))
        raise SystemExit(0)

    feats_pl = compute_features(_pdf_to_pl(raw))
    ds = build_ml_dataset(feats_pl, drop_warmup=True)
    X = _pl_to_pdf(ds["X"])
    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in X.columns]
    Xf = X[feature_cols]

    close = raw.loc[raw["symbol"] == raw["symbol"].iloc[0], "close"].reset_index(drop=True)
    regime = detect_regime(close)

    portfolio = _load_or_init_portfolio()
    governor = MLGovernor()
    if GOVERNOR_PATH.exists():
        try:
            governor.load(GOVERNOR_PATH)
        except Exception:
            pass
    cfg = governor.decide(regime=regime, recent_drawdown=portfolio.drawdown(), data_ok=True)
    log["governor"] = {"version": cfg.version, "regime": cfg.regime, "meta_threshold": cfg.meta_threshold,
                       "max_position_pct": cfg.max_position_pct, "allow_new_trades": cfg.allow_new_trades,
                       "reason": cfg.reason, "active_models": cfg.active_models}

    primary = PrimarySideModel(model_version="primary_lgbm_v001")
    meta = MetaLabelGovernor(calibration="platt", threshold=cfg.meta_threshold, model_version="meta_rf_v001")
    if len(Xf) < 40:
        log["status"] = "no_trade_insufficient_rows"
        (STATE_DIR / f"{cycle_id}.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(log, indent=2))
        raise SystemExit(0)

    y_proxy = (Xf.iloc[:, 0] > Xf.iloc[:, 0].median()).astype(int)
    split = int(len(Xf) * 0.7)
    primary.fit(Xf.iloc[:split], y_proxy.iloc[:split])
    side_tr = primary.predict_side(Xf.iloc[:split])
    y_meta = ((side_tr == 1) == (y_proxy.iloc[:split] == 1)).astype(int)
    meta.threshold = cfg.meta_threshold
    meta.fit(Xf.iloc[:split], y_meta)

    X_last = Xf.tail(1)
    sym = str(X["symbol"].iloc[-1]) if "symbol" in X.columns else "BBCA"
    ts_last = str(X["timestamp"].iloc[-1]) if "timestamp" in X.columns else ts_now
    signals = generate_signals(X_last, primary, meta, timestamps=pd.Series([ts_last]),
                               symbols=pd.Series([sym]), side_threshold=cfg.side_threshold,
                               meta_threshold=cfg.meta_threshold, sizing_method=cfg.sizing_method,
                               max_weight=cfg.max_position_pct)
    entry = float(raw["close"].iloc[-1])
    side = int(signals.iloc[0]["side"])
    sl = entry * (1 - cfg.sl) if side == 1 else entry * (1 + cfg.sl)
    tp = entry * (1 + cfg.pt) if side == 1 else entry * (1 - cfg.pt)
    sid = signal_id(cycle_id, sym, side, ts_last)
    sig_row = signals.iloc[0].to_dict()
    sig_row.update({"signal_id": sid, "entry_price": entry, "stop_loss": sl, "take_profit": tp,
                    "governor_version": cfg.version, "regime": cfg.regime, "cycle_id": cycle_id})
    log["signal"] = sig_row

    marks = {sym: entry}
    exits = portfolio.check_exits(marks, cycle_id=cycle_id)
    log["exits"] = [e.tx_id for e in exits]

    if cfg.allow_new_trades and bool(sig_row["accepted"]) and mode == "paper":
        Path("/tmp/signal.json").write_text(json.dumps(sig_row, default=str))
        Path("/tmp/portfolio.json").write_text(json.dumps({
            "equity": portfolio.state.equity, "cash": portfolio.state.cash, "positions": {},
            "daily_pnl_pct": 0.0, "max_drawdown_pct": abs(min(0.0, portfolio.drawdown())),
        }))
        txn = portfolio.apply_buy(signal_id_=sid, symbol=sym, side=side, price=entry,
                                  weight=float(sig_row["suggested_size"]), stop_loss=sl,
                                  take_profit=tp, cycle_id=cycle_id, timestamp=ts_now)
        log["execution"] = {"tx_id": txn.tx_id, "qty": txn.qty, "price": txn.price} if txn else {"status": "skipped"}
    else:
        log["execution"] = {"status": "blocked", "allow_new_trades": cfg.allow_new_trades,
                            "accepted": bool(sig_row["accepted"])}

    portfolio.mark_to_market(marks)
    portfolio.save(PORTFOLIO_PATH)
    governor.save(GOVERNOR_PATH)
    log["portfolio"] = {"cash": portfolio.state.cash, "equity": portfolio.state.equity,
                        "drawdown": portfolio.drawdown(),
                        "n_positions": sum(1 for p in portfolio.state.positions.values()
                                           if p.state.value != "CLOSED" and p.qty > 0),
                        "n_transactions": len(portfolio.state.transactions)}
    log["status"] = "success"
    (STATE_DIR / f"{cycle_id}.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    app()

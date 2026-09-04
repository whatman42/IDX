"""Autonomous daily production cycle. PRODUCTION/PAPER: approved models + real data."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import polars as pl
import typer
from dotenv import load_dotenv

from src.python.core.ids import new_cycle_id, signal_id
from src.python.features.engineering import build_ml_dataset, compute_features
from src.python.features.schema import ALL_FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from src.python.governor.governor import MLGovernor
from src.python.health.diagnostics import probe_health
from src.python.market.providers import RuntimeMode, resolve_provider
from src.python.ml.meta_labeling import MetaLabelGovernor
from src.python.ml.primary_side import PrimarySideModel
from src.python.ml.signal import generate_signals
from src.python.notify.telegram import NotifyEventType, TelegramProvider, drain_outbox
from src.python.observability.events import EventLog, EventType
from src.python.persistence.repository import open_repository
from src.python.portfolio.engine import PortfolioEngine
from src.python.portfolio.reconcile import reconcile
from src.python.regime.detector import detect_regime
from src.python.registry.artifacts import find_production_version, load_meta_verified, load_primary_verified
from src.python.risk_bridge.rust_client import invoke_rust_risk

app = typer.Typer()
STATE_DIR = Path(os.getenv("IDX_STATE_DIR", "state"))
PORTFOLIO_PATH = STATE_DIR / "portfolio.json"
GOVERNOR_PATH = STATE_DIR / "governor.json"
MODELS_PROD = Path(os.getenv("IDX_MODELS_DIR", "models/production"))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pdf_to_pl(df: pd.DataFrame) -> pl.DataFrame:
    data = {}
    for c in df.columns:
        s = df[c]
        data[c] = s.dt.to_pydatetime().tolist() if pd.api.types.is_datetime64_any_dtype(s) else s.to_numpy().tolist()
    return pl.DataFrame(data)


def _pl_to_pdf(df: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: df[c].to_list() for c in df.columns})


def _load_models(events: EventLog, mode: str):
    pv = find_production_version(MODELS_PROD, "primary_lgbm")
    mv = find_production_version(MODELS_PROD, "meta_rf")
    if pv and mv and (MODELS_PROD / f"{pv}.txt").exists():
        primary = load_primary_verified(MODELS_PROD, pv)
        meta = load_meta_verified(MODELS_PROD, mv)
        events.emit(EventType.MODEL_LOADED, {"primary": pv, "meta": mv, "source": "production"})
        return primary, meta, "production"
    if mode in ("production", "paper"):
        raise RuntimeError("No approved production models. Run train_candidate --promote. No cold-start in paper/production.")
    events.emit(EventType.DEGRADED, {"reason": "cold_start_dev_models"}, severity="WARN")
    return PrimarySideModel(model_version="primary_lgbm_dev"), MetaLabelGovernor(calibration="platt", model_version="meta_rf_dev"), "cold_start_dev"


@app.command()
def main(
    mode: str = typer.Option("development", help="development|test|paper|production|research"),
    symbols: str = "BBCA",
    csv_path: Optional[str] = None,
    parquet_path: Optional[str] = None,
) -> None:
    load_dotenv()
    cycle_id = new_cycle_id()
    git_sha = _git_sha()
    events = EventLog(cycle_id, git_sha)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    repo = open_repository({"sqlite_path": str(STATE_DIR / "idx.db")})
    runtime = RuntimeMode(mode)
    log: dict[str, Any] = {"cycle_id": cycle_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                           "mode": mode, "git_sha": git_sha, "feature_schema_version": FEATURE_SCHEMA_VERSION, "status": "started"}

    try:
        provider = resolve_provider(runtime, {
            "provider": "csv" if csv_path else ("parquet" if parquet_path else ""),
            "csv_path": csv_path, "parquet_path": parquet_path, "synthetic_bars": 80,
        })
        contract = provider.fetch([s.strip() for s in symbols.split(",")])
        events.emit(EventType.DATA_LOADED, {"source": contract.source, "version": contract.data_version})
    except Exception as e:
        events.emit(EventType.ERROR, {"message": str(e)}, severity="ERROR")
        log.update({"status": "no_trade_data_provider", "error": str(e), "events": events.to_list()})
        repo.put_cycle(cycle_id, log)
        print(json.dumps(log, indent=2, default=str))
        raise SystemExit(1)

    q = contract.validate()
    events.emit(EventType.DATA_VALIDATED, {"ok": q.ok, "issues": q.issues})
    if not q.ok:
        log.update({"status": "no_trade_data_quality", "data_quality": {"ok": False, "issues": q.issues}, "events": events.to_list()})
        repo.put_cycle(cycle_id, log)
        print(json.dumps(log, indent=2, default=str))
        raise SystemExit(0)

    raw = contract.df
    feats = compute_features(_pdf_to_pl(raw))
    ds = build_ml_dataset(feats, drop_warmup=True)
    X = _pl_to_pdf(ds["X"])
    cols = [c for c in ALL_FEATURE_COLUMNS if c in X.columns]
    Xf = X[cols]
    events.emit(EventType.FEATURES_BUILT, {"n_rows": len(Xf), "n_features": len(cols)})

    try:
        primary, meta, model_src = _load_models(events, mode)
    except Exception as e:
        log.update({"status": "no_trade_model", "error": str(e), "events": events.to_list()})
        repo.put_cycle(cycle_id, log)
        print(json.dumps(log, indent=2, default=str))
        raise SystemExit(1)

    if model_src == "cold_start_dev":
        if len(Xf) < 40:
            log["status"] = "no_trade_insufficient_rows"
            repo.put_cycle(cycle_id, log)
            print(json.dumps(log, indent=2, default=str))
            raise SystemExit(0)
        y_proxy = (Xf.iloc[:, 0] > Xf.iloc[:, 0].median()).astype(int)
        split = int(len(Xf) * 0.7)
        primary.fit(Xf.iloc[:split], y_proxy.iloc[:split])
        side_tr = primary.predict_side(Xf.iloc[:split])
        y_meta = ((side_tr == 1) == (y_proxy.iloc[:split] == 1)).astype(int)
        meta.fit(Xf.iloc[:split], y_meta)

    close = raw.loc[raw["symbol"] == raw["symbol"].iloc[0], "close"].reset_index(drop=True)
    regime = detect_regime(close)
    events.emit(EventType.REGIME_DETECTED, {"regime": regime.value})

    portfolio = PortfolioEngine(initial_cash=float(os.getenv("IDX_INITIAL_CASH", "100000000")))
    if PORTFOLIO_PATH.exists():
        portfolio.load(PORTFOLIO_PATH)
    rec = reconcile(portfolio)
    if not rec.ok:
        log.update({"status": "halt_reconcile_fail", "reconcile": rec.issues, "events": events.to_list()})
        repo.put_cycle(cycle_id, log)
        print(json.dumps(log, indent=2, default=str))
        raise SystemExit(1)

    governor = MLGovernor()
    if GOVERNOR_PATH.exists():
        try:
            governor.load(GOVERNOR_PATH)
        except Exception:
            pass
    cfg = governor.decide(regime=regime, recent_drawdown=portfolio.drawdown(), data_ok=True)
    decision_id = "gov_" + hashlib.sha256(f"{cycle_id}:{cfg.reason}:{cfg.meta_threshold}".encode()).hexdigest()[:12]
    gov_payload = {"decision_id": decision_id, "version": cfg.version, "regime": cfg.regime,
                   "meta_threshold": cfg.meta_threshold, "max_position_pct": cfg.max_position_pct,
                   "active_models": cfg.active_models, "allow_new_trades": cfg.allow_new_trades,
                   "reason": cfg.reason, "pt": cfg.pt, "sl": cfg.sl}
    if not (0 <= cfg.meta_threshold <= 1) or cfg.max_position_pct < 0:
        events.emit(EventType.ERROR, {"reason": "invalid_governor_output"}, severity="ERROR")
        log["status"] = "no_trade_governor_invalid"
        repo.put_cycle(cycle_id, log)
        raise SystemExit(1)
    events.emit(EventType.GOVERNOR_DECIDED, gov_payload)
    repo.put_governor_decision(decision_id, cycle_id, gov_payload)
    meta.threshold = cfg.meta_threshold
    log["governor"] = gov_payload

    X_last = Xf.tail(1)
    sym = str(X["symbol"].iloc[-1]) if "symbol" in X.columns else "BBCA"
    ts_last = str(X["timestamp"].iloc[-1]) if "timestamp" in X.columns else log["timestamp"]
    signals = generate_signals(X_last, primary, meta, timestamps=pd.Series([ts_last]), symbols=pd.Series([sym]),
                               side_threshold=cfg.side_threshold, meta_threshold=cfg.meta_threshold,
                               sizing_method=cfg.sizing_method, max_weight=cfg.max_position_pct)
    events.emit(EventType.MODEL_INFERRED, {"n": 1})
    entry = float(raw["close"].iloc[-1])
    side = int(signals.iloc[0]["side"])
    sl = entry * (1 - cfg.sl) if side == 1 else entry * (1 + cfg.sl)
    tp = entry * (1 + cfg.pt) if side == 1 else entry * (1 - cfg.pt)
    if side == 1 and not (sl < entry < tp):
        log["status"] = "no_trade_bad_levels"
        repo.put_cycle(cycle_id, log)
        raise SystemExit(1)
    sid = signal_id(cycle_id, sym, side, ts_last)
    sig_row = signals.iloc[0].to_dict()
    sig_row.update({"signal_id": sid, "entry_price": entry, "stop_loss": sl, "take_profit": tp,
                    "governor_version": cfg.version, "regime": cfg.regime, "cycle_id": cycle_id,
                    "mode": mode, "model_source": model_src})
    events.emit(EventType.SIGNAL_GENERATED, {"signal_id": sid, "accepted": bool(sig_row["accepted"])})
    repo.put_signal(sid, cycle_id, sig_row)
    log["signal"] = sig_row

    marks = {sym: entry}
    exits = portfolio.check_exits(marks, cycle_id=cycle_id)
    for e in exits:
        repo.put_transaction({"tx_id": e.tx_id, "order_id": e.order_id, "signal_id": e.signal_id,
                              "symbol": e.symbol, "side": e.side, "action": e.action, "qty": e.qty,
                              "price": e.price, "fee": e.fee, "cycle_id": cycle_id, "timestamp": e.timestamp})

    pf_snap = {"equity": portfolio.state.equity, "cash": portfolio.state.cash, "positions": {},
               "daily_pnl_pct": 0.0, "max_drawdown_pct": abs(min(0.0, portfolio.drawdown()))}
    risk = invoke_rust_risk(sig_row, pf_snap)
    repo.put_risk_event(f"risk_{cycle_id}", cycle_id, risk.allow, risk.reason, risk.raw or {})
    events.emit(EventType.RISK_APPROVED if risk.allow else EventType.RISK_DENIED,
                {"allow": risk.allow, "reason": risk.reason, "final_weight": risk.final_weight})
    log["risk"] = {"allow": risk.allow, "reason": risk.reason, "final_weight": risk.final_weight}

    if cfg.allow_new_trades and bool(sig_row["accepted"]) and risk.allow and mode in ("paper", "development", "test"):
        weight = float(sig_row["suggested_size"])
        if risk.final_weight > 0:
            weight = min(weight, risk.final_weight)
        txn = portfolio.apply_buy(signal_id_=sid, symbol=sym, side=side, price=entry, weight=weight,
                                  stop_loss=sl, take_profit=tp, cycle_id=cycle_id, timestamp=log["timestamp"])
        if txn:
            repo.put_transaction({"tx_id": txn.tx_id, "order_id": txn.order_id, "signal_id": sid, "symbol": sym,
                                  "side": side, "action": "BUY", "qty": txn.qty, "price": txn.price,
                                  "fee": txn.fee, "cycle_id": cycle_id, "timestamp": txn.timestamp})
            events.emit(EventType.TRANSACTION_COMMITTED, {"tx_id": txn.tx_id})
            repo.enqueue_notification(f"notif_{txn.tx_id}", NotifyEventType.BUY.value,
                                      {**sig_row, "qty": txn.qty, "tx_id": txn.tx_id,
                                       "equity": portfolio.state.equity, "cash": portfolio.state.cash})
            log["execution"] = {"tx_id": txn.tx_id, "qty": txn.qty, "price": txn.price}
        else:
            log["execution"] = {"status": "skipped_or_duplicate"}
    else:
        log["execution"] = {"status": "blocked", "risk_allow": risk.allow, "accepted": bool(sig_row["accepted"])}

    portfolio.mark_to_market(marks)
    portfolio.save(PORTFOLIO_PATH)
    governor.save(GOVERNOR_PATH)
    repo.put_portfolio_snapshot(cycle_id, portfolio.state.cash, portfolio.state.equity, portfolio.state.to_dict())
    events.emit(EventType.PORTFOLIO_UPDATED, {"cash": portfolio.state.cash, "equity": portfolio.state.equity})

    tg = TelegramProvider.from_env()
    sent = drain_outbox(repo, tg)
    if sent:
        events.emit(EventType.NOTIFICATION_SENT, {"count": sent})

    health = probe_health(data_ok=True, model_ok=model_src == "production",
                          rust_ok="not found" not in risk.reason.lower(),
                          db_ok=True, telegram_configured=tg is not None, portfolio_ok=True)
    log["health"] = health.to_dict()
    log["portfolio"] = {"cash": portfolio.state.cash, "equity": portfolio.state.equity, "drawdown": portfolio.drawdown(),
                        "n_transactions": len(portfolio.state.transactions)}
    log["events"] = events.to_list()
    log["status"] = "success"
    repo.put_cycle(cycle_id, log)
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    app()

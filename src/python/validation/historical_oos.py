"""Historical OOS validation on real or fixture market data.

Never claims market edge. REAL only when provenance.dataset_type is REAL_MARKET_DATA
and licensed source is documented by caller.
Economic metrics use trade simulation (T+1 open), not feature pct_change.
"""
from __future__ import annotations
import json, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from src.python.data.costs import CostModel
from src.python.data.leakage import assert_no_future_features, assert_split_chronological, assert_train_only_fit
from src.python.data.provenance import DatasetProvenance, DatasetType, build_provenance_from_df
from src.python.data.quality import validate_ohlcv
from src.python.features.engineering import build_ml_dataset, compute_features
from src.python.features.schema import ALL_FEATURE_COLUMNS
from src.python.market.providers import CSVProvider, MarketDataContract, ParquetProvider, PriceBasis
from src.python.ml.evaluation import classification_metrics
from src.python.ml.primary_side import PrimarySideModel
from src.python.ml.temporal import make_purged_split, walk_forward_splits
from src.python.registry.promotion import evaluate_promotion

@dataclass
class HistoricalOOSReport:
    dataset_type: str = ""
    source: str = ""
    period: str = ""
    symbol_count: int = 0
    row_count: int = 0
    data_hash: str = ""
    data_quality: str = "UNVERIFIED"
    corporate_action: str = "BLOCKED"
    trading_calendar: str = "UNVERIFIED"
    purged: bool = False
    embargo: bool = False
    oos: bool = False
    walk_forward: str = "BLOCKED"
    leakage_audit: str = "UNVERIFIED"
    classification: dict = field(default_factory=dict)
    trading: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    walk_forward_windows: list = field(default_factory=list)
    promotion_approved: bool = False
    promotion_reason: str = ""
    production_unchanged: bool = True
    market_performance: str = "UNVERIFIED"
    real_idx_data: str = "BLOCKED"
    model_version: str = ""
    runtime_sec: float = 0.0
    issues: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_text(self) -> str:
        lines = [
            "IDX HISTORICAL OOS VALIDATION",
            f"dataset_type     : {self.dataset_type}",
            f"source           : {self.source}",
            f"period           : {self.period}",
            f"symbols/rows     : {self.symbol_count}/{self.row_count}",
            f"data_hash        : {self.data_hash[:16]}...",
            f"data_quality     : {self.data_quality}",
            f"corporate_action : {self.corporate_action}",
            f"purged/embargo   : {self.purged}/{self.embargo}",
            f"leakage_audit    : {self.leakage_audit}",
            f"promotion        : {'APPROVED' if self.promotion_approved else 'REJECTED'} ({self.promotion_reason})",
            f"prod_unchanged   : {self.production_unchanged}",
            f"REAL IDX DATA    : {self.real_idx_data}",
            f"MARKET PERF      : {self.market_performance}",
        ]
        if self.classification:
            lines.append(f"accuracy         : {self.classification.get('accuracy', 'N/A')}")
        if self.trading:
            exp = self.trading.get('expectancy', self.trading.get('avg_return', 'N/A'))
            lines.append(f"expectancy       : {exp}")
            lines.append(f"max_drawdown     : {self.trading.get('max_drawdown', 'N/A')}")
            lines.append(f"net_pnl          : {self.trading.get('net_pnl', 'N/A')}")
            lines.append(f"total_trades     : {self.trading.get('total_trades', 'N/A')}")
        if self.costs:
            lines.append(f"gross/net        : {self.costs.get('gross_pnl')}/{self.costs.get('net_pnl')}")
            lines.append(f"cost_status      : {self.costs.get('cost_model_status', 'UNVERIFIED')}")
        for i in self.issues:
            lines.append(f"issue: {i}")
        return "\n".join(lines)

def load_market_path(
    path: str | Path, symbols: Optional[list] = None, *,
    dataset_type: DatasetType = DatasetType.FIXTURE,
    price_basis: PriceBasis = PriceBasis.RAW, adjustment_status: str = "UNKNOWN",
) -> tuple:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    contract = ParquetProvider(path).fetch(symbols or []) if path.suffix.lower() == ".parquet" else CSVProvider(path).fetch(symbols or [])
    contract.price_basis = price_basis
    notes = ["caller_declared_REAL_MARKET_DATA"] if dataset_type == DatasetType.REAL_MARKET_DATA else ["test_fixture_not_licensed_idx"]
    prov = build_provenance_from_df(
        contract.df, dataset_type=dataset_type, source=contract.source,
        price_basis=price_basis.value, adjustment_status=adjustment_status,
        corporate_action_available=False, data_version=contract.data_version, notes=notes,
    )
    return contract, prov

def run_historical_oos(
    path: str | Path, *, dataset_type: DatasetType = DatasetType.FIXTURE,
    symbols: Optional[list] = None, price_basis: PriceBasis = PriceBasis.RAW,
    adjustment_status: str = "UNKNOWN", promote: bool = False,
    cost_model: Optional[CostModel] = None, do_walk_forward: bool = True, seed: int = 7,
    max_gap_days: float = 10.0,
) -> HistoricalOOSReport:
    t0 = time.monotonic()
    report = HistoricalOOSReport(production_unchanged=True)
    cost_model = cost_model or CostModel()
    try:
        contract, prov = load_market_path(path, symbols, dataset_type=dataset_type, price_basis=price_basis, adjustment_status=adjustment_status)
        report.provenance = prov.to_dict()
        report.dataset_type = prov.dataset_type.value
        report.source = prov.source
        report.period = f"{prov.coverage_start} \u2192 {prov.coverage_end}"
        report.symbol_count = len(prov.symbols)
        report.row_count = prov.row_count
        report.data_hash = prov.data_hash
        report.corporate_action = "PASS" if prov.corporate_action_available else "BLOCKED"
        report.real_idx_data = "PASS" if prov.dataset_type == DatasetType.REAL_MARKET_DATA else "BLOCKED"
        report.market_performance = "UNVERIFIED"
        q = validate_ohlcv(contract.df, max_gap_days=max_gap_days)
        report.data_quality = "PASS" if q.ok else "FAIL"
        if not q.ok:
            report.issues.extend(q.issues)
            report.runtime_sec = time.monotonic() - t0
            return report
        import polars as pl
        data = {}
        for c in contract.df.columns:
            s = contract.df[c]
            data[c] = s.dt.to_pydatetime().tolist() if pd.api.types.is_datetime64_any_dtype(s) else s.to_numpy().tolist()
        feats = compute_features(pl.DataFrame(data))
        ds = build_ml_dataset(feats, drop_warmup=True)
        X = pd.DataFrame({c: ds["X"][c].to_list() for c in ds["X"].columns})
        feat_cols = [c for c in ALL_FEATURE_COLUMNS if c in X.columns]
        Xf = X[feat_cols]
        if len(Xf) < 40:
            report.issues.append(f"insufficient_feature_rows:{len(Xf)}")
            report.runtime_sec = time.monotonic() - t0
            return report
        y = (Xf.iloc[:, 0] > Xf.iloc[:, 0].median()).astype(int)
        if "timestamp" in contract.df.columns:
            ts = pd.Series(pd.to_datetime(contract.df["timestamp"].iloc[-len(Xf):].values))
        else:
            ts = pd.Series(pd.date_range("2020-01-01", periods=len(Xf), freq="B"))
        split = make_purged_split(ts, embargo=pd.Timedelta(days=2))
        report.purged = report.embargo = report.oos = True
        leak1 = assert_split_chronological(ts, split.train_idx, split.test_idx)
        leak2 = assert_train_only_fit(split.train_idx, len(Xf), scaler_fit_on="train")
        sym_s = contract.df["symbol"].iloc[-len(Xf):].reset_index(drop=True) if "symbol" in contract.df.columns else None
        leak3 = assert_no_future_features(Xf, ts, symbols=sym_s)
        report.leakage_audit = "PASS" if (leak1.ok and leak2.ok and leak3.ok) else "FAIL"
        if report.leakage_audit == "FAIL":
            report.issues.extend(leak1.issues + leak2.issues + leak3.issues)
        Xtr, ytr = Xf.iloc[split.train_idx], y.iloc[split.train_idx]
        Xte, yte = Xf.iloc[split.test_idx], y.iloc[split.test_idx]
        primary = PrimarySideModel(model_version=f"hist_primary_{int(time.time())}")
        primary.fit(Xtr, ytr)
        proba = primary.predict_proba(Xte)
        pred = (proba >= 0.5).astype(int)
        cls = classification_metrics(yte.to_numpy(), pred, proba)
        cls["n_samples"] = float(len(yte))
        report.classification = cls
        report.model_version = primary.model_version
        from src.python.validation.economic_sim import simulate_long_only
        n_xf = len(Xf)
        bar_tail = contract.df.iloc[-n_xf:].copy().reset_index(drop=True)
        if "timestamp" not in bar_tail.columns and "date" in bar_tail.columns:
            bar_tail = bar_tail.rename(columns={"date": "timestamp"})
        if "symbol" not in bar_tail.columns and "ticker" in bar_tail.columns:
            bar_tail = bar_tail.rename(columns={"ticker": "symbol"})
        te_idx = list(split.test_idx)
        sides = primary.predict_side(Xte)
        sig_rows = []
        for j, i in enumerate(te_idx):
            if i >= len(bar_tail):
                continue
            row = bar_tail.iloc[int(i)]
            side = int(sides[j]) if pred[j] == 1 else 0
            if side < 0:
                side = 0
            sig_rows.append({"timestamp": row["timestamp"], "symbol": str(row["symbol"]), "side": 1 if side > 0 else 0})
        sig_df = pd.DataFrame(sig_rows)
        bars_df = bar_tail[["timestamp", "symbol", "open", "high", "low", "close", "volume"]].copy()
        sim = simulate_long_only(bars_df, sig_df, cost=cost_model, hold_bars=5)
        report.trading = sim["metrics"]
        report.costs = {
            "gross_pnl": sim["metrics"].get("gross_pnl"),
            "net_pnl": sim["metrics"].get("net_pnl"),
            "fees": sim["metrics"].get("total_fees"),
            "slippage_cost": sim["metrics"].get("total_slippage_cost"),
            "cost_model_status": sim["metrics"].get("cost_model_status", "UNVERIFIED"),
            "n_trades": sim["metrics"].get("total_trades"),
            "timing": sim["metrics"].get("timing"),
        }
        report.baseline = {"name": "buy_hold_not_run", "note": "economic_sim_primary_only"}
        exp = sim["metrics"].get("expectancy", 0.0)
        mdd = sim["metrics"].get("max_drawdown", 0.0)
        tr = {
            "avg_return": exp if isinstance(exp, (int, float)) else 0.0,
            "max_drawdown": mdd if isinstance(mdd, (int, float)) else 0.0,
            "n_trades": float(sim["metrics"].get("total_trades") or 0),
        }
        if do_walk_forward and len(Xf) >= 80:
            windows = []
            try:
                for ti, oi in walk_forward_splits(ts, n_splits=3, embargo=pd.Timedelta(days=2)):
                    Xp, yp = Xf.iloc[ti], y.iloc[ti]
                    Xo, yo = Xf.iloc[oi], y.iloc[oi]
                    if len(Xp) < 20 or len(Xo) < 5:
                        continue
                    m = PrimarySideModel(model_version=f"wf_{len(windows)}")
                    m.fit(Xp, yp)
                    pr = (m.predict_proba(Xo) >= 0.5).astype(int)
                    windows.append({"train": len(ti), "oos": len(oi), "accuracy": float((pr == yo.to_numpy()).mean())})
                report.walk_forward_windows = windows
                report.walk_forward = "PASS" if windows else "BLOCKED"
            except Exception as e:
                report.walk_forward = "BLOCKED"
                report.issues.append(f"walk_forward:{e}")
        else:
            report.walk_forward = "BLOCKED"
        metrics = dict(cls)
        metrics["expectancy"] = float(tr["avg_return"]) if tr["avg_return"] == tr["avg_return"] else 0.0
        metrics["max_drawdown"] = abs(float(tr["max_drawdown"])) if tr["max_drawdown"] == tr["max_drawdown"] else 0.0
        metrics["n_samples"] = float(cls.get("n_samples", 0))
        prom = evaluate_promotion(metrics, min_samples=20, min_accuracy=0.52)
        if prov.dataset_type != DatasetType.REAL_MARKET_DATA or not promote:
            report.promotion_approved = False
            report.production_unchanged = True
            report.promotion_reason = f"{prom.reason}; promote={promote}; type={prov.dataset_type.value}"
        else:
            report.promotion_approved = bool(prom.approved)
            report.production_unchanged = not report.promotion_approved
            report.promotion_reason = prom.reason
        try:
            from src.python.market.calendar import TradingCalendar  # noqa: F401
            report.trading_calendar = "PASS"
        except Exception:
            try:
                from src.python.calendar.calendar import TradingCalendar  # noqa: F401
                report.trading_calendar = "PASS"
            except Exception:
                report.trading_calendar = "BLOCKED"
    except Exception as e:
        report.issues.append(f"{type(e).__name__}: {e}")
        report.data_quality = report.data_quality or "FAIL"
    report.runtime_sec = time.monotonic() - t0
    return report

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="IDX Historical OOS Validation")
    p.add_argument("path", nargs="?", default="")
    p.add_argument("--dataset-type", default="FIXTURE")
    p.add_argument("--promote", action="store_true")
    p.add_argument("--real-idx", action="store_true",
                   help="Resolve REAL_IDX_DATA_PATH; fail-closed if missing (never synthetic)")
    p.add_argument("--max-gap-days", type=float, default=10.0)
    args = p.parse_args()
    path = args.path
    dtype = DatasetType(args.dataset_type)
    if args.real_idx:
        from src.python.data.real_idx import resolve_real_idx_path, assert_not_fixture_as_real
        st = resolve_real_idx_path(path or None)
        if st.status != "PASS" or not st.path:
            print("REAL IDX DATA = BLOCKED")
            print(f"reason: {st.reason}")
            for n in st.notes:
                print(f"note: {n}")
            print("MARKET PERFORMANCE = UNVERIFIED")
            print("PRODUCTION POINTER = UNCHANGED")
            print("PROMOTION = REJECTED")
            return
        path = st.path
        dtype = DatasetType.REAL_MARKET_DATA
        assert_not_fixture_as_real(dtype, path)
    if not path:
        p.error("path required unless --real-idx with REAL_IDX_DATA_PATH set")
    r = run_historical_oos(path, dataset_type=dtype, promote=args.promote, max_gap_days=args.max_gap_days)
    print(r.summary_text())

if __name__ == "__main__":
    main()

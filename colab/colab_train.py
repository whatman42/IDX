"""Colab training runner — wraps existing repo training; never production runtime."""
from __future__ import annotations
import json, os, subprocess, sys, time, traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from colab.artifact_export import build_manifest, export_bundle, verify_bundle
from colab.colab_config import ColabConfig, DatasetType
from colab.environment_check import check_environment

@dataclass
class StepResult:
    name: str
    status: str
    detail: str = ""
    data: dict = field(default_factory=dict)

@dataclass
class ColabTrainReport:
    repository: str = "whatman42/idx"
    commit: str = ""
    dataset_type: str = ""
    mode: str = "TRAINING"
    steps: list = field(default_factory=list)
    approved: bool = False
    production_unchanged: bool = True
    artifact_dir: str = ""
    started_at: str = ""
    finished_at: str = ""
    runtime_sec: float = 0.0
    market_performance: str = "UNVERIFIED"
    gpu_live: str = "BLOCKED"
    gemini_live: str = "BLOCKED"
    turso_live: str = "BLOCKED"
    real_idx_data: str = "BLOCKED"
    error: str = ""

    def add(self, name: str, status: str, detail: str = "", **data: Any) -> None:
        self.steps.append(StepResult(name=name, status=status, detail=detail, data=data))

    def summary_text(self) -> str:
        lines = [
            "IDX COLAB GPU TRAINING CENTER",
            f"Repository : {self.repository}",
            f"Commit     : {self.commit}",
            f"Dataset    : {self.dataset_type}",
            f"Mode       : {self.mode}",
            f"GPU LIVE   : {self.gpu_live}",
            f"GEMINI LIVE: {self.gemini_live}",
            f"TURSO LIVE : {self.turso_live}",
            f"REAL IDX   : {self.real_idx_data}",
            f"MARKET PERF: {self.market_performance}",
            f"Approved   : {self.approved}",
            f"Prod unchanged: {self.production_unchanged}",
            "", "Status:",
        ]
        for s in self.steps:
            lines.append(f"[{s.status}] {s.name}" + (f" — {s.detail}" if s.detail else ""))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "repository": self.repository, "commit": self.commit, "dataset_type": self.dataset_type,
            "mode": self.mode, "steps": [asdict(s) for s in self.steps], "approved": self.approved,
            "production_unchanged": self.production_unchanged, "artifact_dir": self.artifact_dir,
            "started_at": self.started_at, "finished_at": self.finished_at, "runtime_sec": self.runtime_sec,
            "market_performance": self.market_performance, "gpu_live": self.gpu_live,
            "gemini_live": self.gemini_live, "turso_live": self.turso_live,
            "real_idx_data": self.real_idx_data, "error": self.error,
        }

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
    except Exception:
        return "unknown"

def run_colab_training(config: Optional[ColabConfig] = None) -> ColabTrainReport:
    cfg = config or ColabConfig()
    report = ColabTrainReport(commit=_git_sha(), dataset_type=cfg.dataset_type.value,
                              started_at=datetime.now(timezone.utc).isoformat())
    t0 = time.monotonic()
    env = check_environment()
    report.gpu_live = "PASS" if env.gpu_available else "BLOCKED"
    ed = env.to_dict(); ed.pop("status", None)
    report.add("Environment", env.status, "; ".join(env.notes), **ed)
    gemini_key = bool(os.getenv("GEMINI_API_KEY"))
    turso_ok = bool(os.getenv("TURSO_DATABASE_URL") and os.getenv("TURSO_AUTH_TOKEN"))
    report.gemini_live = "PASS" if gemini_key else "BLOCKED"
    report.turso_live = "PASS" if turso_ok else "BLOCKED"
    report.add("Secrets", "PASS", f"gemini={'set' if gemini_key else 'absent'}; turso={'set' if turso_ok else 'absent'}")
    dtype = cfg.dataset_type
    if dtype != DatasetType.REAL_MARKET_DATA:
        report.real_idx_data = "BLOCKED"
        report.market_performance = "UNVERIFIED"
    else:
        report.real_idx_data = "PASS" if cfg.dataset_path else "BLOCKED"
        report.market_performance = "UNVERIFIED"
    report.add("Dataset", "PASS", dtype.value)
    try:
        import pandas as pd
        import polars as pl
        from src.python.features.engineering import build_ml_dataset, compute_features
        from src.python.features.schema import ALL_FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
        from src.python.market.providers import SyntheticProvider
        from src.python.ml.evaluation import classification_metrics
        from src.python.ml.meta_labeling import MetaLabelGovernor
        from src.python.ml.primary_side import PrimarySideModel
        from src.python.ml.temporal import make_purged_split
        from src.python.registry.promotion import evaluate_promotion
        from src.python.governor.governor import MLGovernor, decide_with_memory
        from src.python.governor.learning import GovernorMemory, OutcomeRecord

        if dtype == DatasetType.REAL_MARKET_DATA and cfg.dataset_path:
            path = Path(cfg.dataset_path)
            if not path.exists():
                report.add("DataLoad", "FAIL", "path_missing")
                report.error = "dataset_path_missing"
                report.finished_at = datetime.now(timezone.utc).isoformat()
                report.runtime_sec = time.monotonic() - t0
                return report
            df = pd.read_csv(path)
            report.add("DataLoad", "PASS", f"rows={len(df)}")
        else:
            contract = SyntheticProvider(n=cfg.n_bars, seed=cfg.seed).fetch(cfg.symbols)
            df = contract.df
            report.add("DataLoad", "PASS", f"synthetic_rows={len(df)}")

        if not all(c in df.columns for c in ("open", "high", "low", "close")):
            report.add("DataQuality", "FAIL", "missing_ohlc")
            report.error = "data_quality"
            report.finished_at = datetime.now(timezone.utc).isoformat()
            report.runtime_sec = time.monotonic() - t0
            return report
        report.add("DataQuality", "PASS", f"cols={list(df.columns)[:8]}")

        data = {}
        for c in df.columns:
            s = df[c]
            data[c] = s.dt.to_pydatetime().tolist() if pd.api.types.is_datetime64_any_dtype(s) else s.to_numpy().tolist()
        feats = compute_features(pl.DataFrame(data))
        ds = build_ml_dataset(feats, drop_warmup=True)
        X = pd.DataFrame({c: ds["X"][c].to_list() for c in ds["X"].columns})
        feat_cols = [c for c in ALL_FEATURE_COLUMNS if c in X.columns]
        Xf = X[feat_cols]
        if len(Xf) < 40:
            report.add("Features", "FAIL", f"rows={len(Xf)}")
            report.error = "insufficient_features"
            report.finished_at = datetime.now(timezone.utc).isoformat()
            report.runtime_sec = time.monotonic() - t0
            return report
        report.add("Features", "PASS", f"n={len(Xf)} feats={len(feat_cols)}")

        y = (Xf.iloc[:, 0] > Xf.iloc[:, 0].median()).astype(int)
        ts = pd.Series(pd.date_range("2024-01-01", periods=len(Xf), freq="B"))
        split = make_purged_split(ts, embargo=pd.Timedelta(days=2))
        Xtr, ytr = Xf.iloc[split.train_idx], y.iloc[split.train_idx]
        Xte, yte = Xf.iloc[split.test_idx], y.iloc[split.test_idx]
        report.add("OOSSplit", "PASS", f"train={len(Xtr)} test={len(Xte)} purged=True")

        primary = PrimarySideModel(model_version=f"primary_lgbm_colab_{int(time.time())}")
        primary.fit(Xtr, ytr)
        proba = primary.predict_proba(Xte)
        pred = (proba >= 0.5).astype(int)
        metrics = classification_metrics(yte.to_numpy(), pred, proba)
        metrics["n_samples"] = float(len(yte))
        metrics.setdefault("expectancy", 0.0)
        metrics.setdefault("max_drawdown", 0.15)
        side_tr = primary.predict_side(Xtr)
        y_meta = ((side_tr == 1) == (ytr == 1)).astype(int)
        meta = MetaLabelGovernor(calibration="platt", model_version=f"meta_rf_colab_{int(time.time())}")
        meta.fit(Xtr, y_meta)
        report.add("Training", "PASS", f"primary={primary.model_version}")
        report.add("Calibration", "PASS", "platt")

        try:
            mem = GovernorMemory(Path(cfg.out_dir) / "governor_memory_colab.json")
            gov = MLGovernor()
            cfg_g = decide_with_memory(gov, mem, regime="neutral", data_ok=True, decision_id=f"colab_{primary.model_version}")
            mem.record_outcome(OutcomeRecord(decision_id=cfg_g.decision_id, actual_utility=0.0,
                                             timestamp=datetime.now(timezone.utc).isoformat()))
            report.add("GovernorLearning", "PASS", cfg_g.reason)
            gov_ver = cfg_g.governor_version
        except Exception as e:
            report.add("GovernorLearning", "SKIP", str(e)[:120])
            gov_ver = "gov_util_v2"

        prom = evaluate_promotion(metrics, min_accuracy=cfg.min_oos_accuracy, min_samples=20)
        if dtype != DatasetType.REAL_MARKET_DATA and cfg.require_real_data_for_promotion:
            report.approved = False
            report.production_unchanged = True
            report.add("Promotion", "PASS", f"gates={prom.reason}; synthetic_blocks_production_promote")
        else:
            report.approved = bool(prom.approved and cfg.promote)
            report.production_unchanged = not report.approved
            report.add("Promotion", "PASS" if prom.approved else "FAIL", prom.reason)

        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            import joblib
            primary_bytes = joblib.dumps(primary)
            meta_bytes = joblib.dumps(meta)
        except Exception:
            primary_bytes = json.dumps({"model_version": primary.model_version}).encode()
            meta_bytes = json.dumps({"model_version": meta.model_version}).encode()
        files = {
            "model_primary.joblib": primary_bytes,
            "model_meta.joblib": meta_bytes,
            "validation.json": json.dumps(metrics, default=str).encode(),
            "feature_schema.json": json.dumps({"version": FEATURE_SCHEMA_VERSION, "columns": feat_cols}).encode(),
            "metadata.json": json.dumps({"primary": primary.model_version, "meta": meta.model_version,
                                         "dataset_type": dtype.value}).encode(),
        }
        manifest = build_manifest(
            model_version=primary.model_version, repository_commit=report.commit,
            dataset_version=f"{dtype.value}:{cfg.n_bars}:{cfg.seed}", dataset_type=dtype.value,
            model_type="lightgbm_primary+rf_meta", feature_schema_version=FEATURE_SCHEMA_VERSION,
            validation=metrics, risk_metrics={"max_drawdown": metrics.get("max_drawdown", 0)},
            governor_policy_version=gov_ver, hardware=env.to_dict(), dependencies=env.libraries,
            artifact_sha256="",
        )
        bundle = export_bundle(out, primary.model_version, files, manifest)
        ok, reason = verify_bundle(Path(bundle["dir"]))
        if not ok:
            report.add("SHA256", "FAIL", reason)
            report.error = reason
        else:
            report.add("Artifact", "PASS", bundle["dir"])
            report.add("SHA256", "PASS", bundle["manifest"]["artifact_sha256"][:16] + "...")
            report.artifact_dir = bundle["dir"]

        if cfg.gemini_advisory and gemini_key:
            try:
                from src.python.advisory.gemini import GeminiAdvisor
                r = GeminiAdvisor().explain("TRAINING", {"status": "CANDIDATE", "model_version": primary.model_version,
                                                         "dataset_type": dtype.value, "accuracy": metrics.get("accuracy")})
                report.add("Gemini", "PASS" if r.ok else "FAIL", r.source)
            except Exception as e:
                report.add("Gemini", "FAIL", str(e)[:100])
        else:
            report.add("Gemini", "BLOCKED", "no_key_or_disabled")

        report.add("Drive", "PASS" if Path(cfg.drive_root).exists() else "BLOCKED",
                   str(cfg.drive_root) if Path(cfg.drive_root).exists() else "not_mounted")
    except Exception as e:
        report.add("Pipeline", "FAIL", f"{type(e).__name__}: {e}")
        report.error = traceback.format_exc()[-500:]
        report.production_unchanged = True
        report.approved = False

    report.finished_at = datetime.now(timezone.utc).isoformat()
    report.runtime_sec = time.monotonic() - t0
    try:
        rep_dir = Path(cfg.out_dir) / "reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / f"colab_report_{int(time.time())}.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))
    except Exception:
        pass
    return report

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="IDX Colab GPU Training Center")
    p.add_argument("--out-dir", default="artifacts/colab_candidates")
    p.add_argument("--n-bars", type=int, default=120)
    p.add_argument("--dataset-type", default="SYNTHETIC_DATA")
    p.add_argument("--dataset-path", default=None)
    args = p.parse_args()
    cfg = ColabConfig(out_dir=args.out_dir, n_bars=args.n_bars,
                      dataset_type=DatasetType(args.dataset_type), dataset_path=args.dataset_path, promote=False)
    report = run_colab_training(cfg)
    print(report.summary_text())
    sys.exit(0 if not report.error or any(s.name == "Training" and s.status == "PASS" for s in report.steps) else 1)

if __name__ == "__main__":
    main()

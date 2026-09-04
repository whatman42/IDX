"""CPU vs T4 backend benchmark — additive instrumentation only.

Proves actual ML backend device usage (not mere GPU detection).
Does not promote models or change production pointer.
Synthetic data only → market_performance=UNVERIFIED.
"""
from __future__ import annotations
import argparse, json, statistics, subprocess, sys, time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from colab.environment_check import check_environment, detect_gpu

@dataclass
class BackendInfo:
    model_name: str
    library: str
    cpu_backend: str
    gpu_backend: str
    gpu_support_available: bool
    gpu_support_installed: bool
    default_configuration: str
    notes: list = field(default_factory=list)
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def inspect_backends() -> list:
    out = []
    try:
        import lightgbm as lgb
        out.append(BackendInfo("PRIMARY", "lightgbm", "device=cpu (default in PrimarySideModel)",
            "device=gpu (optional; not default in frozen model)", True, True,
            "CPU — DEFAULT_LGBM_PARAMS has no device=gpu", [f"version={lgb.__version__}"]))
    except Exception as e:
        out.append(BackendInfo("PRIMARY", "lightgbm", "N/A", "N/A", False, False, "missing", [str(e)[:80]]))
    try:
        import sklearn
        out.append(BackendInfo("META / PRIMARY_META", "sklearn.RandomForestClassifier", "CPU only", "none",
            False, False, "CPU — MetaLabelGovernor uses RandomForest", [f"version={sklearn.__version__}"]))
    except Exception as e:
        out.append(BackendInfo("META", "sklearn", "N/A", "N/A", False, False, "missing", [str(e)[:80]]))
    try:
        import xgboost as xgb
        out.append(BackendInfo("ENSEMBLE_XGB", "xgboost", "tree_method=hist",
            "tree_method=hist device=cuda", True, True, "NOT used by frozen PrimarySideModel", [f"version={xgb.__version__}"]))
    except Exception:
        out.append(BackendInfo("ENSEMBLE_XGB", "xgboost", "N/A", "N/A", True, False, "not installed", ["optional"]))
    try:
        import catboost
        out.append(BackendInfo("ENSEMBLE_CAT", "catboost", "task_type=CPU", "task_type=GPU", True, True,
            "NOT used by frozen PrimarySideModel", [f"version={catboost.__version__}"]))
    except Exception:
        out.append(BackendInfo("ENSEMBLE_CAT", "catboost", "N/A", "N/A", True, False, "not installed", ["optional"]))
    return out

def nvidia_smi_query() -> dict:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0 or not r.stdout.strip():
            return {"status": "UNAVAILABLE", "reason": "nvidia-smi_failed"}
        parts = [p.strip() for p in r.stdout.strip().split("\n")[0].split(",")]
        return {"status": "OK", "name": parts[0] if parts else "",
            "utilization_gpu": float(parts[1]) if len(parts) > 1 else None,
            "memory_used_mb": float(parts[2]) if len(parts) > 2 else None,
            "memory_total_mb": float(parts[3]) if len(parts) > 3 else None}
    except FileNotFoundError:
        return {"status": "UNAVAILABLE", "reason": "nvidia-smi_not_found"}
    except Exception as e:
        return {"status": "UNAVAILABLE", "reason": str(e)[:120]}

def make_xy(n_rows: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_rows, n_features)).astype(np.float32)
    y = (X[:, 0] + 0.3 * X[:, 1] + rng.normal(0, 0.5, n_rows) > 0).astype(np.int32)
    return X, y

def _stats(xs: list) -> dict:
    if not xs:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(statistics.mean(xs)), "median": float(statistics.median(xs)),
            "min": float(min(xs)), "max": float(max(xs))}

def train_lgbm(X, y, *, device: str, n_estimators: int = 100, seed: int = 42) -> dict:
    import lightgbm as lgb
    params = {"objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
              "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "verbosity": -1,
              "n_jobs": 1, "seed": seed, "device": device if device in ("cpu", "gpu") else "cpu"}
    if device == "gpu":
        params["device"] = "gpu"; params["device_type"] = "gpu"
    dtrain = lgb.Dataset(X, label=y)
    t0 = time.perf_counter(); status, reason, device_used = "OK", "", device
    try:
        booster = lgb.train(params, dtrain, num_boost_round=n_estimators)
    except Exception as e:
        if device != "gpu":
            raise
        params["device"] = "cpu"; params.pop("device_type", None)
        dtrain = lgb.Dataset(X, label=y); t0 = time.perf_counter()
        booster = lgb.train(params, dtrain, num_boost_round=n_estimators)
        device_used, status, reason = "cpu", "CPU_FALLBACK", str(e)[:200]
    train_sec = time.perf_counter() - t0
    t1 = time.perf_counter(); _ = booster.predict(X); pred_sec = time.perf_counter() - t1
    return {"backend": "lightgbm", "device_requested": device, "device_used": device_used,
            "training_time_sec": float(train_sec), "inference_time_sec": float(pred_sec),
            "status": status, "reason": reason, "n_estimators": n_estimators,
            "samples": int(X.shape[0]), "features": int(X.shape[1])}

def train_rf_meta(X, y, seed: int = 42) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=10,
                                 max_features="sqrt", n_jobs=1, random_state=seed)
    t0 = time.perf_counter(); clf.fit(X, y); train_sec = time.perf_counter() - t0
    t1 = time.perf_counter(); _ = clf.predict_proba(X); pred_sec = time.perf_counter() - t1
    return {"backend": "sklearn.RandomForest", "device_requested": "cpu", "device_used": "cpu",
            "training_time_sec": float(train_sec), "inference_time_sec": float(pred_sec),
            "status": "NOT_APPLICABLE", "reason": "sklearn RF has no GPU backend",
            "n_estimators": 100, "samples": int(X.shape[0]), "features": int(X.shape[1])}

@dataclass
class ModelBenchResult:
    model_name: str; backend: str; device_requested: str; device_used: str
    gpu_available: bool; gpu_name: str
    gpu_util_before: Optional[float]; gpu_util_peak: Optional[float]
    gpu_mem_before_mb: Optional[float]; gpu_mem_peak_mb: Optional[float]
    training_times: list; inference_times: list; training_stats: dict; inference_stats: dict
    samples: int; features: int; status: str; evidence: list
    speedup_vs_cpu: Optional[float] = None
    def to_dict(self) -> dict:
        return asdict(self)

def _run_repeated(fn, *, warmup: int, runs: int):
    last_meta = {}; train_ts = []; pred_ts = []
    for i in range(warmup + runs):
        meta = fn(); last_meta = meta
        if i >= warmup:
            train_ts.append(meta["training_time_sec"]); pred_ts.append(meta["inference_time_sec"])
    return train_ts, pred_ts, last_meta

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT), text=True).strip()
    except Exception:
        return "unknown"

def benchmark_suite(*, n_rows: int = 2000, n_features: int = 40, seed: int = 42,
                    warmup: int = 1, runs: int = 3, n_estimators: int = 100,
                    force_device: Optional[str] = None) -> dict:
    env = check_environment()
    gpu_ok, gpu_name, vram, cuda = detect_gpu()
    smi0 = nvidia_smi_query()
    X, y = make_xy(n_rows, n_features, seed)
    backends = inspect_backends()
    results = []

    def cpu_primary():
        return train_lgbm(X, y, device="cpu", n_estimators=n_estimators, seed=seed)
    tr, pr, meta = _run_repeated(cpu_primary, warmup=warmup, runs=runs)
    cpu_primary_median = statistics.median(tr) if tr else float("nan")
    results.append(ModelBenchResult("PRIMARY", "lightgbm", "cpu", meta.get("device_used", "cpu"),
        gpu_ok, gpu_name, smi0.get("utilization_gpu") if smi0.get("status") == "OK" else None, None,
        smi0.get("memory_used_mb") if smi0.get("status") == "OK" else None, None,
        tr, pr, _stats(tr), _stats(pr), n_rows, n_features, "CPU_OK", ["forced device=cpu"]))

    want_gpu = (force_device == "gpu" or force_device is None) and gpu_ok
    if force_device == "cpu":
        want_gpu = False
    if want_gpu:
        smi_b = nvidia_smi_query()
        util_peak = smi_b.get("utilization_gpu") if smi_b.get("status") == "OK" else None
        mem_peak = smi_b.get("memory_used_mb") if smi_b.get("status") == "OK" else None
        def gpu_primary():
            return train_lgbm(X, y, device="gpu", n_estimators=n_estimators, seed=seed)
        tr_g, pr_g, meta_g = _run_repeated(gpu_primary, warmup=warmup, runs=runs)
        smi_a = nvidia_smi_query()
        if smi_a.get("status") == "OK":
            if util_peak is not None and smi_a.get("utilization_gpu") is not None:
                util_peak = max(util_peak, smi_a["utilization_gpu"])
            if mem_peak is not None and smi_a.get("memory_used_mb") is not None:
                mem_peak = max(mem_peak, smi_a["memory_used_mb"])
        gpu_median = statistics.median(tr_g) if tr_g else float("nan")
        speedup = (cpu_primary_median / gpu_median) if gpu_median and gpu_median > 0 else None
        status = meta_g.get("status", "OK")
        if meta_g.get("device_used") == "gpu" and status == "OK":
            status = "GPU_CONFIRMED" if (smi_a.get("status") == "OK" and (util_peak or 0) > 0) else "GPU_REQUESTED_NO_UTIL_PROOF"
            evidence = [f"device_used={meta_g.get('device_used')}", f"smi={smi_a.get('status')}"]
        else:
            status = meta_g.get("status", "CPU_FALLBACK")
            evidence = [meta_g.get("reason") or "gpu_train_failed_fallback"]
        results.append(ModelBenchResult("PRIMARY", "lightgbm", "gpu", meta_g.get("device_used", "cpu"),
            gpu_ok, gpu_name, smi_b.get("utilization_gpu") if smi_b.get("status") == "OK" else None, util_peak,
            smi_b.get("memory_used_mb") if smi_b.get("status") == "OK" else None, mem_peak,
            tr_g, pr_g, _stats(tr_g), _stats(pr_g), n_rows, n_features, status, evidence,
            float(speedup) if speedup is not None else None))
    else:
        results.append(ModelBenchResult("PRIMARY", "lightgbm", "gpu", "cpu", gpu_ok, gpu_name,
            None, None, None, None, [], [], {}, {}, n_rows, n_features,
            "GPU_BLOCKED" if not gpu_ok else "SKIPPED", ["gpu_not_available_or_force_cpu"]))

    def light_cpu():
        return train_lgbm(X, y, device="cpu", n_estimators=max(20, n_estimators // 5), seed=seed)
    tr_l, pr_l, _ = _run_repeated(light_cpu, warmup=warmup, runs=runs)
    results.append(ModelBenchResult("LIGHTWEIGHT", "lightgbm", "cpu", "cpu", gpu_ok, gpu_name,
        None, None, None, None, tr_l, pr_l, _stats(tr_l), _stats(pr_l), n_rows, n_features, "CPU_OK", ["n_estimators_reduced"]))

    def meta_fn():
        return train_rf_meta(X, y, seed=seed)
    tr_m, pr_m, _ = _run_repeated(meta_fn, warmup=warmup, runs=runs)
    results.append(ModelBenchResult("PRIMARY_META", "sklearn.RandomForest", "cpu", "cpu", gpu_ok, gpu_name,
        None, None, None, None, tr_m, pr_m, _stats(tr_m), _stats(pr_m), n_rows, n_features, "NOT_APPLICABLE",
        ["RandomForest has no GPU path in frozen MetaLabelGovernor"]))

    def ens_fn():
        a = train_lgbm(X, y, device="cpu", n_estimators=n_estimators, seed=seed)
        b = train_rf_meta(X, y, seed=seed)
        return {"backend": "lightgbm+sklearn", "device_requested": "cpu", "device_used": "cpu",
                "training_time_sec": a["training_time_sec"] + b["training_time_sec"],
                "inference_time_sec": a["inference_time_sec"] + b["inference_time_sec"],
                "status": "CPU_OK", "reason": "frozen ensemble path is CPU sequential",
                "n_estimators": n_estimators, "samples": n_rows, "features": n_features}
    tr_e, pr_e, _ = _run_repeated(ens_fn, warmup=warmup, runs=runs)
    results.append(ModelBenchResult("ENSEMBLE", "lightgbm+sklearn", "cpu", "cpu", gpu_ok, gpu_name,
        None, None, None, None, tr_e, pr_e, _stats(tr_e), _stats(pr_e), n_rows, n_features, "CPU_OK",
        ["no frozen GPU ensemble; sequential CPU"]))

    runtime_records = []
    try:
        from src.python.governor.runtime_estimator import RuntimeMemory, estimate_runtime
        rm = RuntimeMemory(path=Path("artifacts/colab_candidates/runtime_memory_bench.json")); rm.load()
        for r in results:
            if not r.training_times:
                continue
            med = r.training_stats.get("median") or statistics.median(r.training_times)
            est = estimate_runtime(action="PRIMARY", n_rows=n_rows, n_features=n_features,
                gpu_available=gpu_ok and r.device_used == "gpu", vram_gb=float(vram or 0),
                memory=rm if rm.valid else None)
            rm.record(action=r.model_name, n_rows=n_rows, n_features=n_features,
                estimated_seconds=est.estimated_seconds, actual_seconds=float(med),
                hardware="gpu" if r.device_used == "gpu" else "cpu")
            runtime_records.append({"model": r.model_name, "estimated": est.estimated_seconds,
                "actual_median": med, "error": abs(est.estimated_seconds - float(med))})
    except Exception as e:
        runtime_records.append({"error": str(e)[:120]})

    return {
        "commit": _git_sha(), "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "SYNTHETIC", "market_performance": "UNVERIFIED",
        "hardware": {"gpu_available": gpu_ok, "gpu_name": gpu_name, "vram_gb": vram,
            "cuda_available": cuda, "cpu_cores": env.cpu_count, "ram_gb": env.ram_gb, "nvidia_smi": smi0},
        "backends": [b.to_dict() for b in backends],
        "config": {"n_rows": n_rows, "n_features": n_features, "seed": seed, "warmup": warmup,
                   "runs": runs, "n_estimators": n_estimators, "force_device": force_device},
        "benchmarks": [r.to_dict() for r in results],
        "runtime_learning": runtime_records,
        "governor": {"policy": "utility_based", "gpu_does_not_force_all_models": True},
        "production_unchanged": True, "promotion": False,
        "evidence_global": [f"gpu_available={gpu_ok}", f"gpu_name={gpu_name}", f"nvidia_smi={smi0.get('status')}"],
    }

def human_summary(report: dict) -> str:
    hw = report["hardware"]
    lines = ["=== IDX CPU vs T4 BENCHMARK ===",
        f"Commit            : {report['commit'][:12]}",
        f"Dataset           : {report['dataset_type']} (market={report['market_performance']})",
        f"GPU available     : {hw.get('gpu_available')} name={hw.get('gpu_name') or 'N/A'} VRAM={hw.get('vram_gb')}",
        f"nvidia-smi        : {hw.get('nvidia_smi', {}).get('status')}", "",
        f"{'Model':<14} {'Device':<8} {'Train med':>10} {'Status':<24} {'Speedup':>8}", "-" * 72]
    for b in report["benchmarks"]:
        med = b.get("training_stats", {}).get("median")
        med_s = f"{med:.4f}" if isinstance(med, (int, float)) and med == med else "N/A"
        sp = b.get("speedup_vs_cpu")
        sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) and sp == sp else "N/A"
        lines.append(f"{b['model_name']:<14} {b['device_used']:<8} {med_s:>10} {b['status']:<24} {sp_s:>8}")
    lines += ["", f"Production change : NONE", f"Promotion         : REJECTED",
              "Governor          : utility-based (GPU does not force all models)"]
    return "\n".join(lines)

def main() -> None:
    p = argparse.ArgumentParser(description="IDX CPU vs GPU backend benchmark")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--dataset-size", type=int, default=2000)
    p.add_argument("--features", type=int, default=40)
    p.add_argument("--output", default="artifacts/colab_candidates/benchmark_report.json")
    args = p.parse_args()
    force = None if args.device == "auto" else args.device
    if force == "gpu":
        ok, _, _, _ = detect_gpu()
        if not ok:
            report = {"commit": _git_sha(), "dataset_type": "SYNTHETIC", "market_performance": "UNVERIFIED",
                "hardware": {"gpu_available": False}, "benchmarks": [], "production_unchanged": True,
                "promotion": False, "status": "GPU_BLOCKED"}
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2))
            print("GPU_BLOCKED: no GPU available"); sys.exit(0)
    report = benchmark_suite(n_rows=args.dataset_size, n_features=args.features, seed=args.seed,
                             warmup=args.warmup, runs=args.runs, force_device=force)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, default=str))
    print(human_summary(report)); print(f"\nWrote {args.output}")

if __name__ == "__main__":
    main()

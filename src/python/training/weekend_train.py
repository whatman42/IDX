"""Weekend training under hard deadline. Failure → production UNCHANGED."""
from __future__ import annotations
import json, os, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any
import typer
from src.python.governor.governor import MLGovernor, ResourceProfile
from src.python.registry.promotion import evaluate_promotion
from src.python.scheduler.schedule import ScheduleType, SystemClock, TrainingDeadline, classify_schedule, training_run_key

app = typer.Typer()

def _stage_from_plan(stage: str) -> str:
    if stage in ("exploration", "validation"):
        return stage
    plan = classify_schedule(SystemClock())
    if plan.schedule_type == ScheduleType.SATURDAY_EXPLORATION:
        return "exploration"
    if plan.schedule_type == ScheduleType.SUNDAY_VALIDATION:
        return "validation"
    return "exploration"

def _safe_train_once(out_dir: Path, budget_sec: float) -> dict[str, Any]:
    t0 = time.monotonic()
    result: dict[str, Any] = {"status": "FAILED", "promoted": False, "metrics": {}, "reason": "", "runtime_sec": 0.0}
    try:
        remaining = budget_sec - (time.monotonic() - t0)
        if remaining < 30:
            result["status"] = "TIMEOUT"
            result["reason"] = "insufficient_budget"
            return result
        cmd = [sys.executable, "-m", "src.python.training.train_candidate", "--out-dir", str(out_dir), "--min-oos-accuracy", "0.0"]
        proc = subprocess.run(cmd, timeout=max(10, int(remaining - 15)), capture_output=True, text=True)
        result["runtime_sec"] = time.monotonic() - t0
        result["stdout_tail"] = (proc.stdout or "")[-500:]
        result["stderr_tail"] = (proc.stderr or "")[-500:]
        if proc.returncode != 0:
            result["status"] = "FAILED"
            result["reason"] = f"train_exit_{proc.returncode}"
            return result
        result["status"] = "OK"
        result["reason"] = "trained"
        return result
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["reason"] = "subprocess_timeout"
        result["runtime_sec"] = time.monotonic() - t0
        return result
    except Exception as e:
        result["status"] = "FAILED"
        result["reason"] = f"{type(e).__name__}: {e}"
        result["runtime_sec"] = time.monotonic() - t0
        result["trace"] = traceback.format_exc()[-800:]
        return result

@app.command()
def main(stage: str = typer.Option("auto"), budget_sec: int = typer.Option(1200),
         out_dir: str = typer.Option("models/candidates")) -> None:
    clock = SystemClock()
    deadline = TrainingDeadline(internal_budget_sec=budget_sec)
    deadline.start(clock)
    resolved = _stage_from_plan(stage)
    run_date = clock.now().date()
    run_id = training_run_key(run_date, resolved)
    resources = ResourceProfile.detect()
    resources.training_budget_sec = budget_sec
    gov = MLGovernor(resources=resources)
    plan = gov.training_plan(deadline.remaining_sec(clock))
    state_dir = Path(os.getenv("IDX_STATE_DIR", "state"))
    runs_dir = state_dir / "training_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "training_run_id": run_id, "stage": resolved, "started_at": clock.now().isoformat(),
        "budget_sec": budget_sec, "training_plan": plan, "status": "RUNNING",
        "promoted": False, "production_unchanged": True,
    }
    if not plan["allow_train"]:
        record["status"] = "SKIPPED"
        record["reason"] = "budget_too_low"
        (runs_dir / f"{run_id.replace(':', '_')}.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
        raise SystemExit(0)
    train_result = _safe_train_once(out, deadline.remaining_sec(clock))
    record["train_result"] = train_result
    record["finished_at"] = clock.now().isoformat()
    record["runtime_sec"] = train_result.get("runtime_sec", 0)
    if train_result["status"] != "OK":
        record["status"] = train_result["status"]
        record["reason"] = train_result.get("reason", "train_failed")
        (runs_dir / f"{run_id.replace(':', '_')}.json").write_text(json.dumps(record, indent=2, default=str))
        print(json.dumps(record, indent=2, default=str))
        raise SystemExit(0)
    if resolved == "validation" and not deadline.expired(clock):
        metrics = train_result.get("metrics") or {"accuracy": 0.5, "expectancy": 0.0, "max_drawdown": 0.2, "n_samples": 50}
        report = evaluate_promotion(metrics, min_accuracy=0.52, min_expectancy=0.0, max_drawdown=0.25, min_calibration_ok=True)
        record["promotion_report"] = {"approved": report.approved, "reason": report.reason}
        if report.approved:
            record["promoted"] = True
            record["production_unchanged"] = False
            record["status"] = "PROMOTED"
        else:
            record["status"] = "TRAINED_NOT_PROMOTED"
            record["reason"] = report.reason
    else:
        record["status"] = "EXPLORED" if resolved == "exploration" else "TRAINED_NO_PROMOTE_WINDOW"
        record["reason"] = "exploration_no_promote" if resolved == "exploration" else "deadline_or_stage"
    (runs_dir / f"{run_id.replace(':', '_')}.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps(record, indent=2, default=str))

if __name__ == "__main__":
    app()

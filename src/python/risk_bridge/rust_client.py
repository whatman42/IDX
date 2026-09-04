"""Python ↔ Rust risk engine boundary with validated contract."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

CONTRACT_VERSION = "1.0.0"


@dataclass
class RiskDecision:
    allow: bool
    reason: str
    final_weight: float
    kill_switch: bool = False
    contract_version: str = CONTRACT_VERSION
    raw: Optional[dict] = None


class RustRiskError(Exception):
    pass


def _validate_decision(d: dict) -> RiskDecision:
    if "allow" not in d or "reason" not in d:
        raise RustRiskError(f"Malformed risk decision: {d}")
    fw = float(d.get("final_weight", 0.0))
    if fw != fw or fw < 0:
        raise RustRiskError(f"Invalid final_weight: {fw}")
    return RiskDecision(allow=bool(d["allow"]), reason=str(d["reason"]), final_weight=fw,
                        kill_switch=bool(d.get("kill_switch", False)), raw=d)


def invoke_rust_risk(
    signal: dict[str, Any], portfolio: dict[str, Any], *,
    binary: Optional[str] = None, timeout_sec: float = 10.0, work_dir: Path | str = "/tmp",
) -> RiskDecision:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    signal_path = work_dir / "signal.json"
    portfolio_path = work_dir / "portfolio.json"
    output_path = work_dir / "risk_decision.json"

    rust_signal = {
        "timestamp": str(signal.get("timestamp", "")),
        "ticker": str(signal.get("symbol") or signal.get("ticker") or ""),
        "side": int(signal.get("side", 0)),
        "raw_proba": float(signal.get("primary_probability") or signal.get("raw_proba") or 0.0),
        "meta_proba": float(signal.get("meta_probability") or signal.get("meta_proba") or 0.0),
        "suggested_weight": float(signal.get("suggested_size") or signal.get("suggested_weight") or 0.0),
        "mode": str(signal.get("mode", "paper")),
    }
    for k, v in rust_signal.items():
        if isinstance(v, float) and v != v:
            return RiskDecision(allow=False, reason=f"NaN in signal.{k}", final_weight=0.0, kill_switch=True)

    rust_portfolio = {
        "equity": float(portfolio.get("equity", 0)), "cash": float(portfolio.get("cash", 0)),
        "positions": portfolio.get("positions", {}),
        "daily_pnl_pct": float(portfolio.get("daily_pnl_pct", 0)),
        "max_drawdown_pct": float(portfolio.get("max_drawdown_pct", 0)),
    }
    if rust_portfolio["equity"] != rust_portfolio["equity"]:
        return RiskDecision(allow=False, reason="NaN equity", final_weight=0.0, kill_switch=True)

    signal_path.write_text(json.dumps(rust_signal))
    portfolio_path.write_text(json.dumps(rust_portfolio))

    bin_path = binary or _find_binary()
    if not bin_path:
        return RiskDecision(allow=False, reason="Rust risk binary not found — NO NEW TRADE (fail-closed)",
                            final_weight=0.0, kill_switch=False)

    try:
        proc = subprocess.run(
            [bin_path, "--signal-file", str(signal_path), "--portfolio-file", str(portfolio_path),
             "--output", str(output_path)],
            capture_output=True, text=True, timeout=timeout_sec, check=False,
        )
    except subprocess.TimeoutExpired:
        return RiskDecision(allow=False, reason="Rust risk timeout", final_weight=0.0, kill_switch=True)
    except Exception as e:
        return RiskDecision(allow=False, reason=f"Rust invoke error: {e}", final_weight=0.0, kill_switch=True)

    if proc.returncode != 0:
        return RiskDecision(allow=False, reason=f"Rust non-zero exit {proc.returncode}: {proc.stderr[:200]}",
                            final_weight=0.0, kill_switch=True)
    if not output_path.exists():
        return RiskDecision(allow=False, reason="Rust produced no output file", final_weight=0.0, kill_switch=True)
    try:
        return _validate_decision(json.loads(output_path.read_text()))
    except Exception as e:
        return RiskDecision(allow=False, reason=f"Invalid Rust JSON: {e}", final_weight=0.0, kill_switch=True)


def _find_binary() -> Optional[str]:
    for c in ["risk_engine/target/release/risk_engine", "risk_engine/target/debug/risk_engine", shutil.which("risk_engine")]:
        if c and Path(c).exists():
            return str(c)
    return None

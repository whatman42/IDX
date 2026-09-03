"""
Main trading cycle orchestrator (called from GitHub Actions).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv

app = typer.Typer()


@app.command()
def main(mode: str = "paper"):
    load_dotenv()

    print(f"[Cycle] Starting in mode={mode}")

    # 1. Load / generate features (stub)
    # features = build_features(...)

    # 2. Primary side model inference
    # side_proba = primary.predict_proba(features)

    # 3. Meta-labeling size
    # size_proba = governor.predict_proba(...)
    # weights = governor.size_from_proba(size_proba)

    # 4. Write signal for Rust risk engine
    signal = {
        "timestamp": "2026-09-03T12:00:00Z",
        "ticker": "BBCA",
        "side": 1,
        "raw_proba": 0.68,
        "meta_proba": 0.74,
        "suggested_weight": 0.12,
        "mode": mode,
    }

    out = Path("/tmp/signal.json")
    out.write_text(json.dumps(signal, indent=2))
    print(f"[Cycle] Signal written → {out}")

    # Portfolio snapshot stub
    portfolio = {
        "equity": 100_000_000,
        "cash": 85_000_000,
        "positions": {},
        "daily_pnl_pct": 0.0042,
        "max_drawdown_pct": 0.031,
    }
    Path("/tmp/portfolio.json").write_text(json.dumps(portfolio, indent=2))

    print("[Cycle] Done")


if __name__ == "__main__":
    app()

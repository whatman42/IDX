"""
Vectorized paper-trading engine.
Uses portfolio weights rather than event-driven loop (fast on GH Actions CPU).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(risk_decision: str = "/tmp/risk_decision.json"):
    decision = json.loads(Path(risk_decision).read_text())

    if not decision.get("allow", False):
        print(f"[Paper] Blocked by risk engine: {decision.get('reason')}")
        return

    signal = json.loads(Path("/tmp/signal.json").read_text())
    weight = signal["suggested_weight"]
    ticker = signal["ticker"]
    side = signal["side"]

    print(f"[Paper] Executing {ticker} side={side} weight={weight:.4f}")
    # Actual vectorbt / Numba portfolio update goes here

    print("[Paper] Position updated (simulated)")


if __name__ == "__main__":
    app()

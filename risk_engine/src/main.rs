//! Deterministic Risk Engine
//!
//! Receives signal + portfolio state from Python layer,
//! applies hard guardrails, and returns allow/deny decision.
//!
//! Designed for zero GC pauses and absolute determinism.

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to signal JSON produced by ML pipeline
    #[arg(long)]
    signal_file: PathBuf,

    /// Path to current portfolio snapshot
    #[arg(long)]
    portfolio_file: PathBuf,

    /// Output path for risk decision
    #[arg(long, default_value = "/tmp/risk_decision.json")]
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Signal {
    timestamp: String,
    ticker: String,
    side: i8,
    raw_proba: f64,
    meta_proba: f64,
    suggested_weight: f64,
    mode: String,
}

#[derive(Debug, Deserialize)]
struct Portfolio {
    equity: f64,
    cash: f64,
    positions: serde_json::Value,
    daily_pnl_pct: f64,
    max_drawdown_pct: f64,
}

#[derive(Debug, Serialize)]
struct RiskDecision {
    allow: bool,
    reason: String,
    final_weight: f64,
    kill_switch: bool,
}

/// Hard limits – these are constitutional, not tunable at runtime.
const MAX_DAILY_DRAWDOWN: f64 = 0.03; // 3%
const MAX_POSITION_WEIGHT: f64 = 0.20;
const MAX_PORTFOLIO_DRAWDOWN: f64 = 0.12;

fn evaluate(signal: &Signal, portfolio: &Portfolio) -> RiskDecision {
    // 1. Daily drawdown kill-switch
    if portfolio.daily_pnl_pct < -MAX_DAILY_DRAWDOWN {
        return RiskDecision {
            allow: false,
            reason: format!(
                "Daily drawdown {:.2}% breached limit {:.2}%",
                portfolio.daily_pnl_pct * 100.0,
                MAX_DAILY_DRAWDOWN * 100.0
            ),
            final_weight: 0.0,
            kill_switch: true,
        };
    }

    // 2. Portfolio-level max drawdown
    if portfolio.max_drawdown_pct > MAX_PORTFOLIO_DRAWDOWN {
        return RiskDecision {
            allow: false,
            reason: format!(
                "Max drawdown {:.2}% > hard limit {:.2}%",
                portfolio.max_drawdown_pct * 100.0,
                MAX_PORTFOLIO_DRAWDOWN * 100.0
            ),
            final_weight: 0.0,
            kill_switch: true,
        };
    }

    // 3. Position size clamp
    let mut weight = signal.suggested_weight.abs().min(MAX_POSITION_WEIGHT);

    // 4. Contradictory position check (simplified)
    // In full version we inspect portfolio.positions for opposite side on same ticker

    // 5. Minimum confidence gate
    if signal.meta_proba < 0.55 {
        return RiskDecision {
            allow: false,
            reason: format!("Meta probability {:.3} below threshold 0.55", signal.meta_proba),
            final_weight: 0.0,
            kill_switch: false,
        };
    }

    RiskDecision {
        allow: true,
        reason: "All guardrails passed".to_string(),
        final_weight: weight * signal.side.signum() as f64,
        kill_switch: false,
    }
}

fn main() {
    let args = Args::parse();

    let signal_str = fs::read_to_string(&args.signal_file)
        .expect("Failed to read signal file");
    let signal: Signal = serde_json::from_str(&signal_str)
        .expect("Invalid signal JSON");

    let portfolio_str = fs::read_to_string(&args.portfolio_file)
        .expect("Failed to read portfolio file");
    let portfolio: Portfolio = serde_json::from_str(&portfolio_str)
        .expect("Invalid portfolio JSON");

    let decision = evaluate(&signal, &portfolio);

    let out = serde_json::to_string_pretty(&decision).unwrap();
    fs::write(&args.output, out).expect("Failed to write decision");

    println!(
        "[RiskEngine] allow={} | reason={} | weight={:.4}",
        decision.allow, decision.reason, decision.final_weight
    );

    // Exit code can be used by the workflow if desired
    if decision.kill_switch {
        std::process::exit(2);
    }
}

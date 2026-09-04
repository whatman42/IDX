//! Deterministic Risk Engine — integrity boundary. Contract 1.0.0
//! Args: --signal-file PATH --portfolio-file PATH [--output PATH]

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::process;

const CONTRACT_VERSION: &str = "1.0.0";
const MAX_DAILY_DRAWDOWN: f64 = 0.03;
const MAX_POSITION_WEIGHT: f64 = 0.20;
const MAX_PORTFOLIO_DRAWDOWN: f64 = 0.12;
const MIN_META_PROBA: f64 = 0.55;

#[derive(Debug, Deserialize)]
struct Signal {
    #[serde(default)]
    timestamp: String,
    #[serde(alias = "symbol", default)]
    ticker: String,
    side: i8,
    #[serde(alias = "primary_probability", alias = "raw_proba")]
    raw_proba: f64,
    #[serde(alias = "meta_probability", alias = "meta_proba")]
    meta_proba: f64,
    #[serde(alias = "suggested_size", alias = "suggested_weight")]
    suggested_weight: f64,
    #[serde(default)]
    mode: String,
}

#[derive(Debug, Deserialize)]
struct Portfolio {
    equity: f64,
    cash: f64,
    #[serde(default)]
    positions: serde_json::Value,
    #[serde(default)]
    daily_pnl_pct: f64,
    #[serde(default)]
    max_drawdown_pct: f64,
}

#[derive(Debug, Serialize)]
struct RiskDecision {
    allow: bool,
    reason: String,
    final_weight: f64,
    kill_switch: bool,
    contract_version: String,
}

fn is_finite(x: f64) -> bool {
    x.is_finite()
}

pub fn evaluate(signal: &Signal, portfolio: &Portfolio) -> RiskDecision {
    if !is_finite(signal.raw_proba)
        || !is_finite(signal.meta_proba)
        || !is_finite(signal.suggested_weight)
        || !is_finite(portfolio.equity)
        || !is_finite(portfolio.cash)
        || !is_finite(portfolio.daily_pnl_pct)
        || !is_finite(portfolio.max_drawdown_pct)
    {
        return RiskDecision {
            allow: false,
            reason: "NaN/Inf in signal or portfolio".into(),
            final_weight: 0.0,
            kill_switch: true,
            contract_version: CONTRACT_VERSION.into(),
        };
    }
    if signal.side != 1 && signal.side != -1 && signal.side != 0 {
        return RiskDecision {
            allow: false,
            reason: format!("Invalid side {}", signal.side),
            final_weight: 0.0,
            kill_switch: true,
            contract_version: CONTRACT_VERSION.into(),
        };
    }
    if portfolio.equity < 0.0 || portfolio.cash < -1e-9 {
        return RiskDecision {
            allow: false,
            reason: "Negative equity/cash".into(),
            final_weight: 0.0,
            kill_switch: true,
            contract_version: CONTRACT_VERSION.into(),
        };
    }
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
            contract_version: CONTRACT_VERSION.into(),
        };
    }
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
            contract_version: CONTRACT_VERSION.into(),
        };
    }
    if signal.meta_proba < MIN_META_PROBA {
        return RiskDecision {
            allow: false,
            reason: format!(
                "Meta probability {:.3} below threshold {:.2}",
                signal.meta_proba, MIN_META_PROBA
            ),
            final_weight: 0.0,
            kill_switch: false,
            contract_version: CONTRACT_VERSION.into(),
        };
    }
    let weight = signal.suggested_weight.abs().min(MAX_POSITION_WEIGHT);
    let signed = if signal.side == 0 {
        0.0
    } else {
        weight * (signal.side as f64).signum()
    };
    RiskDecision {
        allow: true,
        reason: "All guardrails passed".into(),
        final_weight: signed,
        kill_switch: false,
        contract_version: CONTRACT_VERSION.into(),
    }
}

fn parse_args() -> (String, String, String) {
    let args: Vec<String> = env::args().collect();
    let mut signal = String::new();
    let mut portfolio = String::new();
    let mut output = "/tmp/risk_decision.json".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--signal-file" if i + 1 < args.len() => {
                signal = args[i + 1].clone();
                i += 2;
            }
            "--portfolio-file" if i + 1 < args.len() => {
                portfolio = args[i + 1].clone();
                i += 2;
            }
            "--output" if i + 1 < args.len() => {
                output = args[i + 1].clone();
                i += 2;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                process::exit(1);
            }
        }
    }
    if signal.is_empty() || portfolio.is_empty() {
        eprintln!("Usage: risk_engine --signal-file PATH --portfolio-file PATH [--output PATH]");
        process::exit(1);
    }
    (signal, portfolio, output)
}

fn main() {
    let (signal_path, portfolio_path, output_path) = parse_args();
    let signal_str = fs::read_to_string(&signal_path).unwrap_or_else(|e| {
        eprintln!("Failed to read signal: {e}");
        process::exit(1);
    });
    let signal: Signal = serde_json::from_str(&signal_str).unwrap_or_else(|e| {
        eprintln!("Invalid signal JSON: {e}");
        process::exit(1);
    });
    let portfolio_str = fs::read_to_string(&portfolio_path).unwrap_or_else(|e| {
        eprintln!("Failed to read portfolio: {e}");
        process::exit(1);
    });
    let portfolio: Portfolio = serde_json::from_str(&portfolio_str).unwrap_or_else(|e| {
        eprintln!("Invalid portfolio JSON: {e}");
        process::exit(1);
    });
    let decision = evaluate(&signal, &portfolio);
    let out = serde_json::to_string_pretty(&decision).unwrap();
    fs::write(&output_path, &out).unwrap_or_else(|e| {
        eprintln!("Failed to write decision: {e}");
        process::exit(1);
    });
    println!(
        "[RiskEngine] allow={} | reason={} | weight={:.4} | v={}",
        decision.allow, decision.reason, decision.final_weight, decision.contract_version
    );
    if decision.kill_switch {
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(meta: f64, weight: f64) -> Signal {
        Signal {
            timestamp: "t".into(),
            ticker: "BBCA".into(),
            side: 1,
            raw_proba: 0.6,
            meta_proba: meta,
            suggested_weight: weight,
            mode: "paper".into(),
        }
    }
    fn pf(dd: f64, max_dd: f64) -> Portfolio {
        Portfolio {
            equity: 1e8,
            cash: 1e8,
            positions: serde_json::json!({}),
            daily_pnl_pct: dd,
            max_drawdown_pct: max_dd,
        }
    }

    #[test]
    fn allow_clean() {
        let d = evaluate(&sig(0.70, 0.10), &pf(0.0, 0.01));
        assert!(d.allow);
        assert!((d.final_weight - 0.10).abs() < 1e-9);
    }

    #[test]
    fn deny_low_meta() {
        assert!(!evaluate(&sig(0.40, 0.10), &pf(0.0, 0.01)).allow);
    }

    #[test]
    fn deny_nan() {
        let mut s = sig(0.70, 0.10);
        s.meta_proba = f64::NAN;
        let d = evaluate(&s, &pf(0.0, 0.01));
        assert!(!d.allow && d.kill_switch);
    }

    #[test]
    fn clamp_weight() {
        let d = evaluate(&sig(0.70, 0.50), &pf(0.0, 0.01));
        assert!(d.allow);
        assert!((d.final_weight - MAX_POSITION_WEIGHT).abs() < 1e-9);
    }
}

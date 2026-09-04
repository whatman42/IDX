# IDX Autonomous Quantitative Trading System

Institutional-grade, **100% serverless** quantitative trading engine for IDX (Indonesia Stock Exchange).

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| **Feature Engine** | Python + Polars | Causal multi-symbol feature store |
| **Side Model** | Python + LightGBM | Directional alpha (probability of trend) |
| **Size Model** | Python + Meta-Labeling (RF) | Position sizing + false-positive filter |
| **Labeling** | Triple-Barrier Method | Path-dependent PT / SL / vertical labels |
| **Risk Engine** | Rust | Deterministic kill-switches, drawdown |
| **State** | Turso (libSQL) | Ephemeral → persistent sync |
| **Orchestration** | GitHub Actions + external cron | Zero idle cost |

## Architecture Principles

1. No traditional servers — GitHub Actions runners only.
2. No GitHub native cron — external `repository_dispatch`.
3. Python owns ML, Rust owns risk.
4. Triple-Barrier Method only — no static time labels.
5. Strict causality — features at *t* never use data after *t*.

## Feature Engineering

Polars-first causal pipeline. Groups: returns, trend, volatility, momentum, volume, structure (37 features). Warm-up drop, multi-symbol isolation, `feature_schema_version=1.0.0`.

## ML Layer (Primary Side + Meta-Labeling)

### Primary Side (LightGBM)
- Input: causal features
- Output: `P(up)` → side `+1` / `-1`
- Temporal train/val/test with **purge + embargo** for TBM label overlap
- Class imbalance via `is_unbalance=True`
- Persistence: `primary_lgbm_vNNN.{txt,meta.json}`

### Meta-Labeling (Random Forest)
- Answers: "is the primary signal worth taking?"
- Calibration: **Platt** or **Isotonic** on chronological holdout
- Gate: `meta_probability >= threshold` (default 0.55)
- Sizing: sigmoid or fractional Kelly, clamped — **Risk Engine is final authority**

### Inference contract
```
timestamp | symbol | side | primary_probability | meta_probability
accepted | confidence | suggested_size | model_version | feature_schema_version
```

### Limitations
- Training is **manual / offline** — not scheduled weekly yet
- No ONNX export yet
- Metrics are diagnostic; full vectorized backtest is later

## Quick Start

```bash
git clone https://github.com/whatman42/idx.git && cd idx
pip install -r requirements.txt
pytest -q   # 65+ tests green
```

## Status

- [x] Serverless Actions + Rust risk boilerplate
- [x] Full Triple-Barrier Method + tests
- [x] Causal Feature Engineering Pipeline + tests
- [x] Primary Side + Meta-Labeling + calibration + temporal validation
- [ ] ONNX export + Rust ort inference
- [ ] Turso state integration
- [ ] Live IDX data adapters

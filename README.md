# IDX Autonomous Quantitative Trading System

Institutional-grade, **100% serverless** quantitative trading engine for IDX (Indonesia Stock Exchange).

Built with a strict separation of concerns:

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| **Feature Engine** | Python + Polars | Causal multi-symbol feature store |
| **Side Model** | Python + LightGBM | Directional alpha (probability of trend) |
| **Size Model** | Python + Meta-Labeling (RF/XGBoost) | Position sizing + false-positive filter |
| **Labeling** | Triple-Barrier Method | Path-dependent PT / SL / vertical labels |
| **Risk Engine** | Rust | Deterministic kill-switches, drawdown, regime guard |
| **State** | Turso (libSQL Embedded Replicas) | Ephemeral → persistent sync |
| **Orchestration** | GitHub Actions + external cron | Zero idle cost |
| **Execution** | Vectorized paper trading | Portfolio weight simulation |
| **Reporting** | Telegram (in-memory BytesIO) | Unidirectional push |

## Architecture Principles

1. **No traditional servers** — everything runs on GitHub Actions runners (ephemeral).
2. **No GitHub native cron** — external trigger via `repository_dispatch` (Cron-job.org / similar).
3. **Python owns ML**, Rust owns risk — ONNX bridge for sub-microsecond inference later.
4. **Triple-Barrier Method** only — no static time-based labeling.
5. **Strict causality** — features at time *t* never use data after *t*.
6. **In-memory visualization** — never write `.png` to disk; stream via `io.BytesIO` + Telegram multipart.

## Feature Engineering

Polars-first causal pipeline (`src/python/features/`).

### Feature groups

| Group | Examples |
|-------|----------|
| **Returns** | ret_1/3/5/10/20, log_ret_1, cum_ret_20 |
| **Trend** | SMA/EMA 5/10/20, price_vs_sma_20, ema_spread, trend_slope |
| **Volatility** | rolling std, ATR-14, NATR, vol_ratio, range_expansion |
| **Momentum** | RSI-14, ROC 5/10, momentum 10 |
| **Volume** | vol change, rel volume, z-score, pv interaction |
| **Structure** | HL range, close location, gap, distance to rolling high/low |

### Guarantees

- **Causality**: all rolling / ewm / shift operations are trailing and partitioned by `symbol`.
- **Warm-up**: first `MAX_LOOKBACK` (20) rows per symbol are dropped in `build_ml_dataset`.
- **No silent fill**: null/inf rows are rejected, not zero-filled.
- **Multi-symbol isolation**: BBCA features never see TLKM data.
- **Versioning**: every frame carries `feature_schema_version` (`1.0.0`).
- **Leakage tests**: future-price mutation regression tests in the suite.

### ML-ready interface

```python
from src.python.features import compute_features, build_ml_dataset

feats = compute_features(ohlcv_df)          # full panel + features
ds = build_ml_dataset(ohlcv_df, labels=y)  # {"X", "y", "metadata"}
```

## Quick Start

```bash
git clone https://github.com/whatman42/idx.git
cd idx
pip install -r requirements.txt
cd risk_engine && cargo build --release && cd ..
pytest -q   # should be all green
```

## Triggering the Pipeline

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GH_PAT" \
  https://api.github.com/repos/whatman42/idx/dispatches \
  -d '{"event_type":"run_trading_cycle"}'
```

## Status

- [x] Serverless Actions + Rust risk boilerplate
- [x] Full Triple-Barrier Method + tests
- [x] Causal Feature Engineering Pipeline (Polars) + tests
- [ ] ONNX export + Rust ort inference
- [ ] Turso state integration
- [ ] Live IDX data adapters

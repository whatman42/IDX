# IDX Autonomous Quantitative Signal + Paper Trading System

Institutional-grade, **100% serverless** quant system for Indonesia Stock Exchange (IDX / BEI).

**Signal-only** toward real brokers. Paper execution uses a real ledger (fees, slippage, lots, SL/TP).

## Architecture

```
MARKET DATA → DATA QUALITY → FEATURES (Polars, causal)
  → PRIMARY (LightGBM) → META (RF + calibration)
  → REGIME → ML GOVERNOR → SIGNAL → RUST RISK BOUNDARY
  → PAPER EXECUTION (idempotent) → PORTFOLIO → STATE → TELEGRAM
```

| Layer | Role |
|-------|------|
| Data Quality | Hard reject → NO TRADE |
| Features | 37 causal features, schema versioned |
| TBM | Path-dependent PT/SL/vertical |
| Primary / Meta | LightGBM + RF calibrated gate |
| Regime | HIGH_VOL / LOW_VOL_TREND / MEAN_REVERT |
| ML Governor | Adaptive thresholds × DD × performance × resources |
| Portfolio | Cash, positions, fees, slippage, lot, SL/TP, idempotent IDs |
| Rust Risk | Deterministic boundary |

## Daily production

External cron → `repository_dispatch` → `python -m src.python.pipeline.run_cycle --mode paper`

Target ≤15 min on ~4 CPU / 12 GB. No heavy training on daily runners.

## Quick start

```bash
git clone https://github.com/whatman42/idx.git && cd idx
pip install -r requirements.txt
pytest -q
python -m src.python.pipeline.run_cycle --mode paper
```

## Model promotion

`CANDIDATE → VALIDATED → APPROVED → PRODUCTION → RETIRED` (no destructive overwrite)

## Limitations

Synthetic data unless `data_path` set; Turso is interface-level; no real broker; DL offline only.

## License

Original code. Concepts informed by industry practice (meta-labeling, purged CV). No third-party source copied.

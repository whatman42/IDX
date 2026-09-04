# IDX Autonomous Quantitative Signal + Paper Portfolio Platform

Serverless quant system for Indonesia Stock Exchange (BEI/IDX). **Signal + paper portfolio only**.

## Modes

| Mode | Data | Models | Synthetic |
|------|------|--------|-----------|
| development/test | synthetic or file | production if present else cold-start | yes |
| paper/production | **requires** CSV/Parquet/IDX | **approved artifacts only** | **forbidden** |
| research | any | train candidates | yes |

## Daily production

```
DATA → DQ → FEATURES → LOAD APPROVED MODELS (checksum+schema)
→ INFERENCE → REGIME → GOVERNOR → SIGNAL → RUST RISK (fail-closed)
→ PAPER LEDGER → SQLITE STATE → OUTBOX → TELEGRAM
```

```bash
python -m src.python.pipeline.run_cycle --mode paper --csv-path data/bbca.csv
```

## Weekly research

```bash
python -m src.python.training.train_candidate --out-dir models/candidates --promote
```

## Modules

market, features, labeling, ml, governor, registry (SHA256 manifests), risk_bridge,
portfolio (+reconcile), persistence (SQLite Turso-compatible), notify (outbox),
observability, health, training

## Testing

```bash
pip install -r requirements.txt && pytest -q
```

## Limitations

- Live IDX feed needs licensed endpoint (adapter ready)
- Build Rust: `cd risk_engine && cargo build --release`
- Corporate actions interface-only until real CA feed
- Telegram requires secrets

## License

Original code. Concepts: meta-labeling, purged CV, outbox. No third-party source copied.

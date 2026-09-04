# IDX Autonomous Quantitative Signal + Paper Portfolio Platform

Serverless quant system for Indonesia Stock Exchange (BEI/IDX).
**Signal + paper portfolio only** — no live broker execution.

## Status labels

| Label | Meaning |
|-------|---------|
| ENGINEERING_READY | Architecture, tests, CI paths verified in-repo |
| PAPER_READY | Paper cycle works with CSV/Parquet + approved models |
| PRODUCTION_DATA_REQUIRED | Licensed IDX feed / CA feed not bundled |
| CREDENTIALS_REQUIRED | Turso / Telegram secrets not in repo (by design) |

## Modes

| Mode | Data | Models | Synthetic |
|------|------|--------|-----------|
| development/test | synthetic or file | production if present else cold-start | allowed |
| paper/production | **CSV/Parquet/IDX required** | **approved artifacts only** | **forbidden** |
| research | any | train candidates | allowed |

## Daily production

```
resolve_provider → DQ → features → load approved models (SHA256+schema)
→ regime → ML Governor → signal → Rust risk (fail-closed)
→ paper ledger → StateRepository (SQLite or Turso) → outbox → Telegram
```

```bash
cd risk_engine && cargo test && cargo build --release
python -m src.python.pipeline.run_cycle --mode paper --csv-path data/bbca.csv
```

## Weekly research

```bash
python -m src.python.training.train_candidate --out-dir models/candidates --promote
```

Promotion gates: sample size, OOS accuracy, drawdown, expectancy; rejects accuracy-only perfect scores.

## Persistence authority

- `IDX_STATE_AUTHORITY=auto`: Turso if credentials present, else SQLite
- `turso`: require credentials or fail closed
- Never dual-write. One authority per process.

## Rust risk boundary

Governor sets policy; Rust enforces numerical integrity. Missing binary → NO NEW TRADE.

## CI

`.github/workflows/ci.yml` — pytest + cargo test + release smoke + integration cycle.

## Secrets (never commit)

`TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

## Limitations

- Live IDX API requires licensed endpoint
- Corporate actions: policy + file provider; no fabricated live CA stream
- Telegram/Turso need credentials for live path
- Synthetic OOS ≠ profitability proof

## License

Original implementation. Concepts: meta-labeling, purged CV, outbox. No third-party source copied.

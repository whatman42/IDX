# Architecture Freeze — IDX

**Status:** FROZEN after final hardening pass

## Frozen layers

1. Data → quality → features (causal Polars)
2. Adaptive ML Governor (utility + contextual bandit learning) — strategic only
3. Models primary LightGBM + meta RF (approved artifacts only on weekdays)
4. Signal → Rust risk (deterministic integrity boundary)
5. Paper portfolio (accounting identity, idempotency)
6. Reconciliation fail-closed
7. Gemini advisory (explanation only, no trading authority)
8. Telegram (ID templates + optional Gemini)
9. Persistence SQLite XOR Turso
10. Schedulers weekday EOD production / weekend training (<25 min)

## Explicitly out of scope (post-freeze change request required)

- Live broker execution
- LLM/agent placing orders
- Complex deep RL replacing Governor
- Dual-write databases
- Cold-start training on weekdays
- Hardcoded sole holiday calendar without inject/provider

## Hard safety (immutable)

- No dual-write
- Fail-closed on missing Turso/Gemini/data/models
- Promotion gates
- Rust NaN/Inf rejection
- Accounting: equity = cash + long MV − short MV

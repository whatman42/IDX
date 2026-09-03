# IDX Autonomous Quantitative Trading System

Institutional-grade, **100% serverless** quantitative trading engine for IDX (Indonesia Stock Exchange).

Built with a strict separation of concerns:

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| **Side Model** | Python + LightGBM | Directional alpha (probability of trend) |
| **Size Model** | Python + Meta-Labeling (RF/XGBoost) | Position sizing + false-positive filter |
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
5. **In-memory visualization** — never write `.png` to disk; stream via `io.BytesIO` + Telegram multipart.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/whatman42/idx.git
cd idx

# 2. Python deps
pip install -r requirements.txt

# 3. Rust (for risk engine)
cd risk_engine && cargo build --release && cd ..

# 4. Secrets (GitHub → Settings → Secrets)
# TURSO_DATABASE_URL
# TURSO_AUTH_TOKEN
# TELEGRAM_BOT_TOKEN
# TELEGRAM_CHAT_ID
# GH_PAT (for repository_dispatch if needed)
```

## Triggering the Pipeline

Use an external cron service to call:

```bash
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $GH_PAT" \
  https://api.github.com/repos/whatman42/idx/dispatches \
  -d '{"event_type":"run_trading_cycle"}'
```

## Directory Layout

See the tree below for the full modular structure.

---

**Status**: Boilerplate ready. Next modules to implement:
- Full Triple-Barrier vectorized implementation
- Feature store + Polars pipeline
- ONNX export + Rust ort inference
- Live data adapters (IDX)

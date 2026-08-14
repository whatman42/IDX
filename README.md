# 📈 IDX Stock Analysis Engine & Quantitative Scalping Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/version-2026.Q3.v27.0-blue.svg?style=for-the-badge" alt="Version" />
  <img src="https://img.shields.io/badge/IDX_Compliance-Strict_2026-green.svg?style=for-the-badge" alt="Compliance" />
  <img src="https://img.shields.io/badge/Architecture-Hybrid_Quant--AI-orange.svg?style=for-the-badge" alt="Architecture" />
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg?style=for-the-badge" alt="Build Status" />
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg?style=for-the-badge" alt="Python Version" />
</p>

> **Institutional-Hedge-Fund-Grade Quantitative Feature Engineering, Risk Guard, and Machine Learning Execution Engine** tailored for Indonesia Stock Exchange (IDX / BEI) equities.

Powered by **Vectorized Polars**, **LightGBM/XGBoost Engine**, **Google Gemini AI**, and **Automated Self-Learning Mechanisms**, this engine delivers sub-second intraday scalping signals, real-time drift monitoring, and autonomous risk management.

---

## 🏛️ System Architecture

The engine adopts a **Hybrid Quant-AI Architecture**:
* **Quantitative Heavy Lifting (Local / Fast):** High-speed vector calculations via Polars & NumPy ($0.0033\text{s}$ feature extraction & ML inference time).
* **Qualitative Risk & Meta-Control (Cloud AI):** Google Gemini AI serves as the **Chief Risk Officer (CRO)** for dynamic regime detection and parameter clamping.

```text
                  ┌──────────────────────────────────────────────┐
                  │          Market Data Engine (data.py)        │
                  │       (OHLCV, Staleness & Flatline Filter)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Unified Feature Engine (features.py)      │
                  │  - Cross-Sectional Alpha Ranks               │
                  │  - Technicals & Adaptive Trend (KAMA, HMA)   │
                  │  - Garman-Klass & Parkinson Volatility       │
                  │  - Microstructure & Typical Price ADTV       │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │        Google Gemini AI Meta-Control         │
                  │    (Adaptive Risk Clamping & Regime State)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           Execution & Machine Learning                           │
├──────────────────────────────────────┬───────────────────────────────────────────┤
│    ML Scalping Model Engine          │         Unified Risk Engine (risk.py)      │
│  - LightGBM/XGBoost Inference        │  - Drawdown & Anomaly Circuit Breaker    │
│  - High Confidence Signal Filter     │  - RRR Guard & Position Sizing            │
└──────────────────┬───────────────────┴─────────────────────┬─────────────────────┘
                   │                                         │
                   ▼                                         ▼
┌──────────────────────────────────────┐  ┌────────────────────────────────────────┐
│     Telegram Telemetry Dispatch      │  │   SQLite Audit & Self-Healing Store    │
│   (Explainable AI Signal Reports)    │  │    (WAL Checkpoint & Git Auto-Sync)    │
└──────────────────────────────────────┘  └────────────────────────────────────────┘

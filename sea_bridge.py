# sea_bridge.py
"""
Bridge module for all IDX modules to access S.E.A. (sea.py)
without circular imports. This is the SINGLE SOURCE OF TRUTH for
trading configuration, risk limits, and strategy parameters.
"""

import os
import json
from typing import Any, Dict, Optional

# Global reference to the S.E.A. agent (set once by main.py)
_SEA_AGENT = None


def set_agent(agent) -> None:
    """Inject the S.E.A. GodEntity instance (called by main.py)."""
    global _SEA_AGENT
    _SEA_AGENT = agent


def get_agent():
    """Return the S.E.A. instance. Raises if not set."""
    if _SEA_AGENT is None:
        # Fallback: try to init from sea.py
        try:
            from sea import GodEntity
            agent = GodEntity(base_dir=os.getcwd())
            agent.boot()
            set_agent(agent)
            return agent
        except Exception as e:
            raise RuntimeError("S.E.A. agent not initialized!") from e
    return _SEA_AGENT


def get_trading_config() -> Dict[str, Any]:
    """Get trading configuration (min_adtv, min_conf, rrr, risk_scale) from S.E.A."""
    agent = get_agent()
    cfg = agent.memory.get("trading_config")
    if not cfg:
        # Propose fresh config if not set
        cfg = agent.propose_trading_config()
        agent.memory.set("trading_config", cfg)
    return cfg


def get_risk_limits() -> Dict[str, float]:
    """Get risk limits (max_drawdown, max_position_pct, etc.) from S.E.A."""
    agent = get_agent()
    return agent.memory.get("risk_limits", {
        "max_drawdown": 0.15,
        "max_position_pct": 0.10,
        "max_exposure": 0.50,
        "risk_per_trade": 0.02,
    })


def get_feature_params() -> Dict[str, Any]:
    """Get feature engineering parameters (windows, flags)."""
    agent = get_agent()
    return agent.memory.get("feature_params", {
        "rsi_period": 14,
        "atr_period": 14,
        "ema_fast": 20,
        "ema_slow": 50,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "use_volume": True,
        "use_sentiment": False,
    })


def get_strategy_params() -> Dict[str, Any]:
    """Get strategy-specific parameters."""
    agent = get_agent()
    return agent.memory.get("strategy_params", {
        "scalp_horizon_minutes": 15,
        "swing_horizon_days": 3,
        "position_sizing_mode": "risk_parity",
    })


def update_sea_config(key: str, value: Any) -> None:
    """Update any config in S.E.A. memory and persist."""
    agent = get_agent()
    agent.memory.set(key, value)
    # Optionally, trigger a re-evaluation of the agent's plan if needed.
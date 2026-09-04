"""
Model evaluation metrics — classification + trading-oriented summaries.

Does NOT claim profitability; realistic backtest is a later stage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["tn"] = float(cm[0, 0])
    out["fp"] = float(cm[0, 1])
    out["fn"] = float(cm[1, 0])
    out["tp"] = float(cm[1, 1])
    return out


def trading_metrics(
    returns: np.ndarray,
    accepted: np.ndarray,
    sides: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Lightweight trading-oriented summary on accepted signals only."""
    returns = np.asarray(returns, dtype=float)
    accepted = np.asarray(accepted, dtype=bool)
    n = len(returns)
    n_acc = int(accepted.sum())
    out: dict[str, float] = {
        "n_signals": float(n),
        "n_accepted": float(n_acc),
        "acceptance_rate": float(n_acc / n) if n else 0.0,
    }
    if n_acc == 0:
        out.update({"hit_rate": 0.0, "avg_return": 0.0, "profit_factor": 0.0})
        return out

    r = returns[accepted]
    hits = r > 0
    out["hit_rate"] = float(hits.mean())
    out["avg_return"] = float(r.mean())
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    out["profit_factor"] = float(gains / losses) if losses > 1e-12 else float("inf")
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    out["max_drawdown"] = float(dd.min()) if len(dd) else 0.0
    if r.std() > 1e-12:
        out["sharpe_like"] = float(r.mean() / r.std() * np.sqrt(252))
    else:
        out["sharpe_like"] = 0.0
    return out

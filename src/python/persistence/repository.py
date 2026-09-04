"""Durable state repository — SQLite primary; Turso-compatible schema."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cycles (
  cycle_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, mode TEXT, status TEXT,
  git_sha TEXT, payload TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS signals (
  signal_id TEXT PRIMARY KEY, cycle_id TEXT NOT NULL, symbol TEXT, side INTEGER,
  payload TEXT, created_at TEXT NOT NULL, UNIQUE(signal_id));
CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY, signal_id TEXT NOT NULL, cycle_id TEXT,
  payload TEXT, created_at TEXT NOT NULL, UNIQUE(order_id));
CREATE TABLE IF NOT EXISTS transactions (
  tx_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, signal_id TEXT, symbol TEXT,
  side INTEGER, action TEXT, qty REAL, price REAL, fee REAL, cycle_id TEXT,
  timestamp TEXT, created_at TEXT NOT NULL, UNIQUE(tx_id));
CREATE TABLE IF NOT EXISTS positions (
  position_key TEXT PRIMARY KEY, symbol TEXT, side INTEGER, qty REAL,
  avg_price REAL, state TEXT, payload TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
  snapshot_id TEXT PRIMARY KEY, cycle_id TEXT, cash REAL, equity REAL,
  payload TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS governor_decisions (
  decision_id TEXT PRIMARY KEY, cycle_id TEXT, payload TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_registry (
  model_key TEXT PRIMARY KEY, model_id TEXT, model_version TEXT, status TEXT,
  payload TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS model_promotions (
  promotion_id TEXT PRIMARY KEY, model_id TEXT, model_version TEXT,
  from_status TEXT, to_status TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notification_outbox (
  event_id TEXT PRIMARY KEY, event_type TEXT, payload TEXT, status TEXT,
  attempts INTEGER DEFAULT 0, last_error TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS audit_events (
  event_id TEXT PRIMARY KEY, cycle_id TEXT, event_type TEXT, severity TEXT,
  payload TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS risk_events (
  event_id TEXT PRIMARY KEY, cycle_id TEXT, allow INTEGER, reason TEXT,
  payload TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_tx_order ON transactions(order_id);
CREATE INDEX IF NOT EXISTS idx_signals_cycle ON signals(cycle_id);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON notification_outbox(status);
"""


class StateRepository(ABC):
    @abstractmethod
    def put_cycle(self, cycle_id: str, payload: dict) -> None: ...
    @abstractmethod
    def put_signal(self, signal_id: str, cycle_id: str, payload: dict) -> bool: ...
    @abstractmethod
    def put_transaction(self, tx: dict) -> bool: ...
    @abstractmethod
    def put_portfolio_snapshot(self, cycle_id: str, cash: float, equity: float, payload: dict) -> None: ...
    @abstractmethod
    def put_governor_decision(self, decision_id: str, cycle_id: str, payload: dict) -> None: ...
    @abstractmethod
    def enqueue_notification(self, event_id: str, event_type: str, payload: dict) -> None: ...
    @abstractmethod
    def pending_notifications(self, limit: int = 20) -> list[dict]: ...
    @abstractmethod
    def mark_notification(self, event_id: str, status: str, error: str = "") -> None: ...
    @abstractmethod
    def put_audit(self, event_id: str, cycle_id: str, event_type: str, severity: str, payload: dict) -> None: ...
    @abstractmethod
    def put_risk_event(self, event_id: str, cycle_id: str, allow: bool, reason: str, payload: dict) -> None: ...
    @abstractmethod
    def get_latest_portfolio(self) -> Optional[dict]: ...


class SQLiteStateRepository(StateRepository):
    def __init__(self, path: str | Path = "state/idx.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA_SQL)

    def put_cycle(self, cycle_id: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO cycles(cycle_id,timestamp,mode,status,git_sha,payload,created_at) VALUES(?,?,?,?,?,?,?)",
                (cycle_id, payload.get("timestamp", now), payload.get("mode"), payload.get("status"),
                 payload.get("git_sha", ""), json.dumps(payload, default=str), now),
            )

    def put_signal(self, signal_id: str, cycle_id: str, payload: dict) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO signals(signal_id,cycle_id,symbol,side,payload,created_at) VALUES(?,?,?,?,?,?)",
                    (signal_id, cycle_id, payload.get("symbol"), payload.get("side"),
                     json.dumps(payload, default=str), now),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def put_transaction(self, tx: dict) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO transactions(tx_id,order_id,signal_id,symbol,side,action,qty,price,fee,cycle_id,timestamp,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tx["tx_id"], tx["order_id"], tx.get("signal_id"), tx.get("symbol"), tx.get("side"),
                     tx.get("action"), tx.get("qty"), tx.get("price"), tx.get("fee"), tx.get("cycle_id"),
                     tx.get("timestamp"), now),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def put_portfolio_snapshot(self, cycle_id: str, cash: float, equity: float, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO portfolio_snapshots(snapshot_id,cycle_id,cash,equity,payload,created_at) VALUES(?,?,?,?,?,?)",
                (f"snap_{cycle_id}", cycle_id, cash, equity, json.dumps(payload, default=str), now),
            )

    def put_governor_decision(self, decision_id: str, cycle_id: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO governor_decisions(decision_id,cycle_id,payload,created_at) VALUES(?,?,?,?)",
                (decision_id, cycle_id, json.dumps(payload, default=str), now),
            )

    def enqueue_notification(self, event_id: str, event_type: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO notification_outbox(event_id,event_type,payload,status,attempts,created_at,updated_at) VALUES(?,?,?,'PENDING',0,?,?)",
                (event_id, event_type, json.dumps(payload, default=str), now, now),
            )

    def pending_notifications(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT event_id,event_type,payload,attempts FROM notification_outbox WHERE status IN ('PENDING','FAILED') ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"event_id": r[0], "event_type": r[1], "payload": json.loads(r[2]), "attempts": r[3]} for r in rows]

    def claim_pending_notifications(self, limit: int = 20, worker_id: str = "w") -> list[dict]:
        """Atomically claim PENDING/FAILED rows (status -> PROCESSING) to avoid double delivery."""
        now = datetime.now(timezone.utc).isoformat()
        claimed = []
        with self._conn() as c:
            rows = c.execute(
                "SELECT event_id,event_type,payload,attempts FROM notification_outbox "
                "WHERE status IN ('PENDING','FAILED') ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
            for r in rows:
                cur = c.execute(
                    "UPDATE notification_outbox SET status='PROCESSING', updated_at=? "
                    "WHERE event_id=? AND status IN ('PENDING','FAILED')",
                    (now, r[0]),
                )
                if cur.rowcount == 1:
                    claimed.append(
                        {"event_id": r[0], "event_type": r[1], "payload": json.loads(r[2]), "attempts": r[3]}
                    )
        return claimed

    def mark_notification(self, event_id: str, status: str, error: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "UPDATE notification_outbox SET status=?, last_error=?, attempts=attempts+1, updated_at=? WHERE event_id=?",
                (status, error, now, event_id),
            )

    def put_audit(self, event_id: str, cycle_id: str, event_type: str, severity: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO audit_events(event_id,cycle_id,event_type,severity,payload,created_at) VALUES(?,?,?,?,?,?)",
                (event_id, cycle_id, event_type, severity, json.dumps(payload, default=str), now),
            )

    def put_risk_event(self, event_id: str, cycle_id: str, allow: bool, reason: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO risk_events(event_id,cycle_id,allow,reason,payload,created_at) VALUES(?,?,?,?,?,?)",
                (event_id, cycle_id, int(allow), reason, json.dumps(payload, default=str), now),
            )

    def get_latest_portfolio(self) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT payload FROM portfolio_snapshots ORDER BY created_at DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

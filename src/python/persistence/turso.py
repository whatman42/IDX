"""Turso (libSQL HTTP) state repository. Authority: Turso if creds else SQLite. Never dual-write."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.python.persistence.repository import SCHEMA_SQL, StateRepository


class TursoStateRepository(StateRepository):
    def __init__(self, url: Optional[str] = None, token: Optional[str] = None, timeout: float = 15.0):
        self.url = (url or os.getenv("TURSO_DATABASE_URL") or "").rstrip("/")
        self.token = token or os.getenv("TURSO_AUTH_TOKEN") or ""
        self.timeout = timeout
        if not self.url or not self.token:
            raise RuntimeError("Turso credentials missing (TURSO_DATABASE_URL / TURSO_AUTH_TOKEN)")
        self._api = f"{self.url}/v2/pipeline"
        self._ensure_schema()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def _exec(self, stmts: list[dict]) -> dict:
        payload = {"requests": [{"type": "execute", "stmt": s} for s in stmts] + [{"type": "close"}]}
        resp = httpx.post(self._api, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _ensure_schema(self) -> None:
        for stmt in SCHEMA_SQL.split(";"):
            s = stmt.strip()
            if not s:
                continue
            try:
                self._exec([{"sql": s}])
            except Exception:
                pass

    def put_cycle(self, cycle_id: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._exec([{
            "sql": "INSERT OR REPLACE INTO cycles(cycle_id,timestamp,mode,status,git_sha,payload,created_at) VALUES(?,?,?,?,?,?,?)",
            "args": [
                {"type": "text", "value": cycle_id},
                {"type": "text", "value": payload.get("timestamp", now)},
                {"type": "text", "value": str(payload.get("mode", ""))},
                {"type": "text", "value": str(payload.get("status", ""))},
                {"type": "text", "value": str(payload.get("git_sha", ""))},
                {"type": "text", "value": json.dumps(payload, default=str)},
                {"type": "text", "value": now},
            ],
        }])

    def put_signal(self, signal_id: str, cycle_id: str, payload: dict) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._exec([{
                "sql": "INSERT INTO signals(signal_id,cycle_id,symbol,side,payload,created_at) VALUES(?,?,?,?,?,?)",
                "args": [
                    {"type": "text", "value": signal_id},
                    {"type": "text", "value": cycle_id},
                    {"type": "text", "value": str(payload.get("symbol", ""))},
                    {"type": "integer", "value": str(int(payload.get("side", 0)))},
                    {"type": "text", "value": json.dumps(payload, default=str)},
                    {"type": "text", "value": now},
                ],
            }])
            return True
        except Exception:
            return False

    def put_transaction(self, tx: dict) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._exec([{
                "sql": "INSERT INTO transactions(tx_id,order_id,signal_id,symbol,side,action,qty,price,fee,cycle_id,timestamp,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                "args": [
                    {"type": "text", "value": str(tx["tx_id"])},
                    {"type": "text", "value": str(tx["order_id"])},
                    {"type": "text", "value": str(tx.get("signal_id", ""))},
                    {"type": "text", "value": str(tx.get("symbol", ""))},
                    {"type": "integer", "value": str(int(tx.get("side", 0)))},
                    {"type": "text", "value": str(tx.get("action", ""))},
                    {"type": "float", "value": str(float(tx.get("qty", 0)))},
                    {"type": "float", "value": str(float(tx.get("price", 0)))},
                    {"type": "float", "value": str(float(tx.get("fee", 0)))},
                    {"type": "text", "value": str(tx.get("cycle_id", ""))},
                    {"type": "text", "value": str(tx.get("timestamp", now))},
                    {"type": "text", "value": now},
                ],
            }])
            return True
        except Exception:
            return False

    def put_portfolio_snapshot(self, cycle_id: str, cash: float, equity: float, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._exec([{
            "sql": "INSERT OR REPLACE INTO portfolio_snapshots(snapshot_id,cycle_id,cash,equity,payload,created_at) VALUES(?,?,?,?,?,?)",
            "args": [
                {"type": "text", "value": f"snap_{cycle_id}"},
                {"type": "text", "value": cycle_id},
                {"type": "float", "value": str(cash)},
                {"type": "float", "value": str(equity)},
                {"type": "text", "value": json.dumps(payload, default=str)},
                {"type": "text", "value": now},
            ],
        }])

    def put_governor_decision(self, decision_id: str, cycle_id: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._exec([{
            "sql": "INSERT OR REPLACE INTO governor_decisions(decision_id,cycle_id,payload,created_at) VALUES(?,?,?,?)",
            "args": [
                {"type": "text", "value": decision_id},
                {"type": "text", "value": cycle_id},
                {"type": "text", "value": json.dumps(payload, default=str)},
                {"type": "text", "value": now},
            ],
        }])

    def enqueue_notification(self, event_id: str, event_type: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._exec([{
                "sql": "INSERT OR IGNORE INTO notification_outbox(event_id,event_type,payload,status,attempts,created_at,updated_at) VALUES(?,?,?,'PENDING',0,?,?)",
                "args": [
                    {"type": "text", "value": event_id},
                    {"type": "text", "value": event_type},
                    {"type": "text", "value": json.dumps(payload, default=str)},
                    {"type": "text", "value": now},
                    {"type": "text", "value": now},
                ],
            }])
        except Exception:
            pass

    def pending_notifications(self, limit: int = 20) -> list[dict]:
        return []

    def mark_notification(self, event_id: str, status: str, error: str = "") -> None:
        pass

    def put_audit(self, event_id: str, cycle_id: str, event_type: str, severity: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._exec([{
            "sql": "INSERT OR REPLACE INTO audit_events(event_id,cycle_id,event_type,severity,payload,created_at) VALUES(?,?,?,?,?,?)",
            "args": [
                {"type": "text", "value": event_id},
                {"type": "text", "value": cycle_id},
                {"type": "text", "value": event_type},
                {"type": "text", "value": severity},
                {"type": "text", "value": json.dumps(payload, default=str)},
                {"type": "text", "value": now},
            ],
        }])

    def put_risk_event(self, event_id: str, cycle_id: str, allow: bool, reason: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._exec([{
            "sql": "INSERT OR REPLACE INTO risk_events(event_id,cycle_id,allow,reason,payload,created_at) VALUES(?,?,?,?,?,?)",
            "args": [
                {"type": "text", "value": event_id},
                {"type": "text", "value": cycle_id},
                {"type": "integer", "value": str(int(allow))},
                {"type": "text", "value": reason},
                {"type": "text", "value": json.dumps(payload, default=str)},
                {"type": "text", "value": now},
            ],
        }])

    def get_latest_portfolio(self) -> Optional[dict]:
        return None


def open_repository(config: Optional[dict] = None) -> StateRepository:
    from src.python.persistence.repository import SQLiteStateRepository

    config = config or {}
    url = config.get("turso_url") or os.getenv("TURSO_DATABASE_URL", "")
    token = config.get("turso_token") or os.getenv("TURSO_AUTH_TOKEN", "")
    prefer = (config.get("authority") or os.getenv("IDX_STATE_AUTHORITY", "auto")).lower()
    if prefer == "turso" or (prefer == "auto" and url and token):
        if not url or not token:
            raise RuntimeError("Turso selected as authority but credentials missing — FAIL CLOSED")
        return TursoStateRepository(url, token)
    db_path = config.get("sqlite_path") or os.getenv("IDX_SQLITE_PATH", "state/idx.db")
    return SQLiteStateRepository(db_path)

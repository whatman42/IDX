"""Turso HTTP StateRepository. XOR SQLite authority. Full outbox parity."""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from typing import Any, Optional
import httpx
from src.python.persistence.repository import SCHEMA_SQL, StateRepository

def _arg(value: Any, typ: str = "text") -> dict:
    if typ == "float":
        return {"type": "float", "value": str(value)}
    if typ == "integer":
        return {"type": "integer", "value": str(int(value))}
    return {"type": "text", "value": "" if value is None else str(value)}

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

    def _rows(self, result: dict) -> list[list]:
        rows = []
        try:
            for item in result.get("results", []):
                r = item.get("response", item)
                if r.get("type") == "execute":
                    for row in r.get("result", {}).get("rows", []):
                        if isinstance(row, list):
                            rows.append([c.get("value") if isinstance(c, dict) else c for c in row])
                        elif isinstance(row, dict):
                            rows.append(list(row.values()))
        except Exception:
            pass
        return rows

    def _ensure_schema(self) -> None:
        for stmt in SCHEMA_SQL.split(";"):
            s = stmt.strip()
            if s:
                try:
                    self._exec([{"sql": s}])
                except Exception:
                    pass

    def put_cycle(self, cycle_id: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._exec([{"sql": "INSERT OR REPLACE INTO cycles(cycle_id,timestamp,mode,status,git_sha,payload,created_at) VALUES(?,?,?,?,?,?,?)",
                     "args": [_arg(cycle_id), _arg(payload.get("timestamp", now)), _arg(payload.get("mode", "")),
                              _arg(payload.get("status", "")), _arg(payload.get("git_sha", "")),
                              _arg(json.dumps(payload, default=str)), _arg(now)]}])

    def put_signal(self, signal_id: str, cycle_id: str, payload: dict) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._exec([{"sql": "INSERT INTO signals(signal_id,cycle_id,symbol,side,payload,created_at) VALUES(?,?,?,?,?,?)",
                         "args": [_arg(signal_id), _arg(cycle_id), _arg(payload.get("symbol")),
                                  _arg(payload.get("side"), "integer"), _arg(json.dumps(payload, default=str)), _arg(now)]}])
            return True
        except Exception:
            return False

    def put_transaction(self, tx: dict) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._exec([{"sql": "INSERT INTO transactions(tx_id,order_id,signal_id,symbol,side,action,qty,price,fee,cycle_id,timestamp,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                         "args": [_arg(tx["tx_id"]), _arg(tx["order_id"]), _arg(tx.get("signal_id")), _arg(tx.get("symbol")),
                                  _arg(tx.get("side"), "integer"), _arg(tx.get("action")), _arg(tx.get("qty"), "float"),
                                  _arg(tx.get("price"), "float"), _arg(tx.get("fee"), "float"), _arg(tx.get("cycle_id")),
                                  _arg(tx.get("timestamp")), _arg(now)]}])
            return True
        except Exception:
            return False

    def put_portfolio_snapshot(self, cycle_id: str, cash: float, equity: float, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._exec([{"sql": "INSERT OR REPLACE INTO portfolio_snapshots(snapshot_id,cycle_id,cash,equity,payload,created_at) VALUES(?,?,?,?,?,?)",
                     "args": [_arg(f"snap_{cycle_id}"), _arg(cycle_id), _arg(cash, "float"), _arg(equity, "float"),
                              _arg(json.dumps(payload, default=str)), _arg(now)]}])

    def put_governor_decision(self, decision_id: str, cycle_id: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._exec([{"sql": "INSERT OR REPLACE INTO governor_decisions(decision_id,cycle_id,payload,created_at) VALUES(?,?,?,?)",
                     "args": [_arg(decision_id), _arg(cycle_id), _arg(json.dumps(payload, default=str)), _arg(now)]}])

    def enqueue_notification(self, event_id: str, event_type: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._exec([{"sql": "INSERT OR IGNORE INTO notification_outbox(event_id,event_type,payload,status,attempts,created_at,updated_at) VALUES(?,?,?,'PENDING',0,?,?)",
                         "args": [_arg(event_id), _arg(event_type), _arg(json.dumps(payload, default=str)), _arg(now), _arg(now)]}])
        except Exception:
            pass

    def pending_notifications(self, limit: int = 20) -> list[dict]:
        try:
            result = self._exec([{"sql": "SELECT event_id,event_type,payload,attempts FROM notification_outbox WHERE status IN ('PENDING','FAILED') ORDER BY created_at LIMIT ?",
                                  "args": [_arg(limit, "integer")]}])
            out = []
            for row in self._rows(result):
                if len(row) < 4:
                    continue
                try:
                    payload = json.loads(row[2]) if isinstance(row[2], str) else (row[2] or {})
                except Exception:
                    payload = {}
                out.append({"event_id": row[0], "event_type": row[1], "payload": payload, "attempts": int(row[3] or 0)})
            return out
        except Exception:
            return []

    def claim_pending_notifications(self, limit: int = 20, worker_id: str = "w") -> list[dict]:
        now = datetime.now(timezone.utc).isoformat()
        claimed = []
        for item in self.pending_notifications(limit=limit):
            eid = item["event_id"]
            try:
                self._exec([{"sql": "UPDATE notification_outbox SET status='PROCESSING', updated_at=? WHERE event_id=? AND status IN ('PENDING','FAILED')",
                             "args": [_arg(now), _arg(eid)]}])
                claimed.append(item)
            except Exception:
                continue
        return claimed

    def mark_notification(self, event_id: str, status: str, error: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._exec([{"sql": "UPDATE notification_outbox SET status=?, last_error=?, attempts=attempts+1, updated_at=? WHERE event_id=?",
                         "args": [_arg(status), _arg(error), _arg(now), _arg(event_id)]}])
        except Exception:
            pass

    def put_audit(self, event_id: str, cycle_id: str, event_type: str, severity: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._exec([{"sql": "INSERT OR REPLACE INTO audit_events(event_id,cycle_id,event_type,severity,payload,created_at) VALUES(?,?,?,?,?,?)",
                     "args": [_arg(event_id), _arg(cycle_id), _arg(event_type), _arg(severity),
                              _arg(json.dumps(payload, default=str)), _arg(now)]}])

    def put_risk_event(self, event_id: str, cycle_id: str, allow: bool, reason: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._exec([{"sql": "INSERT OR REPLACE INTO risk_events(event_id,cycle_id,allow,reason,payload,created_at) VALUES(?,?,?,?,?,?)",
                     "args": [_arg(event_id), _arg(cycle_id), _arg(1 if allow else 0, "integer"),
                              _arg(reason), _arg(json.dumps(payload, default=str)), _arg(now)]}])

    def get_latest_portfolio(self) -> Optional[dict]:
        try:
            result = self._exec([{"sql": "SELECT payload FROM portfolio_snapshots ORDER BY created_at DESC LIMIT 1"}])
            rows = self._rows(result)
            if not rows:
                return None
            raw = rows[0][0]
            return json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return None

def open_repository(config: Optional[dict] = None) -> StateRepository:
    from src.python.persistence.repository import SQLiteStateRepository
    config = config or {}
    prefer = (config.get("authority") or os.getenv("IDX_STATE_AUTHORITY", "auto")).lower()
    has_turso = bool(os.getenv("TURSO_DATABASE_URL") and os.getenv("TURSO_AUTH_TOKEN"))
    if prefer == "turso":
        if not has_turso:
            raise RuntimeError("Turso selected as authority but credentials missing — FAIL CLOSED")
        return TursoStateRepository()
    if prefer == "sqlite":
        return SQLiteStateRepository(config.get("sqlite_path") or os.getenv("IDX_SQLITE_PATH", "state/idx.db"))
    if has_turso:
        return TursoStateRepository()
    return SQLiteStateRepository(config.get("sqlite_path") or os.getenv("IDX_SQLITE_PATH", "state/idx.db"))

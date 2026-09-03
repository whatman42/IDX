"""
Turso (libSQL) client – Embedded Replica pattern.

Local SQLite is synced to remote primary at the end of each workflow.
"""

from __future__ import annotations

import os
from typing import Any

# Real implementation will use libsql-client / turso Python SDK
# For boilerplate we keep a thin interface.


class TursoState:
    def __init__(self):
        self.url = os.getenv("TURSO_DATABASE_URL")
        self.token = os.getenv("TURSO_AUTH_TOKEN")

    def pull(self) -> None:
        """Sync remote → local embedded replica."""
        print("[Turso] pull (stub)")

    def push(self) -> None:
        """Sync local changes → remote primary."""
        print("[Turso] push (stub)")

    def get_model_params(self, model_name: str) -> dict[str, Any]:
        return {}

    def save_model_params(self, model_name: str, params: dict[str, Any]) -> None:
        pass

    def log_metric(self, name: str, value: float, ts: str | None = None) -> None:
        pass

import os
import pytest
from src.python.persistence.turso import TursoStateRepository, open_repository
from src.python.persistence.repository import SQLiteStateRepository

HAS_TURSO = bool(os.getenv("TURSO_DATABASE_URL") and os.getenv("TURSO_AUTH_TOKEN"))

def test_contract_methods_exist():
    for m in ("enqueue_notification", "pending_notifications", "claim_pending_notifications", "mark_notification"):
        assert hasattr(TursoStateRepository, m)

@pytest.mark.skipif(not HAS_TURSO, reason="LIVE TURSO VERIFICATION = BLOCKED (no credentials)")
def test_live_turso_outbox_claim():
    repo = TursoStateRepository()
    eid = "test_evt_live_claim_001"
    repo.enqueue_notification(eid, "BUY", {"symbol": "TEST"})
    a = repo.claim_pending_notifications(limit=50)
    ids = [x["event_id"] for x in a]
    b_ids = [x["event_id"] for x in repo.claim_pending_notifications(limit=50)]
    if eid in ids:
        assert eid not in b_ids
        repo.mark_notification(eid, "SENT")

def test_sqlite_claim_still_exclusive(tmp_path):
    repo = SQLiteStateRepository(tmp_path / "t.db")
    repo.enqueue_notification("e1", "BUY", {})
    assert len(repo.claim_pending_notifications()) == 1
    assert len(repo.claim_pending_notifications()) == 0

def test_authority_turso_fail_closed(monkeypatch, tmp_path):
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("IDX_STATE_AUTHORITY", "turso")
    with pytest.raises(RuntimeError):
        open_repository({"sqlite_path": str(tmp_path / "x.db")})

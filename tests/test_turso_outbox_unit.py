from src.python.persistence.repository import SQLiteStateRepository
from src.python.persistence.turso import TursoStateRepository

def test_turso_class_has_outbox_parity_methods():
    for m in ["enqueue_notification", "pending_notifications", "claim_pending_notifications", "mark_notification"]:
        assert hasattr(TursoStateRepository, m)

def test_sqlite_claim_exclusive(tmp_path):
    repo = SQLiteStateRepository(tmp_path / "c.db")
    repo.enqueue_notification("e1", "X", {"a": 1})
    a = repo.claim_pending_notifications(limit=5, worker_id="a")
    b = repo.claim_pending_notifications(limit=5, worker_id="b")
    assert len(a) == 1 and len(b) == 0

def test_open_repository_fail_closed_turso(monkeypatch, tmp_path):
    from src.python.persistence import turso as tmod
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("IDX_STATE_AUTHORITY", "turso")
    try:
        tmod.open_repository({"sqlite_path": str(tmp_path / "x.db")})
        assert False
    except RuntimeError as e:
        assert "FAIL CLOSED" in str(e) or "credentials" in str(e).lower()

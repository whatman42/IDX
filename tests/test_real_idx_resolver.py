"""REAL IDX resolver — fail-closed, no silent synthetic, fixture ≠ REAL."""
from __future__ import annotations
from pathlib import Path
import pytest
from src.python.data.provenance import DatasetType
from src.python.data.real_idx import assert_not_fixture_as_real, load_real_idx_contract, resolve_real_idx_path

def test_missing_env_blocked(monkeypatch):
    monkeypatch.delenv("REAL_IDX_DATA_PATH", raising=False)
    st = resolve_real_idx_path()
    assert st.status == "BLOCKED"
    assert "unset" in st.reason or "REAL_IDX" in st.reason

def test_missing_file_blocked(monkeypatch, tmp_path):
    monkeypatch.setenv("REAL_IDX_DATA_PATH", str(tmp_path / "nope.csv"))
    st = resolve_real_idx_path()
    assert st.status == "BLOCKED" and st.reason == "file_not_found"

def test_fixture_path_cannot_be_real(monkeypatch):
    monkeypatch.delenv("REAL_IDX_DATA_PATH", raising=False)
    fixture = Path("tests/fixtures/idx_ohlcv_fixture.csv")
    if not fixture.exists():
        pytest.skip("fixture missing")
    st = resolve_real_idx_path(str(fixture))
    assert st.status == "BLOCKED" and "fixture" in st.reason

def test_assert_fixture_as_real_raises():
    with pytest.raises(ValueError, match="REAL_MARKET_DATA"):
        assert_not_fixture_as_real(DatasetType.REAL_MARKET_DATA, "tests/fixtures/idx_ohlcv_fixture.csv")
    assert_not_fixture_as_real(DatasetType.FIXTURE, "tests/fixtures/idx_ohlcv_fixture.csv")

def test_valid_csv_outside_fixtures(tmp_path, monkeypatch):
    monkeypatch.delenv("REAL_IDX_DATA_PATH", raising=False)
    csv = tmp_path / "licensed_idx.csv"
    csv.write_text(
        "timestamp,symbol,open,high,low,close,volume\n"
        "2024-01-02,BBCA,9000,9100,8900,9050,1000000\n"
        "2024-01-03,BBCA,9050,9200,9000,9150,1100000\n"
    )
    st = resolve_real_idx_path(str(csv))
    assert st.status == "PASS" and st.format == "csv"
    st2, contract = load_real_idx_contract(str(csv))
    assert st2.status == "PASS" and contract is not None and len(contract.df) == 2
    assert contract.source.startswith("real_idx:")

def test_load_missing_returns_none(monkeypatch):
    monkeypatch.delenv("REAL_IDX_DATA_PATH", raising=False)
    st, contract = load_real_idx_contract()
    assert st.status == "BLOCKED" and contract is None

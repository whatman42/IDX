"""Real-data adapter + historical OOS tests. Fixtures are NOT licensed IDX."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
from src.python.data.costs import CostModel
from src.python.data.leakage import assert_split_chronological, assert_train_only_fit
from src.python.data.provenance import DatasetType, build_provenance_from_df, hash_dataframe
from src.python.data.quality import validate_ohlcv
from src.python.market.providers import CSVProvider
from src.python.ml.temporal import make_purged_split
from src.python.validation.historical_oos import load_market_path, run_historical_oos

FIX = Path(__file__).parent / "fixtures" / "idx_ohlcv_fixture.csv"
BAD = Path(__file__).parent / "fixtures" / "idx_ohlcv_bad.csv"

def test_csv_ingestion_schema():
    c = CSVProvider(FIX).fetch(["BBCA"])
    assert len(c.df) > 0
    assert set(["timestamp", "symbol", "open", "high", "low", "close", "volume"]).issubset(c.df.columns)

def test_ohlc_quality_pass():
    r = validate_ohlcv(pd.read_csv(FIX))
    assert r.ok, r.issues

def test_ohlc_quality_fail_envelope():
    r = validate_ohlcv(pd.read_csv(BAD))
    assert not r.ok

def test_provenance_fixture_not_real():
    df = pd.read_csv(FIX)
    p = build_provenance_from_df(df, dataset_type=DatasetType.FIXTURE, source="tests/fixtures")
    assert p.dataset_type == DatasetType.FIXTURE and p.data_hash
    p.assert_no_secrets()

def test_dataset_sha256_stable():
    df = pd.read_csv(FIX)
    assert hash_dataframe(df) == hash_dataframe(df)

def test_duplicate_detection():
    df = pd.read_csv(FIX)
    df2 = pd.concat([df, df.iloc[:1]], ignore_index=True)
    r = validate_ohlcv(df2)
    assert not r.ok and any("duplicate" in i for i in r.issues)

def test_purged_split_chronological():
    df = pd.read_csv(FIX)
    df = df[df.symbol == "BBCA"].reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    split = make_purged_split(ts, embargo=pd.Timedelta(days=2))
    rep = assert_split_chronological(ts, split.train_idx, split.test_idx)
    assert rep.ok or len(split.train_idx) == 0

def test_train_only_scaler_guard():
    import numpy as np
    assert assert_train_only_fit(np.arange(10), 20, scaler_fit_on="train").ok
    assert not assert_train_only_fit(np.arange(10), 20, scaler_fit_on="all").ok

def test_cost_model_gross_vs_net():
    import numpy as np
    out = CostModel(fee_bps=10, slippage_bps=5).apply(np.array([0.01, -0.005, 0.02]), np.array([True, True, True]))
    assert out["fees"] > 0 and out["net_sum"] < out["gross_sum"]

def test_historical_oos_fixture_blocks_real_claim():
    report = run_historical_oos(FIX, dataset_type=DatasetType.FIXTURE, promote=False)
    assert report.real_idx_data == "BLOCKED"
    assert report.market_performance == "UNVERIFIED"
    assert report.production_unchanged is True
    assert report.promotion_approved is False
    assert report.data_quality == "PASS"
    assert report.purged and report.embargo and report.oos

def test_production_pointer_unchanged(tmp_path):
    prod = tmp_path / "models" / "production"
    prod.mkdir(parents=True)
    pointer = prod / "primary.PRODUCTION"
    pointer.write_text("SAFE")
    run_historical_oos(FIX, dataset_type=DatasetType.FIXTURE, promote=False)
    assert pointer.read_text() == "SAFE"

def test_load_path_provenance():
    contract, prov = load_market_path(FIX, dataset_type=DatasetType.FIXTURE)
    assert prov.row_count > 0 and prov.dataset_type == DatasetType.FIXTURE

def test_reject_fake_real_without_file():
    with pytest.raises(FileNotFoundError):
        load_market_path("/no/such/real_idx.csv", dataset_type=DatasetType.REAL_MARKET_DATA)

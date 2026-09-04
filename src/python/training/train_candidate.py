"""Weekly research/training entrypoint — not for daily Actions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import typer

from src.python.features.engineering import build_ml_dataset, compute_features
from src.python.features.schema import ALL_FEATURE_COLUMNS
from src.python.market.providers import SyntheticProvider
from src.python.ml.evaluation import classification_metrics
from src.python.ml.meta_labeling import MetaLabelGovernor
from src.python.ml.primary_side import PrimarySideModel
from src.python.ml.temporal import make_purged_split
from src.python.registry.artifacts import promote_to_production, save_meta_artifact, save_primary_artifact
from src.python.registry.models import ModelRecord, ModelRegistry, ModelStatus

app = typer.Typer()


def _pl_to_pdf(df):
    return pd.DataFrame({c: df[c].to_list() for c in df.columns})


@app.command()
def main(out_dir: str = "models/candidates", promote: bool = False, min_oos_accuracy: float = 0.52) -> None:
    contract = SyntheticProvider(n=120, seed=7).fetch(["BBCA"])
    data = {}
    for c in contract.df.columns:
        s = contract.df[c]
        data[c] = s.dt.to_pydatetime().tolist() if pd.api.types.is_datetime64_any_dtype(s) else s.to_numpy().tolist()
    feats = compute_features(pl.DataFrame(data))
    ds = build_ml_dataset(feats, drop_warmup=True)
    X = _pl_to_pdf(ds["X"])
    cols = [c for c in ALL_FEATURE_COLUMNS if c in X.columns]
    Xf = X[cols]
    y = (Xf.iloc[:, 0] > Xf.iloc[:, 0].median()).astype(int)
    ts = pd.Series(pd.date_range("2024-01-01", periods=len(Xf), freq="B"))
    split = make_purged_split(ts, embargo=pd.Timedelta(days=2))
    Xtr, ytr = Xf.iloc[split.train_idx], y.iloc[split.train_idx]
    Xte, yte = Xf.iloc[split.test_idx], y.iloc[split.test_idx]
    primary = PrimarySideModel(model_version="primary_lgbm_v002")
    primary.fit(Xtr, ytr)
    proba = primary.predict_proba(Xte)
    pred = (proba >= 0.5).astype(int)
    metrics = classification_metrics(yte.to_numpy(), pred, proba)
    print("OOS primary metrics:", json.dumps(metrics, indent=2))
    side_tr = primary.predict_side(Xtr)
    y_meta = ((side_tr == 1) == (ytr == 1)).astype(int)
    meta = MetaLabelGovernor(calibration="platt", model_version="meta_rf_v002")
    meta.fit(Xtr, y_meta)
    out = Path(out_dir)
    man_p = save_primary_artifact(primary, out, metrics=metrics)
    save_meta_artifact(meta, out)
    reg = ModelRegistry(out.parent if out.name == "candidates" else out)
    reg.register(ModelRecord(model_id="primary_lgbm", model_version=primary.model_version, model_type="lgbm",
                             feature_schema_version=man_p.feature_schema_version, status=ModelStatus.CANDIDATE.value,
                             metrics=metrics, artifact_path=str(out / f"{primary.model_version}.txt")))
    if metrics.get("accuracy", 0) >= min_oos_accuracy and promote:
        reg.promote("primary_lgbm", primary.model_version, ModelStatus.PRODUCTION)
        prod = Path("models/production")
        prod.mkdir(parents=True, exist_ok=True)
        for f in out.glob(f"{primary.model_version}*"):
            (prod / f.name).write_bytes(f.read_bytes())
        for f in out.glob(f"{meta.model_version}*"):
            (prod / f.name).write_bytes(f.read_bytes())
        promote_to_production(prod, "primary_lgbm", primary.model_version)
        promote_to_production(prod, "meta_rf", meta.model_version)
        print("PROMOTED:", primary.model_version, meta.model_version)
    else:
        print("Not promoted")


if __name__ == "__main__":
    app()

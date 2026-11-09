"""Predict from a one-row CSV with 80 speech features plus Age and Sex."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load():
    bundle = joblib.load(ROOT / "notebook_model.joblib")
    return bundle["model"], list(bundle["input_features"])


def predict(df: pd.DataFrame) -> pd.DataFrame:
    model, names = load()
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing[:8]}")
    X = df.loc[:, names].apply(pd.to_numeric, errors="raise")
    p = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return pd.DataFrame({"pd_probability": p, "prediction": model.predict(X)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--json")
    args = ap.parse_args()
    data = pd.DataFrame([json.loads(args.json)]) if args.json else pd.read_csv(args.csv)
    print(predict(data).to_string(index=False))

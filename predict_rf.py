#!/usr/bin/env python3
"""
Predict MBTA bus delay probabilities using a saved RandomForest pipeline.

Usage:
  python predict_rf.py --model artifacts/model.joblib --input samples.csv --output preds.csv --batch 50000

Notes:
- The saved model already contains preprocessing (target/ordinal encoders) and can score new rows directly.
- Input CSV must contain the feature columns used in training (route_id, stop_id, direction_id, point_type,
  weather_condition, season, and the numeric columns such as hour, weekday, time_point_order, weather metrics, etc.).
- Outputs a CSV with the original columns plus: pred_proba_delayed, pred_label (0/1).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load


def parse_args():
    ap = argparse.ArgumentParser(description="Score MBTA delay probability with a saved RF pipeline.")
    ap.add_argument("--model", required=True, help="Path to model.joblib saved by train_rf_bus.py")
    ap.add_argument("--input", required=True, help="Path to input CSV with feature columns")
    ap.add_argument("--output", default="predictions.csv", help="Where to write predictions CSV")
    ap.add_argument("--batch", type=int, default=50000, help="Rows per batch for scoring to control memory")
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for delayed label")
    return ap.parse_args()


def required_columns_from_pre(pre) -> list[str]:
    cols = []
    for _, _, c in pre.transformers_:
        if c is None:
            continue
        if isinstance(c, (list, tuple, np.ndarray, pd.Index)):
            cols.extend(list(c))
    return cols


def main():
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")
    if not input_path.exists():
        sys.exit(f"Input CSV not found: {input_path}")

    pipe = load(model_path)
    pre = getattr(pipe, "named_steps", {}).get("pre")
    if pre is None:
        sys.exit("Loaded model does not contain a 'pre' step; expected a Pipeline with preprocessing.")

    req_cols = required_columns_from_pre(pre)
    # Read in batches to limit memory
    preds = []
    reader = pd.read_csv(input_path, chunksize=args.batch, low_memory=False)
    for chunk in reader:
        missing = [c for c in req_cols if c not in chunk.columns]
        if missing:
            sys.exit(f"Missing required columns: {missing}")
        X = chunk[req_cols].copy()
        proba = pipe.predict_proba(X)[:, 1]
        label = (proba >= args.threshold).astype(int)
        out_chunk = chunk.copy()
        out_chunk["pred_proba_delayed"] = proba
        out_chunk["pred_label"] = label
        preds.append(out_chunk)

    pd.concat(preds, ignore_index=True).to_csv(output_path, index=False)
    print(f"Wrote predictions: {output_path} (threshold={args.threshold})")


if __name__ == "__main__":
    main()

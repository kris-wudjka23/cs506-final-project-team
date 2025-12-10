#!/usr/bin/env python3
"""
Score MBTA bus delay probabilities with a saved linear pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score MBTA bus delay probability with a saved linear pipeline."
    )
    parser.add_argument("--model", required=True, help="Path to the saved joblib pipeline.")
    parser.add_argument("--input", required=True, help="Input CSV containing model features.")
    parser.add_argument(
        "--output",
        default="linear_predictions.csv",
        help="Where to write the scored CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying a row as delayed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="Chunk size for streaming inference.",
    )
    return parser.parse_args()


def required_columns(preprocessor) -> list[str]:
    cols = []
    for _, _, selected in preprocessor.transformers_:
        if selected is None:
            continue
        if isinstance(selected, (list, tuple, np.ndarray, pd.Index)):
            cols.extend([str(c) for c in selected])
    return cols


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")
    if not input_path.exists():
        sys.exit(f"Input CSV not found: {input_path}")

    pipeline = load(model_path)
    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocess")
    if preprocessor is None:
        sys.exit("Loaded pipeline is missing the 'preprocess' step.")
    columns = required_columns(preprocessor)

    reader = pd.read_csv(input_path, chunksize=args.batch_size, low_memory=False)
    outputs = []
    for chunk in reader:
        missing = [c for c in columns if c not in chunk.columns]
        if missing:
            sys.exit(f"Missing required columns: {missing}")
        probs = pipeline.predict_proba(chunk[columns])[:, 1]
        labels = (probs >= args.threshold).astype(int)
        scored = chunk.copy()
        scored["pred_proba_delayed"] = probs
        scored["pred_label"] = labels
        outputs.append(scored)
    pd.concat(outputs, ignore_index=True).to_csv(output_path, index=False)
    print(f"Wrote predictions to {output_path} (threshold={args.threshold:.2f})")


if __name__ == "__main__":
    main()

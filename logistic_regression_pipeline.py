"""
Chronological logistic-regression workflow that mirrors the RF pipeline.

This script reuses the processed cache generated for the random-forest model,
applies the same Jan-Sep / Oct-Dec / 2024 split, persists a fitted pipeline,
and emits the full artifact set (plots + metrics JSON) expected by the project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_loader import load_or_build_processed_df
from linear_artifacts import (
    evaluate_split,
    plot_boxplots,
    plot_calibration,
    plot_coefficients,
    plot_confusion,
    plot_heatmap,
    plot_roc_pr_curves,
    plot_slice_metrics,
    save_metrics,
)
from linear_workflow import (
    MODEL_FEATURES,
    build_preprocessor,
    engineer_linear_features,
    temporal_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Logistic regression baseline with chronological splits."
    )
    parser.add_argument(
        "--csv-path",
        default="bus_weather_clean.csv",
        help="Path to the integrated MBTA + weather CSV.",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/cache",
        help="Directory where the processed Parquet cache should live.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs/logistic_regression",
        help="Directory for metrics, plots, and saved models.",
    )
    parser.add_argument(
        "--delay-threshold",
        type=float,
        default=1.0,
        help="Delay threshold (minutes) for the binary label used by the linear models.",
    )
    parser.add_argument(
        "--tol-sec",
        type=float,
        default=300.0,
        help="Delay tolerance (in seconds) for the cached binary label (default 5 minutes).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Optional row cap after loading the cache (useful for smoke tests).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional fractional sample after loading (0 < frac <= 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Sampling seed for --sample-frac.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force reprocessing of the raw CSV even if the cache looks fresh.",
    )
    return parser.parse_args()


def build_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=None,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])
def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    cache_dir = Path(args.cache_dir)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df_cache = load_or_build_processed_df(
        csv_path, cache_dir, tol_sec=args.tol_sec, rebuild_cache=args.rebuild_cache
    )
    if args.max_rows and args.max_rows > 0:
        df_cache = df_cache.head(args.max_rows).copy()
    if not (0 < args.sample_frac <= 1):
        raise ValueError("--sample-frac must be within (0, 1].")
    if args.sample_frac < 1:
        df_cache = df_cache.sample(frac=args.sample_frac, random_state=args.random_state)

    df = engineer_linear_features(df_cache, args.delay_threshold)
    splits = temporal_split(df)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise RuntimeError("One of the splits is empty; check the date coverage of the CSV.")

    # Stage 1: train on Jan-Sep, evaluate on Oct-Dec.
    stage1_pipeline = build_pipeline()
    stage1_pipeline.fit(train_df[MODEL_FEATURES], train_df["is_delayed"])
    val_metrics, val_pred, val_proba = evaluate_split(stage1_pipeline, val_df)
    plot_confusion(
        val_df["is_delayed"].to_numpy(),
        val_pred,
        outputs_dir / "confusion_matrix_val.png",
        "Confusion Matrix (Validation)",
    )
    plot_roc_pr_curves(stage1_pipeline, val_df, outputs_dir, prefix="Validation")

    # Stage 2: train on Jan-Dec (train+val), evaluate on 2024.
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    final_pipeline = build_pipeline()
    final_pipeline.fit(train_val_df[MODEL_FEATURES], train_val_df["is_delayed"])
    test_metrics, test_pred, test_proba = evaluate_split(final_pipeline, test_df)

    y_test = test_df["is_delayed"].to_numpy()
    plot_confusion(y_test, test_pred, outputs_dir / "confusion_matrix_test.png", "Confusion Matrix (Test)")
    plot_roc_pr_curves(final_pipeline, test_df, outputs_dir, prefix="Test")
    plot_calibration(y_test, test_proba, outputs_dir)
    plot_slice_metrics(test_df, y_test, test_pred, test_proba, "hour", outputs_dir)
    plot_slice_metrics(test_df, y_test, test_pred, test_proba, "precip_mm", outputs_dir)
    plot_coefficients(final_pipeline, outputs_dir, "logistic_coefficients.png")
    plot_boxplots(df, outputs_dir)
    plot_heatmap(df, outputs_dir)

    summary = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "delay_threshold_minutes": args.delay_threshold,
    }
    metrics_payload = {
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_metrics(metrics_payload, summary, outputs_dir, "logistic_regression_metrics.json")

    model_path = outputs_dir / "logistic_regression_model.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"[model] Saved pipeline to {model_path}")


if __name__ == "__main__":
    main()

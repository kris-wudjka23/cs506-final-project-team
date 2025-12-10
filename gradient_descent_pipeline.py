"""
Gradient-descent logistic baseline aligned with the RF training flow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from data_loader import load_or_build_processed_df
from gradient_descent_classifier import GradientDescentLogisticClassifier
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
        description="Gradient-descent logistic regression with chronological splits."
    )
    parser.add_argument("--csv-path", default="bus_weather_clean.csv")
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--outputs-dir", default="outputs/gradient_descent")
    parser.add_argument(
        "--delay-threshold",
        type=float,
        default=5.0,
        help="Delay threshold (minutes) for the binary label.",
    )
    parser.add_argument("--tol-sec", type=float, default=300.0)
    parser.add_argument("--max-rows", type=int, default=-1)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--fit-intercept", action="store_true", default=True)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    preprocessor = build_preprocessor()
    model = GradientDescentLogisticClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        tol=args.tol,
        alpha=args.alpha,
        fit_intercept=args.fit_intercept,
        verbose=args.verbose,
        random_state=args.random_state,
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
        raise RuntimeError("One of the splits is empty; check the CSV coverage.")

    stage1_pipeline = build_pipeline(args)
    stage1_pipeline.fit(train_df[MODEL_FEATURES], train_df["is_delayed"])
    val_metrics, val_pred, _ = evaluate_split(stage1_pipeline, val_df)
    plot_confusion(
        val_df["is_delayed"].to_numpy(),
        val_pred,
        outputs_dir / "confusion_matrix_val.png",
        "Confusion Matrix (Validation)",
    )
    plot_roc_pr_curves(stage1_pipeline, val_df, outputs_dir, prefix="Validation")

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    final_pipeline = build_pipeline(args)
    final_pipeline.fit(train_val_df[MODEL_FEATURES], train_val_df["is_delayed"])
    test_metrics, test_pred, test_proba = evaluate_split(final_pipeline, test_df)

    y_test = test_df["is_delayed"].to_numpy()
    plot_confusion(y_test, test_pred, outputs_dir / "confusion_matrix_test.png", "Confusion Matrix (Test)")
    plot_roc_pr_curves(final_pipeline, test_df, outputs_dir, prefix="Test")
    plot_calibration(y_test, test_proba, outputs_dir)
    plot_slice_metrics(test_df, y_test, test_pred, test_proba, "hour", outputs_dir)
    plot_slice_metrics(test_df, y_test, test_pred, test_proba, "precip_mm", outputs_dir)
    plot_coefficients(final_pipeline, outputs_dir, "gradient_descent_coefficients.png")
    plot_boxplots(df, outputs_dir)
    plot_heatmap(df, outputs_dir)

    summary = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "delay_threshold_minutes": args.delay_threshold,
        "gd_params": {
            "learning_rate": args.learning_rate,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "alpha": args.alpha,
            "fit_intercept": args.fit_intercept,
        },
    }
    metrics_payload = {
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_metrics(metrics_payload, summary, outputs_dir, "gradient_descent_metrics.json")

    model_path = outputs_dir / "gradient_descent_model.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"[model] Saved gradient-descent pipeline to {model_path}")


if __name__ == "__main__":
    main()

"""
Logistic regression workflow for predicting MBTA bus delays and generating visuals.

This script reads the `bus_weather_clean.csv` dataset, engineers a binary delay label,
trains a logistic regression classifier, evaluates performance, and produces the
requested box plots, heatmap, and ROC curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

plt.switch_backend("Agg")

# Columns pulled from the raw CSV to minimize memory pressure.
USECOLS = [
    "service_date",
    "hour",
    "weekday",
    "is_weekend",
    "route_id",
    "direction_id",
    "stop_id",
    "time_point_order",
    "point_type",
    "delay_minutes",
    "air_temp_c",
    "rel_humidity_pct",
    "precip_mm",
    "wind_dir_deg",
    "wind_speed_kmh",
    "pressure_hpa",
    "cloud_cover",
    "weather_condition",
]

# Feature groupings for the sklearn preprocessing pipeline.
NUMERIC_FEATURES = [
    "hour",
    "weekday",
    "is_weekend",
    "time_point_order",
    "air_temp_c",
    "rel_humidity_pct",
    "precip_mm",
    "wind_speed_kmh",
    "pressure_hpa",
    "cloud_cover",
    "wind_dir_sin",
    "wind_dir_cos",
]

CATEGORICAL_FEATURES = [
    "route_id",
    "direction_id",
    "point_type",
    "weather_condition",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Visualizations requested in the proposal.
BOXPLOT_FEATURES = ["air_temp_c", "precip_mm", "wind_speed_kmh", "cloud_cover"]
HEATMAP_MAX_DAYS = 21  # keep the heatmap legible without over-plotting


def build_one_hot_encoder() -> OneHotEncoder:
    """
    Construct a OneHotEncoder compatible with both old and new sklearn versions.

    sklearn >= 1.2 renamed the `sparse` argument to `sparse_output`; this helper
    handles either variant to avoid runtime TypeErrors across environments.
    """

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model on MBTA delay data "
        "and generate exploratory visuals."
    )
    parser.add_argument(
        "--csv-path",
        default="bus_weather_clean.csv",
        help="Path to bus + weather CSV. Defaults to bus_weather_clean.csv in the repo root.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory where metrics and plots will be written.",
    )
    parser.add_argument(
        "--delay-threshold",
        type=float,
        default=1.0,
        help="Delay threshold in minutes to label a trip as delayed (default: 1 minute).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for testing (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500000,
        help="Limit rows loaded into memory (default 500k to keep RAM use manageable). "
        "Set to -1 to load the entire file (if your machine can handle it).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional fractional down-sample applied after loading (0 < frac <= 1).",
    )
    return parser.parse_args()


def load_dataset(
    csv_path: Path, max_rows: int, sample_frac: float, random_state: int
) -> pd.DataFrame:
    nrows = None if max_rows is None or max_rows < 0 else max_rows
    df = pd.read_csv(csv_path, usecols=USECOLS, nrows=nrows)
    if not (0 < sample_frac <= 1):
        raise ValueError("--sample-frac must be within (0, 1].")
    if sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state)
    return df


def engineer_features(df: pd.DataFrame, delay_threshold: float) -> pd.DataFrame:
    data = df.copy()
    data["service_date"] = pd.to_datetime(data["service_date"], errors="coerce")
    data = data.dropna(subset=["service_date", "delay_minutes"])

    data["is_delayed"] = (data["delay_minutes"] >= delay_threshold).astype(int)

    # Treat IDs and categorical descriptors as strings for one-hot encoding.
    data["route_id"] = data["route_id"].astype(str).str.zfill(2)
    data["direction_id"] = data["direction_id"].astype(str)
    data["point_type"] = data["point_type"].astype(str)
    data["weather_condition"] = (
        data["weather_condition"].astype(str).str.strip().replace("", "Unknown")
    )

    # Encode the circular wind direction into sin/cos projections.
    radians = np.deg2rad(data["wind_dir_deg"])
    data["wind_dir_sin"] = np.sin(radians)
    data["wind_dir_cos"] = np.cos(radians)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate_model(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, object]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    conf = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "support": int(y_test.shape[0]),
        "positive_rate": float(y_test.mean()),
        "classification_report": report,
        "confusion_matrix": conf,
    }
    return metrics


def save_metrics(metrics: Dict[str, object], outputs_dir: Path) -> None:
    metrics_path = outputs_dir / "logistic_regression_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[metrics] Saved model metrics to {metrics_path}")


def plot_boxplots(df: pd.DataFrame, outputs_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    for feature in BOXPLOT_FEATURES:
        if feature not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            data=df,
            x="is_delayed",
            y=feature,
            showfliers=False,
        )
        plt.title(f"Delay vs {feature}")
        plt.xlabel("Is Delayed (1 = yes)")
        plt.ylabel(feature.replace("_", " ").title())
        out_path = outputs_dir / f"boxplot_{feature}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[plot] Saved {out_path}")


def plot_heatmap(df: pd.DataFrame, outputs_dir: Path) -> None:
    if df.empty:
        return

    heat_df = df[["service_date", "hour", "delay_minutes"]].dropna()
    if heat_df.empty:
        return

    agg = (
        heat_df.groupby(["service_date", "hour"])["delay_minutes"]
        .mean()
        .reset_index()
    )
    agg["service_date"] = agg["service_date"].dt.date
    pivot = (
        agg.pivot(index="service_date", columns="hour", values="delay_minutes")
        .sort_index()
    )
    if HEATMAP_MAX_DAYS and pivot.shape[0] > HEATMAP_MAX_DAYS:
        pivot = pivot.iloc[-HEATMAP_MAX_DAYS:]

    plt.figure(figsize=(12, max(4, 0.4 * pivot.shape[0])))
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Average Delay (minutes)"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("Average Delay (minutes) by Service Date and Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Service Date")
    plt.tight_layout()
    out_path = outputs_dir / "heatmap_delay_by_date_hour.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved {out_path}")


def plot_roc_curve(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    outputs_dir: Path,
) -> None:
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_estimator(
        pipeline,
        X_test,
        y_test,
        name="Logistic Regression",
    )
    plt.tight_layout()
    out_path = outputs_dir / "roc_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved {out_path}")


def train_and_evaluate(
    df: pd.DataFrame, args: argparse.Namespace
) -> Tuple[Pipeline, Dict[str, object], pd.DataFrame, pd.Series]:
    X = df[MODEL_FEATURES]
    y = df["is_delayed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    print(
        f"[model] Accuracy: {metrics['accuracy']:.3f} | "
        f"ROC-AUC: {metrics['roc_auc']:.3f} | "
        f"Positive rate: {metrics['positive_rate']:.3f}"
    )
    return pipeline, metrics, X_test, y_test


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[info] Loading data from {csv_path} "
        f"(max_rows={args.max_rows}, sample_frac={args.sample_frac})"
    )
    df_raw = load_dataset(csv_path, args.max_rows, args.sample_frac, args.random_state)
    print(f"[info] Loaded {len(df_raw):,} rows.")

    df = engineer_features(df_raw, args.delay_threshold)
    print(
        f"[info] Rows after cleaning/feature engineering: {len(df):,}. "
        f"Delay positives: {df['is_delayed'].sum():,} "
        f"({df['is_delayed'].mean():.2%})."
    )
    if df["is_delayed"].nunique() < 2:
        raise RuntimeError(
            "Binary delay label contains only one class. "
            "Adjust --delay-threshold, --max-rows, or --sample-frac."
        )

    pipeline, metrics, X_test, y_test = train_and_evaluate(df, args)
    save_metrics(metrics, outputs_dir)

    plot_boxplots(df, outputs_dir)
    plot_heatmap(df, outputs_dir)
    plot_roc_curve(pipeline, X_test, y_test, outputs_dir)


if __name__ == "__main__":
    main()

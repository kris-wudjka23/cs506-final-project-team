"""
Reporting helpers shared by the linear baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from linear_workflow import BOXPLOT_FEATURES, HEATMAP_MAX_DAYS, MODEL_FEATURES

plt.switch_backend("Agg")


def evaluate_split(
    pipeline: Pipeline, df: pd.DataFrame
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    X = df[MODEL_FEATURES]
    y_true = df["is_delayed"].to_numpy()
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "positive_rate": float(y_true.mean()),
        "support": int(y_true.shape[0]),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics, y_pred, y_proba


def save_metrics(
    metrics: Dict[str, Dict[str, float]], summary: Dict[str, object], outputs_dir: Path, filename: str
) -> None:
    payload = {"summary": summary, **metrics}
    (outputs_dir / filename).write_text(json.dumps(payload, indent=2))


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
        plt.tight_layout()
        plt.savefig(outputs_dir / f"boxplot_{feature}.png", dpi=150)
        plt.close()


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
        pivot = pivot.iloc[-HEATMAP_MAX_DAYS :]
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
    plt.savefig(outputs_dir / "heatmap_delay_by_date_hour.png", dpi=150)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    labels = ["On time", "Delayed"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(
    pipeline: Pipeline,
    df: pd.DataFrame,
    outputs_dir: Path,
    prefix: str,
) -> None:
    X = df[MODEL_FEATURES]
    y_true = df["is_delayed"]
    RocCurveDisplay.from_estimator(
        pipeline,
        X,
        y_true,
        name=f"{prefix} Linear Model",
    )
    plt.tight_layout()
    plt.savefig(outputs_dir / f"{prefix.lower()}_roc_curve.png", dpi=150)
    plt.close()

    y_proba = pipeline.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outputs_dir / f"{prefix.lower()}_pr_curve.png", dpi=150)
    plt.close()


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray, outputs_dir: Path) -> None:
    plt.figure(figsize=(6, 5))
    CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=10)
    plt.title("Calibration Curve (Test)")
    plt.tight_layout()
    plt.savefig(outputs_dir / "calibration_curve.png", dpi=150)
    plt.close()


def plot_slice_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    column: str,
    outputs_dir: Path,
) -> None:
    if column not in df.columns:
        return
    data = pd.DataFrame(
        {
            column: df[column],
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    ).dropna(subset=[column])
    agg = (
        data.groupby(column)
        .agg(
            support=("y_true", "size"),
            positive_rate=("y_true", "mean"),
            predicted_positive=("y_pred", "mean"),
            avg_proba=("y_proba", "mean"),
        )
        .reset_index()
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=agg, x=column, y="positive_rate", marker="o", label="True positive rate")
    sns.lineplot(data=agg, x=column, y="avg_proba", marker="o", label="Avg predicted prob")
    plt.title(f"Slice Metrics by {column.title()}")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.savefig(outputs_dir / f"slice_metrics_{column}.png", dpi=150)
    plt.close()


def plot_coefficients(pipeline: Pipeline, outputs_dir: Path, filename: str) -> None:
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    if not hasattr(model, "coef_"):
        return
    feature_names = preprocess.get_feature_names_out()
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top = coef_df.sort_values("abs_coef", ascending=False).head(20)
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=top.sort_values("coef"),
        x="coef",
        y="feature",
        palette="coolwarm",
    )
    plt.title("Top Coefficients")
    plt.tight_layout()
    plt.savefig(outputs_dir / filename, dpi=150)
    plt.close()


__all__ = [
    "evaluate_split",
    "plot_boxplots",
    "plot_calibration",
    "plot_confusion",
    "plot_coefficients",
    "plot_heatmap",
    "plot_roc_pr_curves",
    "plot_slice_metrics",
    "save_metrics",
]

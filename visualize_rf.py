#!/usr/bin/env python3
"""
Visualize RF artifacts: confusion matrices, feature importances, and metric summaries.

Usage:
  python visualize_rf.py --out_dir artifacts

Outputs:
  - artifacts/feature_importance.png
  - artifacts/roc_pr_summary.txt
  - artifacts/confusion_matrix.png (already produced by trainer; reused if present)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(fi_path: Path, out_path: Path, top_n: int = 20):
    fi = pd.read_csv(fi_path)
    fi_top = fi.head(top_n)
    plt.figure(figsize=(8, 6))
    plt.barh(fi_top["feature"][::-1], fi_top["importance"][::-1], color="#1f77b4")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def summarize_metrics(metrics_txt: Path, summary_json: Path, out_txt: Path):
    lines = metrics_txt.read_text(encoding="utf-8").strip().splitlines()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    roc_val = summary.get("metrics", {}).get("validation", {}).get("roc_auc")
    pr_val = summary.get("metrics", {}).get("validation", {}).get("pr_auc")
    roc_test = summary.get("metrics", {}).get("test", {}).get("roc_auc")
    pr_test = summary.get("metrics", {}).get("test", {}).get("pr_auc")
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("=== Metrics (from metrics.txt) ===\n")
        f.write("\n".join(lines))
        f.write("\n\n=== Summary JSON ===\n")
        f.write(f"Validation ROC-AUC: {roc_val}\n")
        f.write(f"Validation PR-AUC:  {pr_val}\n")
        f.write(f"Test ROC-AUC:       {roc_test}\n")
        f.write(f"Test PR-AUC:        {pr_test}\n")


def main():
    ap = argparse.ArgumentParser(description="Visualize RF artifacts")
    ap.add_argument("--out_dir", default="artifacts", help="Directory containing metrics.txt, feature_importance.csv, training_summary.json")
    ap.add_argument("--top_n", type=int, default=20, help="Top N features to plot")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fi_csv = out_dir / "feature_importance.csv"
    metrics_txt = out_dir / "metrics.txt"
    summary_json = out_dir / "training_summary.json"

    if fi_csv.exists():
        plot_feature_importance(fi_csv, out_dir / "feature_importance.png", top_n=args.top_n)
        print(f"Wrote feature importance plot: {out_dir / 'feature_importance.png'}")
    else:
        print("feature_importance.csv not found; skipping feature importance plot.")

    if metrics_txt.exists() and summary_json.exists():
        summarize_metrics(metrics_txt, summary_json, out_dir / "roc_pr_summary.txt")
        print(f"Wrote metric summary: {out_dir / 'roc_pr_summary.txt'}")
    else:
        print("metrics.txt or training_summary.json not found; skipping metric summary.")


if __name__ == "__main__":
    main()

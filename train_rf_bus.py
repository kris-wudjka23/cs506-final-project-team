#!/usr/bin/env python3
"""
train_rf_bus.py — Binary Random Forest trainer (delayed vs not_delayed) with processed-data caching.

New:
- Processed DataFrame cache to avoid re-reading and re-processing the big CSV each run.
- Cache lives at <out_dir>/cache/processed.parquet (or .pkl fallback).
- Use --rebuild_cache to force a refresh.

Speed flags:
--max_features {sqrt,log2,auto} or a number
--max_samples (0<frac<=1) to subsample rows per tree (0 or omit = use all rows)

Includes:
- Safe dtype handling for mixed string/int categoricals
- OrdinalEncoder for fast encoding
- Fixed class weights (numpy ndarray) to avoid warnings/errors
- Progress bar for test prediction
- ROC-AUC and PR-AUC metrics

run: python train_rf_bus.py --csv mbta_bus.csv --out_dir artifacts
"""

import argparse, json, time, os
from math import ceil
from pathlib import Path
from typing import Optional, Union

from datetime import datetime
try:
    import holidays  # pip install holidays
except ImportError:
    holidays = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load, dump as joblib_dump, load as joblib_load
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x


# ---------- Helper functions ----------
def ensure_out_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _us_holidays_index(min_year: int, max_year: int) -> pd.DatetimeIndex:
    """Return a DatetimeIndex of US holidays covering [min_year, max_year]."""
    if holidays is None or min_year is None or max_year is None:
        return pd.DatetimeIndex([])
    try:
        yrs = range(int(min_year), int(max_year) + 1)
        # holidays.US(...).keys() yields date objects — wrap in list, then to_datetime
        return pd.to_datetime(list(holidays.US(years=yrs).keys()))
    except Exception:
        return pd.DatetimeIndex([])


def enrich_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features safely, including holiday and school-in-session flags."""
    d = df.copy()
    d["service_date"] = pd.to_datetime(d["service_date"], errors="coerce")

    d["month"] = d["service_date"].dt.month.astype("Int64")
    d["day_of_year"] = d["service_date"].dt.dayofyear.astype("Int64")
    # seasons: 1=winter,2=spring,3=summer,4=fall
    d["season"] = ((d["month"] % 12 + 3) // 3).astype("Int64")

    # Holidays (cover min..max years present)
    years = d["service_date"].dt.year
    miny = int(years.min()) if years.notna().any() else None
    maxy = int(years.max()) if years.notna().any() else None
    us_h = _us_holidays_index(miny or 2020, maxy or 2026)
    d["is_holiday"] = d["service_date"].isin(us_h).astype("Int64")

    # Simple Boston-ish "school in session" heuristic (leak-safe)
    def _school_in_session(ts):
        if pd.isna(ts): return pd.NA
        m, dd = ts.month, ts.day
        return int((m == 1 and dd >= 15) or (2 <= m <= 4) or
                    (m == 5 and dd <= 15) or (9 <= m <= 11) or
                    (m == 12 and dd <= 20))
    d["is_school_in_session"] = d["service_date"].apply(_school_in_session).astype("Int64")
    return d


def make_multiclass_label(df: pd.DataFrame, tol_sec: float) -> pd.Series:
    bins = [-float("inf"), -tol_sec, tol_sec, float("inf")]
    labels = ["early", "on-time", "delayed"]
    return pd.cut(pd.to_numeric(df["delay_seconds"], errors="coerce"), bins=bins, labels=labels)


def to_binary(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("on-time")
    return (s == "delayed").astype(int)


def confusion_matrix_figure(cm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        ylabel="True label", xlabel="Predicted label", title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def predict_with_progress(pre, rf, X_test, batch_size):
    Xte = pre.transform(X_test)
    n = Xte.shape[0]
    preds, proba1 = [], []
    nb = max(1, (n + batch_size - 1) // max(1, batch_size))
    for bi in tqdm(range(nb), total=nb, desc="Testing (batches)", unit="batch"):
        sl = slice(bi * batch_size, min((bi + 1) * batch_size, n))
        p = rf.predict_proba(Xte[sl])
        preds.append(np.argmax(p, axis=1))
        proba1.append(p[:, 1])
    return np.concatenate(preds), np.concatenate(proba1)


def parse_max_features(v):
    if v in (None, "auto", "none"): return None
    v = str(v).lower()
    if v in {"sqrt", "log2"}: return v
    try:
        x = float(v); return int(x) if x >= 1 else x
    except Exception:
        return "log2"


def parse_max_samples(v):
    if v is None or v <= 0: return None
    return float(v) if v <= 1 else int(v)


# ---------- Cache helpers ----------
def _cache_paths(out_dir: Path):
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / "processed.parquet"
    pickle_path = cache_dir / "processed.pkl"
    return cache_dir, parquet_path, pickle_path


def _cache_is_fresh(csv_path: Path, cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    try:
        return cache_path.stat().st_mtime >= csv_path.stat().st_mtime
    except Exception:
        return False


def load_or_build_processed_df(csv_path: Path, out_dir: Path, tol_sec: float, rebuild: bool = False) -> pd.DataFrame:
    cache_dir, pq_path, pkl_path = _cache_paths(out_dir)

    # Try to reuse a fresh cache
    if not rebuild and (_cache_is_fresh(csv_path, pq_path) or _cache_is_fresh(csv_path, pkl_path)):
        try:
            if pq_path.exists():
                print(f"✅ Loading processed cache: {pq_path}")
                return pd.read_parquet(pq_path)
            else:
                print(f"✅ Loading processed cache: {pkl_path}")
                return joblib_load(pkl_path)
        except Exception as e:
            print(f"⚠️  Cache load failed ({e}); rebuilding…")

    # Build fresh from CSV
    print("📂 Loading CSV and building processed cache…")
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        dtype={"route_id": "string", "direction_id": "string",
                "stop_id": "string", "point_type": "string",
                "weather_condition": "string"},
    )

    # Calendar features
    df = enrich_calendar(df)

    # Rainy rush-hour feature
    # Use NA-safe checks
    df["hour"] = pd.to_numeric(df.get("hour"), errors="coerce")
    df["precip_mm"] = pd.to_numeric(df.get("precip_mm"), errors="coerce")
    hr = df["hour"].fillna(-1).astype("Int64")
    pr = df["precip_mm"].fillna(0)
    df["rainy_rush_hour"] = ((pr > 0.2) & (hr.between(7, 9) | hr.between(16, 18))).astype("Int64")

    # Numeric coercion for training columns (keeps NaNs for median impute later)
    num_cols = ["hour","weekday","is_weekend","time_point_order",
                "air_temp_c","rel_humidity_pct","precip_mm",
                "wind_dir_deg","wind_speed_kmh","pressure_hpa","cloud_cover",
                "month", "day_of_year", "is_holiday", "is_school_in_session", 
                "rainy_rush_hour"]
    for c in num_cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # Labels
    df["label_mc"] = make_multiclass_label(df, tol_sec).astype("string")
    df["label"] = to_binary(df["label_mc"])  # 1=delayed, 0=not_delayed

    # Save cache (Parquet preferred)
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(pq_path, engine="pyarrow", compression="snappy", index=False)
        print(f"Processed cache written: {pq_path}")
    except Exception as e:
        print(f"Parquet write unavailable ({e}); saving pickle cache instead.")
        joblib_dump(df, pkl_path)
        print(f"Processed cache written: {pkl_path}")

    return df


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Binary RF on MBTA delays with speed flags + caching.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--tol_sec", type=float, default=300.0)
    ap.add_argument("--n_estimators", type=int, default=100)
    ap.add_argument("--max_depth", type=int, default=14)
    ap.add_argument("--min_samples_leaf", type=int, default=25)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--top_stops", type=int, default=300)
    ap.add_argument("--max_features", default="log2")
    ap.add_argument("--max_samples", type=float, default=0.0)
    ap.add_argument("--test_batch", type=int, default=20000)
    ap.add_argument("--rebuild_cache", action="store_true",
                    help="Force rebuilding the processed-data cache even if up-to-date.")
    args = ap.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    max_features = parse_max_features(args.max_features)
    max_samples = parse_max_samples(args.max_samples)

    # Load processed data (from cache if fresh)
    df = load_or_build_processed_df(Path(args.csv), out_dir, tol_sec=args.tol_sec, rebuild=args.rebuild_cache)

    # Make sure datetime exists (cached already has it, but keep safe)
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")

    num_cols = ["hour","weekday","is_weekend","time_point_order",
                "air_temp_c","rel_humidity_pct","precip_mm",
                "wind_dir_deg","wind_speed_kmh","pressure_hpa","cloud_cover",
                "month", "day_of_year", "is_holiday", "is_school_in_session", 
                "rainy_rush_hour"]

    # Date-based split
    train = df[(df["service_date"] >= "2023-01-01") & (df["service_date"] < "2024-01-01")].copy()
    test  = df[(df["service_date"] >= "2024-01-01") & (df["service_date"] < "2025-01-01")].copy()

    # Reduce stop_id cardinality if requested (depends on training distribution → done after caching)
    if args.top_stops and args.top_stops > 0:
        top_stops = train["stop_id"].value_counts().nlargest(args.top_stops).index
        train.loc[~train["stop_id"].isin(top_stops), "stop_id"] = "other"
        test.loc[~test["stop_id"].isin(top_stops), "stop_id"] = "other"

    cat = ["route_id","direction_id","stop_id","point_type","weather_condition","season"]

    for c in cat:
        train[c] = train[c].astype("string").fillna("<missing>")
        test[c]  = test[c].astype("string").fillna("<missing>")
    for c in num_cols:
        med = pd.to_numeric(train[c], errors="coerce").median()
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(med)
        test[c]  = pd.to_numeric(test[c], errors="coerce").fillna(med)

    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2), cat),
        ("num", "passthrough", num_cols)
    ])

    classes_arr = np.array([0, 1], dtype=int)
    y_train_arr = np.asarray(train["label"], dtype=int)
    cw = compute_class_weight(class_weight="balanced", classes=classes_arr, y=y_train_arr)
    class_weight = {int(k): float(v) for k, v in zip(classes_arr, cw)}

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=args.random_state,
    )
    pipe = Pipeline([("pre", pre), ("rf", rf)])

    # Fit
    t0 = time.time()
    pipe.fit(train[cat+num_cols], y_train_arr)
    print(f"[done] Fit completed in {(time.time()-t0)/60:.1f} min")

    # Evaluate
    pre_fitted = pipe.named_steps["pre"]
    rf_fitted = pipe.named_steps["rf"]
    y_pred, y_prob = predict_with_progress(pre_fitted, rf_fitted, test[cat+num_cols], args.test_batch)

    y_true = np.asarray(test["label"], dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    report = classification_report(y_true, y_pred, target_names=["not_delayed","delayed"], digits=4)
    roc, aupr = roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)

    # Save
    (out_dir / "metrics.txt").write_text(report + f"ROC-AUC: {roc:.4f}\nPR-AUC: {aupr:.4f}\n", encoding="utf-8")
    confusion_matrix_figure(cm, ["not_delayed","delayed"], out_dir / "confusion_matrix.png", "Confusion Matrix (Binary)")

    fi = pd.DataFrame({"feature": cat+num_cols, "importance": rf_fitted.feature_importances_}).sort_values("importance", ascending=False)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)
    dump(pipe, out_dir / "model.joblib")

    summary = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "class_balance_train_binary": train["label"].value_counts(normalize=True).to_dict(),
        "class_balance_test_binary":  test["label"].value_counts(normalize=True).to_dict(),
        "params": {
            "tol_sec": args.tol_sec,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
            "top_stops": args.top_stops,
            "test_batch": args.test_batch,
            "max_features": args.max_features,
            "max_samples": None if max_samples is None else float(max_samples) if isinstance(max_samples, float) else int(max_samples),
        },
        "encoder": "OrdinalEncoder",
        "fit_minutes": round((time.time()-t0)/60, 2),
        "metrics": {"roc_auc": float(roc), "pr_auc": float(aupr)},
        "cache": {
            "used": True,
            "path": str((out_dir / "cache").resolve()),
            "format": "parquet_if_available_else_pickle"
        }
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Binary Classification Report (2024 test) ===")
    print(report)
    print(f"ROC-AUC: {roc:.4f}  |  PR-AUC: {aupr:.4f}")
    print(f"Artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    # Keep native math libs from oversubscribing (helps stability/memory on macOS)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    main()

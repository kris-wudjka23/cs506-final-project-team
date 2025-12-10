#!/usr/bin/env python3
"""
train_rf_bus.py — Binary Random Forest trainer (delayed vs not_delayed) with processed-data caching.

What it does:
- Builds a cached, preprocessed Parquet from the raw CSV (use --rebuild_cache to refresh).
- Drops invalid delay rows, computes calendar features, and uses target encoding (route_id, stop_id) + ordinal encoding for other cats.
- Temporal split: train (2023-01 to 2023-09), val (2023-10 to 2023-12), test (2024).
- Stage 1: fit on train, evaluate on val. Stage 2: fit on train+val, evaluate on test.
- Saves metrics, confusion matrices, feature importances, model.joblib, and training_summary.json.

Speed controls:
- --max_features {sqrt,log2,auto} or a number
- --max_samples (0<frac<=1) to subsample rows per tree (default 0.3 to save memory)

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
from category_encoders import TargetEncoder
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

    # Drop rows with missing/invalid delay_seconds to avoid label noise
    df["delay_seconds"] = pd.to_numeric(df.get("delay_seconds"), errors="coerce")
    mask_valid_delay = df["delay_seconds"].between(-7200, 7200)
    n_before = len(df)
    df = df[mask_valid_delay].copy()
    n_after = len(df)
    if n_after < n_before:
        print(f"Filtered out {n_before - n_after} rows with missing/invalid delay_seconds.")
    df.attrs["filtered_invalid_delay"] = int(n_before - n_after)

    # Calendar features
    df = enrich_calendar(df)

    # Rainy rush-hour feature
    # Use NA-safe checks
    df["hour"] = pd.to_numeric(df.get("hour"), errors="coerce")
    df["precip_mm"] = pd.to_numeric(df.get("precip_mm"), errors="coerce")
    hr = df["hour"].fillna(-1).astype("Int64")
    pr = df["precip_mm"].fillna(0)
    df["rainy_rush_hour"] = ((pr > 0.2) & (hr.between(7, 9) | hr.between(16, 18))).astype("Int64")

    # Delay filtering/clipping
    df["delay_seconds"] = pd.to_numeric(df.get("delay_seconds"), errors="coerce")
    mask_valid_delay = df["delay_seconds"].between(-7200, 7200)
    n_before = len(df)
    df = df[mask_valid_delay].copy()
    n_after = len(df)
    if n_after < n_before:
        print(f"Filtered out {n_before - n_after} rows with missing/invalid delay_seconds.")
    df.attrs["filtered_invalid_delay"] = int(n_before - n_after)
    df["delay_sec_clipped"] = df["delay_seconds"].clip(-7200, 7200)

    # Numeric coercion for training columns (keeps NaNs for median impute later)
    num_cols = ["hour","weekday","is_weekend","time_point_order",
                "air_temp_c","rel_humidity_pct","precip_mm",
                "wind_dir_deg","wind_speed_kmh","pressure_hpa","cloud_cover",
                "month", "day_of_year", "is_holiday", "is_school_in_session", 
                "rainy_rush_hour"]
    for c in num_cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").astype("float32")

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
    ap.add_argument("--max_samples", type=float, default=0.3)
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

    # Date-based split (temporal: train → val → test)
    train = df[(df["service_date"] >= "2023-01-01") & (df["service_date"] < "2023-10-01")].copy()
    val   = df[(df["service_date"] >= "2023-10-01") & (df["service_date"] < "2024-01-01")].copy()
    test  = df[(df["service_date"] >= "2024-01-01") & (df["service_date"] < "2025-01-01")].copy()

    # Keep pristine copies for re-imputation and final train+val fit
    train_raw, val_raw, test_raw = train.copy(), val.copy(), test.copy()

    cat_target = ["route_id","stop_id"]
    cat_other = ["direction_id","point_type","weather_condition","season"]
    cat_all = cat_target + cat_other
    cat = cat_all

    def apply_top_stops(base_train: pd.DataFrame, others: list[pd.DataFrame], top_n: int):
        if not top_n or top_n <= 0:
            return base_train, others, None
        top = base_train["stop_id"].value_counts().nlargest(top_n).index
        base_train.loc[~base_train["stop_id"].isin(top), "stop_id"] = "other"
        for odf in others:
            odf.loc[~odf["stop_id"].isin(top), "stop_id"] = "other"
        return base_train, others, top

    def fill_cats_and_nums(train_df: pd.DataFrame, other_dfs: list[pd.DataFrame]):
        for c in cat_all:
            train_df[c] = train_df[c].astype("string").fillna("<missing>")
            for odf in other_dfs:
                odf[c] = odf[c].astype("string").fillna("<missing>")
        for c in num_cols:
            med = pd.to_numeric(train_df[c], errors="coerce").median()
            train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(med)
            for odf in other_dfs:
                odf[c] = pd.to_numeric(odf[c], errors="coerce").fillna(med)
        return train_df, other_dfs

    def build_preprocessor():
        return ColumnTransformer([
            ("target", TargetEncoder(cols=cat_target), cat_target),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2), cat_other),
            ("num", "passthrough", num_cols)
        ])

    classes_arr = np.array([0, 1], dtype=int)

    metrics_txt_blocks = []
    metrics_summary = {}

    # ---- Stage 1: train-only fit, evaluate on validation ----
    train_stage1 = train_raw.copy()
    val_stage1 = val_raw.copy()
    train_stage1, [val_stage1], _ = apply_top_stops(train_stage1, [val_stage1], args.top_stops)
    train_stage1, [val_stage1] = fill_cats_and_nums(train_stage1, [val_stage1])

    y_train_stage1 = np.asarray(train_stage1["label"], dtype=int)
    cw1 = compute_class_weight(class_weight="balanced", classes=classes_arr, y=y_train_stage1)
    class_weight = {int(k): float(v) for k, v in zip(classes_arr, cw1)}

    pre_stage1 = build_preprocessor()
    rf_stage1 = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=args.random_state,
    )
    pipe = Pipeline([("pre", pre_stage1), ("rf", rf_stage1)])

    t0 = time.time()
    pipe.fit(train_stage1[cat+num_cols], y_train_stage1)
    fit_minutes_stage1 = round((time.time()-t0)/60, 2)
    print(f"[done] Stage1 fit (train only) completed in {fit_minutes_stage1:.1f} min")

    if len(val_stage1):
        pre_fitted = pipe.named_steps["pre"]
        rf_fitted = pipe.named_steps["rf"]
        X_val = val_stage1[cat + num_cols]
        y_val_true = np.asarray(val_stage1["label"], dtype=int)
        y_val_pred, y_val_prob = predict_with_progress(pre_fitted, rf_fitted, X_val, args.test_batch)
        cm_val = confusion_matrix(y_val_true, y_val_pred, labels=[0, 1])
        report_val = classification_report(y_val_true, y_val_pred, target_names=["not_delayed", "delayed"], digits=4)
        roc_val, aupr_val = roc_auc_score(y_val_true, y_val_prob), average_precision_score(y_val_true, y_val_prob)

        metrics_txt_blocks.append(
            f"=== VALIDATION (train-only model) ===\n{report_val}ROC-AUC: {roc_val:.4f}\nPR-AUC: {aupr_val:.4f}\n"
        )
        confusion_matrix_figure(cm_val, ["not_delayed", "delayed"], out_dir / "confusion_matrix_val.png", "Confusion Matrix (Validation)")
        metrics_summary["validation"] = {"roc_auc": float(roc_val), "pr_auc": float(aupr_val)}
    else:
        metrics_txt_blocks.append("=== VALIDATION (train-only model) ===\n(no rows)\n")
        metrics_summary["validation"] = {"roc_auc": None, "pr_auc": None}

    # ---- Stage 2: retrain on train+val, evaluate on test ----
    train_val_final = pd.concat([train_raw, val_raw], axis=0, ignore_index=True)
    test_final = test_raw.copy()
    train_val_final, [test_final], _ = apply_top_stops(train_val_final, [test_final], args.top_stops)
    train_val_final, [test_final] = fill_cats_and_nums(train_val_final, [test_final])

    y_train_final = np.asarray(train_val_final["label"], dtype=int)
    cw2 = compute_class_weight(class_weight="balanced", classes=classes_arr, y=y_train_final)
    class_weight_final = {int(k): float(v) for k, v in zip(classes_arr, cw2)}

    pre_final = build_preprocessor()
    rf_final = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        class_weight=class_weight_final,
        n_jobs=-1,
        random_state=args.random_state,
    )
    pipe_final = Pipeline([("pre", pre_final), ("rf", rf_final)])

    t1 = time.time()
    pipe_final.fit(train_val_final[cat+num_cols], y_train_final)
    fit_minutes_stage2 = round((time.time()-t1)/60, 2)
    print(f"[done] Stage2 fit (train+val) completed in {fit_minutes_stage2:.1f} min")

    pre_fitted_f = pipe_final.named_steps["pre"]
    rf_fitted_f = pipe_final.named_steps["rf"]
    if len(test_final):
        X_test = test_final[cat + num_cols]
        y_test_true = np.asarray(test_final["label"], dtype=int)
        y_test_pred, y_test_prob = predict_with_progress(pre_fitted_f, rf_fitted_f, X_test, args.test_batch)
        cm_test = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1])
        report_test = classification_report(y_test_true, y_test_pred, target_names=["not_delayed", "delayed"], digits=4)
        roc_test, aupr_test = roc_auc_score(y_test_true, y_test_prob), average_precision_score(y_test_true, y_test_prob)
        metrics_txt_blocks.append(
            f"=== TEST (final model: train+val) ===\n{report_test}ROC-AUC: {roc_test:.4f}\nPR-AUC: {aupr_test:.4f}\n"
        )
        confusion_matrix_figure(cm_test, ["not_delayed", "delayed"], out_dir / "confusion_matrix.png", "Confusion Matrix (Test)")
        metrics_summary["test"] = {"roc_auc": float(roc_test), "pr_auc": float(aupr_test)}
    else:
        metrics_txt_blocks.append("=== TEST (final model: train+val) ===\n(no rows)\n")
        metrics_summary["test"] = {"roc_auc": None, "pr_auc": None}

    # Save metrics text
    (out_dir / "metrics.txt").write_text("\n".join(metrics_txt_blocks), encoding="utf-8")

    feature_names = cat_target + cat_other + num_cols
    fi = pd.DataFrame({"feature": feature_names, "importance": rf_fitted_f.feature_importances_}).sort_values("importance", ascending=False)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)
    dump(pipe_final, out_dir / "model.joblib")

    summary = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_raw)),
        "rows_val": int(len(val_raw)),
        "rows_train_val": int(len(train_val_final)),
        "rows_test": int(len(test_raw)),
        "rows_filtered_invalid_delay": int(df.attrs.get("filtered_invalid_delay", 0)),
        "class_balance_train_binary": train_raw["label"].value_counts(normalize=True).to_dict(),
        "class_balance_val_binary":   val_raw["label"].value_counts(normalize=True).to_dict(),
        "class_balance_train_val_binary": train_val_final["label"].value_counts(normalize=True).to_dict(),
        "class_balance_test_binary":  test_raw["label"].value_counts(normalize=True).to_dict(),
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
        "encoder": "TargetEncoder(route_id, stop_id) + OrdinalEncoder(others)",
        "fit_minutes_stage1": fit_minutes_stage1,
        "fit_minutes_stage2": fit_minutes_stage2,
        "metrics": metrics_summary,
        "cache": {
            "used": True,
            "path": str((out_dir / "cache").resolve()),
            "format": "parquet_if_available_else_pickle"
        }
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Validation metrics (train-only model) written to metrics.txt ===")
    print("=== Test metrics (final model) written to metrics.txt ===")
    print(f"Artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    # Keep native math libs from oversubscribing (helps stability/memory on macOS)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    main()

"""
Shared data loading helpers for MBTA bus delay models.

This module replicates the processed-cache flow from train_rf_bus.py so that
the logistic-regression and gradient-descent baselines can reuse the exact same
cleaned dataset without touching the random-forest trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import holidays  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    holidays = None

# Column groups used downstream by both linear baselines and the RF trainer.
NUMERIC_COLUMNS = [
    "hour",
    "weekday",
    "is_weekend",
    "time_point_order",
    "air_temp_c",
    "rel_humidity_pct",
    "precip_mm",
    "wind_dir_deg",
    "wind_speed_kmh",
    "pressure_hpa",
    "cloud_cover",
    "month",
    "day_of_year",
    "is_holiday",
    "is_school_in_session",
    "rainy_rush_hour",
]

CATEGORY_COLUMNS = [
    "route_id",
    "stop_id",
    "direction_id",
    "point_type",
    "weather_condition",
    "season",
]


def _cache_paths(cache_dir: Path) -> Tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / "processed.parquet"
    pickle_path = cache_dir / "processed.pkl"
    return parquet_path, pickle_path


def _us_holidays_index(min_year: int | None, max_year: int | None) -> pd.DatetimeIndex:
    """Return a DatetimeIndex of US holidays covering [min_year, max_year]."""
    if holidays is None or min_year is None or max_year is None:
        return pd.DatetimeIndex([])
    try:
        years = range(int(min_year), int(max_year) + 1)
        keys = holidays.US(years=years).keys()  # type: ignore[attr-defined]
        return pd.to_datetime(list(keys))
    except Exception:
        return pd.DatetimeIndex([])


def _enrich_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features (month/day-of-year/season/holiday/school flags)."""
    data = df.copy()
    data["service_date"] = pd.to_datetime(data["service_date"], errors="coerce")
    data["month"] = data["service_date"].dt.month.astype("Int64")
    data["day_of_year"] = data["service_date"].dt.dayofyear.astype("Int64")
    data["season"] = ((data["month"] % 12 + 3) // 3).astype("Int64")

    years = data["service_date"].dt.year
    min_year = int(years.min()) if years.notna().any() else None
    max_year = int(years.max()) if years.notna().any() else None
    holidays_idx = _us_holidays_index(min_year, max_year)
    data["is_holiday"] = data["service_date"].isin(holidays_idx).astype("Int64")

    def _school_in_session(ts: pd.Timestamp | pd.NaT) -> int | pd.NA:
        if pd.isna(ts):
            return pd.NA
        month, day = ts.month, ts.day
        in_session = (
            (month == 1 and day >= 15)
            or (2 <= month <= 4)
            or (month == 5 and day <= 15)
            or (9 <= month <= 11)
            or (month == 12 and day <= 20)
        )
        return int(in_session)

    data["is_school_in_session"] = data["service_date"].apply(_school_in_session).astype(
        "Int64"
    )
    return data


def _cache_is_fresh(csv_path: Path, cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    try:
        return cache_path.stat().st_mtime >= csv_path.stat().st_mtime
    except Exception:
        return False


def load_or_build_processed_df(
    csv_path: Path,
    cache_dir: Path,
    tol_sec: float = 300.0,
    rebuild_cache: bool = False,
) -> pd.DataFrame:
    """
    Load the processed/cached dataset used by all models.

    Args:
        csv_path: Path to the raw integrated MBTA + weather CSV.
        cache_dir: Directory where processed.parquet or processed.pkl will live.
        tol_sec: Delay tolerance for the binary label (default 5 minutes).
        rebuild_cache: Force cache refresh even if files look fresh.
    """

    pq_path, pkl_path = _cache_paths(cache_dir)
    cache_fresh = _cache_is_fresh(csv_path, pq_path) or _cache_is_fresh(
        csv_path, pkl_path
    )
    if not rebuild_cache and cache_fresh:
        try:
            if pq_path.exists():
                return pd.read_parquet(pq_path)
            if pkl_path.exists():
                from joblib import load as joblib_load

                return joblib_load(pkl_path)
        except Exception:
            # Fall back to rebuilding the cache when load fails.
            pass

    df = pd.read_csv(
        csv_path,
        low_memory=False,
        dtype={
            "route_id": "string",
            "direction_id": "string",
            "stop_id": "string",
            "point_type": "string",
            "weather_condition": "string",
        },
    )

    df["delay_seconds"] = pd.to_numeric(df.get("delay_seconds"), errors="coerce")
    df = df[df["delay_seconds"].between(-7200, 7200)].copy()

    df = _enrich_calendar(df)

    df["hour"] = pd.to_numeric(df.get("hour"), errors="coerce")
    df["precip_mm"] = pd.to_numeric(df.get("precip_mm"), errors="coerce")
    hr = df["hour"].fillna(-1).astype("Int64")
    pr = df["precip_mm"].fillna(0)
    df["rainy_rush_hour"] = (
        (pr > 0.2) & (hr.between(7, 9) | hr.between(16, 18))
    ).astype("Int64")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df.get(column), errors="coerce").astype("float32")

    # Binary label reused by all models (delayed vs not delayed).
    bins = [-float("inf"), -tol_sec, tol_sec, float("inf")]
    labels = ["early", "on-time", "delayed"]
    df["label_mc"] = pd.cut(
        pd.to_numeric(df["delay_seconds"], errors="coerce"), bins=bins, labels=labels
    ).astype("string")
    df["label"] = (df["label_mc"] == "delayed").astype(int)
    df["delay_minutes"] = df["delay_seconds"] / 60.0

    try:
        import pyarrow  # type: ignore

        df.to_parquet(pq_path, engine="pyarrow", compression="snappy", index=False)
    except Exception:
        from joblib import dump as joblib_dump

        joblib_dump(df, pkl_path)

    return df


__all__ = [
    "CATEGORY_COLUMNS",
    "NUMERIC_COLUMNS",
    "load_or_build_processed_df",
]

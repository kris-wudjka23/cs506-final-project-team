"""
Common helpers shared by the linear baselines (logistic regression + GD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    "month",
    "day_of_year",
    "is_holiday",
    "is_school_in_session",
    "rainy_rush_hour",
]

CATEGORICAL_FEATURES = [
    "route_id",
    "stop_id",
    "direction_id",
    "point_type",
    "weather_condition",
    "season",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
BOXPLOT_FEATURES = ["air_temp_c", "precip_mm", "wind_speed_kmh", "cloud_cover"]
HEATMAP_MAX_DAYS = 21


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2023-01-01"
    train_end: str = "2023-10-01"
    val_end: str = "2024-01-01"
    test_end: str = "2025-01-01"


SPLITS = SplitConfig()


def build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # For older sklearn versions.
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
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
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def engineer_linear_features(
    df: pd.DataFrame, delay_threshold_minutes: float
) -> pd.DataFrame:
    data = df.copy()
    data["service_date"] = pd.to_datetime(data["service_date"], errors="coerce")
    data = data.dropna(subset=["service_date", "delay_minutes"])

    if "delay_minutes" not in data.columns:
        data["delay_minutes"] = data["delay_seconds"] / 60.0

    data["is_delayed"] = (data["delay_minutes"] >= delay_threshold_minutes).astype(int)

    for column in ["route_id", "stop_id", "direction_id", "point_type", "weather_condition", "season"]:
        data[column] = data[column].astype(str).str.strip()
        data[column] = data[column].replace({"nan": "<missing>", "": "<missing>"})
    data["route_id"] = data["route_id"].str.zfill(2)

    radians = np.deg2rad(data["wind_dir_deg"])
    data["wind_dir_sin"] = np.sin(radians)
    data["wind_dir_cos"] = np.cos(radians)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    return data


def temporal_split(df: pd.DataFrame, splits: SplitConfig = SPLITS) -> Dict[str, pd.DataFrame]:
    train_mask = (df["service_date"] >= splits.train_start) & (df["service_date"] < splits.train_end)
    val_mask = (df["service_date"] >= splits.train_end) & (df["service_date"] < splits.val_end)
    test_mask = (df["service_date"] >= splits.val_end) & (df["service_date"] < splits.test_end)

    return {
        "train": df[train_mask].copy(),
        "val": df[val_mask].copy(),
        "test": df[test_mask].copy(),
    }


__all__ = [
    "BOXPLOT_FEATURES",
    "CATEGORICAL_FEATURES",
    "HEATMAP_MAX_DAYS",
    "MODEL_FEATURES",
    "NUMERIC_FEATURES",
    "SPLITS",
    "engineer_linear_features",
    "temporal_split",
    "build_preprocessor",
]

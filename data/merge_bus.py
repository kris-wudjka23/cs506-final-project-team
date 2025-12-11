#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from typing import List

import pandas as pd


def list_csv_files(directory: str) -> List[str]:
    """Return all csv file paths under a directory."""
    pattern = os.path.join(directory, "*.csv")
    files = glob.glob(pattern)
    files.sort()
    return files


def load_and_concat(files_2023: List[str], files_2024: List[str]) -> pd.DataFrame:
    """Load all csv files from both lists and concatenate into one DataFrame."""
    all_files = files_2023 + files_2024
    print(f"Number of files: {len(all_files)}")
    for f in all_files:
        print(f)

    dataframes = []
    for i, file in enumerate(all_files):
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"Loaded {i + 1}/{len(all_files)}: {file}")

    bus_all = pd.concat(dataframes, ignore_index=True)
    bus_all = bus_all.drop_duplicates()
    print(f"Concatenated shape (after drop_duplicates): {bus_all.shape}")
    return bus_all


def add_time_features(bus_all: pd.DataFrame) -> pd.DataFrame:
    """Add datetime-related columns and delay features."""

    if "service_date" in bus_all.columns:
        bus_all["service_date"] = pd.to_datetime(bus_all["service_date"])


    bus_all["scheduled_dt_raw"] = pd.to_datetime(
        bus_all["scheduled"], errors="coerce"
    )
    bus_all["actual_dt_raw"] = pd.to_datetime(
        bus_all["actual"], errors="coerce"
    )


    bus_all["scheduled_time"] = bus_all["scheduled_dt_raw"].dt.time
    bus_all["actual_time"] = bus_all["actual_dt_raw"].dt.time


    def combine_date_time(row, date_col: str, time_col: str):
        if pd.notnull(row[date_col]) and pd.notnull(row[time_col]):
            return pd.Timestamp.combine(row[date_col].date(), row[time_col])
        return pd.NaT

    bus_all["scheduled_dt"] = bus_all.apply(
        lambda r: combine_date_time(r, "service_date", "scheduled_time"), axis=1
    )
    bus_all["actual_dt"] = bus_all.apply(
        lambda r: combine_date_time(r, "service_date", "actual_time"), axis=1
    )

    # delay
    bus_all["delay_seconds"] = (
        bus_all["actual_dt"] - bus_all["scheduled_dt"]
    ).dt.total_seconds()
    bus_all["delay_minutes"] = bus_all["delay_seconds"] / 60.0

    # hour, weekday, weekend
    bus_all["hour"] = bus_all["scheduled_dt"].dt.hour
    bus_all["weekday"] = bus_all["scheduled_dt"].dt.weekday  # 0=Mon
    bus_all["is_weekend"] = bus_all["weekday"].isin([5, 6]).astype(int)

    return bus_all


def select_and_save(bus_all: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Select useful columns and save to csv."""
    keep_cols = [
        "service_date",
        "hour",
        "weekday",
        "is_weekend",
        "route_id",
        "direction_id",
        "stop_id",
        "time_point_order",
        "point_type",
        "scheduled_dt",
        "actual_dt",
        "delay_seconds",
        "delay_minutes",
    ]

    bus_clean = bus_all[keep_cols].copy()


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bus_clean.to_csv(output_path, index=False)
    print(f"Clean dataset saved as {output_path}")
    print("Shape:", bus_clean.shape)
    print("\nColumn types:")
    print(bus_clean.dtypes)
    print("\nPreview of cleaned data:")
    print(bus_clean.head(10))

    return bus_clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge MBTA bus CSVs (2023+2024), "
                    "create time/delay features, and output bus_clean.csv"
    )
    parser.add_argument(
        "--path_2023",
        type=str,
        required=True,
        help="Directory containing 2023 CSV files",
    )
    parser.add_argument(
        "--path_2024",
        type=str,
        required=True,
        help="Directory containing 2024 CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="processdata/bus_clean.csv",
        help="Output CSV path (default: processdata/bus_clean.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    files_2023 = list_csv_files(args.path_2023)
    files_2024 = list_csv_files(args.path_2024)

    if not files_2023:
        raise FileNotFoundError(f"No CSV files found under {args.path_2023}")
    if not files_2024:
        raise FileNotFoundError(f"No CSV files found under {args.path_2024}")

    bus_all = load_and_concat(files_2023, files_2024)
    bus_all = add_time_features(bus_all)
    _ = select_and_save(bus_all, args.output)


if __name__ == "__main__":
    main()
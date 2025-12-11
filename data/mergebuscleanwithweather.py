#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge cleaned MBTA bus data with Boston hourly weather data.

Usage example:

    python merge_bus_weather.py \
        --bus-path processdata/bus_clean.csv \
        --weather-2023 rawdata/2023bostonweather.csv \
        --weather-2024 rawdata/2024bostonweather.csv \
        --output-path processdata/bus_weather_clean.csv
"""

import argparse
import pandas as pd
from pathlib import Path


# ------------------------- helper functions ------------------------- #

def load_bus(bus_path: str) -> pd.DataFrame:
    """Load cleaned bus data."""
    bus_path = Path(bus_path)
    print(f"Loading bus data from: {bus_path}")
    bus = pd.read_csv(
        bus_path,
        parse_dates=["service_date", "scheduled_dt", "actual_dt"]
    )
    print("Loaded bus_clean.csv:", bus.shape)
    return bus


def load_raw_weather(weather_2023_path: str,
                     weather_2024_path: str) -> pd.DataFrame:
    """Load raw weather data for 2023 and 2024 and vertically concatenate."""
    w2023_path = Path(weather_2023_path)
    w2024_path = Path(weather_2024_path)

    print(f"Loading weather 2023 from: {w2023_path}")
    weather_2023 = pd.read_csv(w2023_path)

    print(f"Loading weather 2024 from: {w2024_path}")
    weather_2024 = pd.read_csv(w2024_path)

    weather_raw = pd.concat([weather_2023, weather_2024],
                            ignore_index=True)
    print("Loaded weather data:", weather_raw.shape)
    return weather_raw


def clean_weather(weather_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean weather dataframe and prepare hourly timestamp + features."""
    # drop *_source columns and wpgt
    cols_to_drop = [c for c in weather_raw.columns if c.endswith("_source")]
    cols_to_drop += ["wpgt"]
    weather_raw = weather_raw.drop(columns=cols_to_drop, errors="ignore")

    # create hourly timestamp
    weather_raw["timestamp_hour"] = pd.to_datetime(
        weather_raw[["year", "month", "day", "hour"]]
    )

    # keep useful columns
    keep_weather_cols = [
        "timestamp_hour",
        "temp", "rhum", "prcp", "wdir", "wspd", "pres", "cldc", "coco"
    ]
    weather = weather_raw[keep_weather_cols].copy()

    # rename columns
    weather = weather.rename(columns={
        "temp": "air_temp_c",
        "rhum": "rel_humidity_pct",
        "prcp": "precip_mm",
        "wdir": "wind_dir_deg",
        "wspd": "wind_speed_kmh",
        "pres": "pressure_hpa",
        "cldc": "cloud_cover",
        "coco": "weather_condition"
    })

    # map weather_condition codes to labels
    weather_condition_map = {
        1: "Clear",
        2: "Fair",
        3: "Cloudy",
        4: "Overcast",
        5: "Fog",
        6: "Freezing Fog",
        7: "Light Rain",
        8: "Rain",
        9: "Heavy Rain",
        10: "Freezing Rain",
        11: "Heavy Freezing Rain",
        12: "Sleet",
        13: "Heavy Sleet",
        14: "Light Snowfall",
        15: "Snowfall",
        16: "Heavy Snowfall",
        17: "Rain Shower",
        18: "Heavy Rain Shower",
        19: "Sleet Shower",
        20: "Heavy Sleet Shower",
        21: "Snow Shower",
        22: "Heavy Snow Shower",
        23: "Lightning",
        24: "Hail",
        25: "Thunderstorm",
        26: "Heavy Thunderstorm",
        27: "Storm"
    }

    weather["weather_condition"] = weather["weather_condition"].map(
        weather_condition_map
    )

    print("Weather sample after cleaning:")
    print(weather.head(5))
    return weather


def merge_bus_and_weather(bus: pd.DataFrame,
                          weather: pd.DataFrame,
                          drop_na: bool = True) -> pd.DataFrame:
    """Merge bus data with hourly weather, optionally drop rows with NA."""
    # hour of event for bus data
    bus = bus.copy()
    bus["event_hour"] = bus["actual_dt"].dt.floor("h")

    # merge on hour timestamp
    merged = pd.merge(
        bus,
        weather,
        left_on="event_hour",
        right_on="timestamp_hour",
        how="left"
    )
    merged = merged.drop(columns=["timestamp_hour"])

    print("Merged shape (before dropna):", merged.shape)

    if drop_na:
        merged = merged.dropna()
        print("Merged shape (after dropna):", merged.shape)

    return merged


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    """Save dataframe to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved merged dataset to: {output_path}")


# ------------------------------ main ------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge bus_clean.csv with Boston weather data."
    )
    parser.add_argument(
        "--bus-path",
        default="processdata/bus_clean.csv",
        help="Path to bus_clean.csv (default: processdata/bus_clean.csv).",
    )
    parser.add_argument(
        "--weather-2023",
        default="rawdata/2023bostonweather.csv",
        help="Path to 2023 weather CSV (default: rawdata/2023bostonweather.csv).",
    )
    parser.add_argument(
        "--weather-2024",
        default="rawdata/2024bostonweather.csv",
        help="Path to 2024 weather CSV (default: rawdata/2024bostonweather.csv).",
    )
    parser.add_argument(
        "--output-path",
        default="processdata/bus_weather_clean.csv",
        help="Output CSV path (default: processdata/bus_weather_clean.csv).",
    )
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Do NOT drop rows with missing values after merge "
             "(by default rows with NA are dropped).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bus = load_bus(args.bus_path)
    weather_raw = load_raw_weather(args.weather_2023, args.weather_2024)
    weather = clean_weather(weather_raw)

    merged = merge_bus_and_weather(bus, weather, drop_na=not args.keep_na)

    print("\nMerged data preview:")
    print(merged.head(10))

    save_dataframe(merged, args.output_path)


if __name__ == "__main__":
    main()
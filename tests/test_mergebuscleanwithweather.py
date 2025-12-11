import pandas as pd
import pytest

import data.mergebuscleanwithweather as mbw


def test_load_bus_parses_datetime(tmp_path):
    csv_path = tmp_path / "bus_clean.csv"
    df = pd.DataFrame(
        {
            "service_date": ["2023-01-01"],
            "scheduled_dt": ["2023-01-01 08:00:00"],
            "actual_dt": ["2023-01-01 08:05:00"],
            "route_id": [1],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = mbw.load_bus(str(csv_path))
    assert pd.api.types.is_datetime64_any_dtype(loaded["service_date"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["scheduled_dt"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["actual_dt"])


def test_load_raw_weather_concatenates(tmp_path):
    df2023 = pd.DataFrame(
        {
            "year": [2023],
            "month": [1],
            "day": [1],
            "hour": [0],
            "temp": [1.0],
            "rhum": [50],
            "prcp": [0.0],
            "wdir": [180],
            "wspd": [3.0],
            "pres": [1010],
            "cldc": [0.1],
            "coco": [1],
        }
    )
    df2024 = df2023.copy()

    p2023 = tmp_path / "w2023.csv"
    p2024 = tmp_path / "w2024.csv"
    df2023.to_csv(p2023, index=False)
    df2024.to_csv(p2024, index=False)

    weather_raw = mbw.load_raw_weather(str(p2023), str(p2024))
    assert weather_raw.shape[0] == 2  


def _make_sample_weather_raw():
    return pd.DataFrame(
        {
            "year": [2023, 2023],
            "month": [1, 1],
            "day": [1, 1],
            "hour": [0, 1],
            "temp": [1.0, 2.0],
            "rhum": [50, 60],
            "prcp": [0.0, 0.1],
            "wdir": [180, 190],
            "wspd": [3.0, 4.0],
            "pres": [1010, 1011],
            "cldc": [0.1, 0.2],
            "coco": [1, 3],
            "wpgt": [5.0, 6.0],
            "temp_source": ["x", "y"],
        }
    )


def test_clean_weather_drops_source_columns_and_renames():
    raw = _make_sample_weather_raw()
    weather = mbw.clean_weather(raw)


    assert "wpgt" not in weather.columns
    assert "temp_source" not in weather.columns


    expected_cols = [
        "timestamp_hour",
        "air_temp_c",
        "rel_humidity_pct",
        "precip_mm",
        "wind_dir_deg",
        "wind_speed_kmh",
        "pressure_hpa",
        "cloud_cover",
        "weather_condition",
    ]
    assert list(weather.columns) == expected_cols


    assert list(weather["weather_condition"]) == ["Clear", "Cloudy"]


def test_merge_bus_and_weather_matches_on_hour():

    bus = pd.DataFrame(
        {
            "service_date": [pd.to_datetime("2023-01-01")],
            "actual_dt": [pd.to_datetime("2023-01-01 00:30:00")],
            "route_id": [1],
        }
    )
    weather_raw = _make_sample_weather_raw()
    weather = mbw.clean_weather(weather_raw)

    merged = mbw.merge_bus_and_weather(bus, weather, drop_na=False)

    assert merged.shape[0] == 1
    assert merged.loc[0, "air_temp_c"] == 1.0
    assert merged.loc[0, "event_hour"] == pd.Timestamp("2023-01-01 00:00:00")


def test_save_dataframe_creates_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out_path = tmp_path / "merged.csv"

    mbw.save_dataframe(df, str(out_path))

    assert out_path.is_file()
    loaded = pd.read_csv(out_path)
    pd.testing.assert_frame_equal(loaded, df)
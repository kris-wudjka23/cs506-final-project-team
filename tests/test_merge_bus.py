from pathlib import Path

import pandas as pd
import pytest

import data.merge_bus as merge_bus


def test_list_csv_files_sorted(tmp_path):

    b = tmp_path / "b.csv"
    a = tmp_path / "a.csv"
    c = tmp_path / "c.txt"  

    for p in (b, a, c):
        p.write_text("col1\n1\n")

    files = merge_bus.list_csv_files(str(tmp_path))

    assert [Path(f).name for f in files] == ["a.csv", "b.csv"]


def test_load_and_concat_drops_duplicates(tmp_path):

    df1 = pd.DataFrame(
        {
            "service_date": ["2023-01-01", "2023-01-01"],
            "scheduled": ["2023-01-01 08:00:00", "2023-01-01 08:10:00"],
            "actual": ["2023-01-01 08:05:00", "2023-01-01 08:15:00"],
            "route_id": [1, 1],
            "direction_id": [0, 0],
            "stop_id": ["S1", "S2"],
            "time_point_order": [1, 2],
            "point_type": ["DEPARTURE", "DEPARTURE"],
        }
    )
    df2 = df1.copy()

    f1 = tmp_path / "2023_part1.csv"
    f2 = tmp_path / "2024_part1.csv"
    df1.to_csv(f1, index=False)
    df2.to_csv(f2, index=False)

    merged = merge_bus.load_and_concat([str(f1)], [str(f2)])


    assert merged.shape[0] == 2


def _make_minimal_bus_df():

    return pd.DataFrame(
        {
            "service_date": ["2023-01-01", "2023-01-02"],
            "scheduled": ["2023-01-01 08:00:00", "2023-01-02 09:30:00"],
            "actual": ["2023-01-01 08:05:00", "2023-01-02 09:20:00"],
            "route_id": [1, 1],
            "direction_id": [0, 1],
            "stop_id": ["STOP1", "STOP2"],
            "time_point_order": [1, 2],
            "point_type": ["DEPARTURE", "ARRIVAL"],
        }
    )


def test_add_time_features_creates_delay_and_time_columns():
    bus = _make_minimal_bus_df()

    out = merge_bus.add_time_features(bus)


    for col in [
        "scheduled_dt",
        "actual_dt",
        "delay_seconds",
        "delay_minutes",
        "hour",
        "weekday",
        "is_weekend",
    ]:
        assert col in out.columns


    first_delay_sec = out.loc[0, "delay_seconds"]
    first_delay_min = out.loc[0, "delay_minutes"]
    assert first_delay_sec == 5 * 60
    assert first_delay_min == pytest.approx(5.0)


def test_select_and_save_writes_csv(tmp_path):
    bus = _make_minimal_bus_df()
    bus = merge_bus.add_time_features(bus)

    output_path = tmp_path / "bus_clean.csv"
    result = merge_bus.select_and_save(bus, str(output_path))


    assert output_path.is_file()

    expected_cols = [
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
    assert list(result.columns) == expected_cols


    saved = pd.read_csv(
        output_path,
        parse_dates=["service_date", "scheduled_dt", "actual_dt"],
    )
    assert saved.shape == result.shape
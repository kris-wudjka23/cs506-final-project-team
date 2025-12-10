import pandas as pd
from types import SimpleNamespace

from gradient_descent_pipeline import build_pipeline as build_gd_pipeline
from linear_workflow import MODEL_FEATURES
from logistic_regression_pipeline import build_pipeline as build_log_pipeline


def _synth_dataframe(rows: int = 32) -> pd.DataFrame:
    data = {
        "hour": list(range(rows)),
        "weekday": [i % 7 for i in range(rows)],
        "is_weekend": [int(i % 7 in (5, 6)) for i in range(rows)],
        "time_point_order": [i % 5 for i in range(rows)],
        "air_temp_c": [10 + i * 0.1 for i in range(rows)],
        "rel_humidity_pct": [50 + (i % 20) for i in range(rows)],
        "precip_mm": [0.1 * (i % 3) for i in range(rows)],
        "wind_speed_kmh": [5 + i * 0.05 for i in range(rows)],
        "pressure_hpa": [1010 + i * 0.5 for i in range(rows)],
        "cloud_cover": [i % 8 for i in range(rows)],
        "wind_dir_sin": [0.1 for _ in range(rows)],
        "wind_dir_cos": [0.9 for _ in range(rows)],
        "month": [1 + i % 12 for i in range(rows)],
        "day_of_year": [1 + i % 365 for i in range(rows)],
        "is_holiday": [0 for _ in range(rows)],
        "is_school_in_session": [1 for _ in range(rows)],
        "rainy_rush_hour": [int(i % 2 == 0) for i in range(rows)],
        "route_id": [f"{i%3:02d}" for i in range(rows)],
        "stop_id": [f"{i%5:03d}" for i in range(rows)],
        "direction_id": ["0" if i % 2 == 0 else "1" for i in range(rows)],
        "point_type": ["Stop" for _ in range(rows)],
        "weather_condition": ["Clear" if i % 2 == 0 else "Rain" for i in range(rows)],
        "season": ["1" for _ in range(rows)],
    }
    df = pd.DataFrame(data)
    df["is_delayed"] = [int(i % 3 == 0) for i in range(rows)]
    return df


def test_logistic_pipeline_trains_and_predicts():
    df = _synth_dataframe()
    pipeline = build_log_pipeline()
    pipeline.fit(df[MODEL_FEATURES], df["is_delayed"])
    preds = pipeline.predict(df[MODEL_FEATURES])
    assert preds.shape[0] == df.shape[0]
    proba = pipeline.predict_proba(df[MODEL_FEATURES])
    assert proba.shape == (df.shape[0], 2)


def test_gradient_descent_pipeline_trains_and_predicts():
    df = _synth_dataframe()
    args = SimpleNamespace(
        learning_rate=0.05,
        max_iter=50,
        tol=1e-3,
        alpha=0.0,
        fit_intercept=True,
        verbose=0,
        random_state=0,
    )
    pipeline = build_gd_pipeline(args)
    pipeline.fit(df[MODEL_FEATURES], df["is_delayed"])
    preds = pipeline.predict(df[MODEL_FEATURES])
    assert preds.shape[0] == df.shape[0]

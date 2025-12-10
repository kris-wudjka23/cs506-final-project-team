# MBTA Bus Delay Intelligence Hub

Welcome to the documentation for our CS506 final project. This README consolidates our data sources, engineering steps, visualization catalog, and modeling performance for predicting MBTA bus delays with both interpretable (logistic regression) and non-linear (random forest) pipelines.

---

## Quick Links
- **Presentation video:** [YouTube](https://www.youtube.com/watch?v=AJF210YWvrs)
- **Logistic notebook:** `logistic_regression_analysis.ipynb` (outputs in `outputs/logistic_notebook/`)
- **Random-forest trainer:** `train_rf_bus.py` (artifacts in `outputs/`)
- **Exploratory notebook:** `preliminary_feature_exploration.ipynb`
- **Metrics files:** `outputs/logistic_regression_metrics.json`, `metrics.txt`, `training_summary.json`

---

## Problem Statement
MBTA buses are a primary commute option for Boston-area students. Chronic delays ripple into missed lectures, exams, and work shifts. We ingest two years of MBTA schedule/operations logs and Meteostat weather observations to learn which contextual signals (weather, temporal, route-specific effects) drive delays and to predict whether an upcoming trip will be late.

---

## Data Inventory
| Dataset | Period | Rows | Key Fields | Location |
| --- | --- | --- | --- | --- |
| Monthly MBTA bus performance CSVs | Jan 2023 – Dec 2024 | 24 files (~38 GB) | service_date, route_id, stop_id, scheduled_dt, actual_dt, delay_seconds | `data/` (raw) |
| Meteostat hourly weather snapshots | 2023 – 2024 | 2 CSVs | temperature, humidity, precipitation, wind, cloud_cover, weather_condition | `data/` |
| Integrated bus + weather table | 2023 – 2024 | 46,405,028 rows × 22 columns | All bus/time/weather features | `bus_weather_clean.csv` (generated) |
| Sample caches for experimentation | Configurable | ≤ 1,000,000 rows | Subset used in notebooks | `outputs/logistic_notebook/bus_weather_sample.parquet` |

**Summary statistics (from `training_summary.json`):**
- Training set (2023): 22,907,882 rows, 32.7% delayed label.
- Test set (2024): 23,497,146 rows, 34.4% delayed label.
- Feature groups: operational (routes, stops, point_type), temporal (hour, weekday, season, holiday), weather (temp, precip, wind, pressure, cloud cover), engineered aggregates (route-hour delay averages).

---

## End-to-End Pipeline Overview
1. **Bus preprocessing (Notebooks in `data/` folder):**
   - Merge monthly CSVs, drop duplicates, normalize timestamps (`scheduled_dt`, `actual_dt`, `service_date`).
   - Derive `delay_minutes`, `hour`, `weekday`, `is_weekend`, `time_point_order`, and categorical descriptors formatted as zero-padded strings.
2. **Weather preprocessing:**
   - Concatenate Meteostat yearly tables, remove `_source` metadata, assemble hourly timestamps using year/month/day/hour columns.
   - Map condition codes to human-readable labels and keep atmospheric attributes (air temperature, relative humidity, precipitation, pressure, wind direction/speed, cloud cover).
3. **Integration:**
   - Round bus `scheduled_dt` down to the nearest hour and join with weather by timestamp.
   - Remove duplicates/missing rows and export `bus_weather_clean.csv`.
4. **Feature augmentation (shared utilities inside `train_rf_bus.py` and the logistic notebook):**
   - Calendar features: month, day_of_year, season, `is_holiday` (via `holidays` library), heuristic `is_school_in_session`.
   - Encoded wind direction (sin/cos projection), `rainy_rush_hour` indicator, route-hour aggregate stats (mean/median/count of delay).
   - Optional demo dataset generator in the notebook to test workflows without the massive raw CSV.

---

## Dataset Snapshots
### Bus Monthly Volume (2023–2024)
![Raw MBTA tables](image-1.png)

### Weather Annual Volume
![Raw Meteostat tables](image-2.png)

### Cleaned Integrated Frame
![Combined dataset](image-3.png)

### Bus Table Peek
![Partial bus output](image-4.png)

---

## Visualization Gallery
Our notebooks and scripts save plots to either `outputs/` (RF CLI) or `outputs/logistic_notebook/figures/` (notebook). Highlights include:

| Visualization | File | Description |
| --- | --- | --- |
| Delay histogram by class | `outputs/logistic_notebook/figures/delay_histogram.png` | Shows heavy tail beyond 10 minutes, with delayed label capturing long right tail. |
| Hourly delay profile | `.../hourly_profile.png` | Bar/line overlay reveals AM and PM rush-hour spikes in both mean delay and delay rate. |
| Service-date vs. hour heatmap | `.../delay_heatmap.png` | Last 21 days of the sample with patchy hotspots indicating days with systemic issues. |
| Route-level delay ranking | `.../route_delay_share.png` | Top 15 routes by delay percentage; cross-reference with MBTA planning priorities. |
| Weather boxplots | `outputs/boxplot_air_temp_c.png`, etc. | Four-panel view of temperature, precipitation, wind speed, and cloud cover distributions split by on-time vs delayed. |
| Precipitation vs. probability | `.../precip_vs_delay.png` | Bucketized precip intensity vs. delay rate + mean delay. |
| ROC curve (logistic) | `outputs/logistic_notebook/figures/roc_curve.png` | AUC reflects the sampled logistic baseline. |
| PR curve (logistic) | `.../pr_curve.png` | Shows class imbalance challenge—precision drops when chasing higher recall. |
| Calibration plot | `.../calibration_curve.png` | Logistic outputs are well-calibrated near mid-range probabilities but drift at extremes. |
| Confusion matrices | `outputs/confusion_matrix.png` (RF), `.../confusion_matrix.png` (logistic) | Visualize trade-offs between false negatives vs false positives. |
| Feature importance / coefficients | `feature_importance.csv`, `.../logistic_coefficients.png` | Contrasts tree-based importance with linear coefficient magnitudes. |
| Random-forest feature importance plot | `feature_importance.csv` + optional plotting | Top drivers: hour, precipitation, route_id. |
| Composite weather boxplots | `outputs/logistic_notebook/figures/weather_boxplots.png` | Consolidated panel comparing four weather attributes simultaneously. |
| Slice metrics (hour) | `.../slice_metrics_hour.png` | Binned precision/recall by hour-of-day for easy threshold setting. |
| Slice metrics (precip) | `.../slice_metrics_precip.png` | Binned precipitation intensity vs. logistic performance metrics. |

All figures are saved as PNGs for immediate inclusion in slides or reports.

---

## Modeling Results
### Logistic Regression Baseline
Source: `logistic_regression_analysis.ipynb`. The repo snapshot does **not** include `bus_weather_clean.csv`, so the notebook currently auto-generates `outputs/logistic_notebook/demo_bus_weather_clean.csv` (50k synthetic rows) and trains on the cached 25k-row sample noted in the notebook logs. Update `CONFIG["csv_path"]` once the full dataset is available to regenerate real-world metrics/plots.

| Metric | Value (demo sample) |
| --- | --- |
| Accuracy | 0.8056 |
| ROC-AUC | 0.7559 |
| Average Precision | 0.2525 |
| Brier Score | 0.1403 |
| Precision (Delayed) | 0.2210 |
| Recall (Delayed) | 0.5256 |
| Confusion Matrix | `[[9456, 1921], [492, 545]]` on the 12,414-row 2024 test split |

Positive rate in this sample is 8.35% delayed (5-minute threshold). Refresh `outputs/logistic_notebook/logistic_regression_metrics.json` after pointing the notebook at `bus_weather_clean.csv`.

**Interpretation:** Even on the lightweight demo data, the calibrated logistic pipeline keeps probability outputs reliable and highlights temporal + precipitation signals, but class imbalance suppresses precision. Expect metrics to shift once the full dataset is reconnected.

### Random Forest Classifier
Source: `train_rf_bus.py` (chronological split; train Jan–Sep 2023, val Oct–Dec 2023, test 2024). Uses target encoding for `route_id`/`stop_id`, ordinal for other cats, label filtering, and a cached Parquet to speed reruns.

| Metric (latest run) | Value |
| --- | --- |
| Accuracy (test) | 0.6278 |
| Recall (Delayed, test) | 0.5988 |
| Precision (Delayed, test) | 0.4693 |
| ROC-AUC (test) | 0.6668 |
| PR-AUC (test) | 0.4857 |
| Validation PR-AUC | 0.5870 |
| Training time | ~16 minutes total (train-only fit ~2.9 min + train+val fit ~13.1 min) |
| Feature importance leaders | time_point_order, hour, route_id, point_type, day_of_year, stop_id |

**Interpretation:** After label filtering and target encoding, precision improved with a modest recall trade-off; overall PR-AUC and ROC-AUC rose versus the earlier baseline. Artifacts land under `--out_dir` (e.g., `artifacts/`): `metrics.txt`, `confusion_matrix.png`, `feature_importance.csv`, `training_summary.json`, `model.joblib`, and `cache/processed.parquet`. See `README_rf.md` for RF-specific quickstart and inference.

### Visualizations (RF)
- Confusion matrices: `artifacts/confusion_matrix.png` (test), `confusion_matrix_val.png` (validation).
- Feature importances: `artifacts/feature_importance.csv` + `feature_importance.png` (run `python visualize_rf.py --out_dir artifacts`).
- Metric summaries: `artifacts/metrics.txt` and `artifacts/roc_pr_summary.txt` (via `visualize_rf.py`).

---

## Reproducing Results

### Generate Integrated Dataset
1. Download monthly MBTA CSVs and Meteostat weather files into `data/`.
2. Use the notebooks under `data/` (e.g., `merge_bus.ipynb`, `mergebuscleanwithweather.ipynb`) to produce `bus_weather_clean.csv`.

### Logistic Notebook (interactive workflow)
1. Place `bus_weather_clean.csv` in the repo root (or update the config cell to point to the file). If unavailable, enable the built-in demo generator (`auto_create_demo_data=True`).
2. Open `logistic_regression_analysis.ipynb` in Jupyter/VS Code.
3. Run all cells to load data, compute engineered features, train the logistic model, and create plots under `outputs/logistic_notebook/figures/`.
4. Inspect `outputs/logistic_notebook/logistic_regression_metrics.json` for metrics and copy the PNG figures into reports.

### Random-Forest CLI
```bash
python train_rf_bus.py --csv bus_weather_clean.csv --out_dir outputs --n_estimators 200 --max_depth 14 --max_samples 0.3 --top_stops 300
```
Artifacts saved to the directory passed via `--out_dir` (e.g., `outputs/`):
- `metrics.txt` – classification report + ROC/PR AUC.
- `confusion_matrix.png` – annotated heatmap.
- `feature_importance.csv` – sorted list for additional plotting.
- `training_summary.json` – dataset sizes, parameter settings, cache location, runtime.
- `cache/processed.parquet` – cached, feature-engineered table reused across runs.

---

## Key Takeaways from Visuals
- **Temporal signatures:** Delay probability spikes around 7–9 AM and 4–6 PM (see `hourly_profile.png`), reinforcing the need for rush-hour-specific interventions.
- **Weather-driven risk:** Boxplots and precipitation curves show a monotonic increase in both probability and average delay as precipitation intensifies; wind speed also correlates with longer delays.
- **Route heterogeneity:** `route_delay_share.png` identifies high-delay routes. Combined with feature importance (route_id), this supports focusing operations on a small subset of routes causing most incidents.
- **Heatmap hotspots:** The service-date vs. hour heatmap reveals outlier days (e.g., snowstorms) with consistent delays across multiple hours, aiding retrospective analysis.
- **Calibration:** Logistic model probabilities are usable for threshold tuning; calibration curve demonstrates reliability in the 0.2–0.7 range.

---

## Roadmap
1. **Enhanced feature bank:** Introduce rolling averages of delay per stop/hour/week and exposure-based weighting to capture local congestion history.
2. **Model diversification:** Experiment with gradient boosted trees (XGBoost/LightGBM) and temporal networks while monitoring interpretability.
3. **Threshold and cost analysis:** For different MBTA stakeholders (dispatch, riders), vary alert thresholds to trade off false alarms vs. missed delays.
4. **Deployment prep:** Package pipelines in Docker + FastAPI to expose a prediction service that ingests latest MBTA API data and weather forecasts.
5. **Visualization upgrades:** Expand static PNGs into interactive dashboards (e.g., Altair or Plotly) for route planners.
6. **Real-time ingestion:** Replace offline CSV merges with streaming ingestion powered by the MBTA V3 API + weather forecasts to support live alerts.

---

## Appendix: Snippets
### Bus Merge Example
```python
files_2023 = glob.glob(os.path.join(path_2023, "*.csv"))
files_2024 = glob.glob(os.path.join(path_2024, "*.csv"))
all_files = files_2023 + files_2024
bus_frames = [pd.read_csv(file) for file in all_files]
bus_all = pd.concat(bus_frames, ignore_index=True).drop_duplicates()

bus_all["scheduled_dt"] = pd.to_datetime(bus_all["scheduled_dt"])
bus_all["actual_dt"] = pd.to_datetime(bus_all["actual_dt"])
bus_all["delay_minutes"] = (bus_all["actual_dt"] - bus_all["scheduled_dt"]).dt.total_seconds() / 60
bus_all["hour"] = bus_all["scheduled_dt"].dt.hour
bus_all["weekday"] = bus_all["scheduled_dt"].dt.weekday
bus_all["is_weekend"] = bus_all["weekday"].isin([5, 6]).astype(int)
```

### Weather Timestamp Alignment
```python
weather_raw["timestamp_hour"] = pd.to_datetime(
    weather_raw[["year", "month", "day", "hour"]], errors="coerce"
)
weather_filtered = weather_raw.drop(columns=[col for col in weather_raw.columns if col.endswith("_source")])
weather_filtered = weather_filtered.rename(columns={"temperature": "air_temp_c"})
```

---

## Contact
For questions, please reach out via the course Slack or email the project team. We welcome suggestions for additional analyses, visualizations, or deployment strategies.

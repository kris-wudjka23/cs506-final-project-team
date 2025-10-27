## Logistic Regression Workflow

This document explains the logistic regression analysis used to predict MBTA bus delays from the combined schedule and weather dataset (`bus_weather_clean.csv`). The goal is to produce a defendable baseline model, understand feature relationships, and generate visuals that support the project’s proposal.

### Why Logistic Regression?
- Establishes a transparent, interpretable baseline for the binary delay prediction task before exploring heavier models (tree ensembles, deep nets).
- Outputs calibrated class probabilities that feed nicely into evaluation tools such as ROC curves.
- Works well with both numerical and categorical predictors when paired with standard preprocessing.

### Dataset & Label Definition
- Source: `bus_weather_clean.csv` (MBTA stop-level data enriched with weather observations).
- Target: Trips are labelled delayed if `delay_minutes` ≥ 1 minute (default threshold configurable via `--delay-threshold`).
- Sampling: Script limits the initial read (`--max-rows`) and optional random down-sampling (`--sample-frac`) to keep memory usage manageable with the 7.8 GB CSV.

### Feature Engineering & Preprocessing
- Selected columns only (see `USECOLS` in `analysis/logistic_regression_pipeline.py`) to avoid loading unused data.
- Temporal fields: `service_date` converted to datetime so we can group by date/hour for heatmap visuals.
- Categorical identifiers (`route_id`, `direction_id`, `point_type`, `weather_condition`) coerced to strings, cleaned, and one-hot encoded.
- Circular wind direction represented via sine/cosine projections to preserve orientation information.
- Numeric features (hour, humidity, precipitation, etc.) imputed with medians and scaled; categorical features imputed with most-frequent values.
- Pipeline uses `class_weight="balanced"` to compensate for any class imbalance during training.

### Model & Evaluation
- Train/test split: 80/20 with stratification to maintain the delay rate distribution.
- Logistic regression (`lbfgs`, 1000 iterations) fitted within an sklearn `Pipeline`, ensuring preprocessing is identical during train and inference.
- Metrics captured: accuracy, ROC-AUC, confusion matrix, and full classification report; saved to `outputs/logistic_regression_metrics.json`.
- Example run (60k-row sample): Accuracy ≈ 0.65, ROC-AUC ≈ 0.69 (actual numbers depend on sampling).

### Visualizations
All figures are saved under the directory chosen via `--outputs-dir` (defaults to `outputs/`):
- Box plots comparing delayed vs. on-time trips for temperature, precipitation, wind speed, and cloud cover.
- Heatmap showing average delay minutes by service date and hour (capped to recent 21 days to keep it readable).
- ROC curve plot for the logistic regression model.

### Running the Pipeline
Install dependencies in your Python environment:
```bash
pip install pandas seaborn matplotlib scikit-learn
```
Execute the analysis (adjust limits for your hardware):
```bash
python analysis/logistic_regression_pipeline.py --max-rows 300000 --sample-frac 0.2 --outputs-dir outputs
```
Key flags:
- `--max-rows`: Number of rows read from the CSV (set `-1` for full dataset once memory allows).
- `--sample-frac`: Additional random down-sampling (0 < value ≤ 1).
- `--delay-threshold`: Minutes defining a “delay” (default 1.0).
- `--test-size`: Hold-out fraction (default 0.2).

### Interpreting Results
- Use the ROC curve and metrics JSON to evaluate baseline performance and document baseline findings.
- Box plots highlight which weather features shift distributions when delays occur.
- Heatmap reveals temporal hotspots (days/hours) with systemic delays, guiding further investigation.

### Next Steps
- Compare against non-linear models (decision tree, XGBoost, neural nets) using the same engineered features.
- Perform threshold analysis by sweeping `--delay-threshold` to see how the positive rate and performance trade off.
- Incorporate SHAP values or coefficient inspection for deeper feature contribution insights.

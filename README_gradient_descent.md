# Gradient Descent Notebook README
Documentation dedicated to gradient_descent_analysis.ipynb, explaining goals, configuration knobs, and how to reproduce the gradient-descent baseline for MBTA bus delay prediction.
## Table of Contents
## 1. Audience and Scope
- Focused on contributors who need full context without reading every notebook cell.
- Assumes working knowledge of Python, pandas, NumPy, and scikit-learn.
- Mirrors logistic_regression_analysis.ipynb documentation while emphasizing gradient-descent differences.
- Useful for onboarding, reviews, and experiment replication on MBTA bus delay data.
## 2. Notebook Roadmap
- Cells 1-3 import packages, configure plotting defaults, and guard optional dependencies like holidays.
- Cell 4 defines ARTIFACT_DIR=outputs/gradient_descent_notebook, isolating assets from the logistic baseline.
- Cells 5-8 list USECOLS, dtype overrides, and plotting helpers reused later in the workflow.
- Cells 9-15 manage data access via resolve_csv_path, demo generation, chunked CSV reading, and caching.
- Cells 16-20 implement shared calendar, weather, and aggregate feature engineering utilities.
- Cells 21-23 store EDA helpers including histograms, hourly profiles, route delay share, precipitation curves, and slice metrics.
- Cells 24-27 define GradientDescentLogisticClassifier plus pipeline and evaluation functions.
- Remaining cells load data, engineer features, sample for visuals, split chronologically, train, evaluate, plot, and persist metrics.
## 3. Environment Setup
1. Create a virtual environment: python -m venv .venv && .venv\Scripts\activate on Windows or source .venv/bin/activate on Unix.
2. Install requirements: pip install -r requirements_RF.txt jupyter.
3. Add optional libs if missing: pip install holidays seaborn matplotlib scikit-learn pandas numpy pyarrow.
4. Launch Jupyter Lab or VS Code notebooks from the repo root before opening gradient_descent_analysis.ipynb.
5. Use sampling knobs and caches on resource-constrained machines to keep memory usage manageable.
## 4. Data Dependencies
- Primary input: bus_weather_clean.csv, the integrated MBTA plus Meteostat table (~40M rows).
- Optional fallback: outputs/gradient_descent_notebook/demo_bus_weather_clean.csv generated when auto_create_demo_data=True.
- Paths remain configurable via environment variables or CONFIG["csv_search_paths"].
- Notebook outputs stay under outputs/gradient_descent_notebook/ to avoid clobbering logistic artifacts.
- Cached sample bus_weather_sample.parquet accelerates repeated sessions when use_cache is enabled.
## 5. Configuration Reference
- csv_path: default bus_weather_clean.csv at repo root; override with absolute paths or BUS_WEATHER_CSV.
- cache_path plus use_cache/rebuild_cache toggles: control Parquet sampling for faster reloads.
- max_rows and sample_frac: coarse and fine controls over data volume; set max_rows=-1 for full dataset access.
- chunksize: pandas CSV chunk size balancing IO speed and memory footprint.
- delay_threshold_min: defines the binary label cutoff in minutes (default five).
- random_state: shared seed for sampling, splitting, and GD initialization to ensure reproducibility.
- eda_sample_size: subset size for quick visualizations.
- csv_search_paths and auto_search_csv: enable fallback discovery when the main path is missing.
- auto_create_demo_data and demo_rows: control synthetic data generation for lightweight experimentation.
- gd_params: learning rate, max_iter, tolerance, alpha, fit_intercept, verbose, and random_state for the custom optimizer.
## 6. Loading and Sampling Flow
1. resolve_csv_path normalizes user input, environment overrides, and fallback directories while logging the chosen file.
2. If caching is allowed and the Parquet sample exists, it loads instantly, skipping CSV parsing.
3. Otherwise _read_csv_in_chunks iterates through the CSV with the chosen chunksize until max_rows is reached.
4. After concatenation, sample_frac optionally reduces the dataset with a reproducible seed.
5. The resulting frame feeds into feature engineering with transparent logging of row counts.
## 7. Feature Engineering Stack
- add_calendar_features outputs month, day_of_year, season, is_holiday, and is_school_in_session signals.
- Numeric coercion targets hour, precip_mm, delay_minutes, and related columns while dropping corrupt records.
- is_delayed applies delay_threshold_min using a >= comparison for inclusive labeling.
- Route and weather descriptors become zero-padded strings with <missing> placeholders to ease encoding.
- Wind direction uses sine and cosine projections for cyclic continuity.
- rainy_rush_hour flags commute windows with precipitation above 0.2 millimeters.
- attach_route_hour_stats adds mean, median, and count delay aggregates per (route_id, hour, weekday) from training data only.
- Infinite values convert to NaN and are handled later by pipeline imputers.
## 8. Custom Optimizer Details
- GradientDescentLogisticClassifier subclasses BaseEstimator and ClassifierMixin for sklearn compatibility.
- Parameters mirror full-batch GD settings: learning_rate, max_iter, tol, alpha, fit_intercept, verbose, random_state.
- Weights initialize via Normal(0, 0.01) to ensure symmetry breaking and stability.
- Each iteration computes logits, probabilities, residuals, and gradients averaged over all rows.
- L2 regularization adds alpha * weights excluding the bias term when fit_intercept is enabled.
- Loss history records negative log-likelihood plus penalty for diagnostics.
- Convergence stops when gradient L2 norm drops below tol; otherwise loops until max_iter.
- After convergence, intercept_ and coef_ are populated, classes_ is [0,1], and n_features_in_ is cached.
- decision_function, predict_proba, predict, and score behave like standard scikit-learn estimators.
## 9. Pipeline Assembly
- Numeric branch: median imputer followed by StandardScaler to stabilize gradient descent.
- Categorical branch: most-frequent imputer plus build_one_hot_encoder to survive sklearn API changes.
- ColumnTransformer stitches branches in a consistent feature order that powers coefficient charts.
- Final pipeline uses ('preprocess', preprocessor) then ('model', GradientDescentLogisticClassifier(**gd_params)).
- train_model wraps pipeline fitting for clarity and parity with the logistic notebook.
## 10. Exploration Cells
- eda_sample_size guarantees snappy plotting while respecting dataset length.
- Histograms, hourly profiles, service heatmaps, route delay rankings, weather boxplots, and precipitation curves reuse helper functions.
- Slice metrics accept arbitrary numeric columns and bin edges for targeted diagnostics.
- All EDA figures save inside outputs/gradient_descent_notebook/figures to avoid polluting logistic outputs.
## 11. Training and Evaluation
1. Temporal split sorts by service_date; rows before split_date form train_df, others form test_df.
2. attach_route_hour_stats derived from training data merges into both splits for leakage-safe aggregates.
3. train_model executes pipeline.fit and logs [model] Gradient-descent logistic regression fitted.
4. evaluate_model produces y_pred, y_proba, accuracy, ROC-AUC, average precision, Brier score, positive rate, and support.
5. Classification report JSON snippet prints for quick reference while the full dictionary persists.
6. eval_df clones test_df and appends y_true, y_pred, and y_proba for downstream plotting.
7. Plotting cell saves confusion matrix, ROC, PR, calibration, slice metrics, and coefficient bars.
8. Metrics persist to outputs/gradient_descent_notebook/gradient_descent_metrics.json for reproducibility.
## 12. Artifact Directory Layout
```
outputs/gradient_descent_notebook/
|-- bus_weather_sample.parquet
|-- demo_bus_weather_clean.csv
|-- gradient_descent_metrics.json
\-- figures/
    |-- delay_histogram.png
    |-- hourly_profile.png
    |-- delay_heatmap.png
    |-- route_delay_share.png
    |-- weather_boxplots.png
    |-- precip_vs_delay.png
    |-- slice_metrics_hour.png
    |-- slice_metrics_precip.png
    |-- roc_curve.png
    |-- pr_curve.png
    |-- calibration_curve.png
    |-- confusion_matrix.png
    \-- gradient_descent_coefficients.png
```
- Clear cached files when switching between demo and full datasets to avoid stale samples.
- Archive figures externally if you need historical comparisons because each run overwrites PNGs.
## 13. Visualization Products
- Delay histogram now pairs with a cumulative distribution overlay (cdf_delay.png) so you can read tail probabilities directly.
- Hourly profile adds stacked area plots for positive-rate segments plus a weekday-weekend split (hourly_profile_weektype.png) to emphasize service differences.
- Service heatmap sits beside a calendar-style monthly treemap (calendar_heatmap.png) to contrast short-term hotspots with seasonal averages.
- Route delay share extends to a scatter of delay rate vs. trip volume (route_rate_volume.png), highlighting leverage points beyond the top fifteen.
- Precipitation versus delay chart gains a dual-axis violin view for wind speed (weather_violin.png) to surface compounding weather stressors.
- ROC and PR curves are complemented by a lift chart (lift_curve.png) that operations teams can use for threshold calibration.
- Calibration curve sits with a reliability diagram comparing multiple learning rates (calibration_compare.png) after tuning experiments.
- Slice metrics plots now include stop-level quartile bands (slice_metrics_stop.png) to expose spatial fairness issues.
- Coefficient bar chart joins a feature-correlation heatmap (feature_correlation.png) to contextualize multicollinearity behind large weights.
## 14. Metrics JSON Schema
```json
{
  \"accuracy\": float,
  \"roc_auc\": float,
  \"average_precision\": float,
  \"brier_score\": float,
  \"positive_rate\": float,
  \"support\": int,
  \"delay_threshold_minutes\": float,
  \"train_test_split_date\": \"YYYY-MM-DD\",
  \"classification_report\": {class: {precision, recall, f1-score, support}, ...},
  \"confusion_matrix\": [[tn, fp], [fn, tp]]
}
```
- Extend the schema if you log extra diagnostics, but downstream tools expect at least the keys above.
- Diff JSON metrics across runs to track regression or improvement trends.
## 15. Suggested Experiments
1. Sweep learning_rate across {0.01, 0.05, 0.1} to balance convergence speed and stability.
2. Tune alpha regularization strengths (0, 0.001, 0.005, 0.01) to control coefficient shrinkage.
3. Increase max_iter for full-data runs and inspect loss_history_ for diminishing returns.
4. Drop route-hour aggregates or weather features to quantify their contributions.
5. Optimize probability thresholds using eval_df to align precision-recall trade-offs with rider goals.
6. Shift split_date (monthly or quarterly) to evaluate temporal robustness.
7. Compare model quality versus runtime by training on 100, 50, and 10 percent samples.
8. Prototype mini-batch or momentum variants by extending the custom classifier.
## 16. Troubleshooting Tips
- File not found: verify CONFIG['csv_path'] or set BUS_WEATHER_CSV before running the notebook.
- Memory errors: lower max_rows, decrease chunksize, or rely on sample_frac until more RAM is available.
- Convergence stalls: inspect loss_history_; reduce learning_rate or increase alpha for smoother gradients.
- NaN metrics: ensure feature engineering did not drop all rows and check delay_minutes conversions on custom CSVs.
- Plots missing: rerun the setup cell to recreate outputs/gradient_descent_notebook/figures.
- Calibration drift: rerun the calibration cell after major hyperparameter changes to confirm probability quality.
- Demo versus real data: delete bus_weather_sample.parquet when switching between sources to avoid contamination.
- Slow plotting: reduce eda_sample_size or temporarily skip heavy visuals like heatmaps while iterating.
## 17. Next Steps
- Export eval_df to CSV for dashboards or BI pipelines.
- Integrate the gradient pipeline into CLI scripts for automated benchmarking alongside logistic and random-forest baselines.
- Explore cost-sensitive losses to emphasize reducing missed delays.
- Combine predictions with train_rf_bus.py outputs for ensembles or stacking experiments.
## 18. Contact
- File issues or suggestions on the project GitHub board or class Slack.
- Direct data pipeline questions to the integration sub-team maintaining bus_weather_clean.csv.
- Direct modeling and optimizer questions to the ML owners responsible for gradient_descent_analysis.ipynb.
- Reference this README when submitting PRs that touch the gradient notebook to keep documentation in sync.
## Appendix A: Command Cheatsheet
```
# Launch notebook server
jupyter lab gradient_descent_analysis.ipynb
# Clear cached Parquet sample
python - <<'PY'
from pathlib import Path
cache = Path('outputs/gradient_descent_notebook/bus_weather_sample.parquet')
if cache.exists():
    cache.unlink()
PY
# Convert notebook to HTML
jupyter nbconvert --to html gradient_descent_analysis.ipynb
```
## Appendix B: Key Variables Recap
- MODEL_FEATURES: union of numeric and categorical predictors flowing into the pipeline.
- BOXPLOT_FEATURES: weather fields plotted in the multi-panel boxplot utility.
- AGG_KEYS: grouping tuple (route_id, hour, weekday) powering delay aggregates.
- loss_history_: list of per-iteration losses for convergence inspection.
- gd_params: dictionary for learning hyperparameters.
- eval_df: evaluation dataframe containing features, predictions, and probabilities.
## Appendix C: Version Control Tips
- Clear notebook outputs before committing to limit noisy diffs.
- Keep large CSVs out of git and rely on configuration to locate them at runtime.
- Update both logistic and gradient notebooks when editing shared helper functions.
- Record major hyperparameter changes inside this README for traceability.
## Appendix E: Checklist Before Running
- Ensure the Python environment is active and dependencies installed.
- Confirm bus_weather_clean.csv path or enable demo generation in CONFIG.
- Decide on max_rows and sample_frac to match available memory.
- Set gd_params for the experiment you want to run.
- Verify outputs/gradient_descent_notebook/ is writable.
- Clear cached Parquet samples when swapping datasets.
- Execute notebook cells sequentially and monitor console logs.
- Inspect gradient_descent_metrics.json and generated plots upon completion.
---
This README acts as a living companion to gradient_descent_analysis.ipynb; update it whenever the notebook evolves so future contributors remain productive and aligned.

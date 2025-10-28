# Preliminary Findings Report

## Project Framing
- Objective: explore MBTA bus performance alongside weather signals to flag conditions associated with late arrivals.
- Scope so far: built an exploratory notebook (`analysis/preliminary_feature_exploration.ipynb`) for quick feature triage and a reproducible logistic regression pipeline (`analysis/logistic_regression_pipeline.py`).

## Hypotheses at Kickoff
- **Weather-driven delays:** Heavy precipitation, high wind speeds, and low visibility (cloud cover) increase the chance of delays.
- **Temporal patterns:** Rush-hour windows and certain weekdays are more delay-prone due to traffic and ridership peaks.
- **Route/stop variability:** A handful of routes and stops drive most of the observed delays, making localized interventions viable.

## Evidence Gathered
- Load the full 46M-row `bus_weather_clean.csv` by default, with an opt-in sampling knob for lower-memory environments.
- Generate descriptive statistics and missing-value audits to understand data hygiene before modeling.
- Visual inspections (delay histograms, categorical frequency bars, robust hourly median plots, and weather-condition rankings) highlight long-tailed delay distributions and contextualize where congestion and weather stressors appear.
- Logistic regression pipeline encodes categorical features, scales numeric fields, and evaluates on a stratified test split with saved diagnostics (metrics JSON, ROC curve).

## Discoveries So Far
- Delay minutes are highly skewed: most trips arrive on time or a few minutes late, but a meaningful tail reaches 10+ minutes.
- Hour-of-day and weekday effects corroborate the rush-hour hypothesis—early morning and evening windows yield higher median delays than off-peak periods even after filtering low-volume hours.
- Heavy precipitation and elevated wind speeds lead the weather-driven delay rankings, while fair-weather conditions stay near zero; temperature shows a muted relationship.
- **Current logistic regression performance:** Accuracy 0.65 and ROC-AUC 0.69 on the test split. Precision 0.72 and recall 0.65 for the delayed class indicate the model captures most true delays with manageable false positives, yet still misses roughly one-third of actual delays. The model serves as an interpretable baseline with room to grow.

## Challenges and Mitigations
- **File size:** The raw CSV is several gigabytes. Solution: default to full-load for completeness, but keep optional sampling and chunked stats utilities so lower-memory runs remain feasible.
- **Feature completeness:** Mixed data types and missing values required explicit coercion and light engineering (e.g., converting wind direction into sine/cosine components) before modeling.
- **Class imbalance:** Delayed trips are less frequent than on-time trips. Stratified splits and threshold-aware metrics (precision/recall) keep evaluation honest, but the imbalance still affects recall.

## Next Steps
- Expand feature set with temporal aggregates and route-level profiles.
- Trial more expressive models to close the recall gap while monitoring interpretability.
- Operationalize visual dashboards for stakeholders, focusing on high-delay routes/weather combinations surfaced during EDA.
- Investigate real-time inference feasibility once a stronger model is in place.

Airplane Crash Analytics — Predictive Modeling & Survival Analysis, Flight-Hour Risk Assessment
>
> Every year, aviation authorities and airlines make billion-dollar decisions about when to retire aircraft, how to schedule maintenance, and which fleets carry the highest risk — often guided by gut feel or rigid calendar-based rules. This project turns over a century of crash records into a working predictive system: we show exactly when in an aircraft's lifetime crash risk spikes, which operator types are most dangerous, and how a trained XGBoost model can flag high-risk flights before they happen. The model speaks the industry's own language — crashes per million flight hours, D-Check thresholds, Kaplan-Meier survival curves — so findings translate directly into actionable maintenance policy. In short: we move aviation safety from scheduled guesswork to data-driven, condition-based decision-making.
---
Table of Contents
Project Overview
Dataset
Phase 1 — Exploratory Data Analysis
Phase 2 — Predictive Modeling & Survival Analysis
Key Findings
Recommendations
Tech Stack
Setup & Installation
Future Work
---
Project Overview
This notebook-based project analyzes airplane crash data from 1908 to 2009 across two structured phases:
Phase 1 — Exploratory analysis focusing on Boeing vs. Airbus crash distributions, operator type breakdowns (commercial, military, private), and decade-level trends.
Phase 2 — Predictive modeling using time-series forecasting (Prophet), survival analysis (Kaplan-Meier + Cox PH), and a crash-risk classifier (XGBoost), benchmarked against real ICAO industry safety metrics.
The goal is to identify when and why crashes happen — and produce a model that can predict fatal crash probability for hypothetical future flight scenarios.
---
Dataset
Source: Airplane Crashes and Fatalities Since 1908 via Kaggle
Key columns used:
Column	Description
`Date`	Date of the crash
`Type`	Aircraft type (used to extract manufacturer)
`Operator`	Airline or operator name
`Aboard`	Total people on board
`Fatalities`	Number of fatalities
`Ground`	Ground casualties
Engineered features: `Manufacturer`, `Operator_Type`, `Year`, `Decade`, `Aircraft_Age_Est`, `Flight_Hours_Est`, `Survival_Rate`, `Fatal_Flag`, `Hours_Band`
---
Phase 1 — Exploratory Data Analysis
1.1 Boeing vs. Airbus
Filtered all records from 1970 onward (when Airbus entered service)
Compared total crash counts and decade-by-decade trends
Visualizations: pie chart, grouped bar chart (Plotly), decade trend chart
1.2 Top 10 Manufacturers by Crashes (1970+)
Ranked all manufacturers by raw crash count post-1970
Useful baseline for fleet risk assessment
1.3 Operator Type Analysis
Classified all operators into: Military, Private, Commercial
Bar chart of crash volume by operator category
Commercial aviation dominates volume; military and private show distinct risk profiles
---
Phase 2 — Predictive Modeling & Survival Analysis
2.1 Feature Engineering
Beyond raw columns, the following derived features are central to the ML pipeline:
`Aircraft_Age_Est` — years since the manufacturer's first-flight year (sourced from Jane's All the World's Aircraft)
`Flight_Hours_Est` — estimated cumulative hours based on operator type and aircraft age, calibrated to ICAO averages (commercial: ~3,200 hrs/yr; military: ~300 hrs/yr; private: ~400 hrs/yr)
`Hours_Band` — bucketed into industry-standard maintenance bands: `<10k`, `10–20k`, `20–30k`, `30–45k`, `45–60k`, `>60k` hours
`Fatal_Flag` — binary target: 1 if any fatality occurred, 0 otherwise
2.2 Annual Crash Rate Forecasting — Prophet
Meta's Prophet time-series model is fit on annual crash counts from 1920–2009 and projects through 2035.
Captures long-run trend (crash counts declining steeply post-1970 as aviation safety improved)
90% confidence intervals on forecast
Outputs projected crash counts per year with upper/lower bounds
2.3 Industry Safety Metric — Crashes per Million Flight Hours (MFH)
Raw crash counts are misleading without normalizing for total flying activity. This section implements the ICAO standard metric:
```
Crashes per Million Flight Hours = Annual Crashes / (Global Flight Hours in Billions × 1,000)
```
Key industry thresholds contextualized:
~20,000–25,000 hrs → D-Check trigger (heaviest maintenance, mandatory for narrowbodies)
30 years OR 60,000+ hrs → FAA Aging Aircraft Program mandatory structural audit
75,000 cycles / 60,000 hrs → Boeing 737 / 747 design service goals
Finding: Crashes per MFH dropped ~97% from 1930 to 2009.
2.4 Survival Analysis
Using the `lifelines` library (standard in biostatistics and aerospace reliability):
Kaplan-Meier curves by operator type — probability of no fatal crash as a function of estimated flight hours
Cox Proportional Hazards model — hazard ratios for: `Aircraft_Age_Est`, `Flight_Hours_Est`, `Aboard`, `Year`, `Operator_Type`
D-Check (20k hrs) and FAA aging-aircraft (60k hrs) thresholds overlaid for reference
2.5 XGBoost Crash-Risk Classifier
Predicts `Fatal_Flag` (binary: fatal crash or not) from engineered features.
Model config:
```python
xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=<imbalance ratio>,  # handles class imbalance
    eval_metric='logloss',
    random_state=42
)
```
Evaluation:
5-fold stratified cross-validation (AUC-ROC)
Confusion matrix + ROC curve on held-out test set (80/20 split)
2.6 Feature Importance
Horizontal bar chart of XGBoost `feature_importances_` (Gain), color-coded above/below median. Identifies which features most strongly predict a fatal outcome.
2.7 Risk Heatmap
Fatal accident rate (%) broken down by aircraft age band × decade, rendered as a seaborn heatmap. Highlights interaction effects between aircraft age and era-specific safety standards.
2.8 Scenario Predictions
The trained XGBoost model is applied to five hypothetical future flight scenarios (2026–2030), e.g.:
Scenario	Predicted Fatal Probability
New A320 (5 yr / 15k hrs)	Low
Aging A320 (25 yr / 55k hrs)	Elevated
Old private jet (35 yr / 80k hrs)	High
---
Key Findings
Metric	Value
Safety improvement (crashes/MFH, 1930→2009)	~97% reduction
Most dangerous flight-hour band	30,000–45,000 hrs
Industry D-Check threshold	20,000–25,000 hrs (narrowbodies)
FAA Aging Aircraft audit trigger	30 years OR 60,000+ hrs
Prophet forecast trend	Continued decline projected through 2035
---
Recommendations
Flight-hour monitoring — Prioritize inspection of aircraft approaching 30,000+ flight hours. The Kaplan-Meier curve shows the steepest survival probability decline in this band.
Age-based risk flagging — Aircraft over 25 years old show significantly higher fatal rates in the heatmap. Mandatory structural audits should be triggered at this threshold.
Operator-type differentiation — Military and private operators have distinct risk profiles from commercial aviation. Separate maintenance and reporting standards are warranted.
Seasonal audit scheduling — Monthly crash distributions reveal seasonal patterns that can inform timing of additional safety reviews.
Predictive maintenance integration — Pair this model with real-time sensor data (vibration, pressurisation cycles) to shift from calendar-based to condition-based maintenance.
---
Tech Stack
Category	Libraries
Data manipulation	`pandas`, `numpy`
Visualization	`matplotlib`, `seaborn`, `plotly`
Machine learning	`scikit-learn`, `xgboost`
Survival analysis	`lifelines` (Kaplan-Meier, Cox PH)
Time-series forecasting	`prophet` (Meta)
Data source	`kagglehub`
---
Setup & Installation
```bash
# Clone the repository
git clone <repo-url>
cd airplane-crash-analytics

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly xgboost scikit-learn lifelines prophet kagglehub mplcursors

# Launch the notebook
jupyter notebook airplane_crashes.ipynb
```
> **Note:** The first cell will automatically download the dataset from Kaggle via `kagglehub`. A Kaggle account and API token are required.
---
Future Work
Fleet-level survival modelling — Track specific tail numbers across years to model individual aircraft lifecycles
NLP on crash summaries — Extract cause categories (human error, mechanical, weather) using BERT or similar
SHAP values — True per-prediction explainability, beyond global feature importance
Bayesian updating — Update risk estimates dynamically as new flight hours accumulate
ADS-B data integration — Pull real flight hours from live transponder data via OpenSky Network (free API)
---
Dataset covers crashes from 1908–2009. Predictive models are trained on historical data and should not be used as the sole basis for operational safety decisions.

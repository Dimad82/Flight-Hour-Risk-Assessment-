***

# Flight Hour Risk Assessment & Aviation Safety Analysis

## Overview

This project performs **exploratory data analysis (EDA), predictive modeling, and survival analysis** on historical aviation crash data to assess **flight-hour-related risk factors** and improve safety insights.

It combines:

* Data analysis (Pandas, Seaborn, Plotly)
* Machine learning (XGBoost, Scikit-learn)
* Survival analysis (Kaplan-Meier, Cox model)
* Time-series forecasting (Prophet)

***

## Key Objectives

* Compare **Boeing vs Airbus crash trends**
* Analyze **crashes by operator type** (Commercial, Military, Private)
* Estimate **aircraft flight hours & age**
* Identify **high-risk flight-hour intervals**
* Build a **crash risk prediction model**
* Forecast **future crash trends (up to 2035)**

***

## Dataset

* Source: <https://www.kaggle.com/datasets/saurograndi/airplane-crashes-since-1908>
* Contains:
  * Date
  * Aircraft type
  * Operator
  * Fatalities
  * Passengers aboard

***

## Features & Analysis

### 1. Exploratory Data Analysis

* Crash counts by manufacturer
* Decade trends
* Operator type distribution
* Top 10 high-risk manufacturers

### 2. Feature Engineering

* Aircraft age estimation
* Flight-hour estimation
* Survival rate calculation
* Operator classification

### 3. Crash Rate Metrics

* Crashes per million flight hours (industry metric)
* Flight-hour band risk analysis:
  * `<10k`, `10–20k`, `20–30k`, `30–45k`, `45–60k`, `>60k`

### 4. Survival Analysis

* **Kaplan-Meier curves** → survival probability vs flight hours
* **Cox Proportional Hazards model** → risk factors

### 5. Machine Learning

* **XGBoost classifier**
* Predicts probability of fatal outcomes
* Evaluation metrics:
  * AUC-ROC
  * Confusion matrix
  * Classification report

### 6. Time-Series Forecasting

* Prophet model predicts crash trends until **2035**
* Identifies long-term safety improvements

***

## Key Insights

* \~97% reduction in crashes per flight hour (1930–2009)
* Highest risk band: **30,000 – 45,000 flight hours**
* Maintenance threshold:
  * **D-check:** \~20,000–25,000 hours
  * **FAA aging threshold:** \~60,000 hours / 30 years
* Aircraft age significantly impacts crash risk
* Operator type influences safety patterns

***

## Model Output Example

The model predicts fatal crash probability for different scenarios:

* New aircraft → lower risk
* Aging aircraft → higher risk
* Old private jets → highest risk

***

## Technologies Used

### Data & Visualization

* pandas
* numpy
* matplotlib
* seaborn
* plotly

### Machine Learning

* scikit-learn
* xgboost

### Advanced Analysis

* lifelines (survival analysis)
* prophet (forecasting)

***

## Installation

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lifelines prophet mplcursors kagglehub
```

***

## Usage

1. Clone the repository

```bash
git clone https://github.com/your-username/flight-hour-risk-assessment.git
cd flight-hour-risk-assessment
```

2. Run the script/notebook

```bash
python flight_hour_risk_assessment.py
```

***

## Project Structure

```
├── flight_hour_risk_assessment.py
├── README.md
├── plots/
│   ├── annual_crashes.png
│   ├── prophet_forecast.png
│   ├── survival_curves.png
│   └── feature_importance.png
```

***

## Recommendations

* Monitor aircraft approaching **30,000+ flight hours**
* Perform stricter inspections for aircraft **>25 years**
* Use predictive models for **risk-based maintenance**
* Integrate real-time flight data for enhanced accuracy

***

## Future Improvements

* Add **SHAP explainability**
* Include **real flight hour data (ADS-B/OpenSky)**
* Apply **NLP analysis** on crash reports
* Build **fleet-level tracking models**

***

## License

This project is open-source and available under the MIT License.

***

## Contributing

Contributions are welcome!  
Feel free to fork this repo and submit a pull request.

***

If you want, I can also:
✅ Make it more minimal (for recruiters)  
✅ Add badges (build, license, Python version)  
✅ Customize it for a portfolio or CV 🚀

# PulseRetain — AI-Powered Employee Retention Intelligence

> Predict attrition risk before it becomes resignation. PulseRetain gives HR teams and managers an explainable, actionable view of who is at risk and why.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

PulseRetain is an end-to-end machine learning system that predicts employee attrition risk, explains the drivers behind each prediction using SHAP, and surfaces targeted intervention recommendations to managers — all through an interactive Streamlit dashboard.

The system is trained on the IBM HR Analytics dataset (1,470 employees, 35 features) and achieves a **ROC-AUC of 0.76** with a full XGBoost pipeline.

---

## Key Features

| Feature | Description |
|---|---|
| Risk Scoring | Every employee receives a 0–100 risk score and a LOW / MODERATE / HIGH / CRITICAL label |
| Explainability | Per-employee SHAP waterfall drivers — contributing and protective factors |
| Interventions | Rule-based recommendation engine maps top drivers to concrete manager actions |
| Executive Dashboard | Organisation-wide KPIs, risk distribution, department and role breakdowns |
| Analytics | Business analytics + full model performance (ROC, PR curve, confusion matrix, feature importance) |
| Intervention Tracker | Managers can log status, notes, and dates for each at-risk employee |

---

## Project Structure

```
PulseRetain/
├── app/                        # Streamlit application
│   ├── assets/styles.css       # Dark-theme UI styles
│   ├── pages/
│   │   ├── dashboard.py        # Executive KPI dashboard
│   │   ├── employees.py        # Filterable at-risk employee table
│   │   ├── employee_profile.py # Per-employee risk + SHAP drivers
│   │   ├── interventions.py    # Intervention tracker
│   │   └── analytics.py        # Business analytics + model performance
│   ├── app.py                  # Entry point, sidebar navigation
│   └── data_service.py         # Cached data pipeline for the UI
│
├── src/                        # Core ML library
│   ├── config.py               # All paths, thresholds, constants
│   ├── data/
│   │   ├── loader.py           # CSV ingestion
│   │   ├── validator.py        # Data quality checks
│   │   └── preprocessing.py    # Target encoding, column dropping
│   ├── features/
│   │   └── engineering.py      # Derived business features
│   ├── models/
│   │   ├── train.py            # Training pipeline
│   │   └── predict.py          # Inference engine
│   ├── explainability/
│   │   └── shap_explainer.py   # SHAP TreeExplainer wrapper
│   └── interventions/
│       └── recommendation_engine.py  # Driver → action mapping
│
├── data/
│   ├── raw/HR-Employee-Attrition.csv
│   └── processed/employee_features.csv
│
├── models/
│   ├── attrition_model.pkl     # Trained XGBoost pipeline
│   ├── preprocessor.pkl        # Fitted ColumnTransformer
│   └── model_metadata.json     # Metrics + feature list
│
├── docs/                       # Project documentation
├── tests/                      # Unit tests
├── requirements.txt
└── run_app.py
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-org/pulseretain.git
cd pulseretain
pip install -r requirements.txt
```

### 2. Add the dataset

Place `HR-Employee-Attrition.csv` in `data/raw/`. The IBM HR Analytics dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

### 3. Train the model

```bash
python -m src.models.train
```

This saves `attrition_model.pkl`, `preprocessor.pkl`, and `model_metadata.json` to `models/`.

### 4. Launch the dashboard

```bash
streamlit run app/app.py
# or
python run_app.py
```

Open `http://localhost:8501` in your browser.

---

## Model Performance

Evaluated on a stratified 20% hold-out test set (294 employees):

| Metric | Score |
|---|---|
| Accuracy | 0.8265 |
| Precision | 0.4474 |
| Recall | 0.3617 |
| F1 | 0.4000 |
| ROC-AUC | 0.7593 |
| PR-AUC | 0.4505 |

> Recall is the primary business metric — a false negative means an at-risk employee goes undetected.

---

## Risk Thresholds

| Level | Score Range |
|---|---|
| LOW | 0 – 30 |
| MODERATE | 31 – 60 |
| HIGH | 61 – 80 |
| CRITICAL | 81 – 100 |

---

## Configuration

All paths, thresholds, and constants are centralised in `src/config.py`. No values are scattered across the codebase.

```python
RISK_LOW_THRESHOLD      = 30
RISK_MODERATE_THRESHOLD = 60
RISK_HIGH_THRESHOLD     = 80
TEST_SIZE               = 0.2
RANDOM_STATE            = 42
```

---

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
shap>=0.44
joblib>=1.3
plotly>=5.18
streamlit>=1.30
python-dotenv>=1.0
```

---

## Ethical Notice

PulseRetain is a **decision-support tool**. All risk scores and recommendations are probabilistic estimates and must be reviewed by a qualified HR professional before any action is taken. The system does not make employment decisions.

---
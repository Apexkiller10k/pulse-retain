# PulseRetain

PulseRetain is an explainable employee-retention intelligence platform for HR teams and managers. It estimates attrition risk, explains the factors behind each prediction, and helps managers track practical interventions through a professional Streamlit dashboard.

> PulseRetain is a decision-support system. It does not make employment decisions. Risk scores and recommendations must be reviewed by a qualified HR professional and considered alongside human judgment and organisational policy.

## What It Does

- **Executive dashboard:** Workforce size, average risk, critical-risk counts, risk distribution, department risk, job-role risk, and organisation-wide drivers.
- **At-risk employee register:** Search and filter employees by ID, department, job role, and risk level.
- **Employee profiles:** View a selected employee's risk score, risk level, SHAP-based drivers, employee snapshot, and recommended actions.
- **Intervention tracking:** Record manager notes, intervention dates, and statuses: Not Started, In Progress, or Completed.
- **Analytics:** Review risk distributions, business trends, model metrics, ROC and precision-recall curves, confusion matrix results, and feature importance.
- **Explainable predictions:** Use SHAP-based explanations to show factors that increase or reduce predicted attrition risk.

## Technology

- Python 3.10+
- Streamlit
- pandas and NumPy
- scikit-learn
- XGBoost
- SHAP
- Plotly
- joblib

## Project Structure

```text
pulse-retain/
├── app/
│   ├── app.py                  # Streamlit application entry point
│   ├── data_service.py         # Cached data, prediction, and explanation pipeline
│   ├── assets/styles.css        # Shared ERP interface styling
│   └── pages/
│       ├── dashboard.py         # Executive dashboard
│       ├── employees.py         # At-risk employee register
│       ├── employee_profile.py  # Employee detail and explainability view
│       ├── interventions.py     # Intervention tracker
│       └── analytics.py         # Business and model analytics
├── data/raw/HR-Employee-Attrition.csv
├── docs/                        # Architecture and methodology documentation
├── models/
│   ├── attrition_model.pkl      # Trained XGBoost pipeline
│   ├── preprocessor.pkl         # Fitted preprocessing pipeline
│   └── model_metadata.json       # Features, metrics, and training metadata
├── src/
│   ├── config.py                # Paths, thresholds, and shared constants
│   ├── data/                    # Loading, validation, and preprocessing
│   ├── features/                # Derived feature engineering
│   ├── models/                  # Training and prediction
│   ├── explainability/          # SHAP explanation utilities
│   └── interventions/           # Recommendation rules
├── tests/
├── requirements.txt
└── run_app.py
```

## Quick Start

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

Git Bash on Windows:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Add the dataset

Place the IBM HR Analytics CSV at:

```text
data/raw/HR-Employee-Attrition.csv
```

The application expects the standard IBM HR Analytics dataset with 1,470 employee records and an `Attrition` target column. The dataset is commonly available from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

### 4. Train the model

Run training from the repository root:

```bash
python -m src.models.train
```

Training creates or updates `attrition_model.pkl`, `preprocessor.pkl`, and `model_metadata.json` in `models/`.

### 5. Start the dashboard

```bash
streamlit run app/app.py
```

Or use the project launcher:

```bash
python run_app.py
```

Open [http://localhost:8501](http://localhost:8501) in a browser. The dashboard provides navigation for Dashboard, At-Risk Employees, Employee Profile, Interventions, and Analytics.

## Model Pipeline

1. Load and validate the raw CSV.
2. Encode the `Attrition` target and remove administrative identifier columns.
3. Create derived business features such as satisfaction and tenure indicators.
4. Impute missing numeric and categorical values.
5. Scale numeric features and one-hot encode categorical features.
6. Train an XGBoost classifier with class weighting for attrition imbalance.
7. Convert predicted probability to a 0-100 risk score.
8. Assign a risk level and generate SHAP explanations.
9. Map the leading risk driver to an intervention recommendation.

## Risk Levels

Risk score is calculated as predicted attrition probability multiplied by 100.

| Risk level | Score range | Meaning |
|---|---:|---|
| LOW | 0-30 | Lower predicted attrition risk |
| MODERATE | 31-60 | Requires monitoring and context |
| HIGH | 61-80 | Consider proactive manager follow-up |
| CRITICAL | 81-100 | Prioritise human review and intervention |

Thresholds are centralised in `src/config.py`.

## Current Evaluation

The model uses a stratified 80/20 train-test split with `random_state=42`. The latest local training run produced these hold-out results; values may change when the data, dependencies, or training configuration changes.

| Metric | Score |
|---|---:|
| Accuracy | 0.8367 |
| Precision | 0.4865 |
| Recall | 0.3830 |
| F1 | 0.4286 |
| ROC-AUC | 0.7663 |
| PR-AUC | 0.4573 |

Recall is especially important because a false negative may leave an at-risk employee without timely support. Metrics should be interpreted with the dataset's limitations and the organisation's operational context in mind.

## Configuration

Shared configuration is maintained in `src/config.py`, including dataset and model paths, the target and employee identifier columns, risk thresholds, test split size, and random seed.

Optional environment variables can be configured by copying `.env.example` to `.env`. Do not commit credentials or private configuration values.

## Testing and Validation

Run the available tests with:

```bash
python -m pytest
```

Check that all Python modules compile with:

```bash
python -m compileall -q app src
```

## Troubleshooting

### Dataset not found

Confirm that the file is named exactly `HR-Employee-Attrition.csv` and exists under `data/raw/`.

### Model files not found

Run this command from the repository root:

```bash
python -m src.models.train
```

### Activating the environment in a terminal

After activation, your terminal prompt usually starts with `(.venv)`. You can then use `python` and `streamlit` directly:

```text
(.venv) C:\path\to\pulse-retain>
```

If the environment is already created, run only the activation command for your terminal:

```bat
:: Command Prompt
.venv\Scripts\activate.bat
```

```bash
# Git Bash
source .venv/Scripts/activate

# macOS or Linux
source .venv/bin/activate
```

### PowerShell cannot activate the environment

Use the environment interpreter directly without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_app.py
```

### Browser styling appears stale

Refresh the page with `Ctrl+F5` after restarting Streamlit. The local Streamlit configuration is stored in `.streamlit/config.toml`.

## Documentation

- [Architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Proposed system](docs/proposed_system.md)

## Responsible Use

Employee attrition predictions can reflect historical patterns and may reproduce bias in the source data. PulseRetain should support conversations, workload reviews, retention planning, and resource allocation, not automate hiring, firing, promotion, compensation, or disciplinary decisions. Access to employee data should be limited to authorised users, and interventions should be documented and reviewed appropriately.

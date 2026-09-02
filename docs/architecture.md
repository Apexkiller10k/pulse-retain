# PulseRetain — System Architecture

## 1. High-Level Overview

PulseRetain is structured as two distinct layers that share a common `src/` library:

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Application                  │
│  app/app.py  →  app/pages/*  →  app/data_service.py     │
└────────────────────────┬────────────────────────────────┘
                         │ imports
┌────────────────────────▼────────────────────────────────┐
│                     Core ML Library                      │
│  src/data/  src/features/  src/models/                  │
│  src/explainability/  src/interventions/                 │
└────────────────────────┬────────────────────────────────┘
                         │ reads / writes
┌────────────────────────▼────────────────────────────────┐
│                    Filesystem Artefacts                  │
│  data/raw/  data/processed/  models/                    │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Component Diagram

```
data/raw/
  HR-Employee-Attrition.csv
        │
        ▼
src/data/loader.py          ← load_raw_data()
        │
        ▼
src/data/validator.py       ← validate_dataset()
        │
        ▼
src/data/preprocessing.py   ← clean_dataset()
  • encode_target()           Yes/No → 1/0
  • drop_non_predictive_columns()
        │
        ▼
src/features/engineering.py ← add_engineered_features()
  • CareerStagnationRatio
  • RoleTenureRatio
  • ManagerTenureRatio
  • CompanyTenureRatio
  • CompanyChangeRate
  • SatisfactionIndex
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
src/models/train.py                         src/models/predict.py
  • _build_preprocessor()                    • load_model()
      ColumnTransformer                      • predict()
        num: Imputer → Scaler                    → attrition_probability
        cat: Imputer → OHE                       → risk_score
  • XGBClassifier                                → risk_level
  • train_and_save()                             → prediction_timestamp
        │
        ▼
models/
  attrition_model.pkl
  preprocessor.pkl
  model_metadata.json
        │
        ▼
src/explainability/shap_explainer.py
  • explain_employee()   → per-employee SHAP drivers
  • explain_all()        → top driver per employee (bulk)
        │
        ▼
src/interventions/recommendation_engine.py
  • get_recommendation()     → (action_title, detail)
  • get_all_recommendations() → [bullet_1, bullet_2, ...]
        │
        ▼
app/data_service.py
  • load_enriched_data()  @st.cache_data
    Assembles: features + predictions + drivers + recommendations
        │
        ▼
app/pages/
  dashboard.py        → KPIs, risk distribution, dept/role charts
  employees.py        → Filterable risk table, drill-down
  employee_profile.py → Gauge, SHAP bars, recommended actions
  interventions.py    → Status tracking, notes, dates
  analytics.py        → Business analytics + model performance
```

---

## 3. Data Flow

### 3.1 Training Flow (offline, run once)

```
CSV → load_raw_data()
    → validate_dataset()
    → clean_dataset()
    → add_engineered_features()
    → train_test_split (80/20, stratified)
    → _build_preprocessor() → ColumnTransformer.fit()
    → XGBClassifier.fit()
    → evaluate() → metrics dict
    → joblib.dump(model, preprocessor)
    → model_metadata.json
```

### 3.2 Inference Flow (per Streamlit session)

```
CSV → load_raw_data()
    → clean_dataset()
    → add_engineered_features()
    → predict()
        → joblib.load(attrition_model.pkl)
        → model.predict_proba()
        → risk_score = prob × 100
        → risk_level = threshold lookup
    → explain_all()
        → shap.TreeExplainer(clf)
        → shap_values per employee
        → top positive driver label
    → get_recommendation(top_driver)
    → Enriched DataFrame cached in session
```

---

## 4. Module Responsibilities

### `src/config.py`
Single source of truth for all paths, column names, risk thresholds, and ML constants. No magic strings or numbers exist elsewhere in the codebase.

### `src/data/loader.py`
Reads the raw CSV. Strips non-data rows (metadata appended by some CSV exports). Raises informative errors on missing files or empty data.

### `src/data/validator.py`
Runs 8 quality checks against the expected schema. Returns a structured report dict. Does not mutate the DataFrame.

### `src/data/preprocessing.py`
Phase 1 cleaning only: target encoding and column removal. The sklearn imputation/encoding/scaling pipeline is built inside `train.py` to ensure it is fitted only on training data.

### `src/features/engineering.py`
Adds 6 ratio-based features that capture business-meaningful signals not directly present in the raw data. All operations are vectorised pandas; no row-level loops.

### `src/models/train.py`
Builds the full `sklearn.Pipeline` (preprocessor + XGBClassifier), trains it, evaluates on the test split, and persists artefacts. Computes `scale_pos_weight` dynamically from the training split.

### `src/models/predict.py`
Loads the saved pipeline and scores a DataFrame. Applies the risk threshold logic to produce categorical risk levels. Stateless — no side effects beyond reading model files.

### `src/explainability/shap_explainer.py`
Wraps `shap.TreeExplainer`. Handles OHE feature name parsing (strips transformer prefix, maps base name to human-readable label). Provides both single-employee and bulk explanation functions.

### `src/interventions/recommendation_engine.py`
Pure rule-based engine. No ML. Maps SHAP driver labels to manager actions via keyword matching. Easily extensible by adding entries to the `_RULES` list.

### `app/data_service.py`
The single data access point for all Streamlit pages. Runs the full inference pipeline once per session and caches the result. Pages never call `src/` modules directly — they all read from this service.

### `app/pages/`
Each page is a module with a single `render()` function. Pages are stateless except for `interventions.py`, which uses `st.session_state` to persist manager notes and status within a session.

---

## 5. Key Design Decisions

### Separation of training and inference
The preprocessor is fitted only on training data inside `train.py`. The saved pipeline applies the same transformations at inference time without data leakage.

### Centralised configuration
`src/config.py` is the only place where paths, thresholds, and constants are defined. This makes environment-specific changes (e.g. different data paths) a single-file edit.

### SHAP on the XGBoost booster, not the pipeline
`shap.TreeExplainer` requires the raw booster, not the sklearn wrapper. The explainer extracts `model.named_steps["clf"]` and applies the preprocessor transform separately before computing SHAP values.

### Cached data service
`@st.cache_data` on `load_enriched_data()` ensures the model runs once per session regardless of how many pages the user visits. This is critical because SHAP computation over 1,470 employees is expensive.

### Rule-based interventions (not ML)
Intervention recommendations are intentionally rule-based. This makes them auditable, explainable to HR stakeholders, and easy to update without retraining. The rules are driven by the SHAP output, so they remain grounded in the model's reasoning.

---

## 6. File Artefacts

| File | Created by | Used by |
|---|---|---|
| `data/raw/HR-Employee-Attrition.csv` | Manual placement | `loader.py` |
| `data/processed/employee_features.csv` | `preprocessing.py` | Reference only |
| `models/attrition_model.pkl` | `train.py` | `predict.py`, `shap_explainer.py` |
| `models/preprocessor.pkl` | `train.py` | Reference (pipeline includes it) |
| `models/model_metadata.json` | `train.py` | `predict.py`, `shap_explainer.py`, `analytics.py` |

---

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn 1.3+, XGBoost 2.0+ |
| Explainability | SHAP 0.44+ |
| Data | pandas 2.0+, numpy 1.24+ |
| Serialisation | joblib 1.3+ |
| Visualisation | Plotly 5.18+ |
| UI | Streamlit 1.30+ |
| Configuration | python-dotenv 1.0+ |

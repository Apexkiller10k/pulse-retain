# PulseRetain — Methodology

## 1. Problem Statement

Employee attrition is costly. Industry estimates place the cost of replacing a single employee at 50–200% of their annual salary when accounting for recruitment, onboarding, and lost productivity. Most organisations detect attrition risk only after an employee has already decided to leave.

PulseRetain addresses this by shifting the intervention window earlier — predicting which employees are at elevated risk of leaving before they resign, and surfacing the specific drivers behind each prediction so managers can act with context.

---

## 2. Dataset

**Source:** IBM HR Analytics Employee Attrition & Performance dataset  
**Size:** 1,470 employees × 35 features  
**Target:** `Attrition` (Yes / No) — binary classification  
**Class balance:** ~16% attrition (Yes), ~84% retained (No)

The dataset is a synthetic but realistic HR dataset covering demographics, job characteristics, compensation, satisfaction scores, and tenure metrics.

---

## 3. Data Pipeline

### 3.1 Loading and Validation

The raw CSV is loaded via `src/data/loader.py`. Before any transformation, `src/data/validator.py` runs a structured quality check that reports:

- Missing columns against an expected schema
- Duplicate rows
- Missing values per column
- Constant (zero-variance) columns
- Potential identifier columns (all-unique values)
- Target class distribution
- Invalid categorical values

No missing values were found in the IBM HR dataset. Constant columns (`EmployeeCount`, `StandardHours`, `Over18`) and the identifier column (`EmployeeNumber`) are flagged and removed before modelling.

### 3.2 Preprocessing

Performed in `src/data/preprocessing.py`:

1. **Target encoding** — `Attrition`: `Yes → 1`, `No → 0`
2. **Column removal** — Drop `EmployeeNumber`, `EmployeeCount`, `StandardHours`, `Over18` (no predictive signal)

A full sklearn `ColumnTransformer` is built inside the training pipeline:

- **Numerical columns** — `SimpleImputer(strategy="median")` → `StandardScaler`
- **Categorical columns** — `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`

### 3.3 Feature Engineering

Six business-meaningful derived features are added in `src/features/engineering.py`:

| Feature | Formula | Business Meaning |
|---|---|---|
| `CareerStagnationRatio` | `YearsSinceLastPromotion / (YearsAtCompany + 1)` | Proportion of tenure without promotion |
| `RoleTenureRatio` | `YearsInCurrentRole / (YearsAtCompany + 1)` | Time locked in same role relative to tenure |
| `ManagerTenureRatio` | `YearsWithCurrManager / (YearsAtCompany + 1)` | Manager relationship stability |
| `CompanyTenureRatio` | `YearsAtCompany / (TotalWorkingYears + 1)` | Loyalty to current employer vs career length |
| `CompanyChangeRate` | `NumCompaniesWorked / (TotalWorkingYears + 1)` | Historical job-hopping tendency |
| `SatisfactionIndex` | Mean of 4 satisfaction scores | Composite well-being signal |

All divisions use `+1` to guard against zero denominators.

---

## 4. Modelling

### 4.1 Algorithm Selection

**XGBoost (`XGBClassifier`)** was selected for the following reasons:

- Strong performance on tabular data with mixed feature types
- Native handling of class imbalance via `scale_pos_weight`
- Compatible with SHAP `TreeExplainer` for fast, exact Shapley value computation
- Robust to outliers and does not require feature scaling (scaling is applied for consistency with potential future model comparisons)

### 4.2 Hyperparameters

```python
XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,   # ratio of negative to positive class
    eval_metric="logloss",
    random_state=42,
)
```

`scale_pos_weight` is computed dynamically as `count(No) / count(Yes)` on the training split to address the ~84/16 class imbalance.

### 4.3 Train / Test Split

- **Split ratio:** 80% train / 20% test
- **Stratified:** Yes — preserves the 16% positive rate in both splits
- **Train rows:** 1,176 | **Test rows:** 294
- **Random state:** 42 (reproducible)

### 4.4 Evaluation Metrics

| Metric | Value | Notes |
|---|---|---|
| Accuracy | 0.8265 | Misleading on imbalanced data |
| Precision | 0.4474 | Of predicted positives, 45% are true positives |
| Recall | 0.3617 | Primary business metric — detects 36% of true leavers |
| F1 | 0.4000 | Harmonic mean of precision and recall |
| ROC-AUC | 0.7593 | Discrimination ability across all thresholds |
| PR-AUC | 0.4505 | Area under precision-recall curve |

**Recall is the primary business metric.** A false negative (missed at-risk employee) is more costly than a false positive (unnecessary check-in with a retained employee).

---

## 5. Risk Scoring

The model outputs a probability `p ∈ [0, 1]`. This is converted to a 0–100 risk score (`p × 100`) and mapped to a categorical risk level:

| Level | Score Range | Interpretation |
|---|---|---|
| LOW | 0 – 30 | Minimal concern; standard engagement |
| MODERATE | 31 – 60 | Monitor; consider proactive check-in |
| HIGH | 61 – 80 | Elevated risk; manager intervention recommended |
| CRITICAL | 81 – 100 | Immediate action required |

---

## 6. Explainability

### 6.1 SHAP (SHapley Additive exPlanations)

SHAP values are computed using `shap.TreeExplainer`, which provides exact Shapley values for tree-based models in polynomial time.

For each employee, the explainer returns a vector of SHAP values — one per transformed feature. Positive values increase the predicted attrition probability; negative values decrease it.

### 6.2 Feature Label Mapping

Raw feature names (including OHE suffixes like `OverTime_Yes`) are mapped to human-readable labels (e.g. `"Overtime"`) via a lookup dictionary in `shap_explainer.py`. For OHE columns, the base feature name is used for the lookup.

### 6.3 Driver Aggregation

For the risk table and intervention engine, the single top positive SHAP driver per employee is extracted via `explain_all()`. For the employee profile page, the top 5 positive and top 5 protective drivers are shown with proportional bar visualisations.

---

## 7. Intervention Recommendation Engine

The recommendation engine in `src/interventions/recommendation_engine.py` maps the top SHAP driver label to a concrete manager action using keyword matching against a rule table.

Six rule categories are defined:

| Category | Trigger Keywords | Action |
|---|---|---|
| Workload Review | Overtime, Work-Life Balance, Distance from Home, Business Travel | Schedule workload review |
| Satisfaction Check-in | Job Satisfaction, Environment Satisfaction, Satisfaction Index | Structured 1-on-1 |
| Career Progression | Career Stagnation, Years Since Promotion, Years in Current Role | Career development conversation |
| Compensation Review | Monthly Income, Salary Hike %, Stock Option Level, Job Level | Compensation benchmarking |
| Manager Relationship | Years with Manager, Relationship Satisfaction | Confidential HR conversation |
| Retention Conversation | Companies Worked, Company Change Rate, Total Working Years | Open retention discussion |

If no keyword matches, a general manager check-in is recommended as a fallback.

---

## 8. Caching Strategy

The Streamlit application uses `@st.cache_data` on `load_enriched_data()` in `app/data_service.py`. The full pipeline (load → clean → engineer → predict → explain → recommend) runs once per session and is cached in memory. This prevents redundant model inference on every page navigation.

---

## 9. Limitations

- The model is trained on a synthetic dataset. Performance on real organisational data will vary and retraining is recommended.
- SHAP values explain the model's predictions, not ground-truth causal relationships.
- The intervention recommendations are rule-based heuristics, not personalised to individual circumstances.
- The system does not persist intervention records between sessions (session state only).
- Recall of 36% means the majority of true leavers are not flagged at the default 0.5 threshold. Lowering the classification threshold would increase recall at the cost of precision.

# PulseRetain — Proposed System

## 1. Background and Motivation

### 1.1 The Problem

Voluntary employee attrition is one of the most significant and preventable costs in human resources management. When a skilled employee leaves, the organisation incurs:

- Direct costs: recruitment advertising, agency fees, interview time, onboarding
- Indirect costs: lost institutional knowledge, reduced team productivity, morale impact on remaining staff
- Time cost: 3–6 months before a replacement reaches full productivity

The fundamental challenge is timing. By the time an employee submits their resignation, the decision has typically been made weeks or months earlier. Traditional HR processes — annual surveys, exit interviews, performance reviews — are retrospective. They describe attrition after it has happened, not before.

### 1.2 The Opportunity

Modern HR systems collect rich longitudinal data on every employee: tenure, compensation, satisfaction scores, promotion history, overtime patterns, and more. This data contains early signals of disengagement that, when modelled correctly, can identify at-risk employees months before they resign.

The opportunity is to build a system that:
1. Continuously scores every employee's attrition risk
2. Explains *why* each employee is at risk in terms a manager can act on
3. Recommends specific, targeted interventions
4. Tracks whether those interventions were carried out

---

## 2. Proposed System

### 2.1 System Name

**PulseRetain** — AI-Powered Employee Retention Intelligence

### 2.2 Core Proposition

PulseRetain is a decision-support system for HR teams and line managers. It does not make employment decisions. It surfaces probabilistic risk signals and contextual recommendations that enable humans to intervene earlier and more effectively.

### 2.3 Target Users

| User | Primary Need |
|---|---|
| HR Business Partners | Organisation-wide risk overview, intervention tracking |
| Line Managers | Individual employee risk context, actionable recommendations |
| HR Analytics Teams | Model performance monitoring, trend analysis |
| Senior Leadership | Executive KPIs, department-level risk exposure |

---

## 3. Functional Requirements

### 3.1 Risk Prediction

- The system shall score every employee in the dataset with an attrition probability (0–1) and a derived risk score (0–100)
- Risk scores shall be categorised into four levels: LOW, MODERATE, HIGH, CRITICAL
- Predictions shall be reproducible given the same model and input data

### 3.2 Explainability

- For each employee, the system shall identify the top contributing factors (increasing risk) and protective factors (decreasing risk)
- Explanations shall be expressed in human-readable business language, not raw feature names
- Explanations shall be grounded in the model's actual reasoning (SHAP values), not post-hoc heuristics

### 3.3 Intervention Recommendations

- For each at-risk employee, the system shall recommend a specific manager action based on the primary risk driver
- Recommendations shall be concrete and actionable, not generic
- The system shall support tracking of intervention status (Not Started / In Progress / Completed) and manager notes

### 3.4 Dashboard and Reporting

- The system shall provide an executive dashboard with organisation-wide KPIs
- The system shall support filtering of employees by department, job role, and risk level
- The system shall display model performance metrics and diagnostic charts

### 3.5 Data Quality

- The system shall validate input data against an expected schema before processing
- The system shall report data quality issues (missing values, duplicates, unexpected values) without failing silently

---

## 4. Non-Functional Requirements

| Requirement | Specification |
|---|---|
| Performance | Full pipeline (1,470 employees) completes in under 30 seconds on a standard laptop |
| Reproducibility | Fixed random seed (42) ensures identical results across runs |
| Maintainability | All configuration in a single file; no magic strings in business logic |
| Extensibility | New intervention rules added by editing a single list; no code changes elsewhere |
| Transparency | All predictions are explainable; no black-box outputs presented to users |
| Ethical compliance | System is labelled as decision-support; all recommendations require human review |

---

## 5. System Boundaries

### 5.1 In Scope

- Batch attrition risk scoring for all employees in the dataset
- Per-employee SHAP-based driver explanation
- Rule-based intervention recommendation
- Interactive Streamlit dashboard (5 pages)
- Model training and evaluation pipeline
- Data validation and quality reporting

### 5.2 Out of Scope

- Real-time streaming predictions (batch only)
- Integration with HRIS systems (e.g. Workday, SAP SuccessFactors)
- Persistent database storage (session state only)
- User authentication and role-based access control
- Automated email or notification workflows
- Multi-tenant or multi-organisation support

---

## 6. System Pages

### 6.1 Executive Dashboard

Provides a real-time overview of the organisation's attrition risk posture.

- KPI cards: total employees, high/critical count, critical count, average risk score
- Risk level distribution bar chart
- Average risk score by department (horizontal bar)
- Average risk score by job role (horizontal bar)
- Top risk drivers across the organisation (percentage of employees affected)

### 6.2 At-Risk Employees

A filterable table of all employees with their risk scores and recommended actions.

- Filters: employee ID search, department, risk level, job role
- Sortable by risk score (descending by default)
- Colour-coded risk badges
- Direct navigation to individual employee profiles

### 6.3 Employee Profile

A deep-dive view for a single employee.

- Risk gauge (0–100) with colour-coded level badge
- SHAP driver visualisation: top 5 contributing factors and top 5 protective factors with proportional bars
- Recommended action cards (2–3 concrete steps)
- Employee snapshot: income, overtime, satisfaction scores, promotion history, travel frequency

### 6.4 Intervention Tracker

A workflow tool for managers to record and track actions taken.

- Filterable by risk level and intervention status
- Per-employee expandable cards showing risk context and recommendation
- Status selector (Not Started / In Progress / Completed)
- Free-text manager notes field
- Intervention date picker
- Session-level summary: total tracked, in progress, completed

### 6.5 Analytics

Two-tab analytics view for business insights and model validation.

**Business Analytics tab:**
- Risk score distribution histogram
- Tenure vs risk score scatter (colour by risk level)
- Satisfaction index vs risk score scatter
- Average risk by marital status
- Average risk by overtime status

**Model Performance tab:**
- Six metric cards (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
- ROC curve with random baseline
- Precision-recall curve
- Confusion matrix heatmap
- Top 20 feature importances (XGBoost gain)

---

## 7. Data Model

### 7.1 Input Features (36 total after engineering)

**Original features (30):** Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

**Engineered features (6):** CareerStagnationRatio, RoleTenureRatio, ManagerTenureRatio, CompanyTenureRatio, CompanyChangeRate, SatisfactionIndex

### 7.2 Output Fields

| Field | Type | Description |
|---|---|---|
| `attrition_probability` | float [0,1] | Raw model output |
| `risk_score` | float [0,100] | Probability × 100 |
| `risk_level` | string | LOW / MODERATE / HIGH / CRITICAL |
| `top_driver` | string | Human-readable top SHAP driver |
| `recommended_action` | string | Short action title |
| `intervention_status` | string | Not Started / In Progress / Completed |

---

## 8. Intervention Rule Design

The recommendation engine uses a keyword-matching rule table. Each rule maps a set of driver keywords to a short action title and a detailed recommendation text.

This design was chosen over a second ML model for three reasons:

1. **Auditability** — HR stakeholders can read and challenge the rules
2. **Stability** — Rules do not change with model retraining
3. **Simplicity** — Adding a new intervention category requires one list entry

The rules are grounded in established HR practice:
- Workload and overtime → flexible working discussion
- Satisfaction scores → structured 1-on-1
- Promotion stagnation → career development conversation
- Compensation signals → market benchmarking
- Manager relationship → confidential HR mediation
- Job-hopping history → retention conversation

---

## 9. Ethical Considerations

### 9.1 Transparency
Every risk score is accompanied by an explanation. Users are never presented with a score without context.

### 9.2 Human Oversight
The system is explicitly labelled as a decision-support tool throughout the UI. No automated actions are triggered by the system.

### 9.3 Fairness
The model uses demographic features (gender, marital status, age) that are present in the dataset. Organisations deploying this system should audit for disparate impact across protected groups before use in production.

### 9.4 Data Privacy
The system processes employee data. In a production deployment, access should be restricted to authorised HR personnel, and data handling must comply with applicable privacy regulations (GDPR, CCPA, etc.).

---

## 10. Future Enhancements

| Enhancement | Priority | Description |
|---|---|---|
| Threshold tuning | High | Allow HR to adjust the classification threshold to trade precision for recall based on organisational risk appetite |
| HRIS integration | High | Connect to live employee data via API rather than static CSV |
| Persistent storage | Medium | Replace session state with a database for cross-session intervention tracking |
| Retraining pipeline | Medium | Scheduled model retraining as new attrition outcomes are observed |
| Fairness audit | Medium | Automated disparate impact analysis across demographic groups |
| Authentication | Medium | Role-based access (manager sees own team only; HR sees all) |
| Email alerts | Low | Automated notifications when an employee crosses a risk threshold |
| Cohort analysis | Low | Compare risk profiles across hiring cohorts or performance bands |

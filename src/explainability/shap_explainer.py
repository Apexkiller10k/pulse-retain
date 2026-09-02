"""
shap_explainer.py — SHAP-based explainability for PulseRetain.

Produces top positive and protective drivers for each employee
using TreeExplainer on the XGBoost model inside the pipeline.
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import (
    LOG_FORMAT, LOG_DATE_FORMAT,
    MODEL_PATH, MODEL_METADATA_PATH, TARGET_COLUMN,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Human-readable labels for raw feature names
FEATURE_LABELS = {
    "OverTime":                  "Overtime",
    "JobSatisfaction":           "Job Satisfaction",
    "EnvironmentSatisfaction":   "Environment Satisfaction",
    "RelationshipSatisfaction":  "Relationship Satisfaction",
    "WorkLifeBalance":           "Work-Life Balance",
    "YearsSinceLastPromotion":   "Years Since Promotion",
    "YearsInCurrentRole":        "Years in Current Role",
    "YearsAtCompany":            "Years at Company",
    "YearsWithCurrManager":      "Years with Manager",
    "MonthlyIncome":             "Monthly Income",
    "JobLevel":                  "Job Level",
    "Age":                       "Age",
    "DistanceFromHome":          "Distance from Home",
    "NumCompaniesWorked":        "Companies Worked",
    "TotalWorkingYears":         "Total Working Years",
    "BusinessTravel":            "Business Travel",
    "Department":                "Department",
    "JobRole":                   "Job Role",
    "MaritalStatus":             "Marital Status",
    "StockOptionLevel":          "Stock Option Level",
    "PercentSalaryHike":         "Salary Hike %",
    "TrainingTimesLastYear":     "Training Times Last Year",
    "CareerStagnationRatio":     "Career Stagnation",
    "SatisfactionIndex":         "Satisfaction Index",
    "CompanyChangeRate":         "Company Change Rate",
    "RoleTenureRatio":           "Role Tenure Ratio",
}


def _get_explainer_and_features(model_pipeline):
    """Extract the XGB booster and transformed feature names from the pipeline."""
    pre = model_pipeline.named_steps["pre"]
    clf = model_pipeline.named_steps["clf"]

    # Get feature names out of the ColumnTransformer
    try:
        feature_names = pre.get_feature_names_out()
        # Strip transformer prefix (e.g. "num__Age" → "Age", "cat__OverTime_Yes" → "OverTime_Yes")
        feature_names = [n.split("__", 1)[-1] for n in feature_names]
    except Exception:
        feature_names = [f"f{i}" for i in range(clf.n_features_in_)]

    explainer = shap.TreeExplainer(clf)
    return explainer, feature_names, pre


def explain_employee(df_row: pd.DataFrame, top_n: int = 5) -> dict:
    """Compute SHAP values for a single employee row.

    Args:
        df_row: Single-row DataFrame with all feature columns.
        top_n:  Number of top drivers to return.

    Returns:
        Dict with keys:
            drivers_positive  — list of (label, shap_value) increasing risk
            drivers_protective — list of (label, shap_value) decreasing risk
    """
    import joblib
    model = joblib.load(MODEL_PATH)
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    meta = json.loads(MODEL_METADATA_PATH.read_text())
    X = df_row[[c for c in meta["features"] if c in df_row.columns]]
    X_transformed = pre.transform(X)

    try:
        feature_names = pre.get_feature_names_out()
        feature_names = [n.split("__", 1)[-1] for n in feature_names]
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_transformed)[0]

    pairs = sorted(zip(feature_names, shap_vals), key=lambda x: x[1], reverse=True)

    def label(name):
        # For OHE columns like "OverTime_Yes", map base name
        base = name.split("_")[0]
        return FEATURE_LABELS.get(name, FEATURE_LABELS.get(base, name))

    positive   = [(label(n), round(float(v), 4)) for n, v in pairs if v > 0][:top_n]
    protective = [(label(n), round(float(v), 4)) for n, v in pairs if v < 0][:top_n]

    return {"drivers_positive": positive, "drivers_protective": protective}


def explain_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute top positive driver label for every employee (used in risk table).

    Returns a Series of top-driver strings indexed like df.
    """
    import joblib
    model = joblib.load(MODEL_PATH)
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    meta = json.loads(MODEL_METADATA_PATH.read_text())
    X = df[[c for c in meta["features"] if c in df.columns]]
    X_transformed = pre.transform(X)

    try:
        feature_names = pre.get_feature_names_out()
        feature_names = [n.split("__", 1)[-1] for n in feature_names]
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    explainer = shap.TreeExplainer(clf)
    shap_matrix = explainer.shap_values(X_transformed)

    top_drivers = []
    for row in shap_matrix:
        idx = int(np.argmax(row))
        name = feature_names[idx]
        base = name.split("_")[0]
        top_drivers.append(FEATURE_LABELS.get(name, FEATURE_LABELS.get(base, name)))

    return pd.Series(top_drivers, index=df.index, name="top_driver")

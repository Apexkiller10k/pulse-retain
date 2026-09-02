"""
validator.py — Data quality checks for PulseRetain.

Validates the raw DataFrame against known expectations for the
HR-Employee-Attrition dataset and returns a structured report dict.
Problems are logged as warnings; the caller decides whether to abort.
"""
import logging
from typing import Any

import pandas as pd

from src.config import LOG_FORMAT, LOG_DATE_FORMAT, TARGET_COLUMN, EMPLOYEE_ID_COLUMN

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Columns we expect to be present in the raw dataset
EXPECTED_COLUMNS = {
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
    "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StandardHours", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager",
}


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Run all data quality checks and return a structured report.

    Args:
        df: Raw DataFrame loaded from CSV.

    Returns:
        Dictionary with keys: issues, warnings, missing_columns,
        duplicate_rows, missing_values, constant_columns,
        potential_id_columns, target_distribution, status.
    """
    report: dict[str, Any] = {
        "issues": [],
        "warnings": [],
        "missing_columns": [],
        "duplicate_rows": 0,
        "missing_values": {},
        "constant_columns": [],
        "potential_id_columns": [],
        "target_distribution": {},
        "status": "PASS",
    }

    # ── 1. Expected columns ────────────────────────────────────────────────
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        msg = f"Missing expected columns: {sorted(missing_cols)}"
        report["issues"].append(msg)
        report["missing_columns"] = sorted(missing_cols)
        logger.warning(msg)

    # ── 2. Target column ──────────────────────────────────────────────────
    if TARGET_COLUMN not in df.columns:
        msg = f"Target column '{TARGET_COLUMN}' not found."
        report["issues"].append(msg)
        logger.error(msg)
    else:
        dist = df[TARGET_COLUMN].value_counts(normalize=True).mul(100).round(1)
        report["target_distribution"] = dist.to_dict()
        logger.info("Target distribution — %s", dist.to_dict())

        if "Yes" not in dist.index:
            report["warnings"].append("No positive (Yes) attrition cases found.")

    # ── 3. Duplicate rows ─────────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    report["duplicate_rows"] = int(n_dupes)
    if n_dupes > 0:
        msg = f"Found {n_dupes} duplicate rows."
        report["warnings"].append(msg)
        logger.warning(msg)

    # ── 4. Missing values ─────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        report["missing_values"] = missing.to_dict()
        msg = f"Missing values detected: {missing.to_dict()}"
        report["warnings"].append(msg)
        logger.warning(msg)
    else:
        logger.info("No missing values detected.")

    # ── 5. Constant columns ───────────────────────────────────────────────
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    report["constant_columns"] = constant_cols
    if constant_cols:
        msg = f"Constant columns (zero variance): {constant_cols}"
        report["warnings"].append(msg)
        logger.warning(msg)

    # ── 6. Potential ID columns ───────────────────────────────────────────
    n_rows = len(df)
    id_cols = [
        c for c in df.columns
        if df[c].nunique() == n_rows and c != TARGET_COLUMN
    ]
    report["potential_id_columns"] = id_cols
    if id_cols:
        msg = f"Potential identifier columns (all unique): {id_cols}"
        report["warnings"].append(msg)
        logger.warning(msg)

    # ── 7. Data types ─────────────────────────────────────────────────────
    # Age should be numeric — flag if it isn't
    if "Age" in df.columns and not pd.api.types.is_numeric_dtype(df["Age"]):
        msg = "Column 'Age' is not numeric — type conversion may be needed."
        report["warnings"].append(msg)
        logger.warning(msg)

    # ── 8. Invalid values in known categorical columns ────────────────────
    if "OverTime" in df.columns:
        valid_overtime = {"Yes", "No"}
        unexpected = set(df["OverTime"].dropna().unique()) - valid_overtime
        if unexpected:
            msg = f"Unexpected values in 'OverTime': {unexpected}"
            report["warnings"].append(msg)
            logger.warning(msg)

    # ── Final status ──────────────────────────────────────────────────────
    if report["issues"]:
        report["status"] = "FAIL"
    elif report["warnings"]:
        report["status"] = "REVIEW REQUIRED"

    return report

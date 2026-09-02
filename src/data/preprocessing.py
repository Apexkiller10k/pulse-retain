"""
preprocessing.py — Initial data cleaning and quality report for PulseRetain.

Phase 1 scope:
  - Print the human-readable data quality report to stdout.
  - Encode the binary target (Attrition: Yes→1, No→0).
  - Drop constant / identifier columns.
  - Save a clean intermediate CSV for downstream phases.

A full sklearn Pipeline (imputation, encoding, scaling) is built in Phase 2.
"""
import logging
from typing import Any

import pandas as pd

from src.config import (
    LOG_FORMAT, LOG_DATE_FORMAT,
    TARGET_COLUMN, DROP_COLUMNS,
    DATA_PROCESSED_DIR, PROCESSED_DATA_PATH,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ── Report ─────────────────────────────────────────────────────────────────

def generate_data_quality_report(df: pd.DataFrame, report: dict[str, Any]) -> None:
    """Print a formatted data quality report to stdout.

    Args:
        df: Raw DataFrame.
        report: Validation report dict produced by validator.validate_dataset.
    """
    sep = "=" * 48

    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    print(f"\n{sep}")
    print("PulseRetain Data Quality Report")
    print(sep)

    print(f"\nDataset:  HR-Employee-Attrition.csv")
    print(f"Rows:     {len(df):,}")
    print(f"Columns:  {len(df.columns)}")

    total_missing = sum(report["missing_values"].values()) if report["missing_values"] else 0
    print(f"\nMissing values:  {total_missing}")
    print(f"Duplicate rows:  {report['duplicate_rows']}")

    print(f"\nTarget:  {TARGET_COLUMN}")
    if report["target_distribution"]:
        print("Class distribution:")
        for label, pct in report["target_distribution"].items():
            print(f"  {label}: {pct:.1f}%")

    print(f"\nConstant columns:")
    if report["constant_columns"]:
        for c in report["constant_columns"]:
            print(f"  {c}")
    else:
        print("  None")

    print(f"\nPotential ID columns:")
    if report["potential_id_columns"]:
        for c in report["potential_id_columns"]:
            print(f"  {c}")
    else:
        print("  None")

    print(f"\nNumerical columns ({len(numerical_cols)}):")
    for c in numerical_cols:
        print(f"  {c}")

    print(f"\nCategorical columns ({len(categorical_cols)}):")
    for c in categorical_cols:
        unique_vals = df[c].nunique()
        print(f"  {c}  ({unique_vals} unique)")

    if report["warnings"]:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  [!] {w}")

    if report["issues"]:
        print(f"\nIssues ({len(report['issues'])}):")
        for i in report["issues"]:
            print(f"  [X] {i}")

    print(f"\nStatus:  {report['status']}")
    print(f"{sep}\n")


# ── Cleaning ───────────────────────────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Attrition column: Yes→1, No→0.

    Args:
        df: DataFrame containing the TARGET_COLUMN.

    Returns:
        DataFrame with TARGET_COLUMN as integer (0/1).

    Raises:
        KeyError: If TARGET_COLUMN is absent.
        ValueError: If unexpected values are found in TARGET_COLUMN.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    unexpected = set(df[TARGET_COLUMN].dropna().unique()) - {"Yes", "No"}
    if unexpected:
        raise ValueError(f"Unexpected values in '{TARGET_COLUMN}': {unexpected}")

    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    logger.info("Target encoded — Yes→1, No→0")
    return df


def drop_non_predictive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier and constant columns that must not influence the model.

    Args:
        df: DataFrame (may or may not contain all DROP_COLUMNS).

    Returns:
        DataFrame with non-predictive columns removed.
    """
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped non-predictive columns: %s", cols_to_drop)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Phase 1 cleaning: encode target, drop non-predictive columns.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = encode_target(df)
    df = drop_non_predictive_columns(df)
    logger.info("Cleaned dataset — %d rows × %d columns", *df.shape)
    return df


def save_processed_data(df: pd.DataFrame) -> None:
    """Persist the cleaned DataFrame to the processed data directory.

    Args:
        df: Cleaned DataFrame.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Processed dataset saved to: %s", PROCESSED_DATA_PATH)

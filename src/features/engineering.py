"""
engineering.py — Business-relevant feature engineering for PulseRetain.
"""
import logging
import pandas as pd
from src.config import LOG_FORMAT, LOG_DATE_FORMAT

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that capture business-meaningful signals.

    All divisions guard against zero denominators with +1.

    Args:
        df: Cleaned DataFrame (post Phase-1 preprocessing).

    Returns:
        DataFrame with additional engineered columns appended.
    """
    df = df.copy()

    df["CareerStagnationRatio"] = (
        df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    )
    df["RoleTenureRatio"] = (
        df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    )
    df["ManagerTenureRatio"] = (
        df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
    )
    df["CompanyTenureRatio"] = (
        df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    )
    df["CompanyChangeRate"] = (
        df["NumCompaniesWorked"] / (df["TotalWorkingYears"] + 1)
    )
    df["SatisfactionIndex"] = df[[
        "JobSatisfaction", "EnvironmentSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance",
    ]].mean(axis=1)

    logger.info("Engineered features added — %d columns total", len(df.columns))
    return df

"""
data_service.py — Single source of truth for the Streamlit app.

Loads the raw CSV, runs the full pipeline, and returns an enriched
DataFrame that every page reads from. Results are cached so the
model runs only once per session.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.data.loader import load_raw_data
from src.data.preprocessing import clean_dataset
from src.features.engineering import add_engineered_features
from src.models.predict import predict
from src.explainability.shap_explainer import explain_all
from src.interventions.recommendation_engine import get_recommendation
from src.config import TARGET_COLUMN


@st.cache_data(show_spinner="Loading employee data...")
def load_enriched_data() -> pd.DataFrame:
    """Load raw CSV → clean → engineer → predict → explain → recommend.

    Returns:
        DataFrame with all original columns plus:
            risk_score, risk_level, attrition_probability,
            top_driver, recommended_action, intervention_status,
            EmployeeNumber (preserved as display ID)
    """
    raw = load_raw_data()

    # Preserve EmployeeNumber for display before it gets dropped
    emp_ids = raw["EmployeeNumber"].values

    cleaned  = clean_dataset(raw)
    featured = add_engineered_features(cleaned)

    predictions = predict(featured)
    top_drivers = explain_all(featured)

    df = featured.copy()
    df["EmployeeNumber"]   = emp_ids
    df["risk_score"]       = predictions["risk_score"].values
    df["risk_level"]       = predictions["risk_level"].values
    df["attrition_probability"] = predictions["attrition_probability"].values
    df["top_driver"]       = top_drivers.values

    df["recommended_action"] = df["top_driver"].apply(
        lambda d: get_recommendation(d)[0]
    )
    df["intervention_status"] = "Not Started"

    return df

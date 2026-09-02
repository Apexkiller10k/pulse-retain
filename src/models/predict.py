"""
predict.py — Inference engine for PulseRetain.

Loads the trained pipeline and produces risk scores + levels for
every employee in a DataFrame.
"""
import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.config import (
    LOG_FORMAT, LOG_DATE_FORMAT,
    MODEL_PATH, MODEL_METADATA_PATH,
    RISK_LOW_THRESHOLD, RISK_MODERATE_THRESHOLD, RISK_HIGH_THRESHOLD,
    TARGET_COLUMN,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _risk_level(score: float) -> str:
    if score <= RISK_LOW_THRESHOLD:
        return "LOW"
    if score <= RISK_MODERATE_THRESHOLD:
        return "MODERATE"
    if score <= RISK_HIGH_THRESHOLD:
        return "HIGH"
    return "CRITICAL"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run src/models/train.py first."
        )
    return joblib.load(MODEL_PATH)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of employees.

    Args:
        df: Feature DataFrame (same columns used during training,
            TARGET_COLUMN may be present and will be ignored).

    Returns:
        DataFrame with columns:
            attrition_probability, risk_score, risk_level, prediction_timestamp
    """
    model = load_model()
    meta  = json.loads(MODEL_METADATA_PATH.read_text())

    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    # Keep only columns the model was trained on
    X = X[[c for c in meta["features"] if c in X.columns]]

    probs = model.predict_proba(X)[:, 1]
    scores = (probs * 100).round(1)
    ts = datetime.now(timezone.utc).isoformat()

    return pd.DataFrame({
        "attrition_probability": probs.round(4),
        "risk_score":  scores,
        "risk_level":  [_risk_level(s) for s in scores],
        "prediction_timestamp": ts,
    }, index=df.index)

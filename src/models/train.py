"""
train.py — Model training pipeline for PulseRetain.

Trains an XGBoost classifier inside a full sklearn Pipeline
(imputation → encoding → scaling → model).
Saves model, preprocessor, and metadata to models/.
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    LOG_FORMAT, LOG_DATE_FORMAT,
    TARGET_COLUMN, RANDOM_STATE, TEST_SIZE,
    MODEL_PATH, PREPROCESSOR_PATH, MODEL_METADATA_PATH, MODELS_DIR,
)
from src.data.loader import load_raw_data
from src.data.preprocessing import clean_dataset
from src.features.engineering import add_engineered_features

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def _evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_test, y_prob), 4),
    }


def train_and_save() -> dict:
    """Full training run. Returns metadata dict."""
    # ── Data ──────────────────────────────────────────────────────────────
    raw = load_raw_data()
    df  = clean_dataset(raw)
    df  = add_engineered_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train: %d  Test: %d  Positive rate: %.1f%%",
                len(X_train), len(X_test), y.mean() * 100)

    # ── Preprocessor ──────────────────────────────────────────────────────
    preprocessor = _build_preprocessor(X_train)

    # ── Model ─────────────────────────────────────────────────────────────
    scale_pos = int((y_train == 0).sum()) / int((y_train == 1).sum())
    model = Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )),
    ])

    model.fit(X_train, y_train)
    metrics = _evaluate(model, X_test, y_test)
    logger.info("Metrics: %s", metrics)

    # ── Save ──────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # Store feature names for SHAP
    feature_names = X_train.columns.tolist()
    metadata = {
        "model": "XGBClassifier",
        "features": feature_names,
        "metrics": metrics,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": RANDOM_STATE,
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("Model saved to %s", MODEL_PATH)
    return metadata


if __name__ == "__main__":
    meta = train_and_save()
    print("\nTraining complete.")
    for k, v in meta["metrics"].items():
        print(f"  {k}: {v}")

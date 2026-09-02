"""
analytics.py — Analytics page for PulseRetain.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
)

from data_service import load_enriched_data
from src.config import MODEL_METADATA_PATH, TARGET_COLUMN
from src.data.loader import load_raw_data
from src.data.preprocessing import clean_dataset
from src.features.engineering import add_engineered_features
from src.models.predict import predict, load_model

DARK = "#0f1117"
SURFACE = "#161b27"
BORDER = "#1e2535"


def render():
    st.title("Analytics")

    df = load_enriched_data()
    meta = json.loads(MODEL_METADATA_PATH.read_text())

    tab1, tab2 = st.tabs(["Business Analytics", "Model Performance"])

    # ── Tab 1: Business Analytics ─────────────────────────────────────────
    with tab1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            df, x="risk_score", nbins=30, template="plotly_dark",
            color_discrete_sequence=["#4f8ef7"],
            labels={"risk_score": "Risk Score"},
        )
        fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tenure vs Risk Score")
            fig2 = px.scatter(
                df, x="YearsAtCompany", y="risk_score",
                color="risk_level", template="plotly_dark",
                color_discrete_map={
                    "LOW": "#34d399", "MODERATE": "#fbbf24",
                    "HIGH": "#f97316", "CRITICAL": "#f87171",
                },
                labels={"YearsAtCompany": "Years at Company", "risk_score": "Risk Score"},
                opacity=0.6,
            )
            fig2.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                               margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.subheader("Satisfaction Index vs Risk Score")
            fig3 = px.scatter(
                df, x="SatisfactionIndex", y="risk_score",
                color="risk_level", template="plotly_dark",
                color_discrete_map={
                    "LOW": "#34d399", "MODERATE": "#fbbf24",
                    "HIGH": "#f97316", "CRITICAL": "#f87171",
                },
                labels={"SatisfactionIndex": "Satisfaction Index", "risk_score": "Risk Score"},
                opacity=0.6,
            )
            fig3.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                               margin=dict(t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Risk by Marital Status & Overtime")
        col3, col4 = st.columns(2)
        with col3:
            ms = df.groupby("MaritalStatus")["risk_score"].mean().round(1).reset_index()
            fig4 = px.bar(ms, x="MaritalStatus", y="risk_score",
                          template="plotly_dark", color_discrete_sequence=["#4f8ef7"],
                          labels={"risk_score": "Avg Risk Score"})
            fig4.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                               margin=dict(t=10, b=10))
            st.plotly_chart(fig4, use_container_width=True)

        with col4:
            ot = df.groupby("OverTime")["risk_score"].mean().round(1).reset_index()
            fig5 = px.bar(ot, x="OverTime", y="risk_score",
                          template="plotly_dark",
                          color="OverTime",
                          color_discrete_map={"Yes": "#f97316", "No": "#34d399"},
                          labels={"risk_score": "Avg Risk Score"})
            fig5.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                               showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig5, use_container_width=True)

    # ── Tab 2: Model Performance ──────────────────────────────────────────
    with tab2:
        st.subheader("Model Metrics")
        metrics = meta["metrics"]
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy",  metrics["accuracy"])
        m2.metric("Precision", metrics["precision"])
        m3.metric("Recall",    metrics["recall"])
        m4.metric("F1",        metrics["f1"])
        m5.metric("ROC-AUC",   metrics["roc_auc"])
        m6.metric("PR-AUC",    metrics["pr_auc"])

        st.caption(
            "Recall is the key business metric — a false negative means a "
            "at-risk employee goes undetected."
        )
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Recompute curves on test split
        @st.cache_data(show_spinner="Computing model curves...")
        def _get_curves():
            from sklearn.model_selection import train_test_split
            from src.config import TEST_SIZE, RANDOM_STATE
            raw = load_raw_data()
            clean = clean_dataset(raw)
            feat  = add_engineered_features(clean)
            X = feat.drop(columns=[TARGET_COLUMN])
            y = feat[TARGET_COLUMN]
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
            )
            model = load_model()
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred)
            return fpr, tpr, prec, rec, cm, y_test.values, y_prob

        fpr, tpr, prec, rec, cm, y_test, y_prob = _get_curves()

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("ROC Curve")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name="XGBoost",
                line=dict(color="#4f8ef7", width=2),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#6b7280", dash="dash"), name="Random",
            ))
            fig_roc.update_layout(
                template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_b:
            st.subheader("Precision-Recall Curve")
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines", name="XGBoost",
                line=dict(color="#f97316", width=2),
            ))
            fig_pr.update_layout(
                template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                xaxis_title="Recall", yaxis_title="Precision",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        st.subheader("Confusion Matrix")
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Predicted No", "Predicted Yes"],
            y=["Actual No", "Actual Yes"],
            text_auto=True,
            color_continuous_scale=["#161b27", "#4f8ef7"],
            template="plotly_dark",
        )
        fig_cm.update_layout(
            paper_bgcolor=SURFACE, margin=dict(t=10, b=10), width=400
        )
        st.plotly_chart(fig_cm)

        st.subheader("Feature Importance (Top 20)")
        model = load_model()
        clf   = model.named_steps["clf"]
        pre   = model.named_steps["pre"]
        try:
            feat_names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
        except Exception:
            feat_names = [f"f{i}" for i in range(len(clf.feature_importances_))]

        import pandas as pd
        fi = pd.DataFrame({
            "Feature": feat_names,
            "Importance": clf.feature_importances_,
        }).sort_values("Importance", ascending=False).head(20)

        fig_fi = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            template="plotly_dark", color="Importance",
            color_continuous_scale=["#4f8ef7", "#f97316"],
        )
        fig_fi.update_layout(
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            coloraxis_showscale=False, margin=dict(t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

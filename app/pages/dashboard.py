"""
dashboard.py — Executive Dashboard page for PulseRetain.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_service import load_enriched_data

RISK_COLORS = {
    "LOW": "#34d399", "MODERATE": "#fbbf24",
    "HIGH": "#f97316", "CRITICAL": "#f87171",
}


def render():
    st.title("Executive Dashboard")
    st.caption("Organisation-wide attrition risk overview")

    df = load_enriched_data()

    # ── KPIs ──────────────────────────────────────────────────────────────
    total      = len(df)
    high_risk  = int((df["risk_level"].isin(["HIGH", "CRITICAL"])).sum())
    critical   = int((df["risk_level"] == "CRITICAL").sum())
    avg_score  = round(df["risk_score"].mean(), 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-label">Total Employees</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:#f97316">{high_risk}</div>
            <div class="kpi-label">High / Critical Risk</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:#f87171">{critical}</div>
            <div class="kpi-label">Critical Risk</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_score}</div>
            <div class="kpi-label">Avg Risk Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Row 2: Risk distribution + Attrition rate by dept ─────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Level Distribution")
        counts = df["risk_level"].value_counts().reindex(
            ["CRITICAL", "HIGH", "MODERATE", "LOW"], fill_value=0
        ).reset_index()
        counts.columns = ["Risk Level", "Count"]
        fig = px.bar(
            counts, x="Risk Level", y="Count",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#161b27", plot_bgcolor="#161b27",
            showlegend=False, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Avg Risk Score by Department")
        dept = (
            df.groupby("Department")["risk_score"]
            .mean().round(1).reset_index()
            .sort_values("risk_score", ascending=True)
        )
        fig2 = px.bar(
            dept, x="risk_score", y="Department",
            orientation="h", template="plotly_dark",
            color="risk_score",
            color_continuous_scale=["#34d399", "#fbbf24", "#f97316", "#f87171"],
        )
        fig2.update_layout(
            paper_bgcolor="#161b27", plot_bgcolor="#161b27",
            coloraxis_showscale=False, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 3: Job role risk + Top drivers ────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Avg Risk Score by Job Role")
        role = (
            df.groupby("JobRole")["risk_score"]
            .mean().round(1).reset_index()
            .sort_values("risk_score", ascending=True)
        )
        fig3 = px.bar(
            role, x="risk_score", y="JobRole",
            orientation="h", template="plotly_dark",
            color="risk_score",
            color_continuous_scale=["#34d399", "#fbbf24", "#f97316", "#f87171"],
        )
        fig3.update_layout(
            paper_bgcolor="#161b27", plot_bgcolor="#161b27",
            coloraxis_showscale=False, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.subheader("Top Risk Drivers Across Organisation")
        driver_counts = (
            df["top_driver"].value_counts().head(8).reset_index()
        )
        driver_counts.columns = ["Driver", "Count"]
        driver_counts["Pct"] = (driver_counts["Count"] / total * 100).round(1)
        fig4 = px.bar(
            driver_counts, x="Pct", y="Driver",
            orientation="h", template="plotly_dark",
            labels={"Pct": "% of Employees", "Driver": ""},
            color="Pct",
            color_continuous_scale=["#4f8ef7", "#f97316"],
        )
        fig4.update_layout(
            paper_bgcolor="#161b27", plot_bgcolor="#161b27",
            coloraxis_showscale=False, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig4, use_container_width=True)

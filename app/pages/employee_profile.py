"""
employee_profile.py — Employee Profile page for PulseRetain.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

from data_service import load_enriched_data
from src.explainability.shap_explainer import explain_employee
from src.interventions.recommendation_engine import get_all_recommendations

RISK_COLORS = {
    "LOW": "#34d399", "MODERATE": "#fbbf24",
    "HIGH": "#f97316", "CRITICAL": "#f87171",
}


def render():
    st.title("Employee Profile")

    df = load_enriched_data()

    # ── Employee selector ─────────────────────────────────────────────────
    # Pre-select from session state if navigated from employees page
    default_id = st.session_state.get("profile_employee_id", None)
    all_ids = sorted(df["EmployeeNumber"].tolist())
    default_idx = all_ids.index(default_id) if default_id in all_ids else 0

    selected_id = st.selectbox(
        "Select Employee",
        options=all_ids,
        index=default_idx,
        format_func=lambda x: f"Employee {x}",
    )

    row = df[df["EmployeeNumber"] == selected_id].iloc[0]

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Identity ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Employee ID", selected_id)
    c2.metric("Department", row["Department"])
    c3.metric("Job Role", row["JobRole"])
    c4.metric("Job Level", int(row["JobLevel"]))
    c5.metric("Tenure (yrs)", int(row["YearsAtCompany"]))

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Risk score ────────────────────────────────────────────────────────
    level  = row["risk_level"]
    score  = row["risk_score"]
    color  = RISK_COLORS[level]

    col_gauge, col_info = st.columns([1, 2])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"color": color, "size": 48}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                "bar":  {"color": color},
                "bgcolor": "#161b27",
                "steps": [
                    {"range": [0,  30], "color": "#0d3321"},
                    {"range": [30, 60], "color": "#2d2a0a"},
                    {"range": [60, 80], "color": "#2d1a0a"},
                    {"range": [80,100], "color": "#2d0a0a"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "value": score},
            },
            title={"text": "Attrition Risk Score", "font": {"color": "#9aa0b0"}},
        ))
        fig.update_layout(
            paper_bgcolor="#0f1117", font_color="#e8eaf0",
            height=260, margin=dict(t=30, b=0, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"<div style='text-align:center'>"
            f"<span class='badge badge-{level.lower()}' style='font-size:1rem;padding:6px 18px'>"
            f"{level}</span></div>",
            unsafe_allow_html=True,
        )

    with col_info:
        st.subheader("Why is this employee at risk?")

        # SHAP drivers for this employee
        feat_row = df[df["EmployeeNumber"] == selected_id].drop(
            columns=["EmployeeNumber", "risk_score", "risk_level",
                     "attrition_probability", "top_driver",
                     "recommended_action", "intervention_status"],
            errors="ignore",
        )
        with st.spinner("Computing risk drivers..."):
            drivers = explain_employee(feat_row, top_n=5)

        pos = drivers["drivers_positive"]
        prot = drivers["drivers_protective"]

        if pos:
            st.markdown("**Contributing factors (increasing risk)**")
            for label, val in pos:
                bar_w = min(int(abs(val) * 400), 100)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
                    f"<span style='width:200px;color:#c8cdd8;font-size:.85rem'>{label}</span>"
                    f"<div style='background:#f97316;height:10px;width:{bar_w}%;border-radius:4px'></div>"
                    f"<span style='color:#f97316;font-size:.8rem'>{val:+.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if prot:
            st.markdown("**Protective factors (reducing risk)**")
            for label, val in prot:
                bar_w = min(int(abs(val) * 400), 100)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
                    f"<span style='width:200px;color:#c8cdd8;font-size:.85rem'>{label}</span>"
                    f"<div style='background:#34d399;height:10px;width:{bar_w}%;border-radius:4px'></div>"
                    f"<span style='color:#34d399;font-size:.8rem'>{val:+.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Recommended actions ───────────────────────────────────────────────
    st.subheader("Recommended Actions")
    st.caption("These are decision-support suggestions. All actions require human review.")

    actions = get_all_recommendations(row["top_driver"])
    for i, action in enumerate(actions, 1):
        st.markdown(
            f"<div style='background:#161b27;border:1px solid #1e2535;border-radius:8px;"
            f"padding:14px 18px;margin-bottom:10px;color:#c8cdd8;font-size:.9rem'>"
            f"<strong style='color:#4f8ef7'>{i}.</strong> {action}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Key metrics ───────────────────────────────────────────────────────
    st.subheader("Employee Snapshot")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Monthly Income", f"${int(row['MonthlyIncome']):,}")
    m2.metric("Overtime", row["OverTime"])
    m3.metric("Job Satisfaction", int(row["JobSatisfaction"]))
    m4.metric("Work-Life Balance", int(row["WorkLifeBalance"]))
    m5.metric("Yrs Since Promotion", int(row["YearsSinceLastPromotion"]))
    m6.metric("Business Travel", row["BusinessTravel"].replace("Travel_", ""))

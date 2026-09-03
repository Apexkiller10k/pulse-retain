"""
employees.py — At-Risk Employees page for PulseRetain.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from data_service import load_enriched_data

RISK_COLORS = {
    "LOW": "#34d399", "MODERATE": "#fbbf24",
    "HIGH": "#f97316", "CRITICAL": "#f87171",
}


def _badge(level: str) -> str:
    cls = f"badge-{level.lower()}"
    return f'<span class="badge {cls}">{level}</span>'


def render():
    st.title("At-Risk Employees")
    st.caption("Filtered view of employees by attrition risk")

    df = load_enriched_data()

    # ── Filters ───────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    with f1:
        search = st.text_input("Search employee ID", placeholder="e.g. 42")
    with f2:
        dept_opts = ["All"] + sorted(df["Department"].unique().tolist())
        dept = st.selectbox("Department", dept_opts)
    with f3:
        level_opts = ["All", "CRITICAL", "HIGH", "MODERATE", "LOW"]
        risk_filter = st.selectbox("Risk Level", level_opts)
    with f4:
        role_opts = ["All"] + sorted(df["JobRole"].unique().tolist())
        role = st.selectbox("Job Role", role_opts)

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["EmployeeNumber"].astype(str).str.contains(search)]
    if dept != "All":
        filtered = filtered[filtered["Department"] == dept]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"] == risk_filter]
    if role != "All":
        filtered = filtered[filtered["JobRole"] == role]

    filtered = filtered.sort_values("risk_score", ascending=False)

    st.markdown(f"**{len(filtered):,}** employees shown")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Table ─────────────────────────────────────────────────────────────
    display_cols = [
        "EmployeeNumber", "Department", "JobRole",
        "risk_score", "risk_level",
    ]
    table = filtered[display_cols].copy()
    table.columns = [
        "Employee ID", "Department", "Job Role", "Risk Score", "Risk Level",
    ]

    table["Risk Score"] = table["Risk Score"].round(1)
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        height=min(620, max(180, 44 + len(table) * 35)),
        column_config={
            "Employee ID": st.column_config.NumberColumn("Employee ID", width="small"),
            "Risk Score": st.column_config.NumberColumn("Risk Score", format="%.1f", width="small"),
            "Risk Level": st.column_config.TextColumn("Risk Level", width="small"),
        },
    )

    # ── Profile drill-down ────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("View Employee Profile")
    st.caption("Select an employee to open the full risk profile and recommended actions.")
    emp_ids = filtered["EmployeeNumber"].tolist()
    if emp_ids:
        selected_id = st.selectbox(
            "Select Employee ID to view full profile",
            options=emp_ids,
            format_func=lambda x: f"Employee {x}",
        )
        if st.button("View Selected Employee Profile", type="primary"):
            st.session_state["profile_employee_id"] = selected_id
            st.session_state["active_page"] = "Employee Profile"
            st.rerun()

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
        "risk_score", "risk_level", "top_driver", "recommended_action",
    ]
    table = filtered[display_cols].copy()
    table.columns = [
        "Employee ID", "Department", "Job Role",
        "Risk Score", "Risk Level", "Top Driver", "Recommended Action",
    ]

    # Render with HTML badges
    def row_html(row):
        badge = _badge(row["Risk Level"])
        color = RISK_COLORS.get(row["Risk Level"], "#fff")
        score_html = f'<span style="color:{color};font-weight:700">{row["Risk Score"]}</span>'
        return (
            f"<tr>"
            f"<td>{row['Employee ID']}</td>"
            f"<td>{row['Department']}</td>"
            f"<td>{row['Job Role']}</td>"
            f"<td>{score_html}</td>"
            f"<td>{badge}</td>"
            f"<td>{row['Top Driver']}</td>"
            f"<td>{row['Recommended Action']}</td>"
            f"</tr>"
        )

    rows_html = "\n".join(table.apply(row_html, axis=1))
    html = f"""
    <style>
    table.risk-table {{width:100%;border-collapse:collapse;font-size:.85rem;}}
    table.risk-table th {{background:#1e2535;color:#9aa0b0;padding:10px 12px;
        text-align:left;font-weight:500;text-transform:uppercase;font-size:.75rem;letter-spacing:.05em;}}
    table.risk-table td {{padding:10px 12px;border-bottom:1px solid #1e2535;color:#c8cdd8;}}
    table.risk-table tr:hover td {{background:#1a2030;}}
    </style>
    <table class="risk-table">
    <thead><tr>
        <th>Employee ID</th><th>Department</th><th>Job Role</th>
        <th>Risk Score</th><th>Risk Level</th><th>Top Driver</th><th>Recommended Action</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)

    # ── Profile drill-down ────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("View Employee Profile")
    emp_ids = filtered["EmployeeNumber"].tolist()
    if emp_ids:
        selected_id = st.selectbox(
            "Select Employee ID to view full profile",
            options=emp_ids,
            format_func=lambda x: f"Employee {x}",
        )
        if st.button("Open Profile"):
            st.session_state["profile_employee_id"] = selected_id
            st.session_state["active_page"] = "Employee Profile"
            st.rerun()

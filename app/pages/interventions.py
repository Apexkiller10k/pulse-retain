"""
interventions.py — Intervention Tracker page for PulseRetain.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import date
import streamlit as st

from data_service import load_enriched_data

RISK_COLORS = {
    "LOW": "#34d399", "MODERATE": "#fbbf24",
    "HIGH": "#f97316", "CRITICAL": "#f87171",
}
STATUS_OPTIONS = ["Not Started", "In Progress", "Completed"]
STATUS_COLORS  = {"Not Started": "#6b7280", "In Progress": "#fbbf24", "Completed": "#34d399"}


def _get_store() -> dict:
    """Return the session-state intervention store, initialising if needed."""
    if "interventions" not in st.session_state:
        st.session_state["interventions"] = {}
    return st.session_state["interventions"]


def render():
    st.title("Intervention Tracker")
    st.caption("Record and track manager actions for at-risk employees")

    df = load_enriched_data()
    store = _get_store()

    # ── Filters ───────────────────────────────────────────────────────────
    f1, f2 = st.columns([1, 1])
    with f1:
        risk_filter = st.selectbox(
            "Show risk level", ["HIGH & CRITICAL", "ALL", "CRITICAL", "HIGH", "MODERATE", "LOW"]
        )
    with f2:
        status_filter = st.selectbox("Show status", ["All"] + STATUS_OPTIONS)

    if risk_filter == "HIGH & CRITICAL":
        view = df[df["risk_level"].isin(["HIGH", "CRITICAL"])].copy()
    elif risk_filter == "ALL":
        view = df.copy()
    else:
        view = df[df["risk_level"] == risk_filter].copy()

    view = view.sort_values("risk_score", ascending=False)

    # Apply status filter from store
    if status_filter != "All":
        view = view[view["EmployeeNumber"].apply(
            lambda eid: store.get(eid, {}).get("status", "Not Started") == status_filter
        )]

    st.markdown(f"**{len(view)}** employees shown")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Intervention cards ────────────────────────────────────────────────
    for _, row in view.iterrows():
        eid   = row["EmployeeNumber"]
        level = row["risk_level"]
        color = RISK_COLORS[level]

        saved   = store.get(eid, {})
        cur_status = saved.get("status", "Not Started")
        cur_notes  = saved.get("notes", "")
        cur_date   = saved.get("date", date.today())

        with st.expander(
            f"Employee {eid}  |  {row['JobRole']}  |  {row['Department']}  "
            f"|  Risk: {row['risk_score']}",
            expanded=(level == "CRITICAL"),
        ):
            col_l, col_r = st.columns([2, 1])

            with col_l:
                st.markdown(
                    f"<span class='badge badge-{level.lower()}'>{level}</span>"
                    f"&nbsp;&nbsp;<span style='color:#667085;font-size:.85rem'>"
                    f"Top driver: <strong style='color:#172235'>{row['top_driver']}</strong></span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='background:#eef4fb;border:1px solid #d9e5f2;border-radius:6px;padding:10px 14px;"
                    f"margin-top:8px;color:#475467;font-size:.85rem'>"
                    f"<strong style='color:#1769aa'>Recommended:</strong> {row['recommended_action']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                notes = st.text_area(
                    "Manager notes",
                    value=cur_notes,
                    key=f"notes_{eid}",
                    height=80,
                    placeholder="Record what was discussed, agreed actions, follow-up date...",
                )

            with col_r:
                status = st.selectbox(
                    "Intervention status",
                    STATUS_OPTIONS,
                    index=STATUS_OPTIONS.index(cur_status),
                    key=f"status_{eid}",
                )
                sc = STATUS_COLORS[status]
                st.markdown(
                    f"<div style='color:{sc};font-weight:600;font-size:.85rem'>{status}</div>",
                    unsafe_allow_html=True,
                )
                intervention_date = st.date_input(
                    "Intervention date", value=cur_date, key=f"date_{eid}"
                )

                if st.button("Save", key=f"save_{eid}"):
                    store[eid] = {
                        "status": status,
                        "notes":  notes,
                        "date":   intervention_date,
                    }
                    st.success("Saved.")

    # ── Summary ───────────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("Intervention Summary")

    total_tracked = len(store)
    completed = sum(1 for v in store.values() if v.get("status") == "Completed")
    in_progress = sum(1 for v in store.values() if v.get("status") == "In Progress")

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Tracked", total_tracked)
    s2.metric("In Progress", in_progress)
    s3.metric("Completed", completed)

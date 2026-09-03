"""
app.py — PulseRetain Streamlit application entry point.

Run with:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Import all pages up-front so Python's module cache prevents
#    re-execution when Streamlit reruns the script. ─────────────────────────
from pages import dashboard, employees, employee_profile, interventions, analytics

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulseRetain",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────
PAGES = {
    "Dashboard":        dashboard,
    "At-Risk Employees": employees,
    "Employee Profile": employee_profile,
    "Interventions":    interventions,
    "Analytics":        analytics,
}

with st.sidebar:
    st.markdown(
        "<div style='padding:16px 0 24px'>"
        "<span style='font-size:1.4rem;font-weight:700;color:#4f8ef7'>Pulse</span>"
        "<span style='font-size:1.4rem;font-weight:700;color:#e8eaf0'>Retain</span>"
        "<div style='font-size:.72rem;color:#6b7280;margin-top:2px;letter-spacing:.08em'>"
        "AI RETENTION INTELLIGENCE</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    page_names = list(PAGES.keys())
    page = st.session_state.get("active_page", "Dashboard")
    if page not in PAGES:
        page = "Dashboard"

    for page_name in page_names:
        if st.button(
            page_name,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if page_name == page else "secondary",
        ):
            st.session_state["active_page"] = page_name
            st.rerun()

    st.markdown("<hr style='border-color:#1e2535;margin:24px 0 16px'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.72rem;color:#6b7280;line-height:1.6'>"
        "PulseRetain is a decision-support tool.<br>"
        "All recommendations require human review.<br><br>"
        "Data: IBM HR Analytics dataset."
        "</div>",
        unsafe_allow_html=True,
    )

# ── Render selected page ──────────────────────────────────────────────────
PAGES[page].render()

"""
recommendation_engine.py — Rule-based intervention recommendations for PulseRetain.

Maps the top SHAP driver labels to concrete, actionable manager interventions.
"""
from __future__ import annotations

# Driver → (short action, full recommendation)
_RULES: list[tuple[list[str], str, str]] = [
    (
        ["Overtime", "Work-Life Balance", "Distance from Home", "Business Travel"],
        "Workload Review",
        "Schedule a workload review with the employee's manager. "
        "Discuss overtime frequency, travel demands, and flexible working options.",
    ),
    (
        ["Job Satisfaction", "Environment Satisfaction", "Satisfaction Index"],
        "Satisfaction Check-in",
        "Conduct a structured 1-on-1 to understand satisfaction drivers. "
        "Explore role fit, team dynamics, and working environment concerns.",
    ),
    (
        ["Career Stagnation", "Years Since Promotion", "Years in Current Role"],
        "Career Progression Discussion",
        "Initiate a career development conversation. "
        "Review promotion eligibility, skill development opportunities, and growth path.",
    ),
    (
        ["Monthly Income", "Salary Hike %", "Stock Option Level", "Job Level"],
        "Compensation Review",
        "Benchmark the employee's compensation against market rates. "
        "Escalate to HR for a formal compensation review if a gap is identified.",
    ),
    (
        ["Years with Manager", "Relationship Satisfaction"],
        "Manager Relationship Check-in",
        "Facilitate a confidential conversation between the employee and HR. "
        "Assess manager relationship quality and escalate if needed.",
    ),
    (
        ["Companies Worked", "Company Change Rate", "Total Working Years"],
        "Retention Conversation",
        "Have an open retention conversation to understand long-term career goals "
        "and what would make the employee want to stay.",
    ),
]

_FALLBACK_ACTION = "Manager Check-in"
_FALLBACK_DETAIL = (
    "Schedule a general well-being check-in with the employee's manager "
    "to surface any concerns not captured by the model."
)


def get_recommendation(top_driver: str) -> tuple[str, str]:
    """Return (short_action, full_recommendation) for a given top driver label.

    Args:
        top_driver: Human-readable driver label from shap_explainer.

    Returns:
        Tuple of (action_title, recommendation_text).
    """
    for keywords, action, detail in _RULES:
        if any(kw.lower() in top_driver.lower() for kw in keywords):
            return action, detail
    return _FALLBACK_ACTION, _FALLBACK_DETAIL


def get_all_recommendations(top_driver: str) -> list[str]:
    """Return a list of 2-3 concrete action bullet points for the profile page."""
    action, detail = get_recommendation(top_driver)
    bullets = [detail]

    # Always append a general check-in as a second action
    if action != _FALLBACK_ACTION:
        bullets.append(
            "Schedule a manager check-in to discuss the employee's current experience."
        )
    bullets.append(
        "Document the conversation and agreed next steps in the intervention tracker."
    )
    return bullets

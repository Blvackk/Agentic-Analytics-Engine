#ui/components/analyst_statistics.py

"""
Statistics Response Renderer.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_statistics_response(
    response: dict[str, Any],
) -> None:
    """
    Render a statistical analysis response.
    """

    plan = response.get(
        "plan",
        {},
    )

    result = response.get(
        "result",
        {},
    )

    insight = response.get(
        "insight",
        {},
    )

    st.subheader("📊 Statistical Analysis")

    # -----------------------------------------
    # Analysis Summary
    # -----------------------------------------

    if plan:

        st.info(
            f"**Analysis:** {plan.get('analysis_type', 'Unknown').replace('_', ' ').title()}"
        )

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if "correlation" in result:

            st.metric(
                "Correlation",
                round(
                    result["correlation"],
                    4,
                ),
            )

        elif "p_value" in result:

            st.metric(
                "P-Value",
                result["p_value"],
            )

    with col2:

        if "observations" in result:

            st.metric(
                "Observations",
                result["observations"],
            )

    # -----------------------------------------
    # Result
    # -----------------------------------------

    if result:

        with st.expander(
            "Detailed Result"
        ):

            st.json(
                result
            )

    # -----------------------------------------
    # Insight
    # -----------------------------------------

    if insight:

        st.success(
            insight.get(
                "summary",
                "",
            )
        )

        st.write(
            "**Interpretation**"
        )

        st.write(
            insight.get(
                "significance",
                "",
            )
        )

        st.write(
            "**Recommendation**"
        )

        st.write(
            insight.get(
                "recommendation",
                "",
            )
        )


__all__ = [
    "render_statistics_response",
]
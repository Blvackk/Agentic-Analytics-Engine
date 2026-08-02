"""
Machine Learning Response Renderer.
"""

from __future__ import annotations

import streamlit as st


def render_ml_response(
    response: dict,
) -> None:
    """
    Render ML responses.
    """

    plan = response.get("plan", {})
    result = response.get("result", {})
    insight = response.get("insight", {})

    st.subheader("🤖 Machine Learning")

    if plan:

        st.info(
            f"**Problem Type:** "
            f"{plan.get('problem_type', 'Unknown').replace('_', ' ').title()}"
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{result.get('accuracy',0):.2%}",
        )

    with col2:

        st.metric(
            "Precision",
            f"{result.get('precision',0):.2%}",
        )

    with col3:

        st.metric(
            "Recall",
            f"{result.get('recall',0):.2%}",
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{result.get('f1_score',0):.2%}",
        )

    st.success(
        f"🏆 Best Model: "
        f"{result.get('best_model','Unknown').replace('_',' ').title()}"
    )

    with st.expander(
        "Training Details"
    ):

        st.json(result)

    if insight:

        st.write("### Summary")

        st.success(
            insight.get(
                "summary",
                "",
            )
        )

        st.write("### Performance")

        st.write(
            insight.get(
                "performance",
                "",
            )
        )

        st.write("### Interpretation")

        st.write(
            insight.get(
                "interpretation",
                "",
            )
        )

        st.write("### Recommendation")

        st.write(
            insight.get(
                "recommendation",
                "",
            )
        )


__all__ = [
    "render_ml_response",
]
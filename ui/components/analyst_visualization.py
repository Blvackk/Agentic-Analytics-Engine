"""
Visualization Response Renderer.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_visualization_response(
    response: dict,
) -> None:
    """
    Render visualization responses.
    """

    plan = response.get("plan", {})
    result = response.get("result", {})
    insight = response.get("insight", {})

    st.subheader("📈 Visualization")

    if plan:

        st.info(
            f"**Visualization:** "
            f"{plan.get('visualization_type', 'Unknown').replace('_', ' ').title()}"
        )

    chart_path = result.get("chart_path")

    if chart_path:

        path = Path(chart_path)

        if path.exists():

            st.image(
                str(path),
                use_container_width=True,
            )

        else:

            st.warning(
                f"Chart not found:\n{chart_path}"
            )

    with st.expander(
        "Visualization Details"
    ):

        st.json(result)

    if insight:

        st.success(
            insight.get(
                "summary",
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
    "render_visualization_response",
]
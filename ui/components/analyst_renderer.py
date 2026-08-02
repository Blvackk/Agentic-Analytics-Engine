# ui/components/analyst_renderer.py

"""
AI Analyst Renderer.

Dispatches router responses to the
appropriate UI renderer.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.components.analyst_ml import (
    render_ml_response,
)
from ui.components.analyst_statistics import (
    render_statistics_response,
)
from ui.components.analyst_visualization import (
    render_visualization_response,
)


def render_response(
    router_result: dict[str, Any],
) -> None:
    """
    Render the response returned by the
    Agentic Analytics Engine.
    """

    if not isinstance(router_result, dict):

        st.error(
            "Invalid router response."
        )

        return

    route = router_result.get(
        "route",
        "unknown",
    )

    response = router_result.get(
        "response",
        {},
    )

    # -----------------------------------------
    # Statistics
    # -----------------------------------------

    if route == "statistics":

        render_statistics_response(
            response
        )

        return

    # -----------------------------------------
    # Visualization
    # -----------------------------------------

    if route == "visualization":

        render_visualization_response(
            response
        )

        return

    # -----------------------------------------
    # Machine Learning
    # -----------------------------------------

    if route == "machine_learning":

        render_ml_response(
            response
        )

        return

    # -----------------------------------------
    # EDA
    # -----------------------------------------

    if route == "eda":

        st.info(
            response.get(
                "message",
                "EDA Agent is under development.",
            )
        )

        return

    # -----------------------------------------
    # Unknown
    # -----------------------------------------

    st.warning(
        "Unable to determine the appropriate agent."
    )

    if response:

        with st.expander(
            "Raw Response"
        ):

            st.json(
                response
            )


__all__ = [
    "render_response",
]
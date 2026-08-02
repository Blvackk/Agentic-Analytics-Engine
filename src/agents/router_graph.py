"""
Router Graph.

Entry point for the Agentic Analytics Engine.

Receives a user question, determines which
specialized agent should answer it, and
delegates execution.
"""

from typing import Any

import pandas as pd

from src.agents.router import detect_route
from src.agents.statistical_graph import (
    run_statistical_analysis,
)
from src.agents.visualization_graph import (
    run_visualization_analysis,
)


def run_router(
    dataframe: pd.DataFrame,
    question: str,
    semantic_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Route a user request to the appropriate
    analytics agent.
    """

    route = detect_route(question)

    # -----------------------------------------
    # Statistical Agent
    # -----------------------------------------

    if route == "statistics":

        result = run_statistical_analysis(
            dataframe=dataframe,
            question=question,
            semantic_analysis=semantic_analysis,
        )

        return {
            "route": route,
            "response": result,
        }

    # -----------------------------------------
    # EDA Agent (Placeholder)
    # -----------------------------------------

    if route == "eda":

        return {
            "route": route,
            "response": {
                "message": (
                    "EDA Agent is under development."
                )
            },
        }

    # -----------------------------------------
    # Visualization Agent
    # -----------------------------------------

    if route == "visualization":

        result = run_visualization_analysis(
            dataframe=dataframe,
            question=question,
            semantic_analysis=semantic_analysis,
        )

        return {
            "route": route,
            "response": result,
        }

    # -----------------------------------------
    # Machine Learning Agent (Placeholder)
    # -----------------------------------------

    if route == "machine_learning":

        return {
            "route": route,
            "response": {
                "message": (
                    "Machine Learning Agent is under development."
                )
            },
        }

    # -----------------------------------------
    # Unknown
    # -----------------------------------------

    return {
        "route": "unknown",
        "response": {
            "message": (
                "Unable to determine which "
                "agent should answer this question."
            )
        },
    }


__all__ = [
    "run_router",
]
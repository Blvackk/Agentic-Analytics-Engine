#ui/components/analyst_router.py

"""
AI Analyst Router.

Acts as the bridge between the Streamlit UI
and the Agentic Analytics Engine backend.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.agents.router_graph import (
    run_router,
)


def execute_question(
    dataframe: pd.DataFrame,
    question: str,
    semantic_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute a natural-language question through
    the Agentic Analytics Engine.
    """

    if dataframe is None:
        raise ValueError(
            "No dataframe has been loaded."
        )

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if semantic_analysis is None:
        semantic_analysis = {}

    result = run_router(
        dataframe=dataframe,
        question=question,
        semantic_analysis=semantic_analysis,
    )

    return result


def get_route(
    response: dict[str, Any],
) -> str:
    """
    Extract the selected route.
    """

    return response.get(
        "route",
        "unknown",
    )


def get_agent_response(
    response: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract the agent response.
    """

    return response.get(
        "response",
        {},
    )


def is_success(
    response: dict[str, Any],
) -> bool:
    """
    Determine whether routing succeeded.
    """

    route = get_route(
        response
    )

    return route != "unknown"


__all__ = [
    "execute_question",
    "get_route",
    "get_agent_response",
    "is_success",
]
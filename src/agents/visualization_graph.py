#visualization_graph.py
"""
Visualization Graph.

Coordinates the complete visualization workflow.

Question
    ↓
Planner
    ↓
Executor
    ↓
Insight
"""

from typing import Any

import pandas as pd

from src.agents.visualization_executor import (
    execute_visualization_plan,
)
from src.agents.visualization_insight import (
    generate_visualization_insight,
)
from src.agents.visualization_planner import (
    plan_visualization,
)


def run_visualization_analysis(
    dataframe: pd.DataFrame,
    question: str,
    semantic_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute the complete visualization pipeline.
    """

    plan = plan_visualization(
        question=question,
        semantic_analysis=semantic_analysis,
    )

    result = execute_visualization_plan(
        dataframe=dataframe,
        plan=plan,
    )

    insight = generate_visualization_insight(
        result
    )

    return {
        "question": question,
        "plan": plan,
        "result": result,
        "insight": insight,
    }


__all__ = [
    "run_visualization_analysis",
]
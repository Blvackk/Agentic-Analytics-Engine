#src/agents/statistical_graph.py

"""
Statistical Graph.

Coordinates the statistical planning,
execution, and insight generation pipeline.
"""

from typing import Any

import pandas as pd

from src.agents.statistical_planner import (
    plan_statistical_analysis,
)

from src.agents.statistical_executor import (
    execute_statistical_plan,
)

from src.agents.statistical_insight import (
    generate_statistical_insight,
)


def run_statistical_analysis(
    dataframe: pd.DataFrame,
    question: str,
    semantic_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute the complete statistical workflow.

    Workflow
    --------
    Question
        ↓
    Statistical Planner
        ↓
    Statistical Executor
        ↓
    Statistical Insight
        ↓
    Final Result
    """

    # -----------------------------------------
    # Build execution plan
    # -----------------------------------------

    plan = plan_statistical_analysis(
        question=question,
        semantic_analysis=semantic_analysis,
    )

    tool = plan.get("tool")

    if tool is None:
        raise ValueError(
            "Planner did not return a statistical tool."
        )

    # -----------------------------------------
    # Execute statistical function
    # -----------------------------------------

    result = execute_statistical_plan(
        dataframe=dataframe,
        plan=plan,
    )

    # -----------------------------------------
    # Generate business insight
    # -----------------------------------------

    insight = generate_statistical_insight(
        result
    )

    # -----------------------------------------
    # Final response
    # -----------------------------------------

    return {
        "question": question,
        "plan": plan,
        "result": result,
        "insight": insight,
    }


__all__ = [
    "run_statistical_analysis",
]
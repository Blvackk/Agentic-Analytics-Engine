# ml_graph.py

"""
Machine Learning Graph.

Coordinates the complete machine learning
pipeline.
"""

from typing import Any

import pandas as pd

from src.agents.ml_executor import (
    execute_ml_plan,
)

from src.agents.ml_insight import (
    generate_ml_insight,
)

from src.agents.ml_planner import (
    plan_machine_learning,
)


def run_machine_learning(
    dataframe: pd.DataFrame,
    question: str,
    semantic_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute the complete ML pipeline.
    """

    # -----------------------------------------
    # Planning
    # -----------------------------------------

    plan = plan_machine_learning(
        question=question,
        semantic_analysis=semantic_analysis,
    )

    # -----------------------------------------
    # Execution
    # -----------------------------------------

    result = execute_ml_plan(
        dataframe=dataframe,
        plan=plan,
    )

    # -----------------------------------------
    # Insight
    # -----------------------------------------

    insight = generate_ml_insight(
        result
    )

    return {
        "question": question,
        "plan": plan,
        "result": result,
        "insight": insight,
    }


__all__ = [
    "run_machine_learning",
]
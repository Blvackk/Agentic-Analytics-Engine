# src/agents/visualization_planner.py

"""
Visualization Planner.

Determines which visualization should be created
based on the user's natural language request.
"""

import re
from typing import Any


VISUALIZATION_INTENTS: set[str] = {
    "histogram",
    "scatter",
    "boxplot",
    "bar_chart",
    "heatmap",
    "unknown",
}


def detect_visualization_intent(
    question: str,
) -> str:
    """
    Detect the visualization intent.
    """

    if not question.strip():
        return "unknown"

    question = question.lower()

    histogram_patterns = [
        r"\bhistogram\b",
        r"\bdistribution\b",
    ]

    scatter_patterns = [
        r"\bscatter\b",
        r"\bscatter plot\b",
        r"\brelationship\b",
    ]

    boxplot_patterns = [
        r"\bbox\b",
        r"\bboxplot\b",
        r"\bbox plot\b",
    ]

    bar_patterns = [
        r"\bbar\b",
        r"\bbar chart\b",
        r"\bfrequency\b",
        r"\bcount\b",
    ]

    heatmap_patterns = [
        r"\bheatmap\b",
        r"\bcorrelation heatmap\b",
    ]

    pattern_map = {
        "histogram": histogram_patterns,
        "scatter": scatter_patterns,
        "boxplot": boxplot_patterns,
        "bar_chart": bar_patterns,
        "heatmap": heatmap_patterns,
    }

    for intent, patterns in pattern_map.items():

        for pattern in patterns:

            if re.search(pattern, question):

                return intent

    return "unknown"


def build_visualization_plan(
    question: str,
    semantic_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the visualization execution plan.
    """

    intent = detect_visualization_intent(
        question
    )

    selected_columns = semantic_analysis.get(
        "selected_columns",
        []
    )

    numeric_columns = semantic_analysis.get(
        "numeric_columns",
        []
    )

    categorical_columns = semantic_analysis.get(
        "categorical_columns",
        []
    )

    plan: dict[str, Any] = {
        "visualization_type": intent,
        "tool": None,
        "column": None,
        "x_column": None,
        "y_column": None,
        "reason": None,
    }

    # -----------------------------------------
    # Histogram
    # -----------------------------------------

    if intent == "histogram":

        plan["tool"] = "create_histogram"

        if selected_columns:
            plan["column"] = selected_columns[0]

        elif numeric_columns:
            plan["column"] = numeric_columns[0]

        plan["reason"] = (
            "Visualize the distribution "
            "of a numerical variable."
        )

        return plan

    # -----------------------------------------
    # Scatter Plot
    # -----------------------------------------

    if intent == "scatter":

        plan["tool"] = "create_scatter_plot"

        if len(selected_columns) >= 2:

            plan["x_column"] = selected_columns[0]
            plan["y_column"] = selected_columns[1]

        elif len(numeric_columns) >= 2:

            plan["x_column"] = numeric_columns[0]
            plan["y_column"] = numeric_columns[1]

        plan["reason"] = (
            "Visualize the relationship "
            "between two numerical variables."
        )

        return plan

    # -----------------------------------------
    # Box Plot
    # -----------------------------------------

    if intent == "boxplot":

        plan["tool"] = "create_boxplot"

        if selected_columns:

            plan["column"] = selected_columns[0]

        elif numeric_columns:

            plan["column"] = numeric_columns[0]

        plan["reason"] = (
            "Detect spread and potential outliers."
        )

        return plan

    # -----------------------------------------
    # Bar Chart
    # -----------------------------------------

    if intent == "bar_chart":

        plan["tool"] = "create_bar_chart"

        if selected_columns:

            plan["column"] = selected_columns[0]

        elif categorical_columns:

            plan["column"] = categorical_columns[0]

        plan["reason"] = (
            "Visualize category frequencies."
        )

        return plan

    # -----------------------------------------
    # Heatmap
    # -----------------------------------------

    if intent == "heatmap":

        plan["tool"] = (
            "create_correlation_heatmap"
        )

        plan["reason"] = (
            "Visualize correlations among "
            "numerical variables."
        )

        return plan

    plan["reason"] = (
        "Unable to determine an appropriate visualization."
    )

    return plan


def plan_visualization(
    question: str,
    semantic_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Public entry point.
    """

    return build_visualization_plan(
        question,
        semantic_analysis,
    )


__all__ = [
    "detect_visualization_intent",
    "build_visualization_plan",
    "plan_visualization",
]

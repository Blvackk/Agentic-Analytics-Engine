# visualization_executor.py

"""
Visualization Executor.

Receives a visualization execution plan
and dispatches it to the appropriate
visualization function.
"""

from typing import Any

import pandas as pd

from src.tools.visualization import (
    create_histogram,
    create_boxplot,
    create_bar_chart,
    create_scatter_plot,
    create_correlation_heatmap,
)


def _require_parameter(
    plan: dict[str, Any],
    key: str,
) -> str:
    """
    Retrieve a required parameter from the plan.
    """

    value = plan.get(key)

    if value is None:
        raise ValueError(
            f"Missing '{key}' in visualization plan."
        )

    return value


def execute_visualization_plan(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a visualization plan.
    """

    tool = plan.get("tool")

    if tool is None:
        raise ValueError(
            "No visualization tool specified."
        )

    # -----------------------------------------
    # Histogram
    # -----------------------------------------

    if tool == "create_histogram":

        column = _require_parameter(
            plan,
            "column",
        )

        chart_path = create_histogram(
            dataframe=dataframe,
            column=column,
        )

        return {
            "method": "histogram",
            "column": column,
            "chart_path": chart_path,
        }

    # -----------------------------------------
    # Scatter Plot
    # -----------------------------------------

    if tool == "create_scatter_plot":

        x_column = _require_parameter(
            plan,
            "x_column",
        )

        y_column = _require_parameter(
            plan,
            "y_column",
        )

        chart_path = create_scatter_plot(
            dataframe=dataframe,
            x_column=x_column,
            y_column=y_column,
        )

        return {
            "method": "scatter",
            "x_column": x_column,
            "y_column": y_column,
            "chart_path": chart_path,
        }

    # -----------------------------------------
    # Box Plot
    # -----------------------------------------

    if tool == "create_boxplot":

        column = _require_parameter(
            plan,
            "column",
        )

        chart_path = create_boxplot(
            dataframe=dataframe,
            column=column,
        )

        return {
            "method": "boxplot",
            "column": column,
            "chart_path": chart_path,
        }

    # -----------------------------------------
    # Bar Chart
    # -----------------------------------------

    if tool == "create_bar_chart":

        column = _require_parameter(
            plan,
            "column",
        )

        chart_path = create_bar_chart(
            dataframe=dataframe,
            column=column,
        )

        return {
            "method": "bar_chart",
            "column": column,
            "chart_path": chart_path,
        }

    # -----------------------------------------
    # Heatmap
    # -----------------------------------------

    if tool == "create_correlation_heatmap":

        chart_path = create_correlation_heatmap(
            dataframe=dataframe,
        )

        return {
            "method": "heatmap",
            "chart_path": chart_path,
        }

    raise ValueError(
        f"Unsupported visualization tool: {tool}"
    )


__all__ = [
    "execute_visualization_plan",
]




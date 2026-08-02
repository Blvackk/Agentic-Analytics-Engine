# visualization_insight.py
"""
Visualization Insight.

Generate human-readable insights for
visualizations created by the Visualization Agent.
"""

from typing import Any


def generate_visualization_insight(
    result: dict[str, Any],
) -> dict[str, str]:
    """
    Generate an explanation for a visualization.
    """

    method = result.get("method")

    if method == "histogram":

        return {
            "summary":
                "The histogram visualizes the distribution of the selected numerical variable.",
            "interpretation":
                "Inspect the chart for skewness, spread, and potential outliers.",
            "recommendation":
                "Use this plot to understand the variable's distribution before statistical analysis.",
        }

    if method == "scatter":

        return {
            "summary":
                "The scatter plot visualizes the relationship between two numerical variables.",
            "interpretation":
                "Look for positive, negative, or non-linear relationships.",
            "recommendation":
                "If a strong trend exists, consider performing a correlation analysis.",
        }

    if method == "boxplot":

        return {
            "summary":
                "The box plot summarizes the distribution and highlights potential outliers.",
            "interpretation":
                "Points beyond the whiskers may represent unusual observations.",
            "recommendation":
                "Investigate extreme values before training predictive models.",
        }

    if method == "bar_chart":

        return {
            "summary":
                "The bar chart displays the frequency of each category.",
            "interpretation":
                "Compare category counts to identify dominant or underrepresented groups.",
            "recommendation":
                "Useful for understanding class balance and categorical distributions.",
        }

    if method == "heatmap":

        return {
            "summary":
                "The heatmap displays pairwise correlations between numerical variables.",
            "interpretation":
                "Higher absolute correlation values indicate stronger linear relationships.",
            "recommendation":
                "Use this chart to identify correlated features before feature selection.",
        }

    return {
        "summary":
            "Visualization generated.",
        "interpretation":
            "No additional interpretation available.",
        "recommendation":
            "Review the chart manually.",
    }


__all__ = [
    "generate_visualization_insight",
]
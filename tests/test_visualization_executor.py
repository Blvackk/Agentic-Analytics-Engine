# test_visualization_executor.py
"""
Tests for the Visualization Executor.
"""

from pathlib import Path

import pandas as pd

from src.agents.visualization_executor import (
    execute_visualization_plan,
)

print("\n========== VISUALIZATION EXECUTOR ==========\n")

# --------------------------------------------------
# Sample Dataset
# --------------------------------------------------

dataframe = pd.DataFrame(
    {
        "age": [22, 25, 27, 30, 33, 35, 40, 45],
        "income": [
            40000,
            45000,
            48000,
            55000,
            62000,
            67000,
            73000,
            80000,
        ],
        "department": [
            "HR",
            "IT",
            "HR",
            "Sales",
            "IT",
            "Sales",
            "IT",
            "HR",
        ],
    }
)

# ==================================================
# Histogram
# ==================================================

print("Histogram")

plan = {
    "tool": "create_histogram",
    "column": "income",
}

result = execute_visualization_plan(
    dataframe,
    plan,
)

print(result)

assert result["method"] == "histogram"
assert Path(result["chart_path"]).exists()

print("Histogram: PASSED\n")

# ==================================================
# Scatter Plot
# ==================================================

print("Scatter Plot")

plan = {
    "tool": "create_scatter_plot",
    "x_column": "age",
    "y_column": "income",
}

result = execute_visualization_plan(
    dataframe,
    plan,
)

print(result)

assert result["method"] == "scatter"
assert Path(result["chart_path"]).exists()

print("Scatter Plot: PASSED\n")

# ==================================================
# Box Plot
# ==================================================

print("Box Plot")

plan = {
    "tool": "create_boxplot",
    "column": "income",
}

result = execute_visualization_plan(
    dataframe,
    plan,
)

print(result)

assert result["method"] == "boxplot"
assert Path(result["chart_path"]).exists()

print("Box Plot: PASSED\n")

# ==================================================
# Bar Chart
# ==================================================

print("Bar Chart")

plan = {
    "tool": "create_bar_chart",
    "column": "department",
}

result = execute_visualization_plan(
    dataframe,
    plan,
)

print(result)

assert result["method"] == "bar_chart"
assert Path(result["chart_path"]).exists()

print("Bar Chart: PASSED\n")

# ==================================================
# Correlation Heatmap
# ==================================================

print("Heatmap")

plan = {
    "tool": "create_correlation_heatmap",
}

result = execute_visualization_plan(
    dataframe,
    plan,
)

print(result)

assert result["method"] == "heatmap"
assert Path(result["chart_path"]).exists()

print("Heatmap: PASSED\n")

print("All Visualization Executor Tests Passed!")
# tests/test_visualization_planner.py

"""
Tests for the Visualization Planner.
"""

from src.agents.visualization_planner import (
    detect_visualization_intent,
    plan_visualization,
)

print("\n========== VISUALIZATION PLANNER ==========\n")

# --------------------------------------------------
# Sample Semantic Analysis
# --------------------------------------------------

semantic_analysis = {
    "selected_columns": [
        "age",
        "income",
    ],
    "numeric_columns": [
        "age",
        "income",
    ],
    "categorical_columns": [
        "department",
    ],
}

# --------------------------------------------------
# Test Cases
# --------------------------------------------------

test_cases = [

    (
        "Show histogram of income",
        "histogram",
        "create_histogram",
    ),

    (
        "Plot distribution of salary",
        "histogram",
        "create_histogram",
    ),

    (
        "Create scatter plot between age and income",
        "scatter",
        "create_scatter_plot",
    ),

    (
        "Show relationship between age and income",
        "scatter",
        "create_scatter_plot",
    ),

    (
        "Draw boxplot of salary",
        "boxplot",
        "create_boxplot",
    ),

    (
        "Create box plot of salary",
        "boxplot",
        "create_boxplot",
    ),

    (
        "Show bar chart of department",
        "bar_chart",
        "create_bar_chart",
    ),

    (
        "Show frequency of department",
        "bar_chart",
        "create_bar_chart",
    ),

    (
        "Generate correlation heatmap",
        "heatmap",
        "create_correlation_heatmap",
    ),

    (
        "Show heatmap",
        "heatmap",
        "create_correlation_heatmap",
    ),

    (
        "Hello how are you",
        "unknown",
        None,
    ),
]

# --------------------------------------------------
# Execute Tests
# --------------------------------------------------

for question, expected_intent, expected_tool in test_cases:

    print(f"Question : {question}")

    intent = detect_visualization_intent(question)

    print(f"Intent   : {intent}")

    assert intent == expected_intent

    plan = plan_visualization(
        question,
        semantic_analysis,
    )

    print(f"Tool     : {plan['tool']}\n")

    assert plan["tool"] == expected_tool

print("Visualization Planner Tests Passed!")
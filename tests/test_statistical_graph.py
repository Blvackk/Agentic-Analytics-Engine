"""
Integration test for the Statistical Graph.

Pipeline

Question
    ↓
Planner
    ↓
Executor
    ↓
Insight Generator
"""

import pandas as pd

from src.agents.statistical_graph import (
    run_statistical_analysis,
)

print("\n========== STATISTICAL GRAPH ==========\n")

# ------------------------------------------------------
# Sample Dataset
# ------------------------------------------------------

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
    }
)

# ------------------------------------------------------
# Semantic Analysis
# ------------------------------------------------------

semantic_analysis = {
    "selected_columns": [
        "age",
        "income",
    ],
    "numeric_columns": [
        "age",
        "income",
    ],
    "categorical_columns": [],
}

# ------------------------------------------------------
# Question
# ------------------------------------------------------

question = (
    "What is the correlation "
    "between age and income?"
)

# ------------------------------------------------------
# Execute Pipeline
# ------------------------------------------------------

response = run_statistical_analysis(
    dataframe=dataframe,
    question=question,
    semantic_analysis=semantic_analysis,
)

print(response)

# ------------------------------------------------------
# Validate Plan
# ------------------------------------------------------

assert response["plan"]["analysis_type"] == "correlation"

assert (
    response["plan"]["tool"]
    == "pearson_correlation"
)

# ------------------------------------------------------
# Validate Statistics
# ------------------------------------------------------

assert (
    response["result"]["method"]
    == "pearson"
)

assert (
    response["result"]["column_x"]
    == "age"
)

assert (
    response["result"]["column_y"]
    == "income"
)

# ------------------------------------------------------
# Validate Insight
# ------------------------------------------------------

assert (
    "summary"
    in response["insight"]
)

assert (
    "significance"
    in response["insight"]
)

assert (
    "recommendation"
    in response["insight"]
)

print("\nStatistical Graph: PASSED")
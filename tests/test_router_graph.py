"""
Integration test for the Router Graph.
"""

import pandas as pd

from src.agents.router_graph import (
    run_router,
)

print("\n========== ROUTER GRAPH ==========\n")

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
    }
)

# --------------------------------------------------
# Semantic Analysis
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
    "categorical_columns": [],
}

# --------------------------------------------------
# Question
# --------------------------------------------------

question = (
    "What is the correlation "
    "between age and income?"
)

# --------------------------------------------------
# Execute
# --------------------------------------------------

result = run_router(
    dataframe=dataframe,
    question=question,
    semantic_analysis=semantic_analysis,
)

print(result)

# --------------------------------------------------
# Validation
# --------------------------------------------------

assert result["route"] == "statistics"

assert (
    result["response"]["plan"]["tool"]
    == "pearson_correlation"
)

assert (
    result["response"]["result"]["method"]
    == "pearson"
)

assert (
    "summary"
    in result["response"]["insight"]
)

print("\nRouter Graph: PASSED")
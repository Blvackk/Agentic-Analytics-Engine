#test_visualization_graph.py

import os

import pandas as pd

from src.agents.visualization_graph import (
    run_visualization_analysis,
)

print("\n========== VISUALIZATION GRAPH ==========\n")

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

semantic_analysis = {
    "selected_columns": [
        "income",
    ],
    "numeric_columns": [
        "age",
        "income",
    ],
    "categorical_columns": [],
}

question = "Show histogram of income"

result = run_visualization_analysis(
    dataframe=dataframe,
    question=question,
    semantic_analysis=semantic_analysis,
)

print(result)

assert result["plan"]["tool"] == "create_histogram"

assert result["result"]["method"] == "histogram"

assert os.path.exists(
    result["result"]["chart_path"]
)

assert "summary" in result["insight"]

print("\nVisualization Graph: PASSED")
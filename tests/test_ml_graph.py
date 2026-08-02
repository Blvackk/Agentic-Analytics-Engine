import pandas as pd

from src.agents.ml_graph import (
    run_machine_learning,
)

print("\n========== ML GRAPH ==========\n")

dataframe = pd.DataFrame(
    {
        "customer_id": range(1, 11),
        "age": [22, 24, 28, 35, 40, 29, 31, 45, 37, 26],
        "salary": [
            45000,
            48000,
            52000,
            60000,
            70000,
            50000,
            56000,
            75000,
            68000,
            49000,
        ],
        "department": [
            "HR",
            "IT",
            "IT",
            "Sales",
            "HR",
            "IT",
            "Sales",
            "HR",
            "IT",
            "Sales",
        ],
        "churn": [
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
        ],
    }
)

question = "Predict customer churn"

result = run_machine_learning(
    dataframe=dataframe,
    question=question,
    semantic_analysis={},
)

print(result)

assert "plan" in result
assert "result" in result
assert "insight" in result

print("\nML Graph: PASSED")
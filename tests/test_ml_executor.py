
#tests/test_ml_executor.py
import pandas as pd

from src.agents.ml_executor import (
    execute_ml_plan,
)

print("\n========== ML EXECUTOR ==========\n")

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

plan = {
    "problem_type": "classification",
}

result = execute_ml_plan(
    dataframe,
    plan,
)

print(result)

assert result["target"] == "churn"

assert result["training_samples"] == 8

assert result["testing_samples"] == 2

assert result["processed_features"] > 0
assert "best_model" in result
assert result["accuracy"] >= 0

print("\nML Executor: PASSED")
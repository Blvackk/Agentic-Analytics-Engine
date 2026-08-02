# test_statistical_executor.py
import pandas as pd

from src.agents.statistical_executor import (
    execute_statistical_plan,
)


print("\n========== STATISTICAL EXECUTOR ==========\n")

# ------------------------------------------
# Sample Data
# ------------------------------------------

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

# ------------------------------------------
# Execution Plan
# ------------------------------------------

plan = {
    "tool": "pearson_correlation",
    "column_x": "age",
    "column_y": "income",
}

# ------------------------------------------
# Execute
# ------------------------------------------

result = execute_statistical_plan(
    dataframe=dataframe,
    plan=plan,
)

print(result)

# ------------------------------------------
# Validation
# ------------------------------------------

assert result["method"] == "pearson"
assert result["column_x"] == "age"
assert result["column_y"] == "income"

print("\nPearson Dispatcher: PASSED")
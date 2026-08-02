#test_router_ml.py


import pandas as pd

from src.agents.router_graph import (
    run_router,
)

print("\n========== ROUTER + ML ==========\n")

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

result = run_router(
    dataframe=dataframe,
    question="Predict customer churn",
    semantic_analysis={},
)

print(result)

assert result["route"] == "machine_learning"

assert "response" in result

assert "result" in result["response"]

print("\nRouter + ML: PASSED")
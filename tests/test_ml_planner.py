from src.agents.ml_planner import (
    detect_ml_problem,
    plan_machine_learning,
)

print("\n========== ML PLANNER ==========\n")

semantic_analysis = {}

test_cases = [

    (
        "Predict customer churn",
        "classification",
    ),

    (
        "Predict house price",
        "regression",
    ),

    (
        "Cluster customers",
        "clustering",
    ),

    (
        "Forecast sales",
        "time_series",
    ),

    (
        "Detect anomalies",
        "anomaly_detection",
    ),

    (
        "Hello",
        "unknown",
    ),
]

for question, expected in test_cases:

    print(f"Question : {question}")

    problem = detect_ml_problem(
        question
    )

    print(f"Problem  : {problem}")

    assert problem == expected

    plan = plan_machine_learning(
        question,
        semantic_analysis,
    )

    print(plan)
    print()

print("ML Planner Tests Passed!")
#test_ml_insight.py

from src.agents.ml_insight import (
    generate_ml_insight,
)

print("\n========== ML INSIGHT ==========\n")

result = {
    "best_model": "random_forest",
    "accuracy": 0.91,
    "precision": 0.89,
    "recall": 0.88,
    "f1_score": 0.88,
}

insight = generate_ml_insight(
    result
)

print(insight)

assert "summary" in insight
assert "recommendation" in insight
assert "performance" in insight

print("\nML Insight: PASSED")
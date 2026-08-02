#test_visualization_insight.py
from src.agents.visualization_insight import (
    generate_visualization_insight,
)

print("\n========== VISUALIZATION INSIGHT ==========\n")

methods = [
    "histogram",
    "scatter",
    "boxplot",
    "bar_chart",
    "heatmap",
]

for method in methods:

    result = {
        "method": method,
    }

    insight = generate_visualization_insight(result)

    print(method.upper())
    print(insight)
    print()

    assert "summary" in insight
    assert "interpretation" in insight
    assert "recommendation" in insight

print("Visualization Insight Tests Passed!")
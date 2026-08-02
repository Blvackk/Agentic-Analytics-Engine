from src.agents.statistical_planner import (
    detect_statistical_intent,
)
print("\n========== STATISTICAL PLANNER ==========\n")

TEST_CASES = [
    # Correlation
    (
        "What is the correlation between age and income?",
        "correlation",
    ),
    (
        "Show the relationship between sales and profit.",
        "correlation",
    ),

    # Hypothesis Testing
    (
        "Compare salary between males and females.",
        "hypothesis_testing",
    ),
    (
        "Is there a significant difference in income?",
        "hypothesis_testing",
    ),
    (
        "Compare average revenue across groups.",
        "hypothesis_testing",
    ),

    # Normality
    (
        "Is salary normally distributed?",
        "normality",
    ),
    (
        "Perform a Shapiro test.",
        "normality",
    ),

    # Outliers
    (
        "Detect salary outliers.",
        "outlier_detection",
    ),
    (
        "Find anomalies in revenue.",
        "outlier_detection",
    ),

    # Confidence Interval
    (
        "Show confidence interval for salary.",
        "confidence_interval",
    ),
    (
        "Calculate margin of error.",
        "confidence_interval",
    ),

    # Distribution
    (
        "Show income distribution.",
        "distribution",
    ),
    (
        "Calculate skewness.",
        "distribution",
    ),
    (
        "Calculate kurtosis.",
        "distribution",
    ),

    # Unknown
    (
        "Hello, how are you?",
        "unknown",
    ),
]

for question, expected in TEST_CASES:

    predicted = detect_statistical_intent(
        question
    )

    print(
        f"Question : {question}"
    )
    print(
        f"Expected : {expected}"
    )
    print(
        f"Predicted: {predicted}\n"
    )

    assert predicted == expected

print(
    "All Statistical Planner Tests Passed!"
)

"""
Tests for the Semantic Analyzer.
"""

import pandas as pd

from src.agents.semantic_analyzer import (
    analyze_semantics,
)

print("\n========== SEMANTIC ANALYZER ==========\n")

# --------------------------------------------------
# Sample Dataset
# --------------------------------------------------

dataframe = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "age": [22, 31, 28, 40],
        "salary": [45000, 52000, 48000, 70000],
        "department": [
            "HR",
            "IT",
            "HR",
            "Sales",
        ],
        "is_active": [
            True,
            False,
            True,
            True,
        ],
        "churn": [
            0,
            1,
            0,
            1,
        ],
    }
)

# --------------------------------------------------
# Execute (Rule-based only)
# --------------------------------------------------

result = analyze_semantics(
    dataframe=dataframe,
    use_llm=False,
)

print(result)

# --------------------------------------------------
# Validate
# --------------------------------------------------

assert "columns" in result
assert "identifier_columns" in result
assert "feature_columns" in result
assert "target_candidates" in result

assert "customer_id" in result["identifier_columns"]
assert "churn" in result["target_candidates"]

assert result["columns"]["age"]["role"] == "numerical_feature"
assert result["columns"]["salary"]["role"] == "numerical_feature"
assert result["columns"]["department"]["role"] == "categorical_feature"
assert result["columns"]["is_active"]["role"] == "boolean_feature"

print("\nSemantic Analyzer: PASSED")
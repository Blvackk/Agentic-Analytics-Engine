#statistical_planner.py
"""
Statistical Planner.

Responsible for detecting the user's statistical
intent and creating an execution plan for the
Statistical Executor.
"""

import re
from typing import Any

STATISTICAL_INTENTS: set[str] = {
    "correlation",
    "hypothesis_testing",
    "normality",
    "outlier_detection",
    "confidence_interval",
    "distribution",
    "unknown",
}


def detect_statistical_intent(
    question: str,
) -> str:
    """
    Detect the statistical intent from
    a natural language question.
    """

    if not question.strip():
        return "unknown"

    question = question.lower()

    correlation_patterns = [
        r"\bcorrelation\b",
        r"\bcorrelate\b",
        r"\brelationship\b",
        r"\bassociation\b",
    ]

    hypothesis_patterns = [
        r"\bcompare\b",
        r"\bdifference\b",
        r"\bdifferent\b",
        r"\bsignificant\b",
        r"\bmean\b",
    ]

    normality_patterns = [
        r"\bnormal\b",
        r"\bgaussian\b",
        r"\bshapiro\b",
    ]

    outlier_patterns = [
        r"\boutlier\b",
        r"\banomaly\b",
        r"\bextreme\b",
    ]

    confidence_patterns = [
        r"\bconfidence interval\b",
        r"\bmargin of error\b",
    ]

    distribution_patterns = [
        r"\bdistribution\b",
        r"\bskew\b",
        r"\bskewness\b",
        r"\bkurtosis\b",
    ]

    pattern_map = {
        "correlation": correlation_patterns,
        "hypothesis_testing": hypothesis_patterns,
        "normality": normality_patterns,
        "outlier_detection": outlier_patterns,
        "confidence_interval": confidence_patterns,
        "distribution": distribution_patterns,
    }

    for intent, patterns in pattern_map.items():

        for pattern in patterns:

            if re.search(pattern, question):

                return intent

    return "unknown"


def build_statistical_plan(
    question: str,
    semantic_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a statistical execution plan.
    """

    intent = detect_statistical_intent(
        question
    )

    selected_columns = semantic_analysis.get(
        "selected_columns",
        []
    )

    numeric_columns = semantic_analysis.get(
        "numeric_columns",
        []
    )

    categorical_columns = semantic_analysis.get(
        "categorical_columns",
        []
    )

    plan: dict[str, Any] = {
        "analysis_type": intent,
        "tool": None,
        "column_x": None,
        "column_y": None,
        "value_column": None,
        "group_column": None,
        "confidence": 0.95,
        "reason": None,
        "llm_used": False,
        "llm_error": None,
    }

    # =====================================================
    # Correlation
    # =====================================================

    if intent == "correlation":

        plan["tool"] = "pearson_correlation"

        if len(selected_columns) >= 2:

            plan["column_x"] = selected_columns[0]
            plan["column_y"] = selected_columns[1]

        elif len(numeric_columns) >= 2:

            plan["column_x"] = numeric_columns[0]
            plan["column_y"] = numeric_columns[1]

        plan["reason"] = (
            "Analyze the relationship between "
            "two numerical variables."
        )

        return plan

    # =====================================================
    # Hypothesis Testing
    # =====================================================

    if intent == "hypothesis_testing":

        plan["tool"] = "independent_t_test"

        if numeric_columns:

            plan["value_column"] = numeric_columns[0]

        if categorical_columns:

            plan["group_column"] = categorical_columns[0]

        plan["reason"] = (
            "Compare the means of two groups."
        )

        return plan

    # =====================================================
    # Distribution
    # =====================================================

    if intent == "distribution":

        plan["tool"] = "calculate_skewness"

        if numeric_columns:

            plan["column"] = numeric_columns[0]

        plan["reason"] = (
            "Measure the distribution shape."
        )

        return plan

    # =====================================================
    # Outliers
    # =====================================================

    if intent == "outlier_detection":

        plan["tool"] = "detect_iqr_outliers"

        if numeric_columns:

            plan["column"] = numeric_columns[0]

        plan["reason"] = (
            "Detect outliers using IQR."
        )

        return plan

    # =====================================================
    # Confidence Interval
    # =====================================================

    if intent == "confidence_interval":

        plan["tool"] = "confidence_interval_mean"

        if numeric_columns:

            plan["column"] = numeric_columns[0]

        plan["reason"] = (
            "Estimate confidence interval for the mean."
        )

        return plan

    # =====================================================
    # Normality
    # =====================================================

    if intent == "normality":

        plan["tool"] = "normality_test"

        if numeric_columns:

            plan["column"] = numeric_columns[0]

        plan["reason"] = (
            "Assess whether data follows "
            "a normal distribution."
        )

        return plan

    # =====================================================
    # Unknown
    # =====================================================

    plan["reason"] = (
        "Unable to determine the requested "
        "statistical analysis."
    )

    return plan


def plan_statistical_analysis(
    question: str,
    semantic_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Public entry point for statistical planning.
    """

    return build_statistical_plan(
        question,
        semantic_analysis,
    )


__all__ = [
    "detect_statistical_intent",
    "build_statistical_plan",
    "plan_statistical_analysis",
]
"""
Machine Learning Planner.

Determines the type of machine learning
problem and recommends suitable models.
"""

import re
from typing import Any


ML_PROBLEM_TYPES: set[str] = {
    "classification",
    "regression",
    "clustering",
    "time_series",
    "anomaly_detection",
    "unknown",
}


def detect_ml_problem(
    question: str,
) -> str:
    """
    Detect the machine learning problem type.
    """

    if not question.strip():
        return "unknown"

    question = question.lower()

    # -----------------------------------------
    # Classification
    # -----------------------------------------

    classification_patterns = [
        r"\bclassification\b",
        r"\bclassify\b",
        r"\bchurn\b",
        r"\bfraud\b",
        r"\bdefault\b",
        r"\bspam\b",
        r"\bcancel\b",
    ]

    # -----------------------------------------
    # Regression
    # -----------------------------------------

    regression_patterns = [
        r"\bprice\b",
        r"\bhouse price\b",
        r"\bsalary\b",
        r"\bincome\b",
        r"\brevenue\b",
        r"\bcost\b",
        r"\bvalue\b",
    ]

    # -----------------------------------------
    # Clustering
    # -----------------------------------------

    clustering_patterns = [
        r"\bcluster\b",
        r"\bgroup\b",
        r"\bsegment\b",
    ]

    # -----------------------------------------
    # Time Series
    # -----------------------------------------

    time_series_patterns = [
        r"\bforecast\b",
        r"\btime series\b",
        r"\bdemand\b",
        r"\bsales forecast\b",
    ]

    # -----------------------------------------
    # Anomaly Detection
    # -----------------------------------------

    anomaly_patterns = [
        r"\banomaly\b",
        r"\banomalies\b",
        r"\boutlier\b",
        r"\boutliers\b",
        r"\bfraud detection\b",
    ]

    # IMPORTANT:
    # Check regression before classification.
    pattern_map = {
        "regression": regression_patterns,
        "classification": classification_patterns,
        "clustering": clustering_patterns,
        "time_series": time_series_patterns,
        "anomaly_detection": anomaly_patterns,
    }

    for problem_type, patterns in pattern_map.items():

        for pattern in patterns:

            if re.search(pattern, question):
                return problem_type

    return "unknown"


def build_ml_plan(
    question: str,
    semantic_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the machine learning execution plan.
    """

    problem_type = detect_ml_problem(question)

    plan: dict[str, Any] = {
        "problem_type": problem_type,
        "target": None,
        "recommended_models": [],
        "reason": None,
    }

    # -----------------------------------------
    # Classification
    # -----------------------------------------

    if problem_type == "classification":

        plan["recommended_models"] = [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
        ]

        plan["reason"] = (
            "Detected a classification problem."
        )

        return plan

    # -----------------------------------------
    # Regression
    # -----------------------------------------

    if problem_type == "regression":

        plan["recommended_models"] = [
            "linear_regression",
            "random_forest_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
        ]

        plan["reason"] = (
            "Detected a regression problem."
        )

        return plan

    # -----------------------------------------
    # Clustering
    # -----------------------------------------

    if problem_type == "clustering":

        plan["recommended_models"] = [
            "kmeans",
            "dbscan",
        ]

        plan["reason"] = (
            "Detected a clustering problem."
        )

        return plan

    # -----------------------------------------
    # Time Series
    # -----------------------------------------

    if problem_type == "time_series":

        plan["recommended_models"] = [
            "prophet",
            "arima",
        ]

        plan["reason"] = (
            "Detected a forecasting problem."
        )

        return plan

    # -----------------------------------------
    # Anomaly Detection
    # -----------------------------------------

    if problem_type == "anomaly_detection":

        plan["recommended_models"] = [
            "isolation_forest",
            "local_outlier_factor",
        ]

        plan["reason"] = (
            "Detected an anomaly detection problem."
        )

        return plan

    # -----------------------------------------
    # Unknown
    # -----------------------------------------

    plan["reason"] = (
        "Unable to determine the machine learning task."
    )

    return plan


def plan_machine_learning(
    question: str,
    semantic_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Public entry point.
    """

    return build_ml_plan(
        question,
        semantic_analysis,
    )


__all__ = [
    "detect_ml_problem",
    "build_ml_plan",
    "plan_machine_learning",
]
#ml_insight.py

"""
Machine Learning Insight.

Generates human-readable insights from
machine learning results.
"""

from typing import Any


def generate_ml_insight(
    result: dict[str, Any],
) -> dict[str, str]:
    """
    Generate business-friendly insights
    from machine learning results.
    """

    best_model = result["best_model"]

    accuracy = result["accuracy"]
    precision = result["precision"]
    recall = result["recall"]
    f1_score = result["f1_score"]

    # -----------------------------------------
    # Performance Summary
    # -----------------------------------------

    summary = (
        f"{best_model.replace('_', ' ').title()} "
        f"was selected as the best-performing model."
    )

    performance = (
        f"The model achieved an accuracy of "
        f"{accuracy:.2%}."
    )

    # -----------------------------------------
    # Interpretation
    # -----------------------------------------

    if accuracy >= 0.90:

        interpretation = (
            "The model demonstrates excellent predictive performance."
        )

    elif accuracy >= 0.80:

        interpretation = (
            "The model demonstrates good predictive performance."
        )

    elif accuracy >= 0.70:

        interpretation = (
            "The model demonstrates acceptable predictive performance."
        )

    else:

        interpretation = (
            "The predictive performance is relatively low. "
            "Consider improving data quality or feature engineering."
        )

    # -----------------------------------------
    # Recommendation
    # -----------------------------------------

    if (
        precision >= 0.80
        and recall >= 0.80
    ):

        recommendation = (
            "The model appears suitable for deployment "
            "after validation on a larger dataset."
        )

    else:

        recommendation = (
            "Consider collecting more data, engineering "
            "additional features, or tuning model parameters."
        )

    return {
        "summary": summary,
        "performance": performance,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "metrics": (
            f"Precision: {precision:.2%}, "
            f"Recall: {recall:.2%}, "
            f"F1 Score: {f1_score:.2%}"
        ),
    }


__all__ = [
    "generate_ml_insight",
]
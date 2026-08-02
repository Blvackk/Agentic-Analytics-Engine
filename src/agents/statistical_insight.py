#statistical_insight.py

"""
Statistical Insight Generator.

Transforms statistical test results into
human-readable business insights.
"""

from typing import Any


# ==========================================================
# Helper Functions
# ==========================================================

def _interpret_significance(
    p_value: float,
) -> str:
    """
    Interpret statistical significance.
    """

    if p_value < 0.001:
        return (
            "The result is highly statistically significant "
            "(p < 0.001)."
        )

    if p_value < 0.05:
        return (
            "The result is statistically significant "
            "(p < 0.05)."
        )

    return (
        "The result is not statistically significant "
        "(p ≥ 0.05)."
    )


def _interpret_correlation_strength(
    correlation: float,
) -> str:
    """
    Interpret correlation strength.
    """

    absolute = abs(correlation)

    if absolute >= 0.90:
        strength = "very strong"

    elif absolute >= 0.70:
        strength = "strong"

    elif absolute >= 0.50:
        strength = "moderate"

    elif absolute >= 0.30:
        strength = "weak"

    else:
        strength = "very weak"

    direction = (
        "positive"
        if correlation >= 0
        else "negative"
    )

    return f"{strength} {direction}"


# ==========================================================
# Main Insight Generator
# ==========================================================

def generate_statistical_insight(
    result: dict[str, Any],
) -> dict[str, str]:
    """
    Convert statistical results into
    human-readable insights.
    """

    method = result.get("method")

    if method is None:
        raise ValueError(
            "Result dictionary must contain 'method'."
        )

    # ======================================================
    # Pearson Correlation
    # ======================================================

    if method == "pearson":

        strength = _interpret_correlation_strength(
            result["correlation"]
        )

        return {
            "summary": (
                f"There is a {strength} correlation "
                "between the two variables."
            ),
            "significance": _interpret_significance(
                result["p_value"]
            ),
            "recommendation": (
                "Consider using one variable to "
                "help explain or predict the other."
            ),
        }

    # ======================================================
    # Spearman Correlation
    # ======================================================

    if method == "spearman":

        strength = _interpret_correlation_strength(
            result["correlation"]
        )

        return {
            "summary": (
                f"There is a {strength} monotonic "
                "relationship between the variables."
            ),
            "significance": _interpret_significance(
                result["p_value"]
            ),
            "recommendation": (
                "Useful when variables are not "
                "normally distributed."
            ),
        }

    # ======================================================
    # Independent T-Test
    # ======================================================

    if method == "independent_t_test":

        return {
            "summary": (
                "The analysis compares the means "
                "of two independent groups."
            ),
            "significance": _interpret_significance(
                result["p_value"]
            ),
            "recommendation": (
                "If significant, investigate why "
                "the two groups differ."
            ),
        }

    # ======================================================
    # Chi-Square
    # ======================================================

    if method == "chi_square":

        return {
            "summary": (
                "The Chi-Square test evaluates "
                "whether two categorical variables "
                "are associated."
            ),
            "significance": _interpret_significance(
                result["p_value"]
            ),
            "recommendation": (
                "If significant, the variables "
                "are likely associated."
            ),
        }

    # ======================================================
    # ANOVA
    # ======================================================

    if method == "anova":

        return {
            "summary": (
                "ANOVA compares the means of "
                "multiple groups."
            ),
            "significance": _interpret_significance(
                result["p_value"]
            ),
            "recommendation": (
                "If significant, perform post-hoc "
                "analysis to identify which groups "
                "differ."
            ),
        }

    # ======================================================
    # Skewness
    # ======================================================

    if method == "skewness":

        return {
            "summary": (
                f"The distribution is "
                f"{result['interpretation']}."
            ),
            "significance": (
                "Skewness is descriptive and "
                "does not use significance testing."
            ),
            "recommendation": (
                "Consider transformations if "
                "the distribution is highly skewed."
            ),
        }

    # ======================================================
    # Kurtosis
    # ======================================================

    if method == "kurtosis":

        return {
            "summary": (
                f"The distribution is "
                f"{result['interpretation']}."
            ),
            "significance": (
                "Kurtosis is descriptive and "
                "does not use significance testing."
            ),
            "recommendation": (
                "Review the distribution for "
                "heavy or light tails."
            ),
        }

    # ======================================================
    # IQR Outlier Detection
    # ======================================================

    if method == "iqr_outlier_detection":

        count = result["outlier_count"]

        return {
            "summary": (
                f"{count} outlier(s) were detected "
                "using the IQR method."
            ),
            "significance": (
                "Outlier detection does not use "
                "hypothesis testing."
            ),
            "recommendation": (
                "Review outliers before building "
                "predictive models."
            ),
        }

    # ======================================================
    # Confidence Interval
    # ======================================================

    if method == "confidence_interval_mean":

        return {
            "summary": (
                "The confidence interval estimates "
                "the likely range of the population mean."
            ),
            "significance": (
                f"{int(result['confidence_level'] * 100)}% "
                "confidence level."
            ),
            "recommendation": (
                "Narrower intervals indicate "
                "more precise estimates."
            ),
        }

    # ======================================================
    # Unknown
    # ======================================================

    return {
        "summary": (
            "No interpretation is available "
            "for this statistical method."
        ),
        "significance": "",
        "recommendation": "",
    }


__all__ = [
    "generate_statistical_insight",
]
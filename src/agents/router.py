"""
Intent Router.

Determines which analytics agent
should process the user's request.
"""

import re


ROUTES = {
    "eda",
    "statistics",
    "visualization",
    "machine_learning",
    "unknown",
}


def detect_route(
    question: str,
) -> str:
    """
    Detect which analytics agent should
    handle the user's request.
    """

    if not question.strip():
        return "unknown"

    question = question.lower()

    # -----------------------------------------
    # Statistics
    # -----------------------------------------

    statistics_patterns = [
        r"\bcorrelation\b",
        r"\brelationship\b",
        r"\bcompare\b",
        r"\bt-test\b",
        r"\bchi\b",
        r"\banova\b",
        r"\boutlier\b",
        r"\bconfidence\b",
        r"\bskew\b",
        r"\bkurtosis\b",
    ]

    # -----------------------------------------
    # Visualization
    # -----------------------------------------

    visualization_patterns = [
        r"\bplot\b",
        r"\bgraph\b",
        r"\bchart\b",
        r"\bhistogram\b",
        r"\bscatter\b",
        r"\bbar chart\b",
        r"\bboxplot\b",
        r"\bheatmap\b",
    ]

    # -----------------------------------------
    # EDA
    # -----------------------------------------

    eda_patterns = [
        r"\bmissing\b",
        r"\bnull\b",
        r"\bsummary\b",
        r"\bprofile\b",
        r"\bdescribe\b",
        r"\bshape\b",
        r"\bcolumns\b",
        r"\bduplicates\b",
    ]

    # -----------------------------------------
    # Machine Learning
    # -----------------------------------------

    ml_patterns = [
        r"\btrain\b",
        r"\bpredict\b",
        r"\bclassification\b",
        r"\bregression\b",
        r"\bmodel\b",
        r"\bfeature\b",
        r"\bmachine learning\b",
    ]

    for pattern in statistics_patterns:
        if re.search(pattern, question):
            return "statistics"

    for pattern in visualization_patterns:
        if re.search(pattern, question):
            return "visualization"

    for pattern in eda_patterns:
        if re.search(pattern, question):
            return "eda"

    for pattern in ml_patterns:
        if re.search(pattern, question):
            return "machine_learning"

    return "unknown"


__all__ = [
    "detect_route",
]
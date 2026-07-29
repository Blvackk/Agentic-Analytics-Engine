# src/agents/query_analyzer.py

from typing import Any
import re

import pandas as pd

from src.llm.client import generate_json_response


# ==========================================================
# SUPPORTED QUERY INTENTS
# ==========================================================

SUPPORTED_INTENTS = {
    "descriptive_statistics",
    "distribution_analysis",
    "relationship_analysis",
    "comparison_analysis",
    "correlation_analysis",
    "count_analysis",
    "target_analysis",
    "data_quality",
    "dataset_overview",
    "unknown",
}


SUPPORTED_ANALYSIS_TYPES = {
    "summary",
    "distribution",
    "relationship",
    "comparison",
    "correlation",
    "count",
    "target_relationship",
    "quality",
    "overview",
    "unknown",
}


# ==========================================================
# COLUMN MATCHING
# ==========================================================

def _find_mentioned_columns(
    question: str,
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Detect dataset columns explicitly mentioned in the
    user's question.

    Matching uses word boundaries so short column names
    cannot accidentally match inside unrelated words.

    Example:
        column = "age"
        question = "What is the average income?"

    "age" must NOT match the word "average".
    """

    question_lower = question.lower()

    # Normalise separators in the question so columns such
    # as contract_type can match "contract type".
    normalised_question = (
        question_lower
        .replace("_", " ")
        .replace("-", " ")
    )

    matched_columns = []

    for column in dataframe.columns:

        column_name = str(
            column
        )

        normalised_column = (
            column_name.lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        # Escape special regex characters and require
        # boundaries around the complete column name.
        pattern = (
            r"(?<!\w)"
            + re.escape(normalised_column)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            normalised_question,
        ):
            matched_columns.append(
                column_name
            )

    return matched_columns
# ==========================================================
# TARGET MATCHING
# ==========================================================

def _find_mentioned_targets(
    mentioned_columns: list[str],
    semantic_analysis: dict[str, Any],
) -> list[str]:
    """
    Return explicitly mentioned columns that are known
    target candidates according to semantic analysis.
    """

    target_candidates = semantic_analysis.get(
        "target_candidates",
        [],
    )

    target_lookup = {
        str(target).lower():
            str(target)

        for target in target_candidates
    }

    mentioned_targets = []

    for column in mentioned_columns:

        target = target_lookup.get(
            column.lower()
        )

        if target is not None:

            mentioned_targets.append(
                target
            )

    return mentioned_targets


# ==========================================================
# QUERY ANALYZER
# ==========================================================

def analyze_query(
    question: str,
    dataframe: pd.DataFrame,
    semantic_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Analyse a natural-language analytics question.

    The function identifies:

    - analytical intent
    - analysis type
    - requested dataset columns
    - relevant target columns
    - whether computation is required

    The query analyzer DOES NOT answer the question and DOES
    NOT calculate statistics. Its responsibility is only to
    understand what the user is asking.

    Parameters
    ----------
    question : str
        Natural-language question from the user.

    dataframe : pd.DataFrame
        Dataset currently being analysed.

    semantic_analysis : dict | None
        Semantic metadata produced by the Phase 1 semantic
        analyzer.

    Returns
    -------
    dict
        Structured representation of the query.
    """

    # ======================================================
    # VALIDATION
    # ======================================================

    if not isinstance(
        question,
        str,
    ):
        raise TypeError(
            "question must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if dataframe is None:
        raise ValueError(
            "DataFrame cannot be None."
        )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot analyse a query against an empty dataset."
        )

    semantic_analysis = (
        semantic_analysis or {}
    )

    # ======================================================
    # DETERMINISTIC COLUMN DETECTION
    # ======================================================

    mentioned_columns = (
        _find_mentioned_columns(
            question=question,
            dataframe=dataframe,
        )
    )

    mentioned_targets = (
        _find_mentioned_targets(
            mentioned_columns=mentioned_columns,
            semantic_analysis=semantic_analysis,
        )
    )

    identifier_columns = semantic_analysis.get(
        "identifier_columns",
        [],
    )

    target_candidates = semantic_analysis.get(
        "target_candidates",
        [],
    )

    feature_columns = semantic_analysis.get(
        "feature_columns",
        [],
    )

    # ======================================================
    # DATASET CONTEXT FOR LLM
    # ======================================================

    dataset_context = {
        "columns": [
            str(column)
            for column in dataframe.columns
        ],

        "column_types": {
            str(column):
                str(dtype)

            for column, dtype in (
                dataframe.dtypes.items()
            )
        },

        "rows":
            int(
                dataframe.shape[0]
            ),

        "identifier_columns":
            identifier_columns,

        "target_candidates":
            target_candidates,

        "feature_columns":
            feature_columns,

        "explicitly_mentioned_columns":
            mentioned_columns,

        "explicitly_mentioned_targets":
            mentioned_targets,
    }

    # ======================================================
    # QUERY-UNDERSTANDING PROMPT
    # ======================================================

    prompt = f"""
You are the query-understanding component of a data
analytics system.

Your ONLY responsibility is to classify the user's
analytics question.

Do NOT answer the question.
Do NOT calculate statistics.
Do NOT invent dataset columns.
Do NOT invent target columns.
Do NOT infer numerical results.

USER QUESTION:

{question}

DATASET CONTEXT:

{dataset_context}

Classify the question using one of the following intent
values:

- descriptive_statistics
- distribution_analysis
- relationship_analysis
- comparison_analysis
- correlation_analysis
- count_analysis
- target_analysis
- data_quality
- dataset_overview
- unknown

Use one of the following analysis_type values:

- summary
- distribution
- relationship
- comparison
- correlation
- count
- target_relationship
- quality
- overview
- unknown

Guidelines:

1. Questions asking for average, mean, median, minimum,
   maximum, standard deviation, or similar statistics should
   normally use:

   intent = descriptive_statistics
   analysis_type = summary

2. Questions asking how a variable is distributed should
   normally use:

   intent = distribution_analysis
   analysis_type = distribution

3. Questions asking whether variables are related,
   associated, connected, or linked should normally use:

   intent = relationship_analysis
   analysis_type = relationship

4. Questions explicitly asking for correlation should use:

   intent = correlation_analysis
   analysis_type = correlation

5. Questions asking to compare groups, categories, or
   variables should normally use:

   intent = comparison_analysis
   analysis_type = comparison

6. Questions asking "how many", counts, frequencies, or
   totals should normally use:

   intent = count_analysis
   analysis_type = count

7. Questions specifically investigating relationships with
   a known target candidate may use:

   intent = target_analysis
   analysis_type = target_relationship

   However, a general question such as:
   "Which factors are associated with churn?"
   may also be classified as relationship_analysis when
   churn is recorded in target_columns.

8. Questions about missing values, duplicates, cleaning,
   nulls, or data problems should use:

   intent = data_quality
   analysis_type = quality

9. Questions about rows, columns, dataset size, available
   fields, or general dataset structure should use:

   intent = dataset_overview
   analysis_type = overview

10. If the question cannot be reliably classified, use:

    intent = unknown
    analysis_type = unknown

11. requested_columns must contain ONLY columns that exist
    in the supplied dataset.

12. If explicitly_mentioned_columns is not empty, do not add
    unrelated columns to requested_columns.

13. target_columns must contain ONLY target candidates that
    are relevant to the user's question.

14. Identifier columns must not be treated as analytical
    features.

15. An identifier may be selected when it is genuinely
    useful for entity counting. For example, "How many
    customers are there?" may use customer_id if that is the
    dataset's customer identifier.

16. requires_computation should be true when answering the
    question requires a new calculation or analytical tool
    execution.

17. A simple dataset overview that can be answered entirely
    from existing metadata may set requires_computation to
    false.

Return ONLY valid JSON using exactly this structure:

{{
    "intent": "one supported intent",
    "analysis_type": "one supported analysis type",
    "requested_columns": [],
    "target_columns": [],
    "requires_computation": true,
    "reason": "short explanation of why the query was classified this way"
}}
"""

    # ======================================================
    # LLM CLASSIFICATION
    # ======================================================

    try:

        response = generate_json_response(
            prompt=prompt,
            temperature=0.0,
        )

        llm_used = True
        llm_error = None

    except Exception as error:

        response = {}

        llm_used = False
        llm_error = str(
            error
        )

    # ======================================================
    # VALIDATE LLM RESPONSE
    # ======================================================

    if not isinstance(
        response,
        dict,
    ):
        response = {}

    # ======================================================
    # VALIDATE INTENT
    # ======================================================

    intent = response.get(
        "intent",
        "unknown",
    )

    if intent not in SUPPORTED_INTENTS:
        intent = "unknown"

    # ======================================================
    # VALIDATE ANALYSIS TYPE
    # ======================================================

    analysis_type = response.get(
        "analysis_type",
        "unknown",
    )

    if (
        analysis_type
        not in SUPPORTED_ANALYSIS_TYPES
    ):
        analysis_type = "unknown"

    # ======================================================
    # NORMALISE INTENT / ANALYSIS TYPE CONSISTENCY
    # ======================================================

    # Correlation is a specialised relationship analysis.
    # If either field clearly identifies correlation, keep
    # both fields consistent.
    if analysis_type == "correlation":
        intent = "correlation_analysis"

    elif intent == "correlation_analysis":
        analysis_type = "correlation"

    # Keep other strongly defined pairs consistent as well.
    elif intent == "descriptive_statistics":
        analysis_type = "summary"

    elif intent == "distribution_analysis":
        analysis_type = "distribution"

    elif intent == "comparison_analysis":
        analysis_type = "comparison"

    elif intent == "count_analysis":
        analysis_type = "count"

    elif intent == "data_quality":
        analysis_type = "quality"

    elif intent == "dataset_overview":
        analysis_type = "overview"

    elif intent == "target_analysis":
        analysis_type = "target_relationship"

    # ======================================================
    # VALIDATE REQUESTED COLUMNS
    # ======================================================

    valid_columns = {
        str(column)
        for column in dataframe.columns
    }

    llm_requested_columns = response.get(
        "requested_columns",
        [],
    )

    if not isinstance(
        llm_requested_columns,
        list,
    ):
        llm_requested_columns = []

    requested_columns = []

    # ------------------------------------------------------
    # EXPLICIT COLUMN MATCHES
    # ------------------------------------------------------

    # Deterministically detected columns are authoritative.
    for column in mentioned_columns:

        if (
            column in valid_columns
            and column not in requested_columns
        ):
            requested_columns.append(
                column
            )

    # ------------------------------------------------------
    # LLM COLUMN INFERENCE
    # ------------------------------------------------------

    # If the user explicitly mentioned one or more dataset
    # columns, do NOT allow the LLM to add unrelated columns.
    #
    # Example:
    #
    # "What is the average income?"
    #
    # should remain:
    #
    # ["income"]
    #
    # and must not become:
    #
    # ["age", "income"]
    #
    # When no column was explicitly mentioned, LLM inference
    # is allowed, but every inferred column must still exist
    # in the DataFrame.
    if not mentioned_columns:

        for column in llm_requested_columns:

            column = str(
                column
            )

            if (
                column in valid_columns
                and column not in requested_columns
            ):
                requested_columns.append(
                    column
                )

    # ======================================================
    # VALIDATE TARGET COLUMNS
    # ======================================================

    valid_targets = {
        str(target)
        for target in target_candidates
    }

    llm_target_columns = response.get(
        "target_columns",
        [],
    )

    if not isinstance(
        llm_target_columns,
        list,
    ):
        llm_target_columns = []

    target_columns = []

    # Explicitly mentioned known targets take priority.
    for target in mentioned_targets:

        if (
            target in valid_targets
            and target not in target_columns
        ):
            target_columns.append(
                target
            )

    # LLM target inference is allowed only when the target
    # genuinely exists in semantic target candidates.
    for target in llm_target_columns:

        target = str(
            target
        )

        if (
            target in valid_targets
            and target not in target_columns
        ):
            target_columns.append(
                target
            )

    # ======================================================
    # IDENTIFIER SAFETY
    # ======================================================

    identifier_set = {
        str(column)
        for column in identifier_columns
    }

    # Identifiers are useful for entity counting but should
    # not accidentally become analytical features.
    if intent not in {
        "count_analysis",
        "dataset_overview",
    }:

        requested_columns = [
            column
            for column in requested_columns
            if (
                column not in identifier_set
                or column in mentioned_columns
            )
        ]

    # ======================================================
    # COMPUTATION FLAG
    # ======================================================

    requires_computation = response.get(
        "requires_computation",
        True,
    )

    if not isinstance(
        requires_computation,
        bool,
    ):
        requires_computation = True

    # Analytical operations necessarily require computation.
    if intent in {
        "descriptive_statistics",
        "distribution_analysis",
        "relationship_analysis",
        "comparison_analysis",
        "correlation_analysis",
        "count_analysis",
        "target_analysis",
        "data_quality",
    }:
        requires_computation = True

    # ======================================================
    # REASON
    # ======================================================

    reason = response.get(
        "reason",
        "",
    )

    if not isinstance(
        reason,
        str,
    ):
        reason = str(
            reason
        )

    # ======================================================
    # FINAL STRUCTURED QUERY
    # ======================================================

    return {
        "question":
            question,

        "intent":
            intent,

        "analysis_type":
            analysis_type,

        "requested_columns":
            requested_columns,

        "target_columns":
            target_columns,

        "requires_computation":
            requires_computation,

        "reason":
            reason,

        "llm_used":
            llm_used,

        "llm_error":
            llm_error,
    }
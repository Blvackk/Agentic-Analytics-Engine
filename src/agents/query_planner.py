# src/agents/query_planner.py

from typing import Any

import pandas as pd


# ==========================================================
# SUPPORTED QUERY TOOLS
# ==========================================================

SUPPORTED_QUERY_TOOLS = {
    "descriptive_statistics",
    "distribution",
    "correlation",
    "comparison",
    "relationship",
    "target_relationship",
    "count",
    "data_quality",
    "dataset_overview",
}


# ==========================================================
# HELPERS
# ==========================================================

def _is_numeric_column(
    dataframe: pd.DataFrame,
    column: str,
) -> bool:
    """
    Check whether a dataset column is numeric.
    """

    if column not in dataframe.columns:
        return False

    return bool(
        pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    )


def _is_categorical_column(
    dataframe: pd.DataFrame,
    column: str,
) -> bool:
    """
    Check whether a dataset column should be treated as
    categorical for conversational analytics.
    """

    if column not in dataframe.columns:
        return False

    return not _is_numeric_column(
        dataframe=dataframe,
        column=column,
    )


def _get_valid_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    """
    Keep only columns that genuinely exist in the dataset.
    """

    valid_columns = []

    for column in columns:

        column = str(column)

        if (
            column in dataframe.columns
            and column not in valid_columns
        ):
            valid_columns.append(
                column
            )

    return valid_columns


# ==========================================================
# QUERY PLANNER
# ==========================================================

def create_query_plan(
    query_analysis: dict[str, Any],
    dataframe: pd.DataFrame,
    semantic_analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert query-understanding output into an executable
    analytical plan.

    The planner does not calculate statistics and does not
    answer the user's question. It decides which analytical
    operations should be executed.

    Parameters
    ----------
    query_analysis : dict
        Structured output produced by analyze_query().

    dataframe : pd.DataFrame
        Dataset currently being analysed.

    semantic_analysis : dict | None
        Semantic metadata produced during dataset analysis.

    Returns
    -------
    list[dict]
        Ordered analytical tasks.
    """

    # ======================================================
    # VALIDATION
    # ======================================================

    if not isinstance(
        query_analysis,
        dict,
    ):
        raise TypeError(
            "query_analysis must be a dictionary."
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
            "Cannot create a query plan for an empty dataset."
        )

    semantic_analysis = (
        semantic_analysis or {}
    )

    # ======================================================
    # QUERY INFORMATION
    # ======================================================

    intent = query_analysis.get(
        "intent",
        "unknown",
    )

    analysis_type = query_analysis.get(
        "analysis_type",
        "unknown",
    )

    requested_columns = _get_valid_columns(
        dataframe=dataframe,
        columns=query_analysis.get(
            "requested_columns",
            [],
        ),
    )

    target_columns = _get_valid_columns(
        dataframe=dataframe,
        columns=query_analysis.get(
            "target_columns",
            [],
        ),
    )

    identifier_columns = _get_valid_columns(
        dataframe=dataframe,
        columns=semantic_analysis.get(
            "identifier_columns",
            [],
        ),
    )

    feature_columns = _get_valid_columns(
        dataframe=dataframe,
        columns=semantic_analysis.get(
            "feature_columns",
            [],
        ),
    )

    semantic_targets = _get_valid_columns(
        dataframe=dataframe,
        columns=semantic_analysis.get(
            "target_candidates",
            [],
        ),
    )

    plan = []

    # ======================================================
    # DESCRIPTIVE STATISTICS
    # ======================================================

    if intent == "descriptive_statistics":

        for column in requested_columns:

            if column in identifier_columns:
                continue

            plan.append(
                {
                    "tool": "descriptive_statistics",
                    "column": column,
                    "reason":
                        f"Calculate descriptive statistics "
                        f"for '{column}'.",
                }
            )

    # ======================================================
    # DISTRIBUTION ANALYSIS
    # ======================================================

    elif intent == "distribution_analysis":

        for column in requested_columns:

            if column in identifier_columns:
                continue

            distribution_type = (
                "numeric"
                if _is_numeric_column(
                    dataframe,
                    column,
                )
                else "categorical"
            )

            plan.append(
                {
                    "tool": "distribution",
                    "column": column,
                    "distribution_type":
                        distribution_type,
                    "reason":
                        f"Analyse the distribution of "
                        f"'{column}'.",
                }
            )

    # ======================================================
    # CORRELATION ANALYSIS
    # ======================================================

    elif intent == "correlation_analysis":

        numeric_columns = [
            column
            for column in requested_columns
            if (
                _is_numeric_column(
                    dataframe,
                    column,
                )
                and column
                not in identifier_columns
            )
        ]

        if len(numeric_columns) >= 2:

            plan.append(
                {
                    "tool": "correlation",
                    "columns": numeric_columns,
                    "reason":
                        "Calculate correlation between the "
                        "requested numerical variables.",
                }
            )

    # ======================================================
    # COMPARISON ANALYSIS
    # ======================================================

    elif intent == "comparison_analysis":

        analytical_columns = [
            column
            for column in requested_columns
            if column not in identifier_columns
        ]

        if len(analytical_columns) >= 2:

            plan.append(
                {
                    "tool": "comparison",
                    "columns": analytical_columns,
                    "reason":
                        "Compare the requested dataset "
                        "variables.",
                }
            )

    # ======================================================
    # RELATIONSHIP ANALYSIS
    # ======================================================

    elif intent == "relationship_analysis":

        # --------------------------------------------------
        # TARGET-BASED RELATIONSHIP
        # --------------------------------------------------

        effective_targets = (
            target_columns
            if target_columns
            else [
                column
                for column in requested_columns
                if column in semantic_targets
            ]
        )

        if effective_targets:

            for target in effective_targets:

                relevant_features = [
                    column
                    for column in feature_columns
                    if column != target
                ]

                numeric_features = [
                    column
                    for column in relevant_features
                    if _is_numeric_column(
                        dataframe,
                        column,
                    )
                ]

                categorical_features = [
                    column
                    for column in relevant_features
                    if _is_categorical_column(
                        dataframe,
                        column,
                    )
                ]

                plan.append(
                    {
                        "tool": "target_relationship",
                        "target": target,
                        "numeric_features":
                            numeric_features,
                        "categorical_features":
                            categorical_features,
                        "reason":
                            f"Analyse how available features "
                            f"are associated with target "
                            f"'{target}'.",
                    }
                )

        # --------------------------------------------------
        # GENERAL RELATIONSHIP
        # --------------------------------------------------

        else:

            analytical_columns = [
                column
                for column in requested_columns
                if column not in identifier_columns
            ]

            if len(analytical_columns) >= 2:

                plan.append(
                    {
                        "tool": "relationship",
                        "columns":
                            analytical_columns,
                        "reason":
                            "Analyse relationships between "
                            "the requested variables.",
                    }
                )

    # ======================================================
    # TARGET ANALYSIS
    # ======================================================

    elif intent == "target_analysis":

        effective_targets = (
            target_columns
            if target_columns
            else semantic_targets
        )

        for target in effective_targets:

            relevant_features = [
                column
                for column in feature_columns
                if column != target
            ]

            numeric_features = [
                column
                for column in relevant_features
                if _is_numeric_column(
                    dataframe,
                    column,
                )
            ]

            categorical_features = [
                column
                for column in relevant_features
                if _is_categorical_column(
                    dataframe,
                    column,
                )
            ]

            plan.append(
                {
                    "tool": "target_relationship",
                    "target": target,
                    "numeric_features":
                        numeric_features,
                    "categorical_features":
                        categorical_features,
                    "reason":
                        f"Analyse the target '{target}' "
                        f"against available features.",
                }
            )

    # ======================================================
    # COUNT ANALYSIS
    # ======================================================

    elif intent == "count_analysis":

        if requested_columns:

            for column in requested_columns:

                count_type = (
                    "unique"
                    if column in identifier_columns
                    else "non_null"
                )

                plan.append(
                    {
                        "tool": "count",
                        "column": column,
                        "count_type": count_type,
                        "reason":
                            f"Count observations using "
                            f"'{column}'.",
                    }
                )

        else:

            plan.append(
                {
                    "tool": "count",
                    "column": None,
                    "count_type": "rows",
                    "reason":
                        "Count the number of rows in the "
                        "dataset.",
                }
            )

    # ======================================================
    # DATA QUALITY
    # ======================================================

    elif intent == "data_quality":

        plan.append(
            {
                "tool": "data_quality",
                "columns":
                    requested_columns,
                "reason":
                    "Inspect missing values, duplicates, "
                    "and other supported quality metrics.",
            }
        )

    # ======================================================
    # DATASET OVERVIEW
    # ======================================================

    elif intent == "dataset_overview":

        plan.append(
            {
                "tool": "dataset_overview",
                "reason":
                    "Provide structural information about "
                    "the dataset.",
            }
        )

    # ======================================================
    # UNKNOWN QUERY
    # ======================================================

    else:

        return []

    # ======================================================
    # FINAL PLAN VALIDATION
    # ======================================================

    validated_plan = []

    for task in plan:

        tool = task.get(
            "tool"
        )

        if tool not in SUPPORTED_QUERY_TOOLS:
            continue

        validated_plan.append(
            task
        )

    return validated_plan
# src/agents/planner.py

from typing import Any

import pandas as pd


def create_eda_plan(
    dataframe: pd.DataFrame,
    semantic_analysis: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Create an exploratory data analysis plan based on
    dataset structure, data types, and semantic roles.

    If semantic analysis is provided, identifier columns
    are excluded from analytical tasks.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset that will be analysed.

    semantic_analysis : dict | None
        Output produced by analyze_semantics().
        If None, the planner falls back to dtype-based
        behaviour.

    Returns
    -------
    list[dict]
        Ordered list of EDA tasks.
    """

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if dataframe is None:
        raise ValueError(
            "DataFrame cannot be None."
        )

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        return []

    analysis_plan = []

    # --------------------------------------------------
    # SEMANTIC INFORMATION
    # --------------------------------------------------

    identifier_columns = []
    target_columns = []

    if semantic_analysis:

        identifier_columns = semantic_analysis.get(
            "identifier_columns",
            [],
        )

        target_columns = semantic_analysis.get(
            "target_candidates",
            [],
        )

    # Keep only columns that actually exist.
    identifier_columns = [
        column
        for column in identifier_columns
        if column in dataframe.columns
    ]

    target_columns = [
        column
        for column in target_columns
        if column in dataframe.columns
    ]

    # --------------------------------------------------
    # IDENTIFY COLUMN TYPES
    # --------------------------------------------------

    all_numerical_columns = (
        dataframe
        .select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    all_categorical_columns = (
        dataframe
        .select_dtypes(
            include=[
                "object",
                "category",
                "string",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------
    # REMOVE IDENTIFIERS
    # --------------------------------------------------

    numerical_columns = [
        column
        for column in all_numerical_columns
        if column not in identifier_columns
    ]

    categorical_columns = [
        column
        for column in all_categorical_columns
        if column not in identifier_columns
    ]

    # --------------------------------------------------
    # FEATURE COLUMNS
    # --------------------------------------------------

    numerical_features = [
        column
        for column in numerical_columns
        if column not in target_columns
    ]

    categorical_features = [
        column
        for column in categorical_columns
        if column not in target_columns
    ]

    # --------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------

    if numerical_columns:

        analysis_plan.append(
            {
                "tool": "numerical_summary",

                "columns":
                    numerical_columns,

                "reason":
                    "Understand the distribution and "
                    "descriptive statistics of numerical "
                    "analytical columns.",
            }
        )

    # --------------------------------------------------
    # CATEGORICAL SUMMARY
    # --------------------------------------------------

    if categorical_columns:

        analysis_plan.append(
            {
                "tool": "categorical_summary",

                "columns":
                    categorical_columns,

                "reason":
                    "Understand category frequencies and "
                    "dominant values of categorical "
                    "analytical columns.",
            }
        )

    # --------------------------------------------------
    # CORRELATION MATRIX
    # --------------------------------------------------

    if len(numerical_columns) >= 2:

        analysis_plan.append(
            {
                "tool": "correlation_matrix",

                "columns":
                    numerical_columns,

                "reason":
                    "Identify relationships between "
                    "numerical variables while excluding "
                    "identifier columns.",
            }
        )

    # --------------------------------------------------
    # HISTOGRAMS
    # --------------------------------------------------

    for column in numerical_columns:

        analysis_plan.append(
            {
                "tool": "histogram",

                "column":
                    column,

                "reason":
                    f"Inspect the distribution of "
                    f"'{column}'.",
            }
        )

    # --------------------------------------------------
    # BOXPLOTS
    # --------------------------------------------------

    for column in numerical_columns:

        analysis_plan.append(
            {
                "tool": "boxplot",

                "column":
                    column,

                "reason":
                    f"Inspect spread and potential "
                    f"outliers in '{column}'.",
            }
        )

    # --------------------------------------------------
    # BAR CHARTS
    # --------------------------------------------------

    for column in categorical_columns:

        unique_count = (
            dataframe[column]
            .nunique(
                dropna=True
            )
        )

        if unique_count <= 20:

            analysis_plan.append(
                {
                    "tool": "bar_chart",

                    "column":
                        column,

                    "reason":
                        f"Compare category frequencies "
                        f"for '{column}'.",
                }
            )

    # --------------------------------------------------
    # SCATTER PLOTS
    # --------------------------------------------------

    # Use feature columns for feature-vs-feature
    # relationship analysis.
    #
    # We intentionally avoid identifiers and do not
    # generate every possible pair.

    if len(numerical_features) >= 2:

        for index in range(
            len(numerical_features) - 1
        ):

            x_column = numerical_features[
                index
            ]

            y_column = numerical_features[
                index + 1
            ]

            analysis_plan.append(
                {
                    "tool": "scatter_plot",

                    "x_column":
                        x_column,

                    "y_column":
                        y_column,

                    "reason":
                        f"Explore the relationship "
                        f"between '{x_column}' and "
                        f"'{y_column}'.",
                }
            )

    # --------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------

    if len(numerical_columns) >= 2:

        analysis_plan.append(
            {
                "tool": "correlation_heatmap",

                "columns":
                    numerical_columns,

                "reason":
                    "Visualise correlations between "
                    "numerical analytical variables.",
            }
        )

    # --------------------------------------------------
    # TARGET ANALYSIS
    # --------------------------------------------------

    for target_column in target_columns:

        # Categorical target
        if target_column in categorical_columns:

            unique_count = (
                dataframe[target_column]
                .nunique(
                    dropna=True
                )
            )

            if unique_count <= 20:

                analysis_plan.append(
                    {
                        "tool":
                            "target_distribution",

                        "target":
                            target_column,

                        "target_type":
                            "categorical",

                        "reason":
                            f"Inspect the distribution "
                            f"of possible target "
                            f"'{target_column}'.",
                    }
                )

        # Numerical target
        elif target_column in numerical_columns:

            analysis_plan.append(
                {
                    "tool":
                        "target_distribution",

                    "target":
                        target_column,

                    "target_type":
                        "numerical",

                    "reason":
                        f"Inspect the distribution "
                        f"of possible target "
                        f"'{target_column}'.",
                }
            )

    # --------------------------------------------------
    # METADATA TASK
    # --------------------------------------------------

    if identifier_columns:

        analysis_plan.insert(
            0,
            {
                "tool":
                    "semantic_metadata",

                "identifier_columns":
                    identifier_columns,

                "target_columns":
                    target_columns,

                "reason":
                    "Record semantic column roles so "
                    "identifier columns are excluded from "
                    "analytical feature processing.",
            }
        )

    return analysis_plan
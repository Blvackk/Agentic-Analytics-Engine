# src/agents/query_executor.py

from typing import Any

import pandas as pd


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

def _to_python_value(value: Any) -> Any:
    """
    Convert NumPy/Pandas scalar values into ordinary Python
    values so executor results are JSON-friendly.
    """

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    return value


def _numeric_summary(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Calculate descriptive statistics for a numeric Series.
    """

    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": (
            float(clean.std())
            if len(clean) > 1
            else None
        ),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _categorical_distribution(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Calculate frequency and percentage distributions.
    """

    counts = series.value_counts(
        dropna=False
    )

    total = int(len(series))

    frequencies = {}
    percentages = {}

    for value, count in counts.items():

        label = (
            "Missing"
            if pd.isna(value)
            else str(value)
        )

        frequencies[label] = int(count)

        percentages[label] = (
            round(
                (int(count) / total) * 100,
                2,
            )
            if total
            else 0.0
        )

    return {
        "total": total,
        "frequencies": frequencies,
        "percentages": percentages,
    }


def _numeric_distribution(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Produce distribution evidence for a numeric variable.
    """

    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return {
            "count": 0,
            "missing": int(series.isna().sum()),
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "q1": None,
            "q3": None,
            "max": None,
        }

    return {
        "count": int(clean.count()),
        "missing": int(
            len(series) - len(clean)
        ),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": (
            float(clean.std())
            if len(clean) > 1
            else None
        ),
        "min": float(clean.min()),
        "q1": float(clean.quantile(0.25)),
        "q3": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


# ==========================================================
# TOOL EXECUTORS
# ==========================================================

def _execute_descriptive_statistics(
    dataframe: pd.DataFrame,
    task: dict[str, Any],
) -> dict[str, Any]:

    column = task.get("column")

    if column not in dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    series = dataframe[column]

    if pd.api.types.is_numeric_dtype(series):
        statistics = _numeric_summary(
            series
        )
        column_type = "numeric"

    else:
        distribution = _categorical_distribution(
            series
        )

        statistics = {
            "count": int(series.notna().sum()),
            "unique": int(series.nunique()),
            "most_frequent": (
                str(series.mode().iloc[0])
                if not series.mode().empty
                else None
            ),
            "frequencies":
                distribution["frequencies"],
        }

        column_type = "categorical"

    return {
        "tool": "descriptive_statistics",
        "column": column,
        "column_type": column_type,
        "statistics": statistics,
    }


def _execute_distribution(
    dataframe: pd.DataFrame,
    task: dict[str, Any],
) -> dict[str, Any]:

    column = task.get("column")

    if column not in dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    requested_type = task.get(
        "distribution_type"
    )

    if (
        requested_type == "numeric"
        and pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ):
        distribution = _numeric_distribution(
            dataframe[column]
        )

        distribution_type = "numeric"

    else:
        distribution = _categorical_distribution(
            dataframe[column]
        )

        distribution_type = "categorical"

    return {
        "tool": "distribution",
        "column": column,
        "distribution_type":
            distribution_type,
        "distribution": distribution,
    }


def _execute_correlation(
    dataframe: pd.DataFrame,
    task: dict[str, Any],
) -> dict[str, Any]:

    columns = task.get(
        "columns",
        [],
    )

    numeric_columns = [
        column
        for column in columns
        if (
            column in dataframe.columns
            and pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        )
    ]

    if len(numeric_columns) < 2:
        raise ValueError(
            "Correlation requires at least two "
            "numeric columns."
        )

    subset = dataframe[
        numeric_columns
    ]

    matrix = subset.corr(
        method="pearson"
    )

    correlation_matrix = {
        str(row): {
            str(column):
                round(
                    float(value),
                    6,
                )

            for column, value
            in values.items()
        }

        for row, values
        in matrix.to_dict(
            orient="index"
        ).items()
    }

    pairwise = []

    for index, first_column in enumerate(
        numeric_columns
    ):

        for second_column in (
            numeric_columns[index + 1:]
        ):

            valid_pairs = dataframe[
                [
                    first_column,
                    second_column,
                ]
            ].dropna()

            correlation = (
                valid_pairs[
                    first_column
                ].corr(
                    valid_pairs[
                        second_column
                    ]
                )
            )

            pairwise.append(
                {
                    "column_1":
                        first_column,

                    "column_2":
                        second_column,

                    "correlation":
                        (
                            round(
                                float(correlation),
                                6,
                            )
                            if pd.notna(correlation)
                            else None
                        ),

                    "observations_used":
                        int(
                            len(valid_pairs)
                        ),
                }
            )

    return {
        "tool": "correlation",
        "method": "pearson",
        "columns": numeric_columns,
        "correlation_matrix":
            correlation_matrix,
        "pairwise_correlations":
            pairwise,
    }


def _execute_comparison(
    dataframe: pd.DataFrame,
    task: dict[str, Any],
) -> dict[str, Any]:

    columns = [
        column
        for column in task.get(
            "columns",
            [],
        )
        if column in dataframe.columns
    ]

    if len(columns) < 2:
        raise ValueError(
            "Comparison requires at least two columns."
        )

    results = {}

    for column in columns:

        series = dataframe[column]

        if pd.api.types.is_numeric_dtype(
            series
        ):
            results[column] = {
                "type": "numeric",
                "summary":
                    _numeric_summary(
                        series
                    ),
            }

        else:
            results[column] = {
                "type": "categorical",
                "distribution":
                    _categorical_distribution(
                        series
                    ),
            }

    return {
        "tool": "comparison",
        "columns": columns,
        "comparison": results,
    }


def _execute_relationship(
    dataframe: pd.DataFrame,
    task: dict[str, Any],
) -> dict[str, Any]:

    columns = [
        column
        for column in task.get(
            "columns",
            [],
        )
        if column in dataframe.columns
    ]

    if len(columns) < 2:
        raise ValueError(
            "Relationship analysis requires at "
            "least two columns."
        )

    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]

    if len(numeric_columns) == len(columns):

        return _execute_correlation(
            dataframe=dataframe,
            task={
                "columns": columns,
            },
        )

    return {
        "tool": "relationship",
        "columns": columns,
        "column_types": {
            column: (
                "numeric"
                if pd.api.types.is_numeric_dtype(
                    dataframe[column]
                )
                else "categorical"
            )
            for column in columns
        },
        "message":
            "Mixed-type relationship identified. "
            "Use grouped evidence for interpretation.",
        "comparison":
            _execute_comparison(
                dataframe=dataframe,
                task={
                    "columns": columns,
                },
            )["comparison"],
    }


def _execute_target_relationship(
    dataframe: pd.DataFrame,
    task: dict[str, Any],
) -> dict[str, Any]:

    target = task.get(
        "target"
    )

    if target not in dataframe.columns:
        raise ValueError(
            f"Target '{target}' does not exist."
        )

    numeric_features = [
        column
        for column in task.get(
            "numeric_features",
            [],
        )
        if (
            column in dataframe.columns
            and column != target
        )
    ]

    categorical_features = [
        column
        for column in task.get(
            "categorical_features",
            [],
        )
        if (
            column in dataframe.columns
            and column != target
        )
    ]

    target_series = dataframe[
        target
    ]

    target_is_numeric = (
        pd.api.types.is_numeric_dtype(
            target_series
        )
    )

    result = {
        "tool": "target_relationship",
        "target": target,
        "target_type": (
            "numeric"
            if target_is_numeric
            else "categorical"
        ),
        "target_distribution":
            _categorical_distribution(
                target_series
            )
            if not target_is_numeric
            else _numeric_distribution(
                target_series
            ),
        "numeric_feature_relationships": {},
        "categorical_feature_relationships": {},
    }

    # ------------------------------------------------------
    # CATEGORICAL TARGET
    # ------------------------------------------------------

    if not target_is_numeric:

        for feature in numeric_features:

            grouped = (
                dataframe[
                    [target, feature]
                ]
                .dropna()
                .groupby(
                    target,
                    dropna=False,
                )[feature]
                .agg(
                    [
                        "count",
                        "mean",
                        "median",
                        "std",
                        "min",
                        "max",
                    ]
                )
            )

            grouped_results = {}

            for group_name, row in grouped.iterrows():

                label = (
                    "Missing"
                    if pd.isna(group_name)
                    else str(group_name)
                )

                grouped_results[label] = {
                    statistic:
                        _to_python_value(
                            value
                        )

                    for statistic, value
                    in row.to_dict().items()
                }

            result[
                "numeric_feature_relationships"
            ][feature] = {
                "analysis":
                    "grouped_statistics_by_target",

                "groups":
                    grouped_results,
            }

        for feature in categorical_features:

            table = pd.crosstab(
                dataframe[feature],
                dataframe[target],
                dropna=False,
            )

            counts = {
                str(row): {
                    str(column):
                        int(value)

                    for column, value
                    in values.items()
                }

                for row, values
                in table.to_dict(
                    orient="index"
                ).items()
            }

            row_percentages_table = (
                table.div(
                    table.sum(axis=1),
                    axis=0,
                )
                * 100
            )

            row_percentages = {
                str(row): {
                    str(column):
                        round(
                            float(value),
                            2,
                        )

                    for column, value
                    in values.items()
                }

                for row, values
                in row_percentages_table.to_dict(
                    orient="index"
                ).items()
            }

            result[
                "categorical_feature_relationships"
            ][feature] = {
                "analysis":
                    "contingency_table",

                "counts":
                    counts,

                "row_percentages":
                    row_percentages,
            }

    # ------------------------------------------------------
    # NUMERIC TARGET
    # ------------------------------------------------------

    else:

        for feature in numeric_features:

            valid_pairs = dataframe[
                [feature, target]
            ].dropna()

            correlation = (
                valid_pairs[feature].corr(
                    valid_pairs[target]
                )
            )

            result[
                "numeric_feature_relationships"
            ][feature] = {
                "analysis":
                    "pearson_correlation",

                "correlation":
                    (
                        round(
                            float(correlation),
                            6,
                        )
                        if pd.notna(correlation)
                        else None
                    ),

                "observations_used":
                    int(
                        len(valid_pairs)
                    ),
            }

        for feature in categorical_features:

            grouped = (
                dataframe[
                    [feature, target]
                ]
                .dropna()
                .groupby(
                    feature,
                    dropna=False,
                )[target]
                .agg(
                    [
                        "count",
                        "mean",
                        "median",
                        "std",
                        "min",
                        "max",
                    ]
                )
            )

            grouped_results = {}

            for group_name, row in grouped.iterrows():

                label = (
                    "Missing"
                    if pd.isna(group_name)
                    else str(group_name)
                )

                grouped_results[label] = {
                    statistic:
                        _to_python_value(
                            value
                        )

                    for statistic, value
                    in row.to_dict().items()
                }

            result[
                "categorical_feature_relationships"
            ][feature] = {
                "analysis":
                    "target_statistics_by_group",

                "groups":
                    grouped_results,
            }

    return result


def _execute_count(
    dataframe: pd.DataFrame,
    task: dict[str, Any],
) -> dict[str, Any]:

    column = task.get(
        "column"
    )

    count_type = task.get(
        "count_type",
        "rows",
    )

    if count_type == "rows":

        value = int(
            len(dataframe)
        )

    elif count_type == "unique":

        if column not in dataframe.columns:
            raise ValueError(
                f"Column '{column}' does not exist."
            )

        value = int(
            dataframe[column].nunique(
                dropna=True
            )
        )

    elif count_type == "non_null":

        if column not in dataframe.columns:
            raise ValueError(
                f"Column '{column}' does not exist."
            )

        value = int(
            dataframe[column].count()
        )

    else:
        raise ValueError(
            f"Unsupported count type: {count_type}"
        )

    return {
        "tool": "count",
        "column": column,
        "count_type": count_type,
        "value": value,
    }


def _execute_data_quality(
    dataframe: pd.DataFrame,
    task: dict[str, Any],
) -> dict[str, Any]:

    requested_columns = task.get(
        "columns",
        [],
    )

    if requested_columns:

        columns = [
            column
            for column in requested_columns
            if column in dataframe.columns
        ]

    else:
        columns = list(
            dataframe.columns
        )

    missing_values = {
        str(column):
            int(
                dataframe[column]
                .isna()
                .sum()
            )

        for column in columns
    }

    return {
        "tool": "data_quality",
        "rows": int(
            dataframe.shape[0]
        ),
        "columns": int(
            dataframe.shape[1]
        ),
        "missing_values":
            missing_values,
        "total_missing_values":
            int(
                dataframe[
                    columns
                ]
                .isna()
                .sum()
                .sum()
            )
            if columns
            else 0,
        "duplicate_rows":
            int(
                dataframe.duplicated().sum()
            ),
    }


def _execute_dataset_overview(
    dataframe: pd.DataFrame,
    task: dict[str, Any],
) -> dict[str, Any]:

    numeric_columns = [
        str(column)
        for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]

    categorical_columns = [
        str(column)
        for column in dataframe.columns
        if column not in numeric_columns
    ]

    return {
        "tool": "dataset_overview",
        "rows": int(
            dataframe.shape[0]
        ),
        "columns": int(
            dataframe.shape[1]
        ),
        "column_names": [
            str(column)
            for column in dataframe.columns
        ],
        "numeric_columns":
            numeric_columns,
        "categorical_columns":
            categorical_columns,
        "total_missing_values":
            int(
                dataframe.isna()
                .sum()
                .sum()
            ),
        "duplicate_rows":
            int(
                dataframe.duplicated().sum()
            ),
    }


# ==========================================================
# MAIN EXECUTOR
# ==========================================================

def execute_query_plan(
    dataframe: pd.DataFrame,
    query_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Execute a conversational analytics query plan.

    Each task is executed independently. A failed task is
    recorded instead of terminating the entire plan.

    Returns structured evidence that can later be passed to
    the answer generator.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot execute a query against an empty dataset."
        )

    if not isinstance(
        query_plan,
        list,
    ):
        raise TypeError(
            "query_plan must be a list."
        )

    results = []
    errors = []

    executors = {
        "descriptive_statistics":
            _execute_descriptive_statistics,

        "distribution":
            _execute_distribution,

        "correlation":
            _execute_correlation,

        "comparison":
            _execute_comparison,

        "relationship":
            _execute_relationship,

        "target_relationship":
            _execute_target_relationship,

        "count":
            _execute_count,

        "data_quality":
            _execute_data_quality,

        "dataset_overview":
            _execute_dataset_overview,
    }

    for task_number, task in enumerate(
        query_plan,
        start=1,
    ):

        if not isinstance(
            task,
            dict,
        ):
            errors.append(
                {
                    "task_number":
                        task_number,

                    "tool":
                        None,

                    "error":
                        "Task must be a dictionary.",
                }
            )

            continue

        tool = task.get(
            "tool"
        )

        if tool not in SUPPORTED_QUERY_TOOLS:

            errors.append(
                {
                    "task_number":
                        task_number,

                    "tool":
                        tool,

                    "error":
                        f"Unsupported query tool: {tool}",
                }
            )

            continue

        executor = executors[
            tool
        ]

        try:

            evidence = executor(
                dataframe,
                task,
            )

            results.append(
                {
                    "task_number":
                        task_number,

                    "tool":
                        tool,

                    "evidence":
                        evidence,
                }
            )

        except Exception as error:

            errors.append(
                {
                    "task_number":
                        task_number,

                    "tool":
                        tool,

                    "error":
                        str(error),
                }
            )

    execution_report = {
        "total_tasks":
            len(query_plan),

        "successful_tasks":
            len(results),

        "failed_tasks":
            len(errors),
    }

    return {
        "results":
            results,

        "errors":
            errors,

        "execution_report":
            execution_report,
    }
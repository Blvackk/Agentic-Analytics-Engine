# statistics.py

import pandas as pd


def get_numerical_summary(dataframe: pd.DataFrame) -> dict:
    """
    Generate descriptive statistics for numerical columns.

    Returns
    -------
    dict
        Descriptive statistics for each numerical column.
    """

    numerical_columns = dataframe.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if not numerical_columns:
        return {}

    summary = (
        dataframe[numerical_columns]
        .describe()
        .round(2)
        .to_dict()
    )

    return summary


def get_categorical_summary(
    dataframe: pd.DataFrame,
    top_n: int = 10
) -> dict:
    """
    Generate summary information for categorical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to analyse.

    top_n : int
        Maximum number of most frequent categories to return.

    Returns
    -------
    dict
        Summary of categorical columns.
    """

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    summary = {}

    for column in categorical_columns:

        value_counts = (
            dataframe[column]
            .value_counts(dropna=False)
            .head(top_n)
        )

        summary[column] = {
            "unique_values": int(
                dataframe[column].nunique(dropna=True)
            ),

            "most_frequent": (
                None
                if dataframe[column].dropna().empty
                else dataframe[column].mode(dropna=True).iloc[0]
            ),

            "top_values": {
                str(key): int(value)
                for key, value in value_counts.items()
            },
        }

    return summary


def get_correlation_matrix(
    dataframe: pd.DataFrame
) -> dict:
    """
    Calculate Pearson correlations between numerical columns.

    Returns
    -------
    dict
        Correlation matrix represented as a dictionary.
    """

    numerical_dataframe = dataframe.select_dtypes(
        include=["number"]
    )

    if numerical_dataframe.shape[1] < 2:
        return {}

    correlation_matrix = (
        numerical_dataframe
        .corr()
        .round(3)
    )

    return correlation_matrix.to_dict()


def get_group_summary(
    dataframe: pd.DataFrame,
    group_column: str,
    value_column: str,
    aggregation: str = "mean"
) -> dict:
    """
    Compare a numerical variable across groups.

    Supported aggregations:
    - mean
    - median
    - sum
    - count
    """

    if group_column not in dataframe.columns:
        raise ValueError(
            f"Column '{group_column}' does not exist."
        )

    if value_column not in dataframe.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[value_column]
    ):
        raise ValueError(
            f"Column '{value_column}' must be numeric."
        )

    supported_aggregations = {
        "mean",
        "median",
        "sum",
        "count",
    }

    if aggregation not in supported_aggregations:
        raise ValueError(
            "Aggregation must be one of: "
            "mean, median, sum, count."
        )

    grouped = (
        dataframe
        .groupby(group_column, dropna=False)[value_column]
        .agg(aggregation)
        .round(2)
    )

    return {
        str(key): float(value)
        for key, value in grouped.items()
    }
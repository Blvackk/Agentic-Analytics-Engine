# cleaning.py

import pandas as pd


def remove_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove completely duplicated rows from the dataset.

    The original DataFrame is not modified.
    """

    cleaned_dataframe = dataframe.drop_duplicates().copy()

    return cleaned_dataframe


def drop_missing_rows(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Remove rows containing missing values.

    If specific columns are provided, rows are removed only
    when those columns contain missing values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to clean.

    columns : list[str] | None
        Columns to consider when checking for missing values.
        If None, all columns are considered.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the dataset.
    """

    cleaned_dataframe = dataframe.dropna(
        subset=columns
    ).copy()

    return cleaned_dataframe


def fill_numeric_missing(
    dataframe: pd.DataFrame,
    column: str,
    strategy: str = "median"
) -> pd.DataFrame:
    """
    Fill missing values in a numerical column.

    Supported strategies:
    - mean
    - median
    """

    cleaned_dataframe = dataframe.copy()

    if column not in cleaned_dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        cleaned_dataframe[column]
    ):
        raise ValueError(
            f"Column '{column}' is not numeric."
        )

    if strategy == "mean":
        fill_value = cleaned_dataframe[column].mean()

    elif strategy == "median":
        fill_value = cleaned_dataframe[column].median()

    else:
        raise ValueError(
            "Strategy must be either 'mean' or 'median'."
        )

    cleaned_dataframe[column] = (
        cleaned_dataframe[column].fillna(fill_value)
    )

    return cleaned_dataframe


def fill_categorical_missing(
    dataframe: pd.DataFrame,
    column: str,
    strategy: str = "mode",
    fill_value: str = "Unknown"
) -> pd.DataFrame:
    """
    Fill missing values in a categorical column.

    Supported strategies:
    - mode
    - constant
    """

    cleaned_dataframe = dataframe.copy()

    if column not in cleaned_dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if strategy == "mode":

        mode_values = cleaned_dataframe[column].mode()

        if mode_values.empty:
            value = fill_value
        else:
            value = mode_values.iloc[0]

    elif strategy == "constant":
        value = fill_value

    else:
        raise ValueError(
            "Strategy must be either 'mode' or 'constant'."
        )

    cleaned_dataframe[column] = (
        cleaned_dataframe[column].fillna(value)
    )

    return cleaned_dataframe
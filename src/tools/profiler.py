# profiler.py

import pandas as pd


def classify_columns(dataframe: pd.DataFrame) -> dict:
    """
    Classify DataFrame columns based on their data types.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset whose columns need to be classified.

    Returns
    -------
    dict
        Dictionary containing numerical, categorical,
        boolean, and datetime columns.
    """

    numerical_columns = dataframe.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    boolean_columns = dataframe.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    datetime_columns = dataframe.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    return {
        "numerical": numerical_columns,
        "categorical": categorical_columns,
        "boolean": boolean_columns,
        "datetime": datetime_columns,
    }


def profile_dataset(dataframe: pd.DataFrame) -> dict:
    """
    Generate a basic profile of a Pandas DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to analyse.

    Returns
    -------
    dict
        Dictionary containing structural and data-quality
        information about the dataset.
    """

    profile = {
        "rows": dataframe.shape[0],

        "columns": dataframe.shape[1],

        "column_names": dataframe.columns.tolist(),

        "data_types": (
            dataframe.dtypes
            .astype(str)
            .to_dict()
        ),

        "column_types": classify_columns(dataframe),

        "missing_values": (
            dataframe.isnull()
            .sum()
            .to_dict()
        ),

        "missing_percentage": (
            dataframe.isnull()
            .mean()
            .mul(100)
            .round(2)
            .to_dict()
        ),

        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),

        "unique_values": (
            dataframe.nunique(dropna=True)
            .to_dict()
        ),

        "memory_usage_mb": round(
            float(dataframe.memory_usage(deep=True).sum())
            / (1024 ** 2),
            2
        ),
    }

    return profile
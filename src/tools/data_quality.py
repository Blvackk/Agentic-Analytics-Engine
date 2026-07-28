# data_quality.py

import pandas as pd


def get_missing_value_issues(dataframe: pd.DataFrame) -> list[dict]:
    """
    Detect columns containing missing values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to analyse.

    Returns
    -------
    list[dict]
        List of missing-value issues found in the dataset.
    """

    issues = []

    total_rows = len(dataframe)

    if total_rows == 0:
        return issues

    for column in dataframe.columns:

        missing_count = int(
            dataframe[column].isnull().sum()
        )

        if missing_count == 0:
            continue

        missing_percentage = round(
            (missing_count / total_rows) * 100,
            2
        )

        if missing_percentage < 5:
            severity = "low"

        elif missing_percentage < 30:
            severity = "medium"

        else:
            severity = "high"

        issues.append(
            {
                "issue_type": "missing_values",
                "column": column,
                "severity": severity,
                "count": missing_count,
                "percentage": missing_percentage,
            }
        )

    return issues


def get_duplicate_issues(dataframe: pd.DataFrame) -> list[dict]:
    """
    Detect exact duplicate rows in the dataset.

    Returns
    -------
    list[dict]
        Duplicate-row issues found in the dataset.
    """

    issues = []

    duplicate_count = int(
        dataframe.duplicated().sum()
    )

    if duplicate_count == 0:
        return issues

    total_rows = len(dataframe)

    duplicate_percentage = round(
        (duplicate_count / total_rows) * 100,
        2
    )

    issues.append(
        {
            "issue_type": "duplicate_rows",
            "column": None,
            "severity": "medium",
            "count": duplicate_count,
            "percentage": duplicate_percentage,
        }
    )

    return issues


def get_constant_column_issues(
    dataframe: pd.DataFrame
) -> list[dict]:
    """
    Detect columns containing only one unique non-null value.
    """

    issues = []

    for column in dataframe.columns:

        unique_count = int(
            dataframe[column].nunique(dropna=True)
        )

        if unique_count == 1:

            issues.append(
                {
                    "issue_type": "constant_column",
                    "column": column,
                    "severity": "medium",
                    "unique_values": unique_count,
                }
            )

    return issues


def get_identifier_issues(
    dataframe: pd.DataFrame
) -> list[dict]:
    """
    Detect columns that may represent unique identifiers.

    A column is considered a possible identifier when every
    non-null value is unique.
    """

    issues = []

    total_rows = len(dataframe)

    if total_rows == 0:
        return issues

    for column in dataframe.columns:

        non_null_count = int(
            dataframe[column].notna().sum()
        )

        unique_count = int(
            dataframe[column].nunique(dropna=True)
        )

        if (
            non_null_count == total_rows
            and unique_count == total_rows
        ):
            issues.append(
                {
                    "issue_type": "possible_identifier",
                    "column": column,
                    "severity": "info",
                    "unique_values": unique_count,
                }
            )

    return issues


def get_high_cardinality_issues(
    dataframe: pd.DataFrame,
    threshold: float = 0.8
) -> list[dict]:
    """
    Detect categorical columns with a high proportion
    of unique values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to analyse.

    threshold : float
        Minimum unique-value ratio required for a column
        to be considered high-cardinality.
    """

    issues = []

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in categorical_columns:

        non_null_count = int(
            dataframe[column].notna().sum()
        )

        if non_null_count == 0:
            continue

        unique_count = int(
            dataframe[column].nunique(dropna=True)
        )

        unique_ratio = unique_count / non_null_count

        if unique_ratio >= threshold:

            issues.append(
                {
                    "issue_type": "high_cardinality",
                    "column": column,
                    "severity": "medium",
                    "unique_values": unique_count,
                    "unique_ratio": round(
                        unique_ratio,
                        2
                    ),
                }
            )

    return issues


def get_outlier_issues(
    dataframe: pd.DataFrame
) -> list[dict]:
    """
    Detect potential outliers in numerical columns using
    the IQR method.

    Values below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR
    are considered potential outliers.
    """

    issues = []

    numerical_columns = dataframe.select_dtypes(
        include=["number"]
    ).columns

    for column in numerical_columns:

        series = dataframe[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        if outlier_count == 0:
            continue

        outlier_percentage = round(
            (outlier_count / len(series)) * 100,
            2
        )

        issues.append(
            {
                "issue_type": "potential_outliers",
                "column": column,
                "severity": "medium",
                "count": outlier_count,
                "percentage": outlier_percentage,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
        )

    return issues


def analyze_data_quality(
    dataframe: pd.DataFrame
) -> dict:
    """
    Run all data-quality checks and return a structured report.
    """

    issues = []

    issues.extend(
        get_missing_value_issues(dataframe)
    )

    issues.extend(
        get_duplicate_issues(dataframe)
    )

    issues.extend(
        get_constant_column_issues(dataframe)
    )

    issues.extend(
        get_identifier_issues(dataframe)
    )

    issues.extend(
        get_high_cardinality_issues(dataframe)
    )

    issues.extend(
        get_outlier_issues(dataframe)
    )

    return {
        "total_issues": len(issues),
        "issues": issues,
    }
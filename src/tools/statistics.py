# statistics.py

import pandas as pd
import numpy as np

from scipy import stats

from statsmodels.stats.weightstats import DescrStatsW


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

# ==========================================================
 #CORRELATION ANALYSIS
# ==========================================================

# ==========================================================
# CORRELATION ANALYSIS
# ==========================================================
def pearson_correlation(
    dataframe: pd.DataFrame,
    column_x: str,
    column_y: str,
) -> dict:
    """
    Calculate the Pearson correlation coefficient between
    two numerical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    column_x : str
        First numerical column.

    column_y : str
        Second numerical column.

    Returns
    -------
    dict
        Pearson correlation statistics.
    """

    if column_x not in dataframe.columns:
        raise ValueError(
            f"Column '{column_x}' does not exist."
        )

    if column_y not in dataframe.columns:
        raise ValueError(
            f"Column '{column_y}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(dataframe[column_x]):
        raise ValueError(
            f"Column '{column_x}' must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(dataframe[column_y]):
        raise ValueError(
            f"Column '{column_y}' must be numeric."
        )

    clean_dataframe = (
        dataframe[[column_x, column_y]]
        .dropna()
    )

    if len(clean_dataframe) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    correlation, p_value = stats.pearsonr(
        clean_dataframe[column_x],
        clean_dataframe[column_y],
    )

    return {
        "method": "pearson",
        "column_x": column_x,
        "column_y": column_y,
        "correlation": round(float(correlation), 4),
        "p_value": round(float(p_value), 6),
        "observations": len(clean_dataframe),
    }

def spearman_correlation(
    dataframe: pd.DataFrame,
    column_x: str,
    column_y: str,
) -> dict:
    """
    Calculate the Spearman rank correlation coefficient
    between two numerical columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    column_x : str
        First numerical column.

    column_y : str
        Second numerical column.

    Returns
    -------
    dict
        Spearman correlation statistics.
    """

    if column_x not in dataframe.columns:
        raise ValueError(
            f"Column '{column_x}' does not exist."
        )

    if column_y not in dataframe.columns:
        raise ValueError(
            f"Column '{column_y}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[column_x]
    ):
        raise ValueError(
            f"Column '{column_x}' must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[column_y]
    ):
        raise ValueError(
            f"Column '{column_y}' must be numeric."
        )

    clean_dataframe = (
        dataframe[
            [column_x, column_y]
        ]
        .dropna()
    )

    if len(clean_dataframe) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    correlation, p_value = stats.spearmanr(
        clean_dataframe[column_x],
        clean_dataframe[column_y],
    )

    return {
        "method": "spearman",
        "column_x": column_x,
        "column_y": column_y,
        "correlation": round(
            float(correlation),
            4,
        ),
        "p_value": round(
            float(p_value),
            6,
        ),
        "observations": len(
            clean_dataframe
        ),
    }

# ==========================================================
# HYPOTHESIS TESTING
# ==========================================================

def independent_t_test(
    dataframe: pd.DataFrame,
    value_column: str,
    group_column: str,
) -> dict:
    """
    Perform an independent two-sample t-test between
    two groups of a categorical variable.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    value_column : str
        Numeric column.

    group_column : str
        Binary categorical grouping column.

    Returns
    -------
    dict
        Independent t-test results.
    """

    if value_column not in dataframe.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    if group_column not in dataframe.columns:
        raise ValueError(
            f"Column '{group_column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[value_column]
    ):
        raise ValueError(
            f"Column '{value_column}' must be numeric."
        )

    clean_dataframe = (
        dataframe[
            [value_column, group_column]
        ]
        .dropna()
    )

    groups = (
        clean_dataframe[group_column]
        .unique()
        .tolist()
    )

    if len(groups) != 2:
        raise ValueError(
            "Independent t-test requires exactly two groups."
        )

    group_1 = clean_dataframe[
        clean_dataframe[group_column] == groups[0]
    ][value_column]

    group_2 = clean_dataframe[
        clean_dataframe[group_column] == groups[1]
    ][value_column]

    if len(group_1) < 2 or len(group_2) < 2:
        raise ValueError(
            "Each group must contain at least two observations."
        )

    t_statistic, p_value = stats.ttest_ind(
        group_1,
        group_2,
        equal_var=False,
        nan_policy="omit",
    )

    return {
        "method": "independent_t_test",
        "value_column": value_column,
        "group_column": group_column,
        "group_1": str(groups[0]),
        "group_2": str(groups[1]),
        "group_1_size": len(group_1),
        "group_2_size": len(group_2),
        "group_1_mean": round(float(group_1.mean()), 4),
        "group_2_mean": round(float(group_2.mean()), 4),
        "t_statistic": round(float(t_statistic), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
    }

def mann_whitney_test(
    dataframe: pd.DataFrame,
    value_column: str,
    group_column: str,
) -> dict:
    """
    Perform a Mann-Whitney U Test between two groups.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    value_column : str
        Numeric column.

    group_column : str
        Binary categorical grouping column.

    Returns
    -------
    dict
        Mann-Whitney U statistics.
    """

    if value_column not in dataframe.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    if group_column not in dataframe.columns:
        raise ValueError(
            f"Column '{group_column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[value_column]
    ):
        raise ValueError(
            f"Column '{value_column}' must be numeric."
        )

    clean_dataframe = (
        dataframe[
            [value_column, group_column]
        ]
        .dropna()
    )

    groups = (
        clean_dataframe[group_column]
        .unique()
        .tolist()
    )

    if len(groups) != 2:
        raise ValueError(
            "Mann-Whitney U Test requires exactly two groups."
        )

    group_1 = clean_dataframe[
        clean_dataframe[group_column] == groups[0]
    ][value_column]

    group_2 = clean_dataframe[
        clean_dataframe[group_column] == groups[1]
    ][value_column]

    u_statistic, p_value = stats.mannwhitneyu(
        group_1,
        group_2,
        alternative="two-sided",
    )

    return {
        "method": "mann_whitney_u",
        "value_column": value_column,
        "group_column": group_column,
        "group_1": str(groups[0]),
        "group_2": str(groups[1]),
        "group_1_size": len(group_1),
        "group_2_size": len(group_2),
        "u_statistic": round(
            float(u_statistic),
            4,
        ),
        "p_value": round(
            float(p_value),
            6,
        ),
        "significant": bool(
            p_value < 0.05
        ),
    }

def chi_square_test(
    dataframe: pd.DataFrame,
    column_x: str,
    column_y: str,
) -> dict:
    """
    Perform a Chi-Square test of independence between
    two categorical columns.
    """

    if column_x not in dataframe.columns:
        raise ValueError(
            f"Column '{column_x}' does not exist."
        )

    if column_y not in dataframe.columns:
        raise ValueError(
            f"Column '{column_y}' does not exist."
        )

    clean_dataframe = (
        dataframe[[column_x, column_y]]
        .dropna()
    )

    contingency_table = pd.crosstab(
        clean_dataframe[column_x],
        clean_dataframe[column_y],
    )

    if contingency_table.empty:
        raise ValueError(
            "No valid observations available."
        )

    chi2, p_value, dof, expected = stats.chi2_contingency(
        contingency_table
    )

    return {
        "method": "chi_square",
        "column_x": column_x,
        "column_y": column_y,
        "chi_square": round(float(chi2), 4),
        "degrees_of_freedom": int(dof),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "observations": int(len(clean_dataframe)),
        "contingency_table": contingency_table.to_dict(),
        "expected_frequencies": expected.round(4).tolist(),
    }

# ==========================================================
# DISTRIBUTION ANALYSIS
# ==========================================================

def calculate_skewness(
    dataframe: pd.DataFrame,
    column: str,
) -> dict:
    """
    Calculate skewness for a numerical column.
    """

    if column not in dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[column]
    ):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )

    values = dataframe[column].dropna()

    if len(values) < 3:
        raise ValueError(
            "At least three observations are required."
        )

    skewness = stats.skew(
        values,
        bias=False,
    )

    if skewness > 0.5:
        interpretation = "right_skewed"
    elif skewness < -0.5:
        interpretation = "left_skewed"
    else:
        interpretation = "approximately_symmetric"

    return {
        "method": "skewness",
        "column": column,
        "skewness": round(float(skewness), 4),
        "interpretation": interpretation,
        "observations": len(values),
    }

def calculate_kurtosis(
    dataframe: pd.DataFrame,
    column: str,
) -> dict:
    """
    Calculate kurtosis for a numerical column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    column : str
        Numerical column.

    Returns
    -------
    dict
        Kurtosis statistics.
    """

    if column not in dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[column]
    ):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )

    values = dataframe[column].dropna()

    if len(values) < 4:
        raise ValueError(
            "At least four observations are required."
        )

    kurtosis = stats.kurtosis(
        values,
        fisher=True,
        bias=False,
    )

    if kurtosis > 0.5:
        interpretation = "leptokurtic"
    elif kurtosis < -0.5:
        interpretation = "platykurtic"
    else:
        interpretation = "mesokurtic"

    return {
        "method": "kurtosis",
        "column": column,
        "kurtosis": round(
            float(kurtosis),
            4,
        ),
        "interpretation": interpretation,
        "observations": len(values),
    }

# ==========================================================
# OUTLIER DETECTION
# ==========================================================

def detect_iqr_outliers(
    dataframe: pd.DataFrame,
    column: str,
) -> dict:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    column : str
        Numerical column.

    Returns
    -------
    dict
        IQR outlier analysis.
    """

    if column not in dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[column]
    ):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )

    values = dataframe[column].dropna()

    if len(values) < 4:
        raise ValueError(
            "At least four observations are required."
        )

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = values[
        (values < lower_bound)
        | (values > upper_bound)
    ]

    return {
        "method": "iqr_outlier_detection",
        "column": column,
        "q1": round(float(q1), 4),
        "q3": round(float(q3), 4),
        "iqr": round(float(iqr), 4),
        "lower_bound": round(float(lower_bound), 4),
        "upper_bound": round(float(upper_bound), 4),
        "outlier_count": int(len(outliers)),
        "outlier_values": outliers.tolist(),
        "observations": len(values),
    }

# ==========================================================
# CONFIDENCE INTERVALS
# ==========================================================
def confidence_interval_mean(
    dataframe: pd.DataFrame,
    column: str,
    confidence: float = 0.95,
) -> dict:
    """
    Calculate the confidence interval for the mean of a
    numerical column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    column : str
        Numerical column.

    confidence : float
        Confidence level (default = 0.95).

    Returns
    -------
    dict
        Confidence interval statistics.
    """

    if column not in dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[column]
    ):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )

    values = dataframe[column].dropna()

    if len(values) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    descriptive_stats = DescrStatsW(values)

    lower, upper = descriptive_stats.tconfint_mean(
        alpha=1 - confidence
    )

    return {
        "method": "confidence_interval_mean",
        "column": column,
        "confidence_level": confidence,
        "mean": round(float(values.mean()), 4),
        "lower_bound": round(float(lower), 4),
        "upper_bound": round(float(upper), 4),
        "observations": len(values),
    }

# ==========================================================
# ADVANCED HYPOTHESIS TESTING
# ==========================================================
def anova_test(
    dataframe: pd.DataFrame,
    value_column: str,
    group_column: str,
) -> dict:
    """
    Perform a one-way ANOVA test.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    value_column : str
        Numeric column.

    group_column : str
        Categorical grouping column.

    Returns
    -------
    dict
        One-way ANOVA statistics.
    """

    if value_column not in dataframe.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    if group_column not in dataframe.columns:
        raise ValueError(
            f"Column '{group_column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[value_column]
    ):
        raise ValueError(
            f"Column '{value_column}' must be numeric."
        )

    clean_dataframe = dataframe[
        [value_column, group_column]
    ].dropna()

    grouped_data = [
        group[value_column].values
        for _, group in clean_dataframe.groupby(group_column)
    ]

    if len(grouped_data) < 3:
        raise ValueError(
            "ANOVA requires at least three groups."
        )

    f_statistic, p_value = stats.f_oneway(
        *grouped_data
    )

    group_sizes = {
        str(name): len(group)
        for name, group in clean_dataframe.groupby(group_column)
    }

    group_means = {
        str(name): round(
            float(group[value_column].mean()),
            4,
        )
        for name, group in clean_dataframe.groupby(group_column)
    }

    return {
        "method": "anova",
        "value_column": value_column,
        "group_column": group_column,
        "groups": len(grouped_data),
        "group_sizes": group_sizes,
        "group_means": group_means,
        "f_statistic": round(
            float(f_statistic),
            4,
        ),
        "p_value": round(
            float(p_value),
            6,
        ),
        "significant": bool(
            p_value < 0.05
        ),
        "observations": len(
            clean_dataframe
        ),
    }

# ==========================================================
# NORMALITY TESTING
# ==========================================================
def normality_test(
    dataframe: pd.DataFrame,
    column: str,
) -> dict:
    """
    Perform the Shapiro-Wilk normality test.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    column : str
        Numeric column.

    Returns
    -------
    dict
        Shapiro-Wilk test statistics.
    """

    if column not in dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[column]
    ):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )

    values = dataframe[column].dropna()

    if len(values) < 3:
        raise ValueError(
            "Shapiro-Wilk requires at least 3 observations."
        )

    statistic, p_value = stats.shapiro(values)

    return {
        "method": "shapiro_wilk",
        "column": column,
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 6),
        "normal_distribution": bool(
            p_value > 0.05
        ),
        "observations": len(values),
    }

# ==========================================================
# OUTLIER DETECTION
# ==========================================================
def detect_zscore_outliers(
    dataframe: pd.DataFrame,
    column: str,
    threshold: float = 3.0,
) -> dict:
    """
    Detect outliers using Z-score.
    """

    if column not in dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[column]
    ):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )

    values = dataframe[column].dropna()

    if len(values) < 3:
        raise ValueError(
            "At least three observations are required."
        )

    z_scores = np.abs(
        stats.zscore(values)
    )

    outliers = values[
        z_scores > threshold
    ]

    return {
        "method": "zscore_outlier_detection",
        "column": column,
        "threshold": threshold,
        "outlier_count": int(len(outliers)),
        "outlier_values": outliers.tolist(),
        "observations": len(values),
    }









__all__ = [
    "get_numerical_summary",
    "get_categorical_summary",
    "get_correlation_matrix",
    "get_group_summary",
    "pearson_correlation",
    "spearman_correlation",
    "independent_t_test",
    "chi_square_test",
    "calculate_skewness",
    "calculate_kurtosis",
    "detect_iqr_outliers",
    "confidence_interval_mean",
    "anova_test",
    "mann_whitney_test",
    "normality_test",
    "detect_zscore_outliers",
]
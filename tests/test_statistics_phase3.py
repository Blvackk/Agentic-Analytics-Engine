import pandas as pd

from src.tools.statistics import pearson_correlation


def main():

    print("\n========== PHASE 3 ==========\n")

    dataframe = pd.DataFrame(
        {
            "age": [20, 22, 25, 30, 35, 40, 45, 50],
            "income": [30, 35, 40, 55, 60, 70, 80, 95],
        }
    )

    result = pearson_correlation(
        dataframe,
        "age",
        "income",
    )

    print(result)

    assert result["method"] == "pearson"
    assert result["column_x"] == "age"
    assert result["column_y"] == "income"
    assert result["observations"] == 8

    print("\nPearson Correlation: PASSED")


if __name__ == "__main__":
    main()


print("\n========== T-TEST ==========\n")

dataframe = pd.DataFrame(
    {
        "salary": [
            50,
            52,
            48,
            51,
            72,
            74,
            71,
            75,
        ],
        "gender": [
            "Male",
            "Male",
            "Male",
            "Male",
            "Female",
            "Female",
            "Female",
            "Female",
        ],
    }
)

from src.tools.statistics import independent_t_test

result = independent_t_test(
    dataframe,
    value_column="salary",
    group_column="gender",
)

print(result)

assert result["method"] == "independent_t_test"
assert result["group_column"] == "gender"
assert result["value_column"] == "salary"
assert result["group_1_size"] == 4
assert result["group_2_size"] == 4

print("\nIndependent T-Test: PASSED")


print("\n========== CHI-SQUARE TEST ==========\n")

from src.tools.statistics import chi_square_test

dataframe = pd.DataFrame(
    {
        "gender": [
            "Male",
            "Male",
            "Male",
            "Female",
            "Female",
            "Female",
            "Male",
            "Female",
        ],
        "churn": [
            "Yes",
            "No",
            "No",
            "Yes",
            "Yes",
            "No",
            "No",
            "Yes",
        ],
    }
)

result = chi_square_test(
    dataframe,
    column_x="gender",
    column_y="churn",
)

print(result)

assert result["method"] == "chi_square"
assert result["column_x"] == "gender"
assert result["column_y"] == "churn"
assert result["observations"] == 8

print("\nChi-Square Test: PASSED")

print(
    "\n========== SPEARMAN CORRELATION ==========\n"
)

from src.tools.statistics import (
    spearman_correlation,
)

dataframe = pd.DataFrame(
    {
        "study_hours": [
            2,
            3,
            5,
            6,
            8,
            9,
            11,
            12,
        ],
        "exam_score": [
            45,
            50,
            60,
            66,
            75,
            80,
            88,
            92,
        ],
    }
)

result = spearman_correlation(
    dataframe,
    column_x="study_hours",
    column_y="exam_score",
)

print(result)

assert result["method"] == "spearman"
assert result["column_x"] == "study_hours"
assert result["column_y"] == "exam_score"
assert result["observations"] == 8

print(
    "\nSpearman Correlation: PASSED"
)

print(
    "\n========== SKEWNESS ==========\n"
)

from src.tools.statistics import (
    calculate_skewness,
)

dataframe = pd.DataFrame(
    {
        "income": [
            25,
            28,
            31,
            35,
            40,
            42,
            120,
            180,
        ]
    }
)

result = calculate_skewness(
    dataframe,
    "income",
)

print(result)

assert result["method"] == "skewness"
assert result["column"] == "income"
assert result["observations"] == 8

print(
    "\nSkewness: PASSED"
)

print(
    "\n========== KURTOSIS ==========\n"
)

from src.tools.statistics import (
    calculate_kurtosis,
)

dataframe = pd.DataFrame(
    {
        "income": [
            25,
            28,
            31,
            35,
            40,
            42,
            120,
            180,
        ]
    }
)

result = calculate_kurtosis(
    dataframe,
    "income",
)

print(result)

assert result["method"] == "kurtosis"
assert result["column"] == "income"
assert result["observations"] == 8

print(
    "\nKurtosis: PASSED"
)

print(
    "\n========== IQR OUTLIER DETECTION ==========\n"
)

from src.tools.statistics import (
    detect_iqr_outliers,
)

dataframe = pd.DataFrame(
    {
        "salary": [
            50,
            51,
            49,
            48,
            52,
            53,
            54,
            250,
        ]
    }
)

result = detect_iqr_outliers(
    dataframe,
    "salary",
)

print(result)

assert result["method"] == "iqr_outlier_detection"
assert result["column"] == "salary"
assert result["outlier_count"] == 1
assert 250 in result["outlier_values"]

print(
    "\nIQR Outlier Detection: PASSED"
)

print(
    "\n========== CONFIDENCE INTERVAL ==========\n"
)

from src.tools.statistics import (
    confidence_interval_mean,
)

dataframe = pd.DataFrame(
    {
        "salary": [
            50,
            52,
            49,
            51,
            48,
            53,
            50,
            54,
        ]
    }
)

result = confidence_interval_mean(
    dataframe,
    "salary",
)

print(result)

assert result["method"] == "confidence_interval_mean"
assert result["column"] == "salary"
assert result["observations"] == 8
assert result["lower_bound"] < result["mean"]
assert result["upper_bound"] > result["mean"]

print(
    "\nConfidence Interval: PASSED"
)
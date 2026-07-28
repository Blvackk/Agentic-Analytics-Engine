# src/agents/semantic_analyzer.py

from typing import Any

import pandas as pd

from src.llm.client import generate_json_response


# ==========================================================
# SEMANTIC ROLES
# ==========================================================

VALID_ROLES = {
    "identifier",
    "numerical_feature",
    "categorical_feature",
    "boolean_feature",
    "datetime_feature",
    "possible_target",
    "unknown",
}


# Common column names that strongly suggest identifiers.
IDENTIFIER_NAMES = {
    "id",
    "uuid",
    "guid",
    "customer_id",
    "user_id",
    "account_id",
    "transaction_id",
    "order_id",
    "product_id",
    "employee_id",
    "record_id",
}


# Common names that may represent prediction targets.
TARGET_NAMES = {
    "target",
    "label",
    "class",
    "outcome",
    "churn",
    "default",
    "fraud",
    "survived",
    "response",
}


# ==========================================================
# BASIC ROLE INFERENCE
# ==========================================================

def _infer_basic_role(
    dataframe: pd.DataFrame,
    column: str,
) -> str:
    """
    Infer a basic semantic role from dtype.
    """

    series = dataframe[column]

    if pd.api.types.is_bool_dtype(series):
        return "boolean_feature"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime_feature"

    if pd.api.types.is_numeric_dtype(series):
        return "numerical_feature"

    if (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series)
    ):
        return "categorical_feature"

    return "unknown"


# ==========================================================
# IDENTIFIER SIGNALS
# ==========================================================

def _looks_like_identifier(
    dataframe: pd.DataFrame,
    column: str,
) -> bool:
    """
    Determine whether a column has strong identifier signals.

    Uses both the column name and uniqueness ratio.
    """

    normalized_name = column.strip().lower()

    # Strong naming convention.
    if (
        normalized_name in IDENTIFIER_NAMES
        or normalized_name.endswith("_id")
    ):
        return True

    row_count = len(dataframe)

    if row_count == 0:
        return False

    non_missing = dataframe[column].dropna()

    if non_missing.empty:
        return False

    unique_ratio = (
        non_missing.nunique()
        / len(non_missing)
    )

    # High uniqueness alone is not enough for every column,
    # but names containing identifier-like terms strengthen
    # the signal.
    identifier_terms = (
        "identifier",
        "uuid",
        "guid",
        "key",
        "code",
    )

    name_signal = any(
        term in normalized_name
        for term in identifier_terms
    )

    return (
        unique_ratio >= 0.95
        and name_signal
    )


# ==========================================================
# TARGET SIGNALS
# ==========================================================

def _looks_like_target(
    column: str,
) -> bool:
    """
    Check whether a column name resembles a prediction target.
    """

    normalized_name = column.strip().lower()

    return normalized_name in TARGET_NAMES


# ==========================================================
# COLUMN METADATA
# ==========================================================

def _build_column_metadata(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Build compact metadata for every column.

    This metadata is safe to send to the LLM because the
    entire dataset is not included.
    """

    metadata = []

    row_count = len(dataframe)

    for column in dataframe.columns:

        series = dataframe[column]

        non_missing = series.dropna()

        unique_count = int(
            non_missing.nunique()
        )

        if len(non_missing) > 0:

            unique_ratio = round(
                unique_count / len(non_missing),
                4,
            )

        else:

            unique_ratio = 0.0

        missing_count = int(
            series.isna().sum()
        )

        missing_percentage = (
            round(
                missing_count / row_count * 100,
                2,
            )
            if row_count > 0
            else 0.0
        )

        sample_values = (
            non_missing
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        metadata.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "missing_percentage": missing_percentage,
                "sample_values": sample_values,
            }
        )

    return metadata


# ==========================================================
# RULE-BASED SEMANTIC ANALYSIS
# ==========================================================

def _rule_based_analysis(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Generate semantic roles using deterministic rules.

    This acts as both:

    1. initial semantic understanding
    2. fallback if the LLM fails
    """

    results = {}

    for column in dataframe.columns:

        basic_role = _infer_basic_role(
            dataframe,
            column,
        )

        if _looks_like_identifier(
            dataframe,
            column,
        ):

            role = "identifier"
            source = "rule"

        elif _looks_like_target(column):

            role = "possible_target"
            source = "rule"

        else:

            role = basic_role
            source = "dtype"

        results[column] = {
            "role": role,
            "source": source,
        }

    return results


# ==========================================================
# LLM SEMANTIC ANALYSIS
# ==========================================================

def _llm_semantic_analysis(
    dataframe: pd.DataFrame,
) -> dict[str, str]:
    """
    Ask the LLM to classify the semantic role of each column.
    """

    metadata = _build_column_metadata(
        dataframe
    )

    prompt = f"""
You are a data scientist analysing a dataset schema.

Your task is to classify the semantic role of every column.

Allowed roles:

- identifier
- numerical_feature
- categorical_feature
- boolean_feature
- datetime_feature
- possible_target
- unknown

Important rules:

1. An identifier uniquely identifies records and should not
   normally be treated as a predictive numerical feature.

2. Columns such as customer_id, user_id, order_id, UUIDs,
   record keys, and similar fields are usually identifiers.

3. A possible_target is a column that appears to represent
   an outcome that could reasonably be predicted.

4. Do not invent column names.

5. Every provided column must appear exactly once.

6. Use only the allowed roles.

Dataset metadata:

{metadata}

Return ONLY valid JSON using this structure:

{{
    "columns": {{
        "column_name": "role"
    }}
}}
"""

    response = generate_json_response(
        prompt=prompt,
        temperature=0.0,
    )

    if not isinstance(response, dict):

        raise ValueError(
            "LLM semantic response must be a dictionary."
        )

    columns = response.get(
        "columns"
    )

    if not isinstance(columns, dict):

        raise ValueError(
            "LLM response must contain a 'columns' dictionary."
        )

    validated = {}

    dataframe_columns = set(
        dataframe.columns
    )

    for column, role in columns.items():

        if column not in dataframe_columns:
            continue

        if role not in VALID_ROLES:
            continue

        validated[column] = role

    return validated


# ==========================================================
# FINAL SEMANTIC ANALYZER
# ==========================================================

def analyze_semantics(
    dataframe: pd.DataFrame,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Analyse semantic roles of dataset columns.

    The function combines deterministic rules with optional
    LLM-based semantic classification.

    If the LLM fails, the rule-based analysis remains usable.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to analyse.

    use_llm : bool
        Whether to use the LLM for additional semantic
        classification.

    Returns
    -------
    dict
        Semantic information for the dataset.
    """

    if not isinstance(dataframe, pd.DataFrame):

        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:

        raise ValueError(
            "Cannot analyse an empty DataFrame."
        )

    rule_results = _rule_based_analysis(
        dataframe
    )

    llm_results = {}

    llm_used = False

    llm_error = None

    # --------------------------------------------------
    # OPTIONAL LLM ANALYSIS
    # --------------------------------------------------

    if use_llm:

        try:

            llm_results = (
                _llm_semantic_analysis(
                    dataframe
                )
            )

            llm_used = True

        except Exception as error:

            # The analytics workflow should not crash
            # simply because the LLM is unavailable.
            llm_error = str(error)

    # --------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------

    final_columns = {}

    for column in dataframe.columns:

        rule_role = (
            rule_results[column]["role"]
        )

        llm_role = llm_results.get(
            column
        )

        # Strong deterministic identifier detection
        # should override the LLM.
        if rule_role == "identifier":

            final_role = "identifier"
            decision_source = "rule"

        # Strong target naming signal also takes priority.
        elif rule_role == "possible_target":

            final_role = "possible_target"
            decision_source = "rule"

        # Otherwise use valid LLM semantic judgement.
        elif llm_role is not None:

            final_role = llm_role
            decision_source = "llm"

        else:

            final_role = rule_role
            decision_source = (
                rule_results[column]["source"]
            )

        final_columns[column] = {
            "role": final_role,
            "decision_source": decision_source,
            "dtype": str(
                dataframe[column].dtype
            ),
        }

    # --------------------------------------------------
    # CONVENIENT ROLE GROUPS
    # --------------------------------------------------

    identifier_columns = [
        column
        for column, info in final_columns.items()
        if info["role"] == "identifier"
    ]

    target_candidates = [
        column
        for column, info in final_columns.items()
        if info["role"] == "possible_target"
    ]

    feature_columns = [
        column
        for column, info in final_columns.items()
        if info["role"] not in {
            "identifier",
            "possible_target",
        }
    ]

    return {
        "columns": final_columns,

        "identifier_columns":
            identifier_columns,

        "target_candidates":
            target_candidates,

        "feature_columns":
            feature_columns,

        "llm_used":
            llm_used,

        "llm_error":
            llm_error,
    }
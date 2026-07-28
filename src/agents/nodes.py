# src/agents/nodes.py

import pandas as pd

from src.agents.report_generator import (
    generate_report,
    save_report,
)
from src.agents.insight_generator import generate_insights
from src.agents.state import AgentState
from src.agents.semantic_analyzer import analyze_semantics
from src.agents.planner import create_eda_plan
from src.agents.executor import execute_eda_plan

from src.tools.data_loader import load_csv
from src.tools.profiler import profile_dataset
from src.tools.data_quality import analyze_data_quality

from src.tools.cleaning import (
    remove_duplicates,
    fill_numeric_missing,
    fill_categorical_missing,
)

from src.tools.statistics import (
    get_numerical_summary,
    get_categorical_summary,
    get_correlation_matrix,
)


# ==========================================================
# HELPER: SELECT ANALYTICAL DATASET
# ==========================================================

def _get_analysis_dataframe(
    state: AgentState
) -> tuple[pd.DataFrame, str]:
    """
    Select the dataset that downstream analytical nodes
    should use.

    Prefer the cleaned dataset when available.
    Otherwise use the original dataset.
    """

    cleaned_dataframe = state.get(
        "cleaned_dataframe"
    )

    if cleaned_dataframe is not None:
        return (
            cleaned_dataframe,
            "cleaned_dataframe",
        )

    original_dataframe = state.get(
        "dataframe"
    )

    if original_dataframe is not None:
        return (
            original_dataframe,
            "dataframe",
        )

    raise ValueError(
        "No dataframe is available for analysis."
    )


# ==========================================================
# 1. LOAD DATASET NODE
# ==========================================================

def load_dataset_node(
    state: AgentState
) -> dict:
    """
    Load the CSV dataset specified in AgentState.
    """

    dataset_path = state.get(
        "dataset_path"
    )

    if not dataset_path:
        raise ValueError(
            "dataset_path is missing from AgentState."
        )

    dataframe = load_csv(
        dataset_path
    )

    return {
        "dataframe": dataframe
    }


# ==========================================================
# 2. PROFILE DATASET NODE
# ==========================================================

def profile_dataset_node(
    state: AgentState
) -> dict:
    """
    Generate structural information about the
    original dataset.
    """

    dataframe = state.get(
        "dataframe"
    )

    if dataframe is None:
        raise ValueError(
            "dataframe is missing from AgentState."
        )

    profile = profile_dataset(
        dataframe
    )

    return {
        "profile": profile
    }


# ==========================================================
# 3. DATA QUALITY NODE
# ==========================================================

def data_quality_node(
    state: AgentState
) -> dict:
    """
    Analyse the original dataset for common
    data-quality issues.
    """

    dataframe = state.get(
        "dataframe"
    )

    if dataframe is None:
        raise ValueError(
            "dataframe is missing from AgentState."
        )

    quality_report = analyze_data_quality(
        dataframe
    )

    return {
        "quality_report": quality_report
    }


# ==========================================================
# 4. CLEANING ROUTER
# ==========================================================

def route_after_quality(
    state: AgentState
) -> str:
    """
    Decide whether the dataset requires cleaning.

    Supported automatic cleaning:
    - missing values
    - duplicate rows
    """

    quality_report = state.get(
        "quality_report",
        {}
    )

    issues = quality_report.get(
        "issues",
        []
    )

    cleanable_issue_types = {
        "missing_values",
        "duplicate_rows",
    }

    for issue in issues:

        if issue.get(
            "issue_type"
        ) in cleanable_issue_types:

            return "clean"

    return "skip"


# ==========================================================
# 5. CLEAN DATASET NODE
# ==========================================================

def clean_dataset_node(
    state: AgentState
) -> dict:
    """
    Clean supported data-quality issues.

    Rules
    -----
    Numerical missing values -> median
    Categorical missing values -> mode
    Exact duplicate rows -> remove
    """

    dataframe = state.get(
        "dataframe"
    )

    quality_report = state.get(
        "quality_report"
    )

    if dataframe is None:
        raise ValueError(
            "dataframe is missing from AgentState."
        )

    if quality_report is None:
        raise ValueError(
            "quality_report is missing from AgentState."
        )

    cleaned_dataframe = dataframe.copy()

    cleaning_plan = []

    # --------------------------------------------------
    # PROCESS QUALITY ISSUES
    # --------------------------------------------------

    for issue in quality_report.get(
        "issues",
        []
    ):

        issue_type = issue.get(
            "issue_type"
        )

        column = issue.get(
            "column"
        )

        # --------------------------------------------------
        # DUPLICATES
        # --------------------------------------------------

        if issue_type == "duplicate_rows":

            before_rows = len(
                cleaned_dataframe
            )

            cleaned_dataframe = remove_duplicates(
                cleaned_dataframe
            )

            removed_rows = (
                before_rows
                - len(cleaned_dataframe)
            )

            cleaning_plan.append(
                {
                    "action": "remove_duplicates",
                    "rows_removed": removed_rows,
                }
            )

        # --------------------------------------------------
        # MISSING VALUES
        # --------------------------------------------------

        elif issue_type == "missing_values":

            if column is None:
                continue

            if column not in cleaned_dataframe.columns:
                continue

            if pd.api.types.is_numeric_dtype(
                cleaned_dataframe[column]
            ):

                cleaned_dataframe = fill_numeric_missing(
                    dataframe=cleaned_dataframe,
                    column=column,
                    strategy="median",
                )

                cleaning_plan.append(
                    {
                        "action": "fill_numeric_missing",
                        "column": column,
                        "strategy": "median",
                    }
                )

            else:

                cleaned_dataframe = fill_categorical_missing(
                    dataframe=cleaned_dataframe,
                    column=column,
                    strategy="mode",
                )

                cleaning_plan.append(
                    {
                        "action": "fill_categorical_missing",
                        "column": column,
                        "strategy": "mode",
                    }
                )

    # --------------------------------------------------
    # CLEANING REPORT
    # --------------------------------------------------

    original_missing = int(
        dataframe
        .isnull()
        .sum()
        .sum()
    )

    cleaned_missing = int(
        cleaned_dataframe
        .isnull()
        .sum()
        .sum()
    )

    original_duplicates = int(
        dataframe
        .duplicated()
        .sum()
    )

    cleaned_duplicates = int(
        cleaned_dataframe
        .duplicated()
        .sum()
    )

    cleaning_report = {
        "original_rows":
            len(dataframe),

        "cleaned_rows":
            len(cleaned_dataframe),

        "rows_removed":
            len(dataframe)
            - len(cleaned_dataframe),

        "original_missing_values":
            original_missing,

        "cleaned_missing_values":
            cleaned_missing,

        "original_duplicates":
            original_duplicates,

        "cleaned_duplicates":
            cleaned_duplicates,

        "actions_performed":
            len(cleaning_plan),
    }

    return {
        "cleaned_dataframe":
            cleaned_dataframe,

        "cleaning_plan":
            cleaning_plan,

        "cleaning_report":
            cleaning_report,
    }


# ==========================================================
# 6. VALIDATE CLEANING NODE
# ==========================================================

def validate_cleaning_node(
    state: AgentState
) -> dict:
    """
    Validate the result of the cleaning process.
    """

    original_dataframe = state.get(
        "dataframe"
    )

    cleaned_dataframe = state.get(
        "cleaned_dataframe"
    )

    cleaning_report = state.get(
        "cleaning_report"
    )

    if original_dataframe is None:
        raise ValueError(
            "dataframe is missing from AgentState."
        )

    if cleaned_dataframe is None:
        raise ValueError(
            "cleaned_dataframe is missing from AgentState."
        )

    if cleaning_report is None:
        raise ValueError(
            "cleaning_report is missing from AgentState."
        )

    original_rows = len(
        original_dataframe
    )

    original_missing = int(
        original_dataframe
        .isnull()
        .sum()
        .sum()
    )

    original_duplicates = int(
        original_dataframe
        .duplicated()
        .sum()
    )

    cleaned_rows = len(
        cleaned_dataframe
    )

    cleaned_missing = int(
        cleaned_dataframe
        .isnull()
        .sum()
        .sum()
    )

    cleaned_duplicates = int(
        cleaned_dataframe
        .duplicated()
        .sum()
    )

    if original_missing > 0:

        missing_values_improved = (
            cleaned_missing
            < original_missing
        )

    else:

        missing_values_improved = (
            cleaned_missing == 0
        )

    if original_duplicates > 0:

        duplicates_improved = (
            cleaned_duplicates
            < original_duplicates
        )

    else:

        duplicates_improved = (
            cleaned_duplicates == 0
        )

    if original_rows > 0:

        row_retention_percentage = round(
            (
                cleaned_rows
                / original_rows
            )
            * 100,
            2
        )

    else:

        row_retention_percentage = 0.0

    remaining_supported_issues = []

    if cleaned_missing > 0:

        remaining_supported_issues.append(
            "missing_values"
        )

    if cleaned_duplicates > 0:

        remaining_supported_issues.append(
            "duplicate_rows"
        )

    supported_issues_remaining = len(
        remaining_supported_issues
    )

    validation_passed = (
        cleaned_rows > 0
        and supported_issues_remaining == 0
    )

    cleaning_validation = {
        "original_missing_values":
            original_missing,

        "cleaned_missing_values":
            cleaned_missing,

        "missing_values_improved":
            missing_values_improved,

        "original_duplicates":
            original_duplicates,

        "cleaned_duplicates":
            cleaned_duplicates,

        "duplicates_improved":
            duplicates_improved,

        "original_rows":
            original_rows,

        "cleaned_rows":
            cleaned_rows,

        "row_retention_percentage":
            row_retention_percentage,

        "remaining_supported_issues":
            remaining_supported_issues,

        "supported_issues_remaining":
            supported_issues_remaining,

        "validation_passed":
            validation_passed,
    }

    return {
        "cleaning_validation":
            cleaning_validation
    }


# ==========================================================
# 7. SEMANTIC ANALYSIS NODE
# ==========================================================

def semantic_analysis_node(
    state: AgentState
) -> dict:
    """
    Determine semantic roles of dataset columns before
    statistical analysis.

    This allows identifier columns such as customer_id
    to be excluded from analytical statistics.
    """

    (
        analysis_dataframe,
        _,
    ) = _get_analysis_dataframe(
        state
    )

    semantic_analysis = analyze_semantics(
        dataframe=analysis_dataframe,
        use_llm=True,
    )

    identifier_columns = semantic_analysis.get(
        "identifier_columns",
        [],
    )

    target_candidates = semantic_analysis.get(
        "target_candidates",
        [],
    )

    feature_columns = semantic_analysis.get(
        "feature_columns",
        [],
    )

    return {
        "semantic_analysis":
            semantic_analysis,

        "identifier_columns":
            identifier_columns,

        "target_candidates":
            target_candidates,

        "feature_columns":
            feature_columns,
    }


# ==========================================================
# 8. STATISTICAL ANALYSIS NODE
# ==========================================================

def analysis_dataset_node(
    state: AgentState
) -> dict:
    """
    Perform statistical analysis after semantic analysis.

    Identifier columns are excluded before calculating
    descriptive statistics and correlations.

    Target candidates remain included because their
    distributions can be analytically meaningful.
    """

    (
        analysis_dataframe,
        dataset_source,
    ) = _get_analysis_dataframe(
        state
    )

    identifier_columns = state.get(
        "identifier_columns",
        []
    )

    # Keep only identifier names that actually exist.
    valid_identifier_columns = [
        column
        for column in identifier_columns
        if column in analysis_dataframe.columns
    ]

    # --------------------------------------------------
    # REMOVE IDENTIFIERS FROM ANALYTICAL DATA
    # --------------------------------------------------

    analytical_dataframe = (
        analysis_dataframe.drop(
            columns=valid_identifier_columns,
            errors="ignore",
        )
    )

    if analytical_dataframe.empty:

        raise ValueError(
            "No analytical columns remain after "
            "identifier columns were excluded."
        )

    # --------------------------------------------------
    # STATISTICS
    # --------------------------------------------------

    numerical_summary = get_numerical_summary(
        analytical_dataframe
    )

    categorical_summary = get_categorical_summary(
        analytical_dataframe
    )

    correlation_matrix = get_correlation_matrix(
        analytical_dataframe
    )

    # --------------------------------------------------
    # ANALYSIS METADATA
    # --------------------------------------------------

    analysis_results = {
        "dataset_source":
            dataset_source,

        "rows_analyzed":
            len(analytical_dataframe),

        "original_column_count":
            len(analysis_dataframe.columns),

        "columns_analyzed":
            len(analytical_dataframe.columns),

        "analyzed_columns":
            analytical_dataframe.columns.tolist(),

        "excluded_identifier_columns":
            valid_identifier_columns,

        "numerical_summary":
            numerical_summary,

        "categorical_summary":
            categorical_summary,

        "correlation_matrix":
            correlation_matrix,
    }

    return {
        "numerical_summary":
            numerical_summary,

        "categorical_summary":
            categorical_summary,

        "correlation_matrix":
            correlation_matrix,

        "analysis_results":
            analysis_results,
    }


# ==========================================================
# 9. EDA PLANNER NODE
# ==========================================================

def plan_eda_node(
    state: AgentState
) -> dict:
    """
    Create a semantic-aware EDA plan.

    Identifier columns are excluded from normal analytical
    processing while target candidates can receive dedicated
    target analysis.
    """

    (
        planning_dataframe,
        _,
    ) = _get_analysis_dataframe(
        state
    )

    semantic_analysis = state.get(
        "semantic_analysis"
    )

    if semantic_analysis is None:

        raise ValueError(
            "semantic_analysis is missing from AgentState."
        )

    analysis_plan = create_eda_plan(
        dataframe=planning_dataframe,
        semantic_analysis=semantic_analysis,
    )

    return {
        "analysis_plan":
            analysis_plan
    }


# ==========================================================
# 10. EDA TOOL EXECUTOR NODE
# ==========================================================

def execute_eda_node(
    state: AgentState
) -> dict:
    """
    Execute the semantic-aware EDA plan.

    Uses cleaned_dataframe when available.
    """

    (
        execution_dataframe,
        _,
    ) = _get_analysis_dataframe(
        state
    )

    analysis_plan = state.get(
        "analysis_plan"
    )

    if analysis_plan is None:

        raise ValueError(
            "analysis_plan is missing from AgentState."
        )

    execution_result = execute_eda_plan(
        dataframe=execution_dataframe,
        analysis_plan=analysis_plan,
    )

    return {
        "chart_paths":
            execution_result.get(
                "chart_paths",
                [],
            ),

        "executed_tasks":
            execution_result.get(
                "executed_tasks",
                [],
            ),

        "skipped_tasks":
            execution_result.get(
                "skipped_tasks",
                [],
            ),

        "execution_errors":
            execution_result.get(
                "execution_errors",
                [],
            ),

        "execution_report":
            execution_result.get(
                "execution_report",
                {},
            ),

        "semantic_metadata":
            execution_result.get(
                "semantic_metadata",
                {},
            ),

        "target_analysis":
            execution_result.get(
                "target_analysis",
                {},
            ),
    }


# ==========================================================
# 11. INSIGHT GENERATION NODE
# ==========================================================

def generate_insights_node(
    state: AgentState
) -> dict:
    """
    Generate evidence-based analytical insights using
    statistical, semantic, and target analysis results.

    The LLM receives structured analytical results rather
    than the raw DataFrame.
    """

    numerical_summary = state.get(
        "numerical_summary",
        {}
    )

    categorical_summary = state.get(
        "categorical_summary",
        {}
    )

    correlation_matrix = state.get(
        "correlation_matrix",
        {}
    )

    semantic_analysis = state.get(
        "semantic_analysis",
        {}
    )

    target_analysis = state.get(
        "target_analysis",
        {}
    )

    insights = generate_insights(
        numerical_summary=numerical_summary,
        categorical_summary=categorical_summary,
        correlation_matrix=correlation_matrix,
        semantic_analysis=semantic_analysis,
        target_analysis=target_analysis,
    )

    return {
        "insights": insights
    }

# ==========================================================
# 12. FINAL REPORT GENERATION NODE
# ==========================================================

def generate_report_node(
    state: AgentState
) -> dict:
    """
    Generate and save the final analytics report.

    This node combines verified outputs from the complete
    analytics workflow into a Markdown report.
    """

    # --------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------

    report = generate_report(
        profile=state.get(
            "profile",
            {},
        ),

        quality_report=state.get(
            "quality_report",
            {},
        ),

        cleaning_report=state.get(
            "cleaning_report",
            {},
        ),

        cleaning_validation=state.get(
            "cleaning_validation",
            {},
        ),

        analysis_results=state.get(
            "analysis_results",
            {},
        ),

        semantic_analysis=state.get(
            "semantic_analysis",
            {},
        ),

        target_analysis=state.get(
            "target_analysis",
            {},
        ),

        execution_report=state.get(
            "execution_report",
            {},
        ),

        chart_paths=state.get(
            "chart_paths",
            [],
        ),

        insights=state.get(
            "insights",
            {},
        ),
    )

    # --------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------

    report_path = save_report(
        report=report,
    )

    return {
        "final_report":
            report,

        "report_path":
            report_path,
    }
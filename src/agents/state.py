# src/agents/state.py

from typing import Any, TypedDict

import pandas as pd


class AgentState(TypedDict, total=False):
    """
    Shared state used by the Agentic Analytics workflow.

    Every LangGraph node can read values from this state
    and return updates that are merged back into the state.
    """

    # ==================================================
    # DATASET
    # ==================================================

    dataset_path: str

    # Original dataset loaded from the source file.
    dataframe: pd.DataFrame

    # Dataset after cleaning.
    cleaned_dataframe: pd.DataFrame


    # ==================================================
    # DATASET UNDERSTANDING
    # ==================================================

    profile: dict[str, Any]

    quality_report: dict[str, Any]


    # ==================================================
    # CLEANING
    # ==================================================

    cleaning_plan: list[dict[str, Any]]

    cleaning_report: dict[str, Any]

    cleaning_validation: dict[str, Any]


    # ==================================================
    # STATISTICAL ANALYSIS
    # ==================================================

    numerical_summary: dict[str, Any]

    categorical_summary: dict[str, Any]

    correlation_matrix: dict[str, Any]

    analysis_results: dict[str, Any]


    # ==================================================
    # SEMANTIC ANALYSIS
    # ==================================================

    # Complete output produced by analyze_semantics().
    semantic_analysis: dict[str, Any]

    # Convenient copies of important semantic groups.
    identifier_columns: list[str]

    target_candidates: list[str]

    feature_columns: list[str]


    # ==================================================
    # EDA PLANNING
    # ==================================================

    # Tasks produced by create_eda_plan().
    analysis_plan: list[dict[str, Any]]


    # ==================================================
    # TOOL EXECUTION
    # ==================================================

    chart_paths: list[str]

    executed_tasks: list[dict[str, Any]]

    skipped_tasks: list[dict[str, Any]]

    execution_errors: list[dict[str, Any]]

    execution_report: dict[str, Any]


    # ==================================================
    # SEMANTIC EXECUTION METADATA
    # ==================================================

    # Semantic information recorded by the executor.
    semantic_metadata: dict[str, Any]


    # ==================================================
    # TARGET ANALYSIS
    # ==================================================

    # Distribution/statistics generated for possible
    # prediction targets.
    target_analysis: dict[str, Any]


    # ==================================================
    # AGENT REASONING / INSIGHTS
    # ==================================================

    insights: list[str, Any]


    # ==================================================
    # FINAL OUTPUT
    # ==================================================

    final_report: str
    report_path: str


    # ==================================================
    # WORKFLOW ERROR TRACKING
    # ==================================================

    errors: list[str]
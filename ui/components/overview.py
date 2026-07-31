# ui/components/overview.py

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from ui.utils.html import render_html


# ============================================================
# HELPERS
# ============================================================

def _safe_list(value: Any) -> list:
    """Return value as a list when possible."""
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    return []


def _safe_dict(value: Any) -> dict:
    """Return value as a dictionary when possible."""
    return value if isinstance(value, dict) else {}


def _format_number(value: Any) -> str:
    """Format numeric values for compact dashboard display."""
    if value is None:
        return "—"

    try:
        value = float(value)

        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"

        if value >= 1_000:
            return f"{value / 1_000:.1f}K"

        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:,.1f}"

    except (TypeError, ValueError):
        return str(value)


def _get_dataframe(state: dict[str, Any]) -> pd.DataFrame | None:
    """
    Prefer the cleaned dataframe because this represents
    the dataset used by the analytical workflow.
    """

    cleaned_df = state.get("cleaned_dataframe")

    if isinstance(cleaned_df, pd.DataFrame):
        return cleaned_df

    dataframe = state.get("dataframe")

    if isinstance(dataframe, pd.DataFrame):
        return dataframe

    return None


def _missing_count(dataframe: pd.DataFrame | None) -> int:
    if dataframe is None:
        return 0

    return int(dataframe.isna().sum().sum())


def _duplicate_count(dataframe: pd.DataFrame | None) -> int:
    if dataframe is None:
        return 0

    return int(dataframe.duplicated().sum())


def _quality_score(dataframe: pd.DataFrame | None) -> float:
    """
    Produce a simple presentation-level completeness score.

    This is NOT intended to replace the backend data-quality
    analysis. It gives the dashboard a compact health metric.
    """

    if dataframe is None or dataframe.empty:
        return 0.0

    total_cells = dataframe.shape[0] * dataframe.shape[1]

    if total_cells == 0:
        return 0.0

    missing = int(
        dataframe.isna().sum().sum()
    )

    completeness = (
        (total_cells - missing)
        / total_cells
    ) * 100

    return round(completeness, 1)


def _semantic_groups(
    state: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:

    semantic = _safe_dict(
        state.get("semantic_analysis")
    )

    identifiers = _safe_list(
        state.get("identifier_columns")
    )

    targets = _safe_list(
        state.get("target_candidates")
    )

    features = _safe_list(
        state.get("feature_columns")
    )

    # Fallback to semantic_analysis if the convenient
    # state copies are unavailable.

    if not identifiers:
        identifiers = _safe_list(
            semantic.get("identifier_columns")
        )

    if not targets:
        targets = _safe_list(
            semantic.get("target_candidates")
        )

    if not features:
        features = _safe_list(
            semantic.get("feature_columns")
        )

    return identifiers, targets, features


# ============================================================
# HTML COMPONENTS
# ============================================================

def _render_page_header() -> None:

    render_html(
        """
        <div class="aa-page-header">
            <div class="aa-eyebrow">
                Analytics workspace
            </div>

            <div class="aa-page-title">
                Overview
            </div>

            <div class="aa-page-description">
                A consolidated view of dataset health,
                semantic structure and the most important
                findings discovered by the analytics engine.
            </div>
        </div>
        """
    )


def _render_dataset_header(
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> None:

    rows, columns = dataframe.shape

    safe_name = escape(dataset_name)

    render_html(
        f"""
        <div class="aa-dataset-header">

            <div>
                <div class="aa-dataset-name">
                    {safe_name}
                </div>

                <div class="aa-dataset-meta">
                    {rows:,} rows &nbsp;·&nbsp;
                    {columns:,} columns
                    &nbsp;·&nbsp;
                    Analysis complete
                </div>
            </div>

            <div class="aa-status">
                <span class="aa-status-dot"></span>
                Ready
            </div>

        </div>
        """
    )


def _render_kpis(
    dataframe: pd.DataFrame,
) -> None:

    rows, columns = dataframe.shape

    missing = _missing_count(dataframe)

    duplicates = _duplicate_count(dataframe)

    quality = _quality_score(dataframe)

    render_html(
        f"""
        <div class="aa-kpi-grid">

            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Observations
                </div>

                <div class="aa-kpi-value">
                    {_format_number(rows)}
                </div>

                <div class="aa-kpi-detail">
                    Rows available for analysis
                </div>
            </div>


            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Variables
                </div>

                <div class="aa-kpi-value">
                    {_format_number(columns)}
                </div>

                <div class="aa-kpi-detail">
                    Fields in the active dataset
                </div>
            </div>


            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Missing values
                </div>

                <div class="aa-kpi-value">
                    {_format_number(missing)}
                </div>

                <div class="aa-kpi-detail">
                    Null cells after preprocessing
                </div>
            </div>


            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Completeness
                </div>

                <div class="aa-kpi-value">
                    {quality:.1f}%
                </div>

                <div class="aa-kpi-detail">
                    {duplicates:,} duplicate rows detected
                </div>
            </div>

        </div>
        """
    )


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

def _render_executive_overview(
    state: dict[str, Any],
    dataframe: pd.DataFrame,
) -> None:

    identifiers, targets, features = (
        _semantic_groups(state)
    )

    rows, columns = dataframe.shape

    missing = _missing_count(dataframe)

    target_text = (
        ", ".join(map(str, targets))
        if targets
        else "No explicit target detected"
    )

    identifier_text = (
        ", ".join(map(str, identifiers))
        if identifiers
        else "No identifier detected"
    )

    render_html(
        """
        <div class="aa-section">
            <div class="aa-section-label">
                Executive overview
            </div>

            <div class="aa-section-description">
                Structural summary of the active analytical dataset.
            </div>
        </div>
        """
    )

    render_html(
        f"""
        <div class="aa-panel">

            <div class="aa-panel-title">
                Dataset profile
            </div>

            <div class="aa-panel-description">
                The active dataset contains
                <strong>{rows:,} observations</strong>
                across
                <strong>{columns:,} variables</strong>.

                The analytical pipeline currently recognises
                <strong>{len(features):,} feature variables</strong>.

                The detected target context is
                <strong>{escape(target_text)}</strong>,
                while
                <strong>{escape(identifier_text)}</strong>
                represents the identifier context.

                There are currently
                <strong>{missing:,} missing values</strong>
                in the dataframe used for analysis.
            </div>

        </div>
        """
    )


# ============================================================
# SEMANTIC STRUCTURE
# ============================================================

def _render_semantic_structure(
    state: dict[str, Any],
) -> None:

    identifiers, targets, features = (
        _semantic_groups(state)
    )

    render_html(
        """
        <div class="aa-section">
            <div class="aa-section-label">
                Semantic structure
            </div>

            <div class="aa-section-description">
                How the engine interprets the role of dataset fields.
            </div>
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        render_html(
            """
            <div class="aa-panel">
                <div class="aa-panel-title">
                    Identifiers
                </div>
                <div class="aa-panel-description">
                    Fields recognised as record identifiers.
                </div>
            </div>
            """
    )

        if identifiers:
            for column in identifiers:
                st.code(
                    str(column),
                    language=None,
                )
        else:
            st.caption(
                "No identifier columns detected."
            )

    with col2:

        render_html(
            """
            <div class="aa-panel">
                <div class="aa-panel-title">
                    Targets
                </div>
                <div class="aa-panel-description">
                    Variables that may represent analytical outcomes.
                </div>
            </div>
            """
    )

        if targets:
            for column in targets:
                st.code(
                    str(column),
                    language=None,
                )
        else:
            st.caption(
                "No target candidates detected."
            )

    with col3:

        render_html(
            """
            <div class="aa-panel">
                <div class="aa-panel-title">
                    Features
                </div>
                <div class="aa-panel-description">
                    Variables available for explanatory analysis.
                </div>
            </div>
            """
    )

        if features:
            preview = features[:8]

            for column in preview:
                st.code(
                    str(column),
                    language=None,
                )

            remaining = (
                len(features)
                - len(preview)
            )

            if remaining > 0:
                st.caption(
                    f"+ {remaining} additional features"
                )
        else:
            st.caption(
                "No feature columns detected."
            )


# ============================================================
# INSIGHTS
# ============================================================

def _render_insights(
    state: dict[str, Any],
) -> None:

    insights = _safe_list(
        state.get("insights")
    )

    render_html(
        """
        <div class="aa-section">
            <div class="aa-section-label">
                Key findings
            </div>

            <div class="aa-section-description">
                Findings generated from statistical and
                exploratory analysis.
            </div>
        </div>
        """
    )

    if not insights:

        render_html(
            """
            <div class="aa-panel">
                <div class="aa-panel-title">
                    No findings available
                </div>

                <div class="aa-panel-description">
                    The workflow completed without returning
                    displayable insight statements.
                </div>
            </div>
            """
    )

        return

    for index, insight in enumerate(
        insights[:8],
        start=1,
    ):

        if isinstance(insight, dict):

            insight_text = (
                insight.get("insight")
                or insight.get("text")
                or insight.get("message")
                or str(insight)
            )

        else:
            insight_text = str(insight)

        render_html(
            f"""
            <div class="aa-insight">

                <div class="aa-insight-index">
                    {index:02d}
                </div>

                <div class="aa-insight-text">
                    {escape(insight_text)}
                </div>

            </div>
            """
    )

    if len(insights) > 8:

        st.caption(
            f"{len(insights) - 8} additional findings "
            "are available in the full report."
        )


# ============================================================
# PIPELINE STATUS
# ============================================================

def _render_pipeline_status(
    state: dict[str, Any],
) -> None:

    execution_report = _safe_dict(
        state.get("execution_report")
    )

    executed_tasks = _safe_list(
        state.get("executed_tasks")
    )

    skipped_tasks = _safe_list(
        state.get("skipped_tasks")
    )

    execution_errors = _safe_list(
        state.get("execution_errors")
    )

    render_html(
        """
        <div class="aa-section">
            <div class="aa-section-label">
                Analysis pipeline
            </div>

            <div class="aa-section-description">
                Execution status from the autonomous analytical workflow.
            </div>
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Executed tasks",
            len(executed_tasks),
        )

    with col2:
        st.metric(
            "Skipped tasks",
            len(skipped_tasks),
        )

    with col3:
        st.metric(
            "Execution errors",
            len(execution_errors),
        )

    if execution_report:

        with st.expander(
            "Pipeline execution details"
        ):
            st.json(execution_report)


# ============================================================
# DATASET PREVIEW
# ============================================================

def _render_dataset_preview(
    dataframe: pd.DataFrame,
) -> None:

    render_html(
        """
        <div class="aa-section">
            <div class="aa-section-label">
                Dataset preview
            </div>

            <div class="aa-section-description">
                First records from the dataframe currently used
                by the analytics engine.
            </div>
        </div>
        """
    )

    st.dataframe(
        dataframe.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Showing up to 10 of {len(dataframe):,} rows."
    )


# ============================================================
# PUBLIC COMPONENT
# ============================================================

def render_overview(
    state: dict[str, Any],
    dataset_name: str = "Active dataset",
) -> None:
    """
    Render the Overview workspace.

    Parameters
    ----------
    state:
        Final state returned by the Phase 1 LangGraph workflow.

    dataset_name:
        Name of the uploaded dataset displayed in the interface.
    """

    if not isinstance(state, dict):
        st.error(
            "The analytics workflow state is unavailable."
        )
        return

    dataframe = _get_dataframe(state)

    if dataframe is None:
        st.warning(
            "No dataframe is available for the overview."
        )
        return

    _render_page_header()

    _render_dataset_header(
        dataset_name=dataset_name,
        dataframe=dataframe,
    )

    _render_kpis(
        dataframe=dataframe,
    )

    _render_executive_overview(
        state=state,
        dataframe=dataframe,
    )

    _render_semantic_structure(
        state=state,
    )

    _render_insights(
        state=state,
    )

    _render_pipeline_status(
        state=state,
    )

    _render_dataset_preview(
        dataframe=dataframe,
    )
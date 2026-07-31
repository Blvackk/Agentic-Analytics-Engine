# ui/components/quality.py

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from ui.utils.html import render_html


# ============================================================
# HELPERS
# ============================================================

def _safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    return []


def _get_original_dataframe(
    state: dict[str, Any],
) -> pd.DataFrame | None:

    dataframe = state.get("dataframe")

    return (
        dataframe
        if isinstance(dataframe, pd.DataFrame)
        else None
    )


def _get_cleaned_dataframe(
    state: dict[str, Any],
) -> pd.DataFrame | None:

    dataframe = state.get(
        "cleaned_dataframe"
    )

    if isinstance(dataframe, pd.DataFrame):
        return dataframe

    return _get_original_dataframe(state)


def _count_missing(
    dataframe: pd.DataFrame | None,
) -> int:

    if dataframe is None:
        return 0

    return int(
        dataframe.isna().sum().sum()
    )


def _count_duplicates(
    dataframe: pd.DataFrame | None,
) -> int:

    if dataframe is None:
        return 0

    return int(
        dataframe.duplicated().sum()
    )


def _completeness(
    dataframe: pd.DataFrame | None,
) -> float:

    if dataframe is None:
        return 0.0

    rows, columns = dataframe.shape

    total_cells = rows * columns

    if total_cells == 0:
        return 0.0

    missing = _count_missing(
        dataframe
    )

    return round(
        (
            (total_cells - missing)
            / total_cells
        )
        * 100,
        1,
    )


def _format_delta(
    before: int,
    after: int,
) -> str:

    difference = after - before

    if difference == 0:
        return "No change"

    if difference > 0:
        return f"+{difference:,}"

    return f"{difference:,}"


# ============================================================
# HEADER
# ============================================================

def _render_header() -> None:

    render_html(
        """
        <div class="aa-page-header">

            <div class="aa-eyebrow">
                Dataset diagnostics
            </div>

            <div class="aa-page-title">
                Data Quality
            </div>

            <div class="aa-page-description">
                Inspect completeness, duplication and preprocessing
                outcomes before relying on downstream analytical
                results.
            </div>

        </div>
        """
    )


# ============================================================
# HEALTH SUMMARY
# ============================================================

def _render_health_summary(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> None:

    original_missing = _count_missing(
        original
    )

    cleaned_missing = _count_missing(
        cleaned
    )

    original_duplicates = _count_duplicates(
        original
    )

    cleaned_duplicates = _count_duplicates(
        cleaned
    )

    completeness = _completeness(
        cleaned
    )

    render_html(
        f"""
        <div class="aa-kpi-grid">

            <div class="aa-kpi">

                <div class="aa-kpi-label">
                    Completeness
                </div>

                <div class="aa-kpi-value">
                    {completeness:.1f}%
                </div>

                <div class="aa-kpi-detail">
                    Non-null cells in active data
                </div>

            </div>


            <div class="aa-kpi">

                <div class="aa-kpi-label">
                    Missing values
                </div>

                <div class="aa-kpi-value">
                    {cleaned_missing:,}
                </div>

                <div class="aa-kpi-detail">
                    {_format_delta(
                        original_missing,
                        cleaned_missing,
                    )} after preprocessing
                </div>

            </div>


            <div class="aa-kpi">

                <div class="aa-kpi-label">
                    Duplicate rows
                </div>

                <div class="aa-kpi-value">
                    {cleaned_duplicates:,}
                </div>

                <div class="aa-kpi-detail">
                    {_format_delta(
                        original_duplicates,
                        cleaned_duplicates,
                    )} after preprocessing
                </div>

            </div>


            <div class="aa-kpi">

                <div class="aa-kpi-label">
                    Active rows
                </div>

                <div class="aa-kpi-value">
                    {len(cleaned):,}
                </div>

                <div class="aa-kpi-detail">
                    {len(original):,} rows originally loaded
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# QUALITY STATUS
# ============================================================

def _render_quality_status(
    dataframe: pd.DataFrame,
) -> None:

    missing = _count_missing(
        dataframe
    )

    duplicates = _count_duplicates(
        dataframe
    )

    completeness = _completeness(
        dataframe
    )

    if missing == 0 and duplicates == 0:

        title = "Dataset is analysis-ready"

        description = (
            "No missing values or duplicate rows are "
            "present in the active dataframe."
        )

        css_class = "aa-quality-good"

    elif completeness >= 95:

        title = "Dataset quality is strong"

        description = (
            "Minor quality issues remain, but overall "
            "completeness is high."
        )

        css_class = "aa-quality-warning"

    else:

        title = "Quality issues require attention"

        description = (
            "The active dataset contains quality issues "
            "that may materially affect interpretation."
        )

        css_class = "aa-quality-danger"

    render_html(
        f"""
        <div class="aa-panel">

            <div class="aa-panel-title {css_class}">
                {title}
            </div>

            <div class="aa-panel-description">
                {description}
            </div>

            <div style="margin-top: 1rem;">
                <div class="aa-quality-bar">
                    <div
                        class="aa-quality-fill"
                        style="width: {completeness}%;">
                    </div>
                </div>
            </div>

        </div>
        """
    )


# ============================================================
# COLUMN QUALITY
# ============================================================

def _render_column_quality(
    dataframe: pd.DataFrame,
) -> None:

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Field diagnostics
            </div>

            <div class="aa-section-description">
                Completeness and uniqueness for each variable
                in the active dataset.
            </div>

        </div>
        """
    )

    rows = []

    total_rows = len(dataframe)

    for column in dataframe.columns:

        series = dataframe[column]

        missing = int(
            series.isna().sum()
        )

        non_null = int(
            series.notna().sum()
        )

        unique = int(
            series.nunique(
                dropna=True
            )
        )

        if total_rows > 0:

            completeness = round(
                (
                    non_null
                    / total_rows
                )
                * 100,
                1,
            )

        else:
            completeness = 0.0

        rows.append(
            {
                "Field": column,
                "Type": str(series.dtype),
                "Missing": missing,
                "Unique": unique,
                "Completeness": (
                    f"{completeness:.1f}%"
                ),
            }
        )

    quality_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# BEFORE / AFTER
# ============================================================

def _render_cleaning_comparison(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> None:

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Preprocessing impact
            </div>

            <div class="aa-section-description">
                Compare dataset condition before and after
                autonomous cleaning.
            </div>

        </div>
        """
    )

    before_missing = _count_missing(
        original
    )

    after_missing = _count_missing(
        cleaned
    )

    before_duplicates = _count_duplicates(
        original
    )

    after_duplicates = _count_duplicates(
        cleaned
    )

    comparison = pd.DataFrame(
        [
            {
                "Metric": "Rows",
                "Before": len(original),
                "After": len(cleaned),
            },
            {
                "Metric": "Columns",
                "Before": original.shape[1],
                "After": cleaned.shape[1],
            },
            {
                "Metric": "Missing values",
                "Before": before_missing,
                "After": after_missing,
            },
            {
                "Metric": "Duplicate rows",
                "Before": before_duplicates,
                "After": after_duplicates,
            },
            {
                "Metric": "Completeness",
                "Before": (
                    f"{_completeness(original):.1f}%"
                ),
                "After": (
                    f"{_completeness(cleaned):.1f}%"
                ),
            },
        ]
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CLEANING ACTIONS
# ============================================================

def _render_cleaning_activity(
    state: dict[str, Any],
) -> None:

    cleaning_plan = _safe_list(
        state.get("cleaning_plan")
    )

    cleaning_report = _safe_dict(
        state.get("cleaning_report")
    )

    cleaning_validation = _safe_dict(
        state.get("cleaning_validation")
    )

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Cleaning activity
            </div>

            <div class="aa-section-description">
                Actions selected and validated by the
                preprocessing workflow.
            </div>

        </div>
        """
    )

    if not cleaning_plan:

        render_html(
            """
            <div class="aa-panel">

                <div class="aa-panel-title">
                    No cleaning actions required
                </div>

                <div class="aa-panel-description">
                    The workflow did not record a cleaning plan
                    for this dataset.
                </div>

            </div>
            """
    )

    else:

        for index, action in enumerate(
            cleaning_plan,
            start=1,
        ):

            if isinstance(action, dict):

                action_name = (
                    action.get("action")
                    or action.get("tool")
                    or action.get("operation")
                    or "Cleaning action"
                )

                reason = (
                    action.get("reason")
                    or action.get("description")
                    or ""
                )

            else:

                action_name = str(action)
                reason = ""

            render_html(
                f"""
                <div class="aa-insight">

                    <div class="aa-insight-index">
                        {index:02d}
                    </div>

                    <div class="aa-insight-text">

                        <strong>
                            {escape(str(action_name))}
                        </strong>

                        {
                            "<br>" + escape(str(reason))
                            if reason
                            else ""
                        }

                    </div>

                </div>
                """
    )

    col1, col2 = st.columns(2)

    with col1:

        with st.expander(
            "Cleaning report"
        ):

            if cleaning_report:
                st.json(
                    cleaning_report
                )
            else:
                st.caption(
                    "No cleaning report recorded."
                )

    with col2:

        with st.expander(
            "Validation report"
        ):

            if cleaning_validation:
                st.json(
                    cleaning_validation
                )
            else:
                st.caption(
                    "No cleaning validation recorded."
                )


# ============================================================
# BACKEND QUALITY REPORT
# ============================================================

def _render_backend_report(
    state: dict[str, Any],
) -> None:

    quality_report = _safe_dict(
        state.get("quality_report")
    )

    if not quality_report:
        return

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Diagnostic details
            </div>

            <div class="aa-section-description">
                Raw quality evidence generated by the
                analytics pipeline.
            </div>

        </div>
        """
    )

    with st.expander(
        "View quality report"
    ):
        st.json(
            quality_report
        )


# ============================================================
# PUBLIC COMPONENT
# ============================================================

def render_quality(
    state: dict[str, Any],
) -> None:
    """
    Render the Data Quality workspace.
    """

    if not isinstance(state, dict):

        st.error(
            "The analytics workflow state is unavailable."
        )
        return

    original = _get_original_dataframe(
        state
    )

    cleaned = _get_cleaned_dataframe(
        state
    )

    if original is None:

        st.warning(
            "The original dataframe is unavailable."
        )
        return

    if cleaned is None:
        cleaned = original

    _render_header()

    _render_health_summary(
        original=original,
        cleaned=cleaned,
    )

    _render_quality_status(
        dataframe=cleaned,
    )

    _render_column_quality(
        dataframe=cleaned,
    )

    _render_cleaning_comparison(
        original=original,
        cleaned=cleaned,
    )

    _render_cleaning_activity(
        state=state,
    )

    _render_backend_report(
        state=state,
    )
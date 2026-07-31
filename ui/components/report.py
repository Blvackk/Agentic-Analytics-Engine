from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from ui.utils.html import render_html


# ============================================================
# HELPERS
# ============================================================

def _safe_list(value: Any) -> list:
    """
    Safely convert supported collection values to a list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    return []


def _safe_dict(value: Any) -> dict:
    """
    Return value when it is a dictionary, otherwise {}.
    """

    return value if isinstance(value, dict) else {}


def _report_text(
    state: dict[str, Any],
) -> str:
    """
    Extract the generated report body from workflow state.
    """

    report = state.get("final_report")

    if report is None:
        return ""

    return str(report).strip()


def _project_root() -> Path:
    """
    Return the project root directory.

    Expected structure:

        project/
            outputs/
            src/
            ui/
                components/
                    report.py
    """

    return Path(__file__).resolve().parents[2]


def _resolve_path(
    value: Any,
) -> Path | None:
    """
    Resolve a workflow-generated path.

    Supports:
    - absolute paths
    - project-relative paths
    - Windows-style relative paths
    """

    if not value:
        return None

    try:

        raw_value = str(value).strip()

        if not raw_value:
            return None

        path = Path(raw_value)

        # --------------------------------------------
        # Absolute path
        # --------------------------------------------

        if path.is_absolute():

            if path.is_file():
                return path.resolve()

            return None

        # --------------------------------------------
        # Relative path
        # --------------------------------------------

        candidate = (
            _project_root()
            / path
        )

        if candidate.is_file():
            return candidate.resolve()

        # --------------------------------------------
        # Handle slash differences defensively
        # --------------------------------------------

        normalized = raw_value.replace(
            "\\",
            "/",
        )

        candidate = (
            _project_root()
            / Path(normalized)
        )

        if candidate.is_file():
            return candidate.resolve()

    except (
        TypeError,
        ValueError,
        OSError,
    ):
        return None

    return None


def _report_path(
    state: dict[str, Any],
) -> Path | None:
    """
    Resolve the generated report artifact.
    """

    return _resolve_path(
        state.get("report_path")
    )


def _report_stats(
    report: str,
) -> tuple[int, int, int]:
    """
    Calculate basic report statistics.
    """

    if not report:
        return 0, 0, 0

    words = len(
        report.split()
    )

    characters = len(
        report
    )

    sections = sum(
        1
        for line in report.splitlines()
        if line.strip().startswith("#")
    )

    return (
        words,
        characters,
        sections,
    )


def _chart_title(
    path: Path,
) -> str:
    """
    Convert a chart filename into a readable title.

    Example:
        CreditScore_vs_Age_scatter.png
        ->
        CreditScore vs Age scatter
    """

    title = (
        path.stem
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )

    if not title:
        return "Visualization"

    return title


# ============================================================
# HEADER
# ============================================================

def _render_header() -> None:

    render_html(
        """
        <div class="aa-page-header">

            <div class="aa-eyebrow">
                Analytical deliverable
            </div>

            <div class="aa-page-title">
                Report
            </div>

            <div class="aa-page-description">
                Review and export the analytical report produced
                from profiling, quality assessment, semantic
                interpretation, exploratory analysis and
                generated insights.
            </div>

        </div>
        """
    )


# ============================================================
# REPORT SUMMARY
# ============================================================

def _render_report_summary(
    state: dict[str, Any],
    report: str,
) -> None:

    words, characters, sections = (
        _report_stats(report)
    )

    insights = _safe_list(
        state.get("insights")
    )

    charts = _safe_list(
        state.get("chart_paths")
    )

    render_html(
        f"""
        <div class="aa-kpi-grid">

            <div class="aa-kpi">

                <div class="aa-kpi-label">
                    Report words
                </div>

                <div class="aa-kpi-value">
                    {words:,}
                </div>

                <div class="aa-kpi-detail">
                    Generated analytical narrative
                </div>

            </div>


            <div class="aa-kpi">

                <div class="aa-kpi-label">
                    Sections
                </div>

                <div class="aa-kpi-value">
                    {sections:,}
                </div>

                <div class="aa-kpi-detail">
                    Structured report sections
                </div>

            </div>


            <div class="aa-kpi">

                <div class="aa-kpi-label">
                    Findings
                </div>

                <div class="aa-kpi-value">
                    {len(insights):,}
                </div>

                <div class="aa-kpi-detail">
                    Generated analytical insights
                </div>

            </div>


            <div class="aa-kpi">

                <div class="aa-kpi-label">
                    Visuals
                </div>

                <div class="aa-kpi-value">
                    {len(charts):,}
                </div>

                <div class="aa-kpi-detail">
                    Charts produced by EDA
                </div>

            </div>

        </div>
        """
    )

    if characters:

        st.caption(
            f"Report contains "
            f"{characters:,} characters."
        )


# ============================================================
# KEY FINDINGS
# ============================================================

def _render_findings(
    state: dict[str, Any],
) -> None:

    insights = _safe_list(
        state.get("insights")
    )

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Findings included
            </div>

            <div class="aa-section-description">
                Primary analytical observations supporting
                the generated report.
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
                    No standalone insight statements were
                    recorded by the workflow.
                </div>

            </div>
            """
        )

        return

    for index, insight in enumerate(
        insights[:6],
        start=1,
    ):

        if isinstance(
            insight,
            dict,
        ):

            text = (
                insight.get("insight")
                or insight.get("text")
                or insight.get("message")
                or str(insight)
            )

        else:

            text = str(
                insight
            )

        render_html(
            f"""
            <div class="aa-insight">

                <div class="aa-insight-index">
                    {index:02d}
                </div>

                <div class="aa-insight-text">
                    {escape(text)}
                </div>

            </div>
            """
        )

    if len(insights) > 6:

        st.caption(
            f"+ {len(insights) - 6} "
            "additional findings are included "
            "in the analysis."
        )


# ============================================================
# VISUAL ANALYSIS
# ============================================================

def _render_charts(
    state: dict[str, Any],
) -> None:
    """
    Render chart files generated by the EDA executor.

    chart_paths may contain values such as:

        outputs/charts/Age_histogram.png

    or:

        outputs\\charts\\Age_histogram.png

    or absolute paths.
    """

    chart_paths = _safe_list(
        state.get("chart_paths")
    )

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Visual analysis
            </div>

            <div class="aa-section-description">
                Visualizations generated automatically during
                exploratory analysis of the active dataset.
            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # NO CHART REFERENCES
    # --------------------------------------------------------

    if not chart_paths:

        render_html(
            """
            <div class="aa-panel">

                <div class="aa-panel-title">
                    No visualizations available
                </div>

                <div class="aa-panel-description">
                    The analytical workflow did not generate
                    chart artifacts for this dataset.
                </div>

            </div>
            """
        )

        return

    # --------------------------------------------------------
    # RESOLVE FILES
    # --------------------------------------------------------

    valid_charts: list[Path] = []

    missing_charts: list[str] = []

    seen: set[str] = set()

    for chart_path in chart_paths:

        if not chart_path:
            continue

        raw_path = str(
            chart_path
        ).strip()

        if not raw_path:
            continue

        # Avoid rendering duplicates.
        if raw_path in seen:
            continue

        seen.add(
            raw_path
        )

        resolved = _resolve_path(
            raw_path
        )

        if resolved is None:

            missing_charts.append(
                raw_path
            )

            continue

        # Only render supported image files.
        if resolved.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:

            missing_charts.append(
                raw_path
            )

            continue

        valid_charts.append(
            resolved
        )

    # --------------------------------------------------------
    # REFERENCES EXIST BUT FILES DO NOT
    # --------------------------------------------------------

    if not valid_charts:

        st.warning(
            "The workflow generated chart references, "
            "but the corresponding image files could "
            "not be located."
        )

        if missing_charts:

            with st.expander(
                "View unresolved chart paths"
            ):

                st.caption(
                    "These paths were returned by the "
                    "analytics workflow but could not "
                    "be resolved."
                )

                for path in missing_charts:

                    st.code(
                        path,
                        language=None,
                    )

        return

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    chart_count = len(
        valid_charts
    )

    st.caption(
        f"{chart_count} visualization"
        f"{'s' if chart_count != 1 else ''} "
        "generated by the analytical workflow."
    )

    # --------------------------------------------------------
    # DISPLAY CHARTS
    # --------------------------------------------------------

    for start_index in range(
        0,
        chart_count,
        2,
    ):

        columns = st.columns(
            2,
            gap="medium",
        )

        batch = valid_charts[
            start_index:start_index + 2
        ]

        for column, path in zip(
            columns,
            batch,
        ):

            with column:

                title = _chart_title(
                    path
                )

                render_html(
                    f"""
                    <div class="aa-panel">

                        <div class="aa-panel-title">
                            {escape(title)}
                        </div>

                        <div class="aa-panel-description">
                            Generated exploratory visualization
                        </div>

                    </div>
                    """
                )

                try:

                    st.image(
                        str(path),
                        width="stretch",
                    )

                except Exception as error:

                    st.warning(
                        "Unable to display "
                        f"{path.name}."
                    )

                    with st.expander(
                        "Chart rendering error"
                    ):

                        st.code(
                            str(error),
                            language=None,
                        )

    # --------------------------------------------------------
    # MISSING CHARTS
    # --------------------------------------------------------

    if missing_charts:

        with st.expander(
            f"{len(missing_charts)} chart "
            "reference(s) could not be loaded"
        ):

            st.caption(
                "The remaining analysis is unaffected."
            )

            for path in missing_charts:

                st.code(
                    path,
                    language=None,
                )


# ============================================================
# REPORT PREVIEW
# ============================================================

def _render_report_preview(
    report: str,
) -> None:

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Report preview
            </div>

            <div class="aa-section-description">
                Final narrative generated by the reporting agent.
            </div>

        </div>
        """
    )

    if not report:

        render_html(
            """
            <div class="aa-panel">

                <div class="aa-panel-title">
                    Report unavailable
                </div>

                <div class="aa-panel-description">
                    The workflow completed without returning
                    a final report body.
                </div>

            </div>
            """
        )

        return

    # Do not try to wrap Streamlit Markdown inside an HTML
    # container opened by another Streamlit call.
    #
    # Streamlit renders each call independently, so opening
    # <div> with render_html(), then calling st.markdown(),
    # then closing the div can produce broken markup.

    st.markdown(
        report
    )


# ============================================================
# EXPORT
# ============================================================

def _render_export(
    state: dict[str, Any],
    report: str,
) -> None:

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Export
            </div>

            <div class="aa-section-description">
                Save the generated analytical report for
                documentation or further review.
            </div>

        </div>
        """
    )

    report_path = _report_path(
        state
    )

    col1, col2 = st.columns(
        2,
        gap="medium",
    )

    # --------------------------------------------------------
    # MARKDOWN EXPORT
    # --------------------------------------------------------

    with col1:

        if report:

            st.download_button(
                label="Download Markdown report",
                data=report,
                file_name="analytics_report.md",
                mime="text/markdown",
                width="stretch",
            )

        else:

            st.button(
                "Markdown unavailable",
                disabled=True,
                width="stretch",
            )

    # --------------------------------------------------------
    # GENERATED ARTIFACT EXPORT
    # --------------------------------------------------------

    with col2:

        if report_path:

            try:

                report_bytes = (
                    report_path.read_bytes()
                )

                suffix = (
                    report_path
                    .suffix
                    .lower()
                )

                mime_map = {
                    ".md":
                        "text/markdown",

                    ".txt":
                        "text/plain",

                    ".html":
                        "text/html",

                    ".pdf":
                        "application/pdf",
                }

                mime = mime_map.get(
                    suffix,
                    "application/octet-stream",
                )

                st.download_button(
                    label="Download generated file",
                    data=report_bytes,
                    file_name=report_path.name,
                    mime=mime,
                    width="stretch",
                )

            except OSError as error:

                st.button(
                    "Generated file unavailable",
                    disabled=True,
                    width="stretch",
                )

                with st.expander(
                    "Artifact error"
                ):

                    st.code(
                        str(error),
                        language=None,
                    )

        else:

            st.button(
                "No generated file",
                disabled=True,
                width="stretch",
            )

    if report_path:

        st.caption(
            f"Generated artifact: "
            f"{report_path.name}"
        )


# ============================================================
# ANALYSIS PROVENANCE
# ============================================================

def _render_provenance(
    state: dict[str, Any],
) -> None:

    execution_report = _safe_dict(
        state.get(
            "execution_report"
        )
    )

    cleaning_report = _safe_dict(
        state.get(
            "cleaning_report"
        )
    )

    semantic_metadata = _safe_dict(
        state.get(
            "semantic_metadata"
        )
    )

    errors = _safe_list(
        state.get(
            "errors"
        )
    )

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Report provenance
            </div>

            <div class="aa-section-description">
                Supporting workflow metadata retained for
                transparency and debugging.
            </div>

        </div>
        """
    )

    with st.expander(
        "View report provenance"
    ):

        tab1, tab2, tab3 = st.tabs(
            [
                "Execution",
                "Preprocessing",
                "Semantic context",
            ]
        )

        # ----------------------------------------------------
        # EXECUTION
        # ----------------------------------------------------

        with tab1:

            if execution_report:

                st.json(
                    execution_report
                )

            else:

                st.caption(
                    "No execution report available."
                )

            if errors:

                st.markdown(
                    "**Workflow errors**"
                )

                st.json(
                    errors
                )

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        with tab2:

            if cleaning_report:

                st.json(
                    cleaning_report
                )

            else:

                st.caption(
                    "No preprocessing report available."
                )

        # ----------------------------------------------------
        # SEMANTICS
        # ----------------------------------------------------

        with tab3:

            if semantic_metadata:

                st.json(
                    semantic_metadata
                )

            else:

                st.caption(
                    "No semantic execution metadata available."
                )


# ============================================================
# PUBLIC COMPONENT
# ============================================================

def render_report(
    state: dict[str, Any],
) -> None:
    """
    Render the analytical Report workspace.
    """

    # --------------------------------------------------------
    # VALIDATE STATE
    # --------------------------------------------------------

    if not isinstance(
        state,
        dict,
    ):

        st.error(
            "The analytics workflow state is unavailable."
        )

        return

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = _report_text(
        state
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    _render_header()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    _render_report_summary(
        state=state,
        report=report,
    )

    # --------------------------------------------------------
    # FINDINGS
    # --------------------------------------------------------

    _render_findings(
        state=state,
    )

    # --------------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------------

    _render_charts(
        state=state,
    )

    # --------------------------------------------------------
    # REPORT BODY
    # --------------------------------------------------------

    _render_report_preview(
        report=report,
    )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    _render_export(
        state=state,
        report=report,
    )

    # --------------------------------------------------------
    # PROVENANCE
    # --------------------------------------------------------

    _render_provenance(
        state=state,
    )
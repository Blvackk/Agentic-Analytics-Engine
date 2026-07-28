# src/agents/report_generator.py

from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path(
    "outputs/reports"
)


def _format_section(
    title: str,
    content: str,
) -> str:
    """
    Create a Markdown report section.
    """

    return (
        f"\n## {title}\n\n"
        f"{content.strip()}\n"
    )


def _format_dictionary(
    data: dict[str, Any],
) -> str:
    """
    Convert a dictionary into readable Markdown.
    """

    if not data:
        return "No information available."

    lines = []

    for key, value in data.items():

        readable_key = (
            str(key)
            .replace("_", " ")
            .title()
        )

        lines.append(
            f"- **{readable_key}:** {value}"
        )

    return "\n".join(lines)


def generate_report(
    profile: dict[str, Any],
    quality_report: dict[str, Any],
    cleaning_report: dict[str, Any] | None = None,
    cleaning_validation: dict[str, Any] | None = None,
    analysis_results: dict[str, Any] | None = None,
    semantic_analysis: dict[str, Any] | None = None,
    target_analysis: dict[str, Any] | None = None,
    execution_report: dict[str, Any] | None = None,
    chart_paths: list[str] | None = None,
    insights: dict[str, Any] | None = None,
) -> str:
    """
    Build a grounded Markdown analytics report from
    workflow results.

    The report is constructed from results already produced
    by Python tools and the insight generator.

    Returns
    -------
    str
        Complete Markdown report.
    """

    profile = profile or {}
    quality_report = quality_report or {}
    cleaning_report = cleaning_report or {}
    cleaning_validation = cleaning_validation or {}
    analysis_results = analysis_results or {}
    semantic_analysis = semantic_analysis or {}
    target_analysis = target_analysis or {}
    execution_report = execution_report or {}
    chart_paths = chart_paths or []
    insights = insights or {}

    sections = []

    # ==================================================
    # TITLE
    # ==================================================

    sections.append(
        "# Agentic Analytics Report\n"
    )

    # ==================================================
    # EXECUTIVE SUMMARY
    # ==================================================

    summary = insights.get(
        "summary",
        ""
    )

    if not summary:

        summary = (
            "The dataset was processed through the "
            "Agentic Analytics Engine."
        )

    sections.append(
        _format_section(
            "Executive Summary",
            summary,
        )
    )

    # ==================================================
    # DATASET OVERVIEW
    # ==================================================

    rows = profile.get(
        "rows",
        "Unknown",
    )

    columns = profile.get(
        "columns",
        "Unknown",
    )

    duplicate_rows = profile.get(
        "duplicate_rows",
        "Unknown",
    )

    dataset_overview = (
        f"- **Original rows:** {rows}\n"
        f"- **Columns:** {columns}\n"
        f"- **Duplicate rows detected:** "
        f"{duplicate_rows}"
    )

    sections.append(
        _format_section(
            "Dataset Overview",
            dataset_overview,
        )
    )

    # ==================================================
    # DATA QUALITY
    # ==================================================

    quality_issues = quality_report.get(
        "issues",
        []
    )

    quality_lines = [
        (
            f"- **Total quality issues:** "
            f"{quality_report.get('total_issues', 0)}"
        )
    ]

    if quality_issues:

        quality_lines.append(
            "\n### Detected Issues\n"
        )

        for issue in quality_issues:

            issue_type = issue.get(
                "issue_type",
                "unknown",
            )

            column = issue.get(
                "column"
            )

            count = issue.get(
                "count",
                "Unknown",
            )

            severity = issue.get(
                "severity",
                "Unknown",
            )

            if column:

                quality_lines.append(
                    f"- `{column}`: {issue_type} "
                    f"(count={count}, severity={severity})"
                )

            else:

                quality_lines.append(
                    f"- {issue_type} "
                    f"(count={count}, severity={severity})"
                )

    else:

        quality_lines.append(
            "- No supported quality issues detected."
        )

    sections.append(
        _format_section(
            "Data Quality",
            "\n".join(
                quality_lines
            ),
        )
    )

    # ==================================================
    # CLEANING
    # ==================================================

    if cleaning_report:

        cleaning_lines = [
            (
                f"- **Original rows:** "
                f"{cleaning_report.get('original_rows')}"
            ),
            (
                f"- **Cleaned rows:** "
                f"{cleaning_report.get('cleaned_rows')}"
            ),
            (
                f"- **Rows removed:** "
                f"{cleaning_report.get('rows_removed')}"
            ),
            (
                f"- **Original missing values:** "
                f"{cleaning_report.get('original_missing_values')}"
            ),
            (
                f"- **Remaining missing values:** "
                f"{cleaning_report.get('cleaned_missing_values')}"
            ),
            (
                f"- **Original duplicates:** "
                f"{cleaning_report.get('original_duplicates')}"
            ),
            (
                f"- **Remaining duplicates:** "
                f"{cleaning_report.get('cleaned_duplicates')}"
            ),
        ]

        if cleaning_validation:

            cleaning_lines.extend(
                [
                    (
                        f"- **Validation passed:** "
                        f"{cleaning_validation.get('validation_passed')}"
                    ),
                    (
                        f"- **Row retention:** "
                        f"{cleaning_validation.get('row_retention_percentage')}%"
                    ),
                ]
            )

        sections.append(
            _format_section(
                "Data Cleaning",
                "\n".join(
                    cleaning_lines
                ),
            )
        )

    else:

        sections.append(
            _format_section(
                "Data Cleaning",
                (
                    "No automatic cleaning operations "
                    "were required."
                ),
            )
        )

    # ==================================================
    # SEMANTIC ANALYSIS
    # ==================================================

    identifiers = semantic_analysis.get(
        "identifier_columns",
        []
    )

    targets = semantic_analysis.get(
        "target_candidates",
        []
    )

    features = semantic_analysis.get(
        "feature_columns",
        []
    )

    semantic_content = (
        f"- **Identifier columns:** "
        f"{', '.join(identifiers) if identifiers else 'None'}\n"
        f"- **Target candidates:** "
        f"{', '.join(targets) if targets else 'None'}\n"
        f"- **Feature columns:** "
        f"{', '.join(features) if features else 'None'}"
    )

    sections.append(
        _format_section(
            "Semantic Analysis",
            semantic_content,
        )
    )

    # ==================================================
    # STATISTICAL ANALYSIS
    # ==================================================

    numerical_summary = analysis_results.get(
        "numerical_summary",
        {}
    )

    categorical_summary = analysis_results.get(
        "categorical_summary",
        {}
    )

    correlation_matrix = analysis_results.get(
        "correlation_matrix",
        {}
    )

    statistics_lines = []

    # --------------------------------------------------
    # NUMERICAL
    # --------------------------------------------------

    if numerical_summary:

        statistics_lines.append(
            "### Numerical Summary\n"
        )

        for column, statistics in (
            numerical_summary.items()
        ):

            statistics_lines.append(
                f"#### {column}\n"
            )

            statistics_lines.append(
                _format_dictionary(
                    statistics
                )
            )

            statistics_lines.append(
                ""
            )

    # --------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------

    if categorical_summary:

        statistics_lines.append(
            "### Categorical Summary\n"
        )

        for column, statistics in (
            categorical_summary.items()
        ):

            statistics_lines.append(
                f"#### {column}\n"
            )

            statistics_lines.append(
                _format_dictionary(
                    statistics
                )
            )

            statistics_lines.append(
                ""
            )

    # --------------------------------------------------
    # CORRELATIONS
    # --------------------------------------------------

    if correlation_matrix:

        statistics_lines.append(
            "### Correlation Matrix\n"
        )

        for column, correlations in (
            correlation_matrix.items()
        ):

            statistics_lines.append(
                f"- **{column}:** "
                f"{correlations}"
            )

    if not statistics_lines:

        statistics_lines.append(
            "No statistical results available."
        )

    sections.append(
        _format_section(
            "Statistical Analysis",
            "\n".join(
                statistics_lines
            ),
        )
    )

    # ==================================================
    # TARGET ANALYSIS
    # ==================================================

    if target_analysis:

        target_lines = []

        for target, result in (
            target_analysis.items()
        ):

            target_lines.append(
                f"### {target}\n"
            )

            target_lines.append(
                f"- **Type:** "
                f"{result.get('target_type')}"
            )

            distribution = result.get(
                "distribution"
            )

            if distribution is not None:

                target_lines.append(
                    f"- **Distribution:** "
                    f"{distribution}"
                )

            target_lines.append(
                ""
            )

        sections.append(
            _format_section(
                "Target Analysis",
                "\n".join(
                    target_lines
                ),
            )
        )

    # ==================================================
    # AI INSIGHTS
    # ==================================================

    key_insights = insights.get(
        "key_insights",
        []
    )

    insight_lines = []

    if key_insights:

        for index, insight in enumerate(
            key_insights,
            start=1,
        ):

            insight_lines.append(
                f"### {index}. "
                f"{insight.get('title', 'Insight')}\n"
            )

            insight_lines.append(
                insight.get(
                    "insight",
                    "",
                )
            )

            evidence = insight.get(
                "evidence"
            )

            if evidence:

                insight_lines.append(
                    f"\n**Evidence:** {evidence}"
                )

            importance = insight.get(
                "importance"
            )

            if importance:

                insight_lines.append(
                    f"\n**Importance:** {importance}"
                )

            insight_lines.append(
                ""
            )

    else:

        insight_lines.append(
            "No AI-generated insights available."
        )

    sections.append(
        _format_section(
            "Key Insights",
            "\n".join(
                insight_lines
            ),
        )
    )

    # ==================================================
    # TARGET INSIGHTS
    # ==================================================

    target_insights = insights.get(
        "target_insights",
        []
    )

    if target_insights:

        target_insight_lines = []

        for insight in target_insights:

            target = insight.get(
                "target",
                "Unknown target",
            )

            target_insight_lines.append(
                f"### {target}\n"
            )

            target_insight_lines.append(
                insight.get(
                    "insight",
                    "",
                )
            )

            evidence = insight.get(
                "evidence"
            )

            if evidence:

                target_insight_lines.append(
                    f"\n**Evidence:** {evidence}"
                )

            target_insight_lines.append(
                ""
            )

        sections.append(
            _format_section(
                "Target Insights",
                "\n".join(
                    target_insight_lines
                ),
            )
        )

    # ==================================================
    # DATA CAUTIONS
    # ==================================================

    cautions = insights.get(
        "data_cautions",
        []
    )

    if cautions:

        caution_content = "\n".join(
            f"- {caution}"
            for caution in cautions
        )

    else:

        caution_content = (
            "No additional analytical cautions "
            "were generated."
        )

    sections.append(
        _format_section(
            "Data Cautions",
            caution_content,
        )
    )

    # ==================================================
    # EDA EXECUTION
    # ==================================================

    execution_content = (
        f"- **Tasks planned:** "
        f"{execution_report.get('total_planned_tasks', 0)}\n"
        f"- **Tasks executed:** "
        f"{execution_report.get('executed_tasks', 0)}\n"
        f"- **Tasks skipped:** "
        f"{execution_report.get('skipped_tasks', 0)}\n"
        f"- **Tasks failed:** "
        f"{execution_report.get('failed_tasks', 0)}\n"
        f"- **Charts generated:** "
        f"{execution_report.get('charts_generated', 0)}"
    )

    sections.append(
        _format_section(
            "EDA Execution",
            execution_content,
        )
    )

    # ==================================================
    # GENERATED VISUALISATIONS
    # ==================================================

    if chart_paths:

        chart_lines = []

        for path in chart_paths:

            chart_lines.append(
                f"- `{path}`"
            )

        chart_content = "\n".join(
            chart_lines
        )

    else:

        chart_content = (
            "No visualisations were generated."
        )

    sections.append(
        _format_section(
            "Generated Visualisations",
            chart_content,
        )
    )

    # ==================================================
    # BUILD REPORT
    # ==================================================

    report = "\n".join(
        sections
    )

    return report


def save_report(
    report: str,
    filename: str = "analytics_report.md",
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> str:
    """
    Save a generated Markdown report to disk.

    Returns
    -------
    str
        Path of the saved report.
    """

    if not isinstance(
        report,
        str
    ):

        raise TypeError(
            "report must be a string."
        )

    if not report.strip():

        raise ValueError(
            "Report cannot be empty."
        )

    output_directory = Path(
        output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / filename
    )

    output_path.write_text(
        report,
        encoding="utf-8",
    )

    return str(
        output_path
    )
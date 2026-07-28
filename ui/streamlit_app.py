# ui/streamlit_app.py

import sys
from pathlib import Path


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ==========================================================
# IMPORTS
# ==========================================================

import pandas as pd
import streamlit as st

from src.agents.graph import build_graph


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Agentic Analytics Engine",
    page_icon="📊",
    layout="wide",
)


# ==========================================================
# APPLICATION HEADER
# ==========================================================

st.title(
    "📊 Agentic Analytics Engine"
)

st.write(
    "Upload a CSV dataset and let the analytics engine "
    "automatically inspect, clean, analyse, visualise, "
    "and generate AI-powered insights."
)

st.divider()


# ==========================================================
# FILE UPLOAD
# ==========================================================

st.subheader(
    "Upload Dataset"
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
)


# ==========================================================
# HANDLE UPLOADED DATASET
# ==========================================================

if uploaded_file is not None:

    # ------------------------------------------------------
    # READ CSV
    # ------------------------------------------------------

    try:

        dataframe = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            f"Could not read the CSV file: {error}"
        )

        st.stop()

    # ------------------------------------------------------
    # VALIDATE DATASET
    # ------------------------------------------------------

    if dataframe.empty:

        st.error(
            "The uploaded CSV does not contain any rows."
        )

        st.stop()

    if dataframe.shape[1] == 0:

        st.error(
            "The uploaded CSV does not contain any columns."
        )

        st.stop()

    # ------------------------------------------------------
    # UPLOAD SUCCESS
    # ------------------------------------------------------

    st.success(
        "Dataset uploaded successfully."
    )

    # ------------------------------------------------------
    # DATASET INFORMATION
    # ------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            dataframe.shape[0],
        )

    with col2:

        st.metric(
            "Columns",
            dataframe.shape[1],
        )

    with col3:

        st.metric(
            "Missing Values",
            int(
                dataframe
                .isna()
                .sum()
                .sum()
            ),
        )

    # ------------------------------------------------------
    # DATASET PREVIEW
    # ------------------------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        dataframe.head(20),
        use_container_width=True,
    )

    # ======================================================
    # RUN ANALYSIS
    # ======================================================

    if st.button(
        "Run Analysis",
        type="primary",
    ):

        # --------------------------------------------------
        # CREATE UPLOAD DIRECTORY
        # --------------------------------------------------

        upload_directory = (
            PROJECT_ROOT
            / "data"
            / "uploads"
        )

        upload_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # SAFE FILE NAME
        # --------------------------------------------------

        safe_filename = Path(
            uploaded_file.name
        ).name

        dataset_path = (
            upload_directory
            / safe_filename
        )

        # --------------------------------------------------
        # SAVE DATASET
        # --------------------------------------------------

        try:

            dataframe.to_csv(
                dataset_path,
                index=False,
            )

        except Exception as error:

            st.error(
                "Could not save the uploaded dataset."
            )

            st.exception(
                error
            )

            st.stop()

        # --------------------------------------------------
        # BUILD LANGGRAPH WORKFLOW
        # --------------------------------------------------

        try:

            graph = build_graph()

        except Exception as error:

            st.error(
                "Could not build the LangGraph workflow."
            )

            st.exception(
                error
            )

            st.stop()

        # --------------------------------------------------
        # INITIAL AGENT STATE
        # --------------------------------------------------

        initial_state = {
            "dataset_path":
                str(dataset_path)
        }

        # --------------------------------------------------
        # RUN AGENTIC WORKFLOW
        # --------------------------------------------------

        with st.spinner(
            "Running Agentic Analytics workflow..."
        ):

            try:

                final_state = graph.invoke(
                    initial_state
                )

            except Exception as error:

                st.error(
                    "The analytics workflow failed."
                )

                st.exception(
                    error
                )

                st.stop()

        # --------------------------------------------------
        # STORE RESULT
        # --------------------------------------------------

        st.session_state[
            "analysis_result"
        ] = final_state

        st.session_state[
            "analysed_dataset"
        ] = safe_filename

        # --------------------------------------------------
        # SUCCESS
        # --------------------------------------------------

        st.success(
            "Analysis completed successfully!"
        )


# ==========================================================
# ANALYSIS RESULTS
# ==========================================================

if "analysis_result" in st.session_state:

    result = st.session_state[
        "analysis_result"
    ]

    st.divider()

    st.header(
        "Analysis Results"
    )

    analysed_dataset = st.session_state.get(
        "analysed_dataset",
        "Dataset",
    )

    st.caption(
        f"Analysed dataset: {analysed_dataset}"
    )

    # ======================================================
    # EXTRACT WORKFLOW RESULTS
    # ======================================================

    profile = result.get(
        "profile",
        {}
    )

    quality_report = result.get(
        "quality_report",
        {}
    )

    cleaning_report = result.get(
        "cleaning_report",
        {}
    )

    cleaning_validation = result.get(
        "cleaning_validation",
        {}
    )

    analysis_results = result.get(
        "analysis_results",
        {}
    )

    semantic_analysis = result.get(
        "semantic_analysis",
        {}
    )

    execution_report = result.get(
        "execution_report",
        {}
    )

    target_analysis = result.get(
        "target_analysis",
        {}
    )

    insights = result.get(
        "insights",
        {}
    )

    chart_paths = result.get(
        "chart_paths",
        []
    )

    final_report = result.get(
        "final_report",
        ""
    )

    report_path = result.get(
        "report_path",
        ""
    )

    numerical_summary = result.get(
        "numerical_summary",
        {}
    )

    categorical_summary = result.get(
        "categorical_summary",
        {}
    )

    correlation_matrix = result.get(
        "correlation_matrix",
        {}
    )

    # ======================================================
    # TOP-LEVEL METRICS
    # ======================================================

    st.subheader(
        "Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Original Rows",
            profile.get(
                "rows",
                0,
            ),
        )

    with col2:

        st.metric(
            "Columns",
            profile.get(
                "columns",
                0,
            ),
        )

    with col3:

        st.metric(
            "Quality Issues",
            quality_report.get(
                "total_issues",
                0,
            ),
        )

    with col4:

        st.metric(
            "Rows After Cleaning",
            cleaning_report.get(
                "cleaned_rows",
                profile.get(
                    "rows",
                    0,
                ),
            ),
        )

    # ======================================================
    # SEMANTIC SUMMARY
    # ======================================================

    st.subheader(
        "Semantic Analysis"
    )

    identifiers = result.get(
        "identifier_columns",
        semantic_analysis.get(
            "identifier_columns",
            [],
        ),
    )

    targets = result.get(
        "target_candidates",
        semantic_analysis.get(
            "target_candidates",
            [],
        ),
    )

    features = result.get(
        "feature_columns",
        semantic_analysis.get(
            "feature_columns",
            [],
        ),
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            "**Identifier Columns**"
        )

        if identifiers:

            for column in identifiers:

                st.write(
                    f"• {column}"
                )

        else:

            st.write(
                "None detected"
            )

    with col2:

        st.markdown(
            "**Feature Columns**"
        )

        if features:

            for column in features:

                st.write(
                    f"• {column}"
                )

        else:

            st.write(
                "None detected"
            )

    with col3:

        st.markdown(
            "**Target Candidates**"
        )

        if targets:

            for column in targets:

                st.write(
                    f"• {column}"
                )

        else:

            st.write(
                "None detected"
            )

    # ======================================================
    # EDA EXECUTION SUMMARY
    # ======================================================

    st.subheader(
        "EDA Execution"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Planned Tasks",
            execution_report.get(
                "total_planned_tasks",
                0,
            ),
        )

    with col2:

        st.metric(
            "Executed",
            execution_report.get(
                "executed_tasks",
                0,
            ),
        )

    with col3:

        st.metric(
            "Charts",
            execution_report.get(
                "charts_generated",
                0,
            ),
        )

    with col4:

        failed_tasks = execution_report.get(
            "failed_tasks",
            0,
        )

        st.metric(
            "Failed",
            failed_tasks,
        )

    if failed_tasks:

        st.warning(
            f"{failed_tasks} EDA task(s) failed. "
            "Check the workflow output for details."
        )

    # ======================================================
    # RESULTS TABS
    # ======================================================

    (
        overview_tab,
        quality_tab,
        stats_tab,
        charts_tab,
        insights_tab,
        report_tab,
    ) = st.tabs(
        [
            "Overview",
            "Data Quality",
            "Statistics",
            "Visualizations",
            "AI Insights",
            "Report",
        ]
    )

    # ======================================================
    # OVERVIEW TAB
    # ======================================================

    with overview_tab:

        st.subheader(
            "Workflow Overview"
        )

        st.write(
            "The Agentic Analytics workflow completed "
            "for the uploaded dataset."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Rows Analysed",
                analysis_results.get(
                    "rows_analyzed",
                    0,
                ),
            )

        with col2:

            st.metric(
                "EDA Tasks Executed",
                execution_report.get(
                    "executed_tasks",
                    0,
                ),
            )

        with col3:

            st.metric(
                "Charts Generated",
                execution_report.get(
                    "charts_generated",
                    0,
                ),
            )

        # --------------------------------------------------
        # TARGET ANALYSIS
        # --------------------------------------------------

        st.markdown(
            "### Target Analysis"
        )

        if target_analysis:

            for target, information in (
                target_analysis.items()
            ):

                st.markdown(
                    f"#### {target}"
                )

                if not isinstance(
                    information,
                    dict,
                ):

                    st.write(
                        information
                    )

                    continue

                target_type = information.get(
                    "target_type",
                    "Unknown",
                )

                st.write(
                    f"**Target Type:** {target_type}"
                )

                distribution = information.get(
                    "distribution"
                )

                if distribution:

                    st.write(
                        "**Distribution**"
                    )

                    try:

                        distribution_df = pd.DataFrame(
                            {
                                "Category":
                                    list(
                                        distribution.keys()
                                    ),

                                "Count":
                                    list(
                                        distribution.values()
                                    ),
                            }
                        )

                        st.dataframe(
                            distribution_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    except Exception:

                        st.json(
                            distribution
                        )

        else:

            st.info(
                "No target candidate was analysed."
            )

    # ======================================================
    # DATA QUALITY TAB
    # ======================================================

    with quality_tab:

        st.subheader(
            "Data Quality Analysis"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Quality Issues",
                quality_report.get(
                    "total_issues",
                    0,
                ),
            )

        with col2:

            st.metric(
                "Rows Removed",
                cleaning_report.get(
                    "rows_removed",
                    0,
                ),
            )

        with col3:

            st.metric(
                "Cleaned Rows",
                cleaning_report.get(
                    "cleaned_rows",
                    profile.get(
                        "rows",
                        0,
                    ),
                ),
            )

        # --------------------------------------------------
        # DETECTED ISSUES
        # --------------------------------------------------

        st.markdown(
            "### Detected Issues"
        )

        issues = quality_report.get(
            "issues",
            []
        )

        if issues:

            for issue in issues:

                if isinstance(
                    issue,
                    dict,
                ):

                    column = issue.get(
                        "column"
                    )

                    issue_type = issue.get(
                        "issue",
                        issue.get(
                            "type",
                            "Unknown issue",
                        ),
                    )

                    count = issue.get(
                        "count",
                        "N/A",
                    )

                    severity = issue.get(
                        "severity"
                    )

                    if column:

                        message = (
                            f"{column}: "
                            f"{issue_type} "
                            f"(count: {count})"
                        )

                    else:

                        message = (
                            f"{issue_type} "
                            f"(count: {count})"
                        )

                    if severity:

                        message += (
                            f" | severity: "
                            f"{severity}"
                        )

                    st.warning(
                        message
                    )

                else:

                    st.warning(
                        str(issue)
                    )

        else:

            st.success(
                "No data quality issues detected."
            )

        # --------------------------------------------------
        # CLEANING REPORT
        # --------------------------------------------------

        st.markdown(
            "### Cleaning Report"
        )

        if cleaning_report:

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Original Rows",
                    cleaning_report.get(
                        "original_rows",
                        profile.get(
                            "rows",
                            0,
                        ),
                    ),
                )

                st.metric(
                    "Rows Removed",
                    cleaning_report.get(
                        "rows_removed",
                        0,
                    ),
                )

                st.metric(
                    "Original Missing Values",
                    cleaning_report.get(
                        "original_missing_values",
                        0,
                    ),
                )

            with col2:

                st.metric(
                    "Cleaned Rows",
                    cleaning_report.get(
                        "cleaned_rows",
                        profile.get(
                            "rows",
                            0,
                        ),
                    ),
                )

                st.metric(
                    "Remaining Missing Values",
                    cleaning_report.get(
                        "remaining_missing_values",
                        0,
                    ),
                )

                st.metric(
                    "Remaining Duplicates",
                    cleaning_report.get(
                        "remaining_duplicates",
                        0,
                    ),
                )

            with st.expander(
                "Full Cleaning Report"
            ):

                st.json(
                    cleaning_report
                )

        else:

            st.info(
                "Automatic cleaning was not required."
            )

        # --------------------------------------------------
        # CLEANING VALIDATION
        # --------------------------------------------------

        st.markdown(
            "### Cleaning Validation"
        )

        if cleaning_validation:

            validation_passed = (
                cleaning_validation.get(
                    "validation_passed"
                )
            )

            if validation_passed is True:

                st.success(
                    "Cleaning validation passed."
                )

            elif validation_passed is False:

                st.error(
                    "Cleaning validation failed."
                )

            st.json(
                cleaning_validation
            )

        else:

            st.info(
                "No cleaning validation was required."
            )

    # ======================================================
    # STATISTICS TAB
    # ======================================================

    with stats_tab:

        st.subheader(
            "Statistical Analysis"
        )

        # --------------------------------------------------
        # NUMERICAL SUMMARY
        # --------------------------------------------------

        st.markdown(
            "### Numerical Summary"
        )

        if numerical_summary:

            try:

                numerical_df = pd.DataFrame(
                    numerical_summary
                ).T

                st.dataframe(
                    numerical_df,
                    use_container_width=True,
                )

            except Exception as error:

                st.warning(
                    "Could not convert the numerical "
                    f"summary into a table: {error}"
                )

                st.json(
                    numerical_summary
                )

        else:

            st.info(
                "No numerical columns were analysed."
            )

        # --------------------------------------------------
        # CATEGORICAL SUMMARY
        # --------------------------------------------------

        st.markdown(
            "### Categorical Summary"
        )

        if categorical_summary:

            categorical_rows = []

            for column, information in (
                categorical_summary.items()
            ):

                if not isinstance(
                    information,
                    dict,
                ):

                    continue

                categorical_rows.append(
                    {
                        "Column":
                            column,

                        "Unique Values":
                            information.get(
                                "unique_values"
                            ),

                        "Most Frequent":
                            information.get(
                                "most_frequent"
                            ),

                        "Top Values":
                            str(
                                information.get(
                                    "top_values",
                                    {},
                                )
                            ),
                    }
                )

            if categorical_rows:

                categorical_df = pd.DataFrame(
                    categorical_rows
                )

                st.dataframe(
                    categorical_df,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.json(
                    categorical_summary
                )

        else:

            st.info(
                "No categorical columns were analysed."
            )

        # --------------------------------------------------
        # CORRELATION MATRIX
        # --------------------------------------------------

        st.markdown(
            "### Correlation Matrix"
        )

        if correlation_matrix:

            try:

                correlation_df = pd.DataFrame(
                    correlation_matrix
                )

                st.dataframe(
                    correlation_df.round(3),
                    use_container_width=True,
                )

            except Exception as error:

                st.warning(
                    "Could not convert the correlation "
                    f"matrix into a table: {error}"
                )

                st.json(
                    correlation_matrix
                )

        else:

            st.info(
                "A correlation matrix could not "
                "be generated."
            )

    # ======================================================
    # VISUALIZATIONS TAB
    # ======================================================

    with charts_tab:

        st.subheader(
            "Generated Visualizations"
        )

        if chart_paths:

            st.write(
                f"{len(chart_paths)} visualizations "
                "were generated automatically."
            )

            # ----------------------------------------------
            # RESOLVE CHART PATHS
            # ----------------------------------------------

            valid_charts = []
            missing_charts = []

            for chart_path in chart_paths:

                path = Path(
                    chart_path
                )

                if not path.is_absolute():

                    path = (
                        PROJECT_ROOT
                        / path
                    )

                if path.exists():

                    valid_charts.append(
                        path
                    )

                else:

                    missing_charts.append(
                        str(chart_path)
                    )

            # ----------------------------------------------
            # DISPLAY CHARTS
            # ----------------------------------------------

            if valid_charts:

                for index in range(
                    0,
                    len(valid_charts),
                    2,
                ):

                    chart_columns = st.columns(
                        2
                    )

                    # --------------------------------------
                    # LEFT CHART
                    # --------------------------------------

                    left_chart = valid_charts[
                        index
                    ]

                    with chart_columns[0]:

                        left_title = (
                            left_chart
                            .stem
                            .replace(
                                "_",
                                " ",
                            )
                            .title()
                        )

                        st.markdown(
                            f"#### {left_title}"
                        )

                        st.image(
                            str(left_chart),
                            use_container_width=True,
                        )

                    # --------------------------------------
                    # RIGHT CHART
                    # --------------------------------------

                    if (
                        index + 1
                        < len(valid_charts)
                    ):

                        right_chart = valid_charts[
                            index + 1
                        ]

                        with chart_columns[1]:

                            right_title = (
                                right_chart
                                .stem
                                .replace(
                                    "_",
                                    " ",
                                )
                                .title()
                            )

                            st.markdown(
                                f"#### {right_title}"
                            )

                            st.image(
                                str(right_chart),
                                use_container_width=True,
                            )

            else:

                st.error(
                    "The workflow reported generated "
                    "charts, but none of the files "
                    "could be found."
                )

            # ----------------------------------------------
            # MISSING CHART FILES
            # ----------------------------------------------

            if missing_charts:

                with st.expander(
                    "Missing Chart Files"
                ):

                    for missing_chart in (
                        missing_charts
                    ):

                        st.code(
                            missing_chart
                        )

        else:

            st.info(
                "No visualizations were generated "
                "for this dataset."
            )

    # ======================================================
    # AI INSIGHTS TAB
    # ======================================================

    with insights_tab:

        st.subheader(
            "AI-Generated Insights"
        )

        if insights and isinstance(
            insights,
            dict,
        ):

            # ----------------------------------------------
            # OVERALL SUMMARY
            # ----------------------------------------------

            st.markdown(
                "### Overall Summary"
            )

            summary = insights.get(
                "summary",
                "",
            )

            if summary:

                st.write(
                    summary
                )

            else:

                st.info(
                    "No overall summary was generated."
                )

            # ----------------------------------------------
            # KEY INSIGHTS
            # ----------------------------------------------

            st.markdown(
                "### Key Insights"
            )

            key_insights = insights.get(
                "key_insights",
                [],
            )

            if key_insights:

                for index, insight in enumerate(
                    key_insights,
                    start=1,
                ):

                    if not isinstance(
                        insight,
                        dict,
                    ):

                        st.write(
                            insight
                        )

                        continue

                    title = insight.get(
                        "title",
                        f"Insight {index}",
                    )

                    with st.expander(
                        f"{index}. {title}",
                        expanded=True,
                    ):

                        insight_text = insight.get(
                            "insight",
                            "",
                        )

                        if insight_text:

                            st.write(
                                insight_text
                            )

                        evidence = insight.get(
                            "evidence"
                        )

                        if evidence:

                            st.markdown(
                                f"**Evidence:** "
                                f"{evidence}"
                            )

                        importance = insight.get(
                            "importance"
                        )

                        if importance:

                            st.markdown(
                                f"**Importance:** "
                                f"{importance}"
                            )

            else:

                st.info(
                    "No key insights were generated."
                )

            # ----------------------------------------------
            # TARGET INSIGHTS
            # ----------------------------------------------

            st.markdown(
                "### Target Insights"
            )

            target_insights = insights.get(
                "target_insights",
                [],
            )

            if target_insights:

                for target_insight in (
                    target_insights
                ):

                    if not isinstance(
                        target_insight,
                        dict,
                    ):

                        st.write(
                            target_insight
                        )

                        continue

                    target = target_insight.get(
                        "target",
                        "Target",
                    )

                    st.markdown(
                        f"#### {target}"
                    )

                    target_text = (
                        target_insight.get(
                            "insight",
                            "",
                        )
                    )

                    if target_text:

                        st.write(
                            target_text
                        )

                    evidence = (
                        target_insight.get(
                            "evidence"
                        )
                    )

                    if evidence:

                        st.caption(
                            f"Evidence: {evidence}"
                        )

            else:

                st.info(
                    "No target insights were generated."
                )

            # ----------------------------------------------
            # DATA CAUTIONS
            # ----------------------------------------------

            st.markdown(
                "### Data Cautions"
            )

            data_cautions = insights.get(
                "data_cautions",
                [],
            )

            if data_cautions:

                for caution in data_cautions:

                    st.warning(
                        str(caution)
                    )

            else:

                st.success(
                    "No additional data cautions "
                    "were generated."
                )

        else:

            st.warning(
                "No AI insights are available."
            )

    # ======================================================
    # REPORT TAB
    # ======================================================

    with report_tab:

        st.subheader(
            "Final Analytics Report"
        )

        if final_report:

            st.success(
                "Final report generated successfully."
            )

            # ----------------------------------------------
            # REPORT INFORMATION
            # ----------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Report Characters",
                    len(final_report),
                )

            with col2:

                st.metric(
                    "Report Words",
                    len(
                        final_report.split()
                    ),
                )

            # ----------------------------------------------
            # REPORT PREVIEW
            # ----------------------------------------------

            st.markdown(
                "### Report Preview"
            )

            with st.container(
                border=True
            ):

                st.markdown(
                    final_report
                )

            # ----------------------------------------------
            # DOWNLOAD REPORT
            # ----------------------------------------------

            st.download_button(
                label="Download Markdown Report",
                data=final_report,
                file_name="analytics_report.md",
                mime="text/markdown",
                type="primary",
            )

        else:

            st.warning(
                "No final report is available."
            )

        # ----------------------------------------------
        # SAVED REPORT PATH
        # ----------------------------------------------

        if report_path:

            report_file_path = Path(
                report_path
            )

            if not report_file_path.is_absolute():

                report_file_path = (
                    PROJECT_ROOT
                    / report_file_path
                )

            st.caption(
                f"Report saved to: {report_path}"
            )

            if not report_file_path.exists():

                st.warning(
                    "The report path exists in the "
                    "workflow state, but the file "
                    "could not be found on disk."
                )
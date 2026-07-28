# app.py

from pprint import pprint

from src.agents.graph import build_graph


def main():
    """
    Run the complete Agentic Analytics Engine.
    """

    # ==================================================
    # BUILD WORKFLOW
    # ==================================================

    graph = build_graph()

    # ==================================================
    # INITIAL STATE
    # ==================================================

    initial_state = {
        "dataset_path":
            "data/samples/dirty_customers.csv"
    }

    print(
        "\n========== STARTING ANALYTICS WORKFLOW ==========\n"
    )

    # ==================================================
    # EXECUTE WORKFLOW
    # ==================================================

    final_state = graph.invoke(
        initial_state
    )

    # ==================================================
    # ORIGINAL DATASET
    # ==================================================

    print(
        "\n========== ORIGINAL DATASET ==========\n"
    )

    print(
        final_state.get(
            "dataframe"
        )
    )

    # ==================================================
    # DATASET PROFILE
    # ==================================================

    profile = final_state.get(
        "profile",
        {}
    )

    print(
        "\n========== DATASET PROFILE ==========\n"
    )

    pprint(
        profile
    )

    # ==================================================
    # DATA QUALITY
    # ==================================================

    quality_report = final_state.get(
        "quality_report",
        {}
    )

    print(
        "\n========== DATA QUALITY REPORT ==========\n"
    )

    pprint(
        quality_report
    )

    # ==================================================
    # CLEANING
    # ==================================================

    if "cleaned_dataframe" in final_state:

        print(
            "\n========== CLEANING PLAN ==========\n"
        )

        pprint(
            final_state.get(
                "cleaning_plan",
                []
            )
        )

        print(
            "\n========== CLEANED DATASET ==========\n"
        )

        print(
            final_state.get(
                "cleaned_dataframe"
            )
        )

        print(
            "\n========== CLEANING REPORT ==========\n"
        )

        pprint(
            final_state.get(
                "cleaning_report",
                {}
            )
        )

        print(
            "\n========== CLEANING VALIDATION ==========\n"
        )

        pprint(
            final_state.get(
                "cleaning_validation",
                {}
            )
        )

    else:

        print(
            "\n========== CLEANING ==========\n"
        )

        print(
            "Dataset did not require supported "
            "automatic cleaning."
        )

    # ==================================================
    # STATISTICAL ANALYSIS
    # ==================================================

    numerical_summary = final_state.get(
        "numerical_summary",
        {}
    )

    categorical_summary = final_state.get(
        "categorical_summary",
        {}
    )

    correlation_matrix = final_state.get(
        "correlation_matrix",
        {}
    )

    print(
        "\n========== NUMERICAL SUMMARY ==========\n"
    )

    pprint(
        numerical_summary
    )

    print(
        "\n========== CATEGORICAL SUMMARY ==========\n"
    )

    pprint(
        categorical_summary
    )

    print(
        "\n========== CORRELATION MATRIX ==========\n"
    )

    pprint(
        correlation_matrix
    )

    # ==================================================
    # ANALYSIS INFORMATION
    # ==================================================

    analysis_results = final_state.get(
        "analysis_results",
        {}
    )

    print(
        "\n========== ANALYSIS INFORMATION ==========\n"
    )

    print(
        "Dataset used:",
        analysis_results.get(
            "dataset_source"
        )
    )

    print(
        "Rows analysed:",
        analysis_results.get(
            "rows_analyzed"
        )
    )

    print(
        "Columns analysed:",
        analysis_results.get(
            "columns_analyzed"
        )
    )

    # ==================================================
    # SEMANTIC ANALYSIS
    # ==================================================

    semantic_analysis = final_state.get(
        "semantic_analysis",
        {}
    )

    identifier_columns = final_state.get(
        "identifier_columns",
        []
    )

    target_candidates = final_state.get(
        "target_candidates",
        []
    )

    feature_columns = final_state.get(
        "feature_columns",
        []
    )

    print(
        "\n========== SEMANTIC ANALYSIS ==========\n"
    )

    print(
        "Identifier columns:",
        identifier_columns
    )

    print(
        "Target candidates:",
        target_candidates
    )

    print(
        "Feature columns:",
        feature_columns
    )

    print(
        "\nColumn semantic roles:\n"
    )

    column_roles = semantic_analysis.get(
        "columns",
        {}
    )

    if column_roles:

        for column, information in column_roles.items():

            if isinstance(
                information,
                dict
            ):

                role = information.get(
                    "role",
                    "unknown"
                )

                source = information.get(
                    "decision_source",
                    "unknown"
                )

                print(
                    f"{column:<20} "
                    f"-> {role:<25} "
                    f"[source: {source}]"
                )

            else:

                print(
                    f"{column:<20} "
                    f"-> {information}"
                )

    else:

        print(
            "No semantic roles available."
        )

    print(
        "\nLLM used:",
        semantic_analysis.get(
            "llm_used"
        )
    )

    print(
        "LLM error:",
        semantic_analysis.get(
            "llm_error"
        )
    )

    # ==================================================
    # EDA PLAN
    # ==================================================

    analysis_plan = final_state.get(
        "analysis_plan",
        []
    )

    print(
        "\n========== EDA PLAN ==========\n"
    )

    if analysis_plan:

        for index, task in enumerate(
            analysis_plan,
            start=1
        ):

            print(
                f"{index}. {task}"
            )

    else:

        print(
            "No EDA tasks were planned."
        )

    print(
        f"\nTotal planned tasks: "
        f"{len(analysis_plan)}"
    )

    # ==================================================
    # EDA EXECUTION REPORT
    # ==================================================

    execution_report = final_state.get(
        "execution_report",
        {}
    )

    print(
        "\n========== EDA EXECUTION REPORT ==========\n"
    )

    pprint(
        execution_report
    )

    # ==================================================
    # SEMANTIC METADATA
    # ==================================================

    semantic_metadata = final_state.get(
        "semantic_metadata",
        {}
    )

    print(
        "\n========== SEMANTIC METADATA ==========\n"
    )

    pprint(
        semantic_metadata
    )

    # ==================================================
    # TARGET ANALYSIS
    # ==================================================

    target_analysis = final_state.get(
        "target_analysis",
        {}
    )

    print(
        "\n========== TARGET ANALYSIS ==========\n"
    )

    if target_analysis:

        pprint(
            target_analysis
        )

    else:

        print(
            "No target analysis generated."
        )

    # ==================================================
    # GENERATED CHARTS
    # ==================================================

    chart_paths = final_state.get(
        "chart_paths",
        []
    )

    print(
        "\n========== GENERATED CHARTS ==========\n"
    )

    if chart_paths:

        for chart_path in chart_paths:

            print(
                chart_path
            )

    else:

        print(
            "No charts were generated."
        )

    # ==================================================
    # EXECUTED TASKS
    # ==================================================

    executed_tasks = final_state.get(
        "executed_tasks",
        []
    )

    print(
        "\n========== EXECUTED EDA TASKS ==========\n"
    )

    if executed_tasks:

        for task in executed_tasks:

            pprint(
                task
            )

    else:

        print(
            "No EDA tasks were executed."
        )

    # ==================================================
    # SKIPPED TASKS
    # ==================================================

    skipped_tasks = final_state.get(
        "skipped_tasks",
        []
    )

    print(
        "\n========== SKIPPED EDA TASKS ==========\n"
    )

    if skipped_tasks:

        for task in skipped_tasks:

            pprint(
                task
            )

    else:

        print(
            "No EDA tasks were skipped."
        )

    # ==================================================
    # EXECUTION ERRORS
    # ==================================================

    execution_errors = final_state.get(
        "execution_errors",
        []
    )

    print(
        "\n========== EDA EXECUTION ERRORS ==========\n"
    )

    if execution_errors:

        for error in execution_errors:

            pprint(
                error
            )

    else:

        print(
            "No execution errors."
        )

    # ==================================================
    # AI GENERATED INSIGHTS
    # ==================================================

    insights = final_state.get(
        "insights",
        {}
    )

    print(
        "\n========== AI GENERATED INSIGHTS ==========\n"
    )

    if insights:

        # ----------------------------------------------
        # OVERALL SUMMARY
        # ----------------------------------------------

        print(
            "Overall Summary:\n"
        )

        print(
            insights.get(
                "summary",
                "No summary generated."
            )
        )

        # ----------------------------------------------
        # KEY INSIGHTS
        # ----------------------------------------------

        print(
            "\nKey Insights:\n"
        )

        key_insights = insights.get(
            "key_insights",
            []
        )

        if key_insights:

            for index, insight in enumerate(
                key_insights,
                start=1
            ):

                print(
                    f"{index}. "
                    f"{insight.get('title', 'Insight')}"
                )

                print(
                    "   Insight:",
                    insight.get(
                        "insight",
                        ""
                    )
                )

                print(
                    "   Evidence:",
                    insight.get(
                        "evidence",
                        ""
                    )
                )

                print(
                    "   Importance:",
                    insight.get(
                        "importance",
                        ""
                    )
                )

                print()

        else:

            print(
                "No key insights generated."
            )

        # ----------------------------------------------
        # TARGET INSIGHTS
        # ----------------------------------------------

        print(
            "\nTarget Insights:\n"
        )

        target_insights = insights.get(
            "target_insights",
            []
        )

        if target_insights:

            for index, insight in enumerate(
                target_insights,
                start=1
            ):

                print(
                    f"{index}. Target:",
                    insight.get(
                        "target",
                        ""
                    )
                )

                print(
                    "   Insight:",
                    insight.get(
                        "insight",
                        ""
                    )
                )

                print(
                    "   Evidence:",
                    insight.get(
                        "evidence",
                        ""
                    )
                )

                print()

        else:

            print(
                "No target insights generated."
            )

        # ----------------------------------------------
        # DATA CAUTIONS
        # ----------------------------------------------

        print(
            "\nData Cautions:\n"
        )

        data_cautions = insights.get(
            "data_cautions",
            []
        )

        if data_cautions:

            for index, caution in enumerate(
                data_cautions,
                start=1
            ):

                print(
                    f"{index}. {caution}"
                )

        else:

            print(
                "No data cautions generated."
            )

    else:

        print(
            "No AI insights generated."
        )

    # ==================================================
    # FINAL ANALYTICS REPORT
    # ==================================================

    final_report = final_state.get(
        "final_report",
        ""
    )

    report_path = final_state.get(
        "report_path",
        ""
    )

    print(
        "\n========== FINAL ANALYTICS REPORT ==========\n"
    )

    if final_report:

        print(
            "Report generation: PASSED"
        )

        print(
            "Report length:",
            len(final_report),
            "characters"
        )

    else:

        print(
            "Report generation: FAILED"
        )

    if report_path:

        print(
            "Report saved to:",
            report_path
        )

    else:

        print(
            "Report path unavailable."
        )

    # ==================================================
    # WORKFLOW SUMMARY
    # ==================================================

    print(
        "\n========== WORKFLOW SUMMARY ==========\n"
    )

    print(
        "Dataset:",
        final_state.get(
            "dataset_path"
        )
    )

    print(
        "Original rows:",
        profile.get(
            "rows"
        )
    )

    print(
        "Columns:",
        profile.get(
            "columns"
        )
    )

    print(
        "Quality issues detected:",
        quality_report.get(
            "total_issues",
            0
        )
    )

    # --------------------------------------------------
    # CLEANING SUMMARY
    # --------------------------------------------------

    if "cleaning_report" in final_state:

        cleaning_report = final_state.get(
            "cleaning_report",
            {}
        )

        print(
            "Cleaned rows:",
            cleaning_report.get(
                "cleaned_rows"
            )
        )

        print(
            "Rows removed:",
            cleaning_report.get(
                "rows_removed"
            )
        )

        cleaning_validation = final_state.get(
            "cleaning_validation",
            {}
        )

        print(
            "Cleaning validation passed:",
            cleaning_validation.get(
                "validation_passed"
            )
        )

    else:

        print(
            "Cleaning skipped."
        )

    # --------------------------------------------------
    # ANALYSIS SUMMARY
    # --------------------------------------------------

    print(
        "Rows analysed:",
        analysis_results.get(
            "rows_analyzed"
        )
    )

    # --------------------------------------------------
    # SEMANTIC SUMMARY
    # --------------------------------------------------

    print(
        "Identifiers detected:",
        len(
            identifier_columns
        )
    )

    print(
        "Target candidates detected:",
        len(
            target_candidates
        )
    )

    print(
        "Feature columns detected:",
        len(
            feature_columns
        )
    )

    # --------------------------------------------------
    # EDA SUMMARY
    # --------------------------------------------------

    print(
        "EDA tasks planned:",
        len(
            analysis_plan
        )
    )

    print(
        "EDA tasks executed:",
        execution_report.get(
            "executed_tasks",
            0
        )
    )

    print(
        "EDA tasks skipped:",
        execution_report.get(
            "skipped_tasks",
            0
        )
    )

    print(
        "EDA tasks failed:",
        execution_report.get(
            "failed_tasks",
            0
        )
    )

    print(
        "Charts generated:",
        execution_report.get(
            "charts_generated",
            0
        )
    )

    print(
        "Target analyses:",
        execution_report.get(
            "target_analyses",
            0
        )
    )

    # --------------------------------------------------
    # AI SUMMARY
    # --------------------------------------------------

    print(
        "AI key insights generated:",
        len(
            insights.get(
                "key_insights",
                []
            )
        )
    )

    # --------------------------------------------------
    # REPORT SUMMARY
    # --------------------------------------------------

    print(
        "Final report generated:",
        bool(
            final_report
        )
    )

    print(
        "Report path:",
        report_path
        or "Not available"
    )

    # ==================================================
    # FINAL WORKFLOW VALIDATION
    # ==================================================

    print(
        "\n========== WORKFLOW VALIDATION ==========\n"
    )

    # --------------------------------------------------
    # EDA VALIDATION
    # --------------------------------------------------

    failed_tasks = execution_report.get(
        "failed_tasks",
        0
    )

    if failed_tasks == 0:

        print(
            "EDA execution: PASSED"
        )

    else:

        print(
            "EDA execution: FAILED"
        )

    # --------------------------------------------------
    # IDENTIFIER VALIDATION
    # --------------------------------------------------

    if identifier_columns:

        print(
            "Semantic identifier detection: PASSED"
        )

    else:

        print(
            "Semantic identifier detection: "
            "NO IDENTIFIER DETECTED"
        )

    # --------------------------------------------------
    # TARGET VALIDATION
    # --------------------------------------------------

    if target_candidates:

        print(
            "Target detection: PASSED"
        )

    else:

        print(
            "Target detection: "
            "NO TARGET DETECTED"
        )

    # --------------------------------------------------
    # TARGET ANALYSIS VALIDATION
    # --------------------------------------------------

    if target_analysis:

        print(
            "Target analysis: PASSED"
        )

    else:

        print(
            "Target analysis: "
            "NOT GENERATED"
        )

    # --------------------------------------------------
    # AI INSIGHT VALIDATION
    # --------------------------------------------------

    if (
        insights
        and insights.get(
            "summary"
        )
    ):

        print(
            "AI insight generation: PASSED"
        )

    else:

        print(
            "AI insight generation: FAILED"
        )

    # --------------------------------------------------
    # FINAL REPORT VALIDATION
    # --------------------------------------------------

    if (
        final_report
        and report_path
    ):

        print(
            "Final report generation: PASSED"
        )

    else:

        print(
            "Final report generation: FAILED"
        )

    # ==================================================
    # WORKFLOW COMPLETE
    # ==================================================

    print(
        "\nLangGraph workflow executed successfully."
    )


if __name__ == "__main__":
    main()
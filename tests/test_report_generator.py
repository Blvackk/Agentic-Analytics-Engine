# tests/test_report_generator.py

from pathlib import Path

from src.agents.graph import build_graph
from src.agents.report_generator import (
    generate_report,
    save_report,
)


def main():

    print(
        "\n========== REPORT GENERATOR TEST ==========\n"
    )

    # --------------------------------------------------
    # RUN EXISTING ANALYTICS WORKFLOW
    # --------------------------------------------------

    graph = build_graph()

    final_state = graph.invoke(
        {
            "dataset_path":
                "data/samples/dirty_customers.csv"
        }
    )

    print(
        "Analytics workflow: COMPLETE"
    )

    # --------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------

    report = generate_report(
        profile=final_state.get(
            "profile",
            {},
        ),

        quality_report=final_state.get(
            "quality_report",
            {},
        ),

        cleaning_report=final_state.get(
            "cleaning_report",
            {},
        ),

        cleaning_validation=final_state.get(
            "cleaning_validation",
            {},
        ),

        analysis_results=final_state.get(
            "analysis_results",
            {},
        ),

        semantic_analysis=final_state.get(
            "semantic_analysis",
            {},
        ),

        target_analysis=final_state.get(
            "target_analysis",
            {},
        ),

        execution_report=final_state.get(
            "execution_report",
            {},
        ),

        chart_paths=final_state.get(
            "chart_paths",
            [],
        ),

        insights=final_state.get(
            "insights",
            {},
        ),
    )

    print(
        "Report generation: COMPLETE"
    )

    # --------------------------------------------------
    # VALIDATE REPORT
    # --------------------------------------------------

    assert isinstance(
        report,
        str,
    )

    assert report.strip()

    assert (
        "# Agentic Analytics Report"
        in report
    )

    assert (
        "## Executive Summary"
        in report
    )

    assert (
        "## Dataset Overview"
        in report
    )

    assert (
        "## Data Quality"
        in report
    )

    assert (
        "## Semantic Analysis"
        in report
    )

    assert (
        "## Statistical Analysis"
        in report
    )

    assert (
        "## Key Insights"
        in report
    )

    # Identifier should be documented semantically.
    assert (
        "customer_id"
        in report
    )

    # --------------------------------------------------
    # IMPORTANT IDENTIFIER CHECK
    # --------------------------------------------------

    numerical_summary = final_state.get(
        "numerical_summary",
        {},
    )

    correlation_matrix = final_state.get(
        "correlation_matrix",
        {},
    )

    assert (
        "customer_id"
        not in numerical_summary
    )

    assert (
        "customer_id"
        not in correlation_matrix
    )

    # --------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------

    report_path = save_report(
        report
    )

    print(
        "Report saved:",
        report_path,
    )

    assert Path(
        report_path
    ).exists()

    # --------------------------------------------------
    # PREVIEW
    # --------------------------------------------------

    print(
        "\n========== REPORT PREVIEW ==========\n"
    )

    print(
        report[:3000]
    )

    print(
        "\n========== VALIDATION ==========\n"
    )

    print(
        "Report type: PASSED"
    )

    print(
        "Report content: PASSED"
    )

    print(
        "Required sections: PASSED"
    )

    print(
        "Identifier handling: PASSED"
    )

    print(
        "Report saving: PASSED"
    )

    print(
        "\n========== REPORT GENERATOR TEST COMPLETE ==========\n"
    )


if __name__ == "__main__":
    main()
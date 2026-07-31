# tests/test_churn_workflow.py

import time
import traceback
from pathlib import Path

from src.agents.graph import build_graph


DATASET_PATH = Path(
    "data/uploads/Churn_Modelling.csv"
)


def main():

    print(
        "\n========== CHURN FULL WORKFLOW TEST ==========\n"
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.resolve()}"
        )

    print(
        f"Dataset: {DATASET_PATH.resolve()}"
    )

    graph = build_graph()

    print(
        "Graph compiled successfully."
    )

    initial_state = {
        "dataset_path": str(DATASET_PATH),
        "errors": [],
    }

    print(
        "\nStarting full workflow...\n"
    )

    start_time = time.perf_counter()

    try:

        result = graph.invoke(
            initial_state
        )

    except Exception as error:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            "\n========== WORKFLOW FAILED ==========\n"
        )

        print(
            f"Elapsed: {elapsed:.2f} seconds"
        )

        print(
            f"Exception type: {type(error).__name__}"
        )

        print(
            f"Exception: {error}"
        )

        print(
            "\nFull traceback:\n"
        )

        traceback.print_exc()

        raise

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        "\n========== WORKFLOW COMPLETE ==========\n"
    )

    print(
        f"Total time: {elapsed:.2f} seconds"
    )

    print(
        "\nResult keys:"
    )

    for key in result.keys():
        print(
            f"  - {key}"
        )

    print(
        "\nErrors:"
    )

    print(
        result.get(
            "errors",
            [],
        )
    )

    print(
        "\nSemantic Analysis:"
    )

    semantic = result.get(
        "semantic_analysis",
        {},
    )

    print(
        "Identifiers:",
        semantic.get(
            "identifier_columns",
            [],
        ),
    )

    print(
        "Targets:",
        semantic.get(
            "target_candidates",
            [],
        ),
    )

    print(
        "\nEDA Execution:"
    )

    execution_report = result.get(
        "execution_report",
        {},
    )

    print(
        execution_report
    )

    print(
        "\nInsights:"
    )

    insights = result.get(
        "insights",
        [],
    )

    print(
        f"Generated insights: {len(insights)}"
    )

    print(
        "\nReport:"
    )

    print(
        "Report path:",
        result.get(
            "report_path"
        ),
    )

    print(
        "\n========== TEST PASSED ==========\n"
    )


if __name__ == "__main__":
    main()
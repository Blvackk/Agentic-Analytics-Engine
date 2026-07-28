# tests/test_executor.py

from pprint import pprint

from src.tools.data_loader import load_csv
from src.agents.semantic_analyzer import analyze_semantics
from src.agents.planner import create_eda_plan
from src.agents.executor import execute_eda_plan


def main():
    """
    Test the complete semantic-aware EDA execution pipeline.

    Pipeline:

        Dataset
           ↓
        Semantic Analyzer
           ↓
        EDA Planner
           ↓
        EDA Executor
    """

    print(
        "\n========== EDA EXECUTOR TEST ==========\n"
    )

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    dataframe = load_csv(
        "data/samples/dirty_customers.csv"
    )

    print(
        "Dataset loaded:",
        dataframe.shape
    )

    # --------------------------------------------------
    # SEMANTIC ANALYSIS
    # --------------------------------------------------

    print(
        "\n========== SEMANTIC ANALYSIS ==========\n"
    )

    semantic_result = analyze_semantics(
        dataframe=dataframe,
        use_llm=True,
    )

    print(
        "Identifiers:",
        semantic_result["identifier_columns"]
    )

    print(
        "Target candidates:",
        semantic_result["target_candidates"]
    )

    print(
        "Features:",
        semantic_result["feature_columns"]
    )

    # --------------------------------------------------
    # CREATE SEMANTIC-AWARE PLAN
    # --------------------------------------------------

    plan = create_eda_plan(
        dataframe=dataframe,
        semantic_analysis=semantic_result,
    )

    print(
        "\n========== EDA PLAN ==========\n"
    )

    print(
        f"Plan created: {len(plan)} tasks"
    )

    for index, task in enumerate(
        plan,
        start=1,
    ):
        print(
            f"{index}. {task}"
        )

    # --------------------------------------------------
    # EXECUTE PLAN
    # --------------------------------------------------

    result = execute_eda_plan(
        dataframe=dataframe,
        analysis_plan=plan,
    )

    # --------------------------------------------------
    # EXECUTION REPORT
    # --------------------------------------------------

    print(
        "\n========== EXECUTION REPORT ==========\n"
    )

    pprint(
        result["execution_report"]
    )

    # --------------------------------------------------
    # GENERATED CHARTS
    # --------------------------------------------------

    print(
        "\n========== GENERATED CHARTS ==========\n"
    )

    if result["chart_paths"]:

        for path in result["chart_paths"]:
            print(path)

    else:

        print(
            "No charts generated."
        )

    # --------------------------------------------------
    # TARGET ANALYSIS
    # --------------------------------------------------

    print(
        "\n========== TARGET ANALYSIS ==========\n"
    )

    if result["target_analysis"]:

        pprint(
            result["target_analysis"]
        )

    else:

        print(
            "No target analysis generated."
        )

    # --------------------------------------------------
    # SEMANTIC METADATA
    # --------------------------------------------------

    print(
        "\n========== SEMANTIC METADATA ==========\n"
    )

    pprint(
        result["semantic_metadata"]
    )

    # --------------------------------------------------
    # SKIPPED TASKS
    # --------------------------------------------------

    print(
        "\n========== SKIPPED TASKS ==========\n"
    )

    if result["skipped_tasks"]:

        for task in result["skipped_tasks"]:
            pprint(task)

    else:

        print(
            "No skipped tasks."
        )

    # --------------------------------------------------
    # ERRORS
    # --------------------------------------------------

    print(
        "\n========== ERRORS ==========\n"
    )

    if result["execution_errors"]:

        for error in result["execution_errors"]:
            pprint(error)

    else:

        print(
            "No execution errors."
        )

    # ==================================================
    # VALIDATION
    # ==================================================

    print(
        "\n========== VALIDATION ==========\n"
    )

    # --------------------------------------------------
    # 1. NO EXECUTION FAILURES
    # --------------------------------------------------

    assert (
        result["execution_report"]["failed_tasks"]
        == 0
    ), (
        "One or more EDA tasks failed."
    )

    print(
        "Execution failures: PASSED"
    )

    # --------------------------------------------------
    # 2. IDENTIFIER DETECTION
    # --------------------------------------------------

    assert (
        "customer_id"
        in semantic_result["identifier_columns"]
    ), (
        "customer_id was not detected as an identifier."
    )

    print(
        "Identifier detection: PASSED"
    )

    # --------------------------------------------------
    # 3. NO CUSTOMER_ID ANALYTICAL CHARTS
    # --------------------------------------------------

    identifier_columns = set(
        semantic_result["identifier_columns"]
    )

    chart_tools = {
        "histogram",
        "boxplot",
        "bar_chart",
        "scatter_plot",
        "correlation_heatmap",
    }

    identifier_chart_tasks = []

    for task in result["executed_tasks"]:

        if task.get("tool") not in chart_tools:
            continue

        task_columns = set()

        if task.get("column"):
            task_columns.add(
                task["column"]
            )

        if task.get("x_column"):
            task_columns.add(
                task["x_column"]
            )

        if task.get("y_column"):
            task_columns.add(
                task["y_column"]
            )

        if task.get("columns"):
            task_columns.update(
                task["columns"]
            )

        if identifier_columns & task_columns:

            identifier_chart_tasks.append(
                task
            )

    assert not identifier_chart_tasks, (
        "Identifier columns were used in charts:\n"
        f"{identifier_chart_tasks}"
    )

    print(
        "Identifier chart prevention: PASSED"
    )

    # --------------------------------------------------
    # 4. TARGET ANALYSIS EXISTS
    # --------------------------------------------------

    assert (
        "churn"
        in result["target_analysis"]
    ), (
        "Target analysis was not generated for churn."
    )

    print(
        "Target analysis: PASSED"
    )

    # --------------------------------------------------
    # 5. TARGET TYPE
    # --------------------------------------------------

    assert (
        result[
            "target_analysis"
        ]["churn"]["target_type"]
        == "categorical"
    ), (
        "churn should be treated as a categorical target."
    )

    print(
        "Target type detection: PASSED"
    )

    # --------------------------------------------------
    # 6. SEMANTIC METADATA
    # --------------------------------------------------

    assert (
        "customer_id"
        in result[
            "semantic_metadata"
        ]["identifier_columns"]
    )

    assert (
        "churn"
        in result[
            "semantic_metadata"
        ]["target_columns"]
    )

    print(
        "Semantic metadata: PASSED"
    )

    # --------------------------------------------------
    # 7. EXPECTED FEATURES USED
    # --------------------------------------------------

    expected_features = {
        "age",
        "income",
        "tenure",
        "contract_type",
    }

    planned_columns = set()

    for task in plan:

        if task.get("tool") == "semantic_metadata":
            continue

        if task.get("column"):
            planned_columns.add(
                task["column"]
            )

        if task.get("columns"):
            planned_columns.update(
                task["columns"]
            )

        if task.get("x_column"):
            planned_columns.add(
                task["x_column"]
            )

        if task.get("y_column"):
            planned_columns.add(
                task["y_column"]
            )

    missing_features = (
        expected_features
        - planned_columns
    )

    assert not missing_features, (
        "Expected features missing from EDA plan: "
        f"{missing_features}"
    )

    print(
        "Feature coverage: PASSED"
    )

    # --------------------------------------------------
    # COMPLETE
    # --------------------------------------------------

    print(
        "\n========== EXECUTOR TEST COMPLETE ==========\n"
    )

    print(
        "Semantic Analyzer -> PASSED"
    )

    print(
        "EDA Planner       -> PASSED"
    )

    print(
        "EDA Executor      -> PASSED"
    )


if __name__ == "__main__":
    main()
# tests/test_planner.py

from pprint import pprint

from src.tools.data_loader import load_csv
from src.agents.semantic_analyzer import analyze_semantics
from src.agents.planner import create_eda_plan


def main():
    """
    Test the semantic-aware EDA planner.
    """

    print(
        "\n========== EDA PLANNER TEST ==========\n"
    )

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    dataframe = load_csv(
        "data/samples/dirty_customers.csv"
    )

    print(
        f"Dataset loaded: {dataframe.shape}"
    )

    print(
        f"Columns: {dataframe.columns.tolist()}"
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
    # CREATE EDA PLAN
    # --------------------------------------------------

    plan = create_eda_plan(
        dataframe=dataframe,
        semantic_analysis=semantic_result,
    )

    print(
        "\n========== EDA PLAN ==========\n"
    )

    for index, task in enumerate(
        plan,
        start=1,
    ):
        print(
            f"{index}. {task}"
        )

    print(
        f"\nTotal tasks: {len(plan)}"
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print(
        "\n========== VALIDATION ==========\n"
    )

    identifier_columns = set(
        semantic_result["identifier_columns"]
    )

    # Collect columns that the planner intends
    # to analyse statistically or visually.
    analysed_columns = set()

    for task in plan:

        tool = task.get("tool")

        # Metadata intentionally contains identifiers,
        # so it must not count as analytical usage.
        if tool == "semantic_metadata":
            continue

        if "column" in task:
            analysed_columns.add(
                task["column"]
            )

        if "columns" in task:
            analysed_columns.update(
                task["columns"]
            )

        if "x_column" in task:
            analysed_columns.add(
                task["x_column"]
            )

        if "y_column" in task:
            analysed_columns.add(
                task["y_column"]
            )

        if "target" in task:
            analysed_columns.add(
                task["target"]
            )

    incorrectly_analysed_identifiers = (
        identifier_columns
        & analysed_columns
    )

    assert not incorrectly_analysed_identifiers, (
        "Identifier columns were incorrectly included "
        "in analytical EDA tasks: "
        f"{incorrectly_analysed_identifiers}"
    )

    print(
        "Identifier exclusion: PASSED"
    )

    # --------------------------------------------------
    # VERIFY CUSTOMER_ID
    # --------------------------------------------------

    assert (
        "customer_id"
        not in analysed_columns
    ), (
        "customer_id should not be used in "
        "analytical EDA tasks."
    )

    print(
        "customer_id exclusion: PASSED"
    )

    # --------------------------------------------------
    # VERIFY EXPECTED NUMERICAL FEATURES
    # --------------------------------------------------

    expected_numerical_features = {
        "age",
        "income",
        "tenure",
    }

    missing_features = (
        expected_numerical_features
        - analysed_columns
    )

    assert not missing_features, (
        "Expected numerical features were not "
        "included in the EDA plan: "
        f"{missing_features}"
    )

    print(
        "Numerical feature inclusion: PASSED"
    )

    # --------------------------------------------------
    # VERIFY TARGET
    # --------------------------------------------------

    target_tasks = [
        task
        for task in plan
        if task.get("tool")
        == "target_distribution"
    ]

    target_names = {
        task["target"]
        for task in target_tasks
    }

    assert (
        "churn"
        in target_names
    ), (
        "churn should have a target_distribution task."
    )

    print(
        "Target recognition: PASSED"
    )

    # --------------------------------------------------
    # VERIFY NO ID CHARTS
    # --------------------------------------------------

    id_chart_tasks = []

    chart_tools = {
        "histogram",
        "boxplot",
        "bar_chart",
        "scatter_plot",
        "correlation_heatmap",
    }

    for task in plan:

        if task.get("tool") not in chart_tools:
            continue

        task_columns = set()

        if "column" in task:
            task_columns.add(
                task["column"]
            )

        if "columns" in task:
            task_columns.update(
                task["columns"]
            )

        if "x_column" in task:
            task_columns.add(
                task["x_column"]
            )

        if "y_column" in task:
            task_columns.add(
                task["y_column"]
            )

        if identifier_columns & task_columns:
            id_chart_tasks.append(
                task
            )

    assert not id_chart_tasks, (
        "Charts were planned for identifier columns:\n"
        f"{id_chart_tasks}"
    )

    print(
        "Identifier chart prevention: PASSED"
    )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print(
        "\n========== PLANNER TEST COMPLETE ==========\n"
    )

    print(
        f"Total planned tasks: {len(plan)}"
    )

    print(
        "Analysed columns:"
    )

    pprint(
        sorted(analysed_columns)
    )


if __name__ == "__main__":
    main()
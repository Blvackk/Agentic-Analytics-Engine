# tests/test_query_executor.py

from pprint import pprint

from src.agents.query_analyzer import (
    analyze_query,
)

from src.agents.query_executor import (
    execute_query_plan,
)

from src.agents.query_planner import (
    create_query_plan,
)

from src.agents.semantic_analyzer import (
    analyze_semantics,
)

from src.tools.data_loader import (
    load_csv,
)


def main():

    print(
        "\n========== QUERY EXECUTOR TEST ==========\n"
    )

    # ======================================================
    # LOAD DATASET
    # ======================================================

    dataframe = load_csv(
        "data/samples/dirty_customers.csv"
    )

    print(
        "Dataset loaded:",
        dataframe.shape,
    )

    # ======================================================
    # SEMANTIC ANALYSIS
    # ======================================================

    semantic_analysis = analyze_semantics(
        dataframe
    )

    print(
        "Identifiers:",
        semantic_analysis.get(
            "identifier_columns",
            [],
        ),
    )

    print(
        "Targets:",
        semantic_analysis.get(
            "target_candidates",
            [],
        ),
    )

    print(
        "Features:",
        semantic_analysis.get(
            "feature_columns",
            [],
        ),
    )

    # ======================================================
    # TEST QUERIES
    # ======================================================

    questions = [
        "What is the average income?",
        "Show me the distribution of tenure.",
        "Which factors are associated with churn?",
        "Compare income and age.",
        "What is the correlation between age and income?",
        "How many customers are there?",
        "Are there any missing values or duplicates?",
        "Give me an overview of this dataset.",
    ]

    test_results = []

    # ======================================================
    # ANALYSE -> PLAN -> EXECUTE
    # ======================================================

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print(
            f"\n========== QUERY {index} ==========\n"
        )

        print(
            "Question:",
            question,
        )

        query_analysis = analyze_query(
            question=question,
            dataframe=dataframe,
            semantic_analysis=semantic_analysis,
        )

        query_plan = create_query_plan(
            query_analysis=query_analysis,
            dataframe=dataframe,
            semantic_analysis=semantic_analysis,
        )

        execution = execute_query_plan(
            dataframe=dataframe,
            query_plan=query_plan,
        )

        print(
            "\nQuery Analysis:"
        )
        pprint(
            query_analysis
        )

        print(
            "\nQuery Plan:"
        )
        pprint(
            query_plan
        )

        print(
            "\nExecution:"
        )
        pprint(
            execution
        )

        test_results.append(
            {
                "question":
                    question,

                "analysis":
                    query_analysis,

                "plan":
                    query_plan,

                "execution":
                    execution,
            }
        )

    # ======================================================
    # VALIDATION
    # ======================================================

    print(
        "\n========== VALIDATION ==========\n"
    )

    # ------------------------------------------------------
    # EXECUTION SUCCESS
    # ------------------------------------------------------

    for result in test_results:

        report = result[
            "execution"
        ]["execution_report"]

        assert (
            report["failed_tasks"]
            == 0
        ), (
            f"Execution failed for: "
            f"{result['question']}"
        )

        assert (
            report["successful_tasks"]
            >= 1
        )

    print(
        "Execution success: PASSED"
    )

    # ------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # ------------------------------------------------------

    descriptive = (
        test_results[0]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        descriptive["tool"]
        == "descriptive_statistics"
    )

    assert (
        descriptive["column"]
        == "income"
    )

    assert (
        descriptive["statistics"]["mean"]
        is not None
    )

    print(
        "Descriptive statistics: PASSED"
    )

    # ------------------------------------------------------
    # DISTRIBUTION
    # ------------------------------------------------------

    distribution = (
        test_results[1]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        distribution["tool"]
        == "distribution"
    )

    assert (
        distribution["column"]
        == "tenure"
    )

    assert (
        distribution["distribution_type"]
        == "numeric"
    )

    print(
        "Distribution execution: PASSED"
    )

    # ------------------------------------------------------
    # TARGET RELATIONSHIP
    # ------------------------------------------------------

    target_relationship = (
        test_results[2]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        target_relationship["tool"]
        == "target_relationship"
    )

    assert (
        target_relationship["target"]
        == "churn"
    )

    assert (
        target_relationship["target_type"]
        == "categorical"
    )

    numeric_relationships = (
        target_relationship[
            "numeric_feature_relationships"
        ]
    )

    categorical_relationships = (
        target_relationship[
            "categorical_feature_relationships"
        ]
    )

    assert "age" in numeric_relationships
    assert "income" in numeric_relationships
    assert "tenure" in numeric_relationships

    assert (
        "contract_type"
        in categorical_relationships
    )

    assert (
        "customer_id"
        not in numeric_relationships
    )

    assert (
        "customer_id"
        not in categorical_relationships
    )

    print(
        "Target relationship execution: PASSED"
    )

    # ------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------

    comparison = (
        test_results[3]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        comparison["tool"]
        == "comparison"
    )

    assert set(
        comparison["columns"]
    ) == {
        "age",
        "income",
    }

    assert (
        "age"
        in comparison["comparison"]
    )

    assert (
        "income"
        in comparison["comparison"]
    )

    print(
        "Comparison execution: PASSED"
    )

    # ------------------------------------------------------
    # CORRELATION
    # ------------------------------------------------------

    correlation = (
        test_results[4]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        correlation["tool"]
        == "correlation"
    )

    assert (
        correlation["method"]
        == "pearson"
    )

    assert (
        correlation[
            "pairwise_correlations"
        ]
    )

    pair = correlation[
        "pairwise_correlations"
    ][0]

    assert (
        pair["correlation"]
        is not None
    )

    assert (
        pair["observations_used"]
        > 0
    )

    print(
        "Correlation execution: PASSED"
    )

    # ------------------------------------------------------
    # COUNT
    # ------------------------------------------------------

    count_result = (
        test_results[5]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        count_result["tool"]
        == "count"
    )

    assert (
        count_result["column"]
        == "customer_id"
    )

    assert (
        count_result["count_type"]
        == "unique"
    )

    assert (
        count_result["value"]
        > 0
    )

    print(
        "Count execution: PASSED"
    )

    # ------------------------------------------------------
    # DATA QUALITY
    # ------------------------------------------------------

    quality = (
        test_results[6]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        quality["tool"]
        == "data_quality"
    )

    assert (
        "missing_values"
        in quality
    )

    assert (
        "duplicate_rows"
        in quality
    )

    print(
        "Data quality execution: PASSED"
    )

    # ------------------------------------------------------
    # DATASET OVERVIEW
    # ------------------------------------------------------

    overview = (
        test_results[7]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        overview["tool"]
        == "dataset_overview"
    )

    assert (
        overview["rows"]
        == dataframe.shape[0]
    )

    assert (
        overview["columns"]
        == dataframe.shape[1]
    )

    print(
        "Dataset overview execution: PASSED"
    )

    # ------------------------------------------------------
    # IDENTIFIER SAFETY
    # ------------------------------------------------------

    assert (
        "customer_id"
        not in target_relationship[
            "numeric_feature_relationships"
        ]
    )

    assert (
        "customer_id"
        not in target_relationship[
            "categorical_feature_relationships"
        ]
    )

    print(
        "Identifier safety: PASSED"
    )

    # ------------------------------------------------------
    # ERROR ISOLATION
    # ------------------------------------------------------

    invalid_plan = [
        {
            "tool": "not_a_real_tool",
        }
    ]

    invalid_execution = execute_query_plan(
        dataframe=dataframe,
        query_plan=invalid_plan,
    )

    assert (
        invalid_execution[
            "execution_report"
        ]["successful_tasks"]
        == 0
    )

    assert (
        invalid_execution[
            "execution_report"
        ]["failed_tasks"]
        == 1
    )

    assert (
        invalid_execution["errors"]
    )

    print(
        "Error isolation: PASSED"
    )

    # ======================================================
    # COMPLETE
    # ======================================================

    print(
        "\n========== QUERY EXECUTOR TEST COMPLETE ==========\n"
    )

    print(
        "Query Analyzer       -> PASSED"
    )

    print(
        "Query Planner        -> PASSED"
    )

    print(
        "Query Executor       -> PASSED"
    )

    print(
        "Evidence Generation  -> PASSED"
    )

    print(
        "Identifier Safety    -> PASSED"
    )

    print(
        "Error Handling       -> PASSED"
    )


if __name__ == "__main__":
    main()
# tests/test_query_planner.py

from pprint import pprint

from src.agents.query_analyzer import (
    analyze_query,
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
        "\n========== QUERY PLANNER TEST ==========\n"
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

    results = []

    # ======================================================
    # ANALYSE + PLAN
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

        print(
            "\nQuery Analysis:"
        )

        pprint(
            query_analysis
        )

        plan = create_query_plan(
            query_analysis=query_analysis,
            dataframe=dataframe,
            semantic_analysis=semantic_analysis,
        )

        print(
            "\nExecution Plan:"
        )

        pprint(
            plan
        )

        results.append(
            {
                "question":
                    question,

                "analysis":
                    query_analysis,

                "plan":
                    plan,
            }
        )

    # ======================================================
    # VALIDATION
    # ======================================================

    print(
        "\n========== VALIDATION ==========\n"
    )

    # ------------------------------------------------------
    # EVERY QUERY SHOULD HAVE A PLAN
    # ------------------------------------------------------

    for result in results:

        assert result["plan"], (
            "Planner returned an empty plan for: "
            f"{result['question']}"
        )

    print(
        "Plan generation: PASSED"
    )

    # ------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # ------------------------------------------------------

    assert (
        results[0]["plan"][0]["tool"]
        == "descriptive_statistics"
    )

    assert (
        results[0]["plan"][0]["column"]
        == "income"
    )

    print(
        "Descriptive statistics planning: PASSED"
    )

    # ------------------------------------------------------
    # DISTRIBUTION
    # ------------------------------------------------------

    assert (
        results[1]["plan"][0]["tool"]
        == "distribution"
    )

    assert (
        results[1]["plan"][0]["column"]
        == "tenure"
    )

    assert (
        results[1]["plan"][0][
            "distribution_type"
        ]
        == "numeric"
    )

    print(
        "Distribution planning: PASSED"
    )

    # ------------------------------------------------------
    # TARGET RELATIONSHIP
    # ------------------------------------------------------

    target_task = (
        results[2]["plan"][0]
    )

    assert (
        target_task["tool"]
        == "target_relationship"
    )

    assert (
        target_task["target"]
        == "churn"
    )

    assert (
        "age"
        in target_task[
            "numeric_features"
        ]
    )

    assert (
        "income"
        in target_task[
            "numeric_features"
        ]
    )

    assert (
        "tenure"
        in target_task[
            "numeric_features"
        ]
    )

    assert (
        "contract_type"
        in target_task[
            "categorical_features"
        ]
    )

    assert (
        "customer_id"
        not in target_task[
            "numeric_features"
        ]
    )

    print(
        "Target relationship planning: PASSED"
    )

    # ------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------

    assert (
        results[3]["plan"][0]["tool"]
        == "comparison"
    )

    assert set(
        results[3]["plan"][0]["columns"]
    ) == {
        "age",
        "income",
    }

    print(
        "Comparison planning: PASSED"
    )

    # ------------------------------------------------------
    # CORRELATION
    # ------------------------------------------------------

    assert (
        results[4]["plan"][0]["tool"]
        == "correlation"
    )

    assert set(
        results[4]["plan"][0]["columns"]
    ) == {
        "age",
        "income",
    }

    print(
        "Correlation planning: PASSED"
    )

    # ------------------------------------------------------
    # COUNT
    # ------------------------------------------------------

    count_task = (
        results[5]["plan"][0]
    )

    assert (
        count_task["tool"]
        == "count"
    )

    assert (
        count_task["column"]
        == "customer_id"
    )

    assert (
        count_task["count_type"]
        == "unique"
    )

    print(
        "Count planning: PASSED"
    )

    # ------------------------------------------------------
    # DATA QUALITY
    # ------------------------------------------------------

    assert (
        results[6]["plan"][0]["tool"]
        == "data_quality"
    )

    print(
        "Data quality planning: PASSED"
    )

    # ------------------------------------------------------
    # DATASET OVERVIEW
    # ------------------------------------------------------

    assert (
        results[7]["plan"][0]["tool"]
        == "dataset_overview"
    )

    print(
        "Dataset overview planning: PASSED"
    )

    # ------------------------------------------------------
    # TOOL VALIDATION
    # ------------------------------------------------------

    supported_tools = {
        "descriptive_statistics",
        "distribution",
        "correlation",
        "comparison",
        "relationship",
        "target_relationship",
        "count",
        "data_quality",
        "dataset_overview",
    }

    for result in results:

        for task in result["plan"]:

            assert (
                task["tool"]
                in supported_tools
            )

    print(
        "Supported tools only: PASSED"
    )

    # ======================================================
    # COMPLETE
    # ======================================================

    print(
        "\n========== QUERY PLANNER TEST COMPLETE ==========\n"
    )

    print(
        "Query Analysis      -> PASSED"
    )

    print(
        "Plan Generation     -> PASSED"
    )

    print(
        "Target Planning     -> PASSED"
    )

    print(
        "Identifier Safety   -> PASSED"
    )

    print(
        "Tool Validation     -> PASSED"
    )


if __name__ == "__main__":
    main()
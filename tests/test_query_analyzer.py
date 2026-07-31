# tests/test_query_analyzer.py

from pprint import pprint

from src.agents.query_analyzer import (
    analyze_query,
)

from src.agents.semantic_analyzer import (
    analyze_semantics,
)

from src.tools.data_loader import (
    load_csv,
)


def main():

    print(
        "\n========== QUERY ANALYZER TEST ==========\n"
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

    print(
        "Columns:",
        dataframe.columns.tolist(),
    )

    # ======================================================
    # SEMANTIC ANALYSIS
    # ======================================================

    print(
        "\n========== SEMANTIC ANALYSIS ==========\n"
    )

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

    test_queries = [
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

    print(
        "\n========== QUERY TESTS ==========\n"
    )

    for index, question in enumerate(
        test_queries,
        start=1,
    ):

        print(
            f"\n---------- QUERY {index} ----------\n"
        )

        print(
            "Question:",
            question,
        )

        result = analyze_query(
            question=question,
            dataframe=dataframe,
            semantic_analysis=semantic_analysis,
        )

        pprint(
            result
        )

        results.append(
            result
        )

    # ======================================================
    # VALIDATION
    # ======================================================

    print(
        "\n========== VALIDATION ==========\n"
    )

    # ------------------------------------------------------
    # RESULT STRUCTURE
    # ------------------------------------------------------

    required_fields = {
        "question",
        "intent",
        "analysis_type",
        "requested_columns",
        "target_columns",
        "requires_computation",
        "reason",
        "llm_used",
        "llm_error",
    }

    for result in results:

        assert required_fields.issubset(
            result.keys()
        )

    print(
        "Required fields: PASSED"
    )

    # ------------------------------------------------------
    # VALID INTENTS
    # ------------------------------------------------------

    supported_intents = {
        "descriptive_statistics",
        "distribution_analysis",
        "relationship_analysis",
        "comparison_analysis",
        "correlation_analysis",
        "count_analysis",
        "target_analysis",
        "data_quality",
        "dataset_overview",
        "unknown",
    }

    for result in results:

        assert (
            result["intent"]
            in supported_intents
        )

    print(
        "Intent validation: PASSED"
    )

    # ------------------------------------------------------
    # COLUMN SAFETY
    # ------------------------------------------------------

    dataset_columns = set(
        dataframe.columns.tolist()
    )

    for result in results:

        for column in result[
            "requested_columns"
        ]:

            assert (
                column
                in dataset_columns
            )

    print(
        "Column validation: PASSED"
    )

    # ------------------------------------------------------
    # TARGET SAFETY
    # ------------------------------------------------------

    target_candidates = set(
        semantic_analysis.get(
            "target_candidates",
            [],
        )
    )

    for result in results:

        for target in result[
            "target_columns"
        ]:

            assert (
                target
                in target_candidates
            )

    print(
        "Target validation: PASSED"
    )

    # ------------------------------------------------------
    # SPECIFIC COLUMN DETECTION
    # ------------------------------------------------------

    assert (
        "income"
        in results[0]["requested_columns"]
    )

    print(
        "Income detection: PASSED"
    )

    assert (
        "tenure"
        in results[1]["requested_columns"]
    )

    print(
        "Tenure detection: PASSED"
    )

    assert (
        "churn"
        in results[2]["requested_columns"]
    )

    print(
        "Churn detection: PASSED"
    )

    assert (
        "churn"
        in results[2]["target_columns"]
    )

    print(
        "Churn target detection: PASSED"
    )

    assert (
        "income"
        in results[3]["requested_columns"]
    )

    assert (
        "age"
        in results[3]["requested_columns"]
    )

    print(
        "Comparison column detection: PASSED"
    )

    assert (
        "age"
        in results[4]["requested_columns"]
    )

    assert (
        "income"
        in results[4]["requested_columns"]
    )

    print(
        "Correlation column detection: PASSED"
    )

        # ------------------------------------------------------
    # IDENTIFIER SAFETY
    # ------------------------------------------------------

    identifier_columns = set(
        semantic_analysis.get(
            "identifier_columns",
            [],
        )
    )

    for result in results:

        requested_identifiers = (
            identifier_columns.intersection(
                result["requested_columns"]
            )
        )

        # Identifier columns may legitimately be used for
        # entity counting or dataset overview, but should
        # not become normal analytical features.
        if requested_identifiers:

            assert (
                result["intent"]
                in {
                    "count_analysis",
                    "dataset_overview",
                }
            )

    print(
        "Identifier safety: PASSED"
    )

    # ------------------------------------------------------
    # EXPECTED INTENTS
    # ------------------------------------------------------

    expected_intents = [
        "descriptive_statistics",
        "distribution_analysis",
        "relationship_analysis",
        "comparison_analysis",
        "correlation_analysis",
        "count_analysis",
        "data_quality",
        "dataset_overview",
    ]

    for result, expected_intent in zip(
        results,
        expected_intents,
    ):
        assert (
            result["intent"]
            == expected_intent
        ), (
            f"Expected {expected_intent}, "
            f"got {result['intent']} "
            f"for question: {result['question']}"
        )

    print(
        "Expected intent classification: PASSED"
    )

    # ------------------------------------------------------
    # EXACT COLUMN CHECKS
    # ------------------------------------------------------

    assert (
        results[0]["requested_columns"]
        == ["income"]
    )

    print(
        "Unrelated column prevention: PASSED"
    )

    assert set(
        results[4]["requested_columns"]
    ) == {
        "age",
        "income",
    }

    print(
        "Correlation columns: PASSED"
    )

    # ------------------------------------------------------
    # INTENT / ANALYSIS TYPE CONSISTENCY
    # ------------------------------------------------------

    assert (
        results[4]["intent"]
        == "correlation_analysis"
    )

    assert (
        results[4]["analysis_type"]
        == "correlation"
    )

    print(
        "Correlation classification: PASSED"
    )
    # ======================================================
    # COMPLETE
    # ======================================================

    print(
        "\n========== QUERY ANALYZER TEST COMPLETE ==========\n"
    )

    print(
        "Semantic Context   -> PASSED"
    )

    print(
        "Intent Detection   -> PASSED"
    )

    print(
        "Column Detection   -> PASSED"
    )

    print(
        "Target Detection   -> PASSED"
    )

    print(
        "Safety Validation  -> PASSED"
    )


if __name__ == "__main__":
    main()
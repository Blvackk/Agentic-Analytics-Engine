# tests/test_answer_generator.py

from pprint import pprint

from src.agents.answer_generator import (
    generate_query_answer,
)

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
        "\n========== ANSWER GENERATOR TEST ==========\n"
    )

    # ======================================================
    # DATASET
    # ======================================================

    dataframe = load_csv(
        "data/samples/dirty_customers.csv"
    )

    print(
        "Dataset loaded:",
        dataframe.shape,
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

    # ======================================================
    # QUESTIONS
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
    # FULL CONVERSATIONAL PIPELINE
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

        answer = generate_query_answer(
            question=question,
            query_analysis=query_analysis,
            query_plan=query_plan,
            execution=execution,
        )

        print(
            "\nAnswer:"
        )

        pprint(
            answer
        )

        results.append(
            {
                "question":
                    question,

                "analysis":
                    query_analysis,

                "plan":
                    query_plan,

                "execution":
                    execution,

                "answer":
                    answer,
            }
        )

    # ======================================================
    # VALIDATION
    # ======================================================

    print(
        "\n========== VALIDATION ==========\n"
    )

    # ------------------------------------------------------
    # BASIC STRUCTURE
    # ------------------------------------------------------

    required_fields = {
        "question",
        "answer",
        "key_points",
        "cautions",
        "evidence_tools",
        "llm_used",
        "llm_error",
    }

    for result in results:

        answer = result[
            "answer"
        ]

        assert isinstance(
            answer,
            dict,
        )

        assert required_fields.issubset(
            answer.keys()
        )

        assert isinstance(
            answer["answer"],
            str,
        )

        assert answer[
            "answer"
        ].strip()

        assert isinstance(
            answer["key_points"],
            list,
        )

        assert isinstance(
            answer["cautions"],
            list,
        )

        assert isinstance(
            answer["evidence_tools"],
            list,
        )

    print(
        "Answer structure: PASSED"
    )

    # ------------------------------------------------------
    # EVIDENCE TOOL TRACEABILITY
    # ------------------------------------------------------

    expected_tools = [
        "descriptive_statistics",
        "distribution",
        "target_relationship",
        "comparison",
        "correlation",
        "count",
        "data_quality",
        "dataset_overview",
    ]

    for result, expected_tool in zip(
        results,
        expected_tools,
    ):

        assert (
            expected_tool
            in result[
                "answer"
            ]["evidence_tools"]
        )

    print(
        "Evidence traceability: PASSED"
    )

    # ------------------------------------------------------
    # AVERAGE INCOME
    # ------------------------------------------------------

    income_execution = (
        results[0]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    income_mean = (
        income_execution[
            "statistics"
        ]["mean"]
    )

    assert income_mean is not None

    assert (
        results[0]
        ["answer"]
        ["answer"]
    )

    print(
        "Descriptive answer: PASSED"
    )

    # ------------------------------------------------------
    # CORRELATION EVIDENCE
    # ------------------------------------------------------

    correlation_execution = (
        results[4]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    pair = correlation_execution[
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

    assert (
        "correlation"
        in results[4]
        ["answer"]
        ["evidence_tools"]
    )

    print(
        "Correlation answer: PASSED"
    )

    # ------------------------------------------------------
    # TARGET RELATIONSHIP
    # ------------------------------------------------------

    target_evidence = (
        results[2]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        target_evidence["target"]
        == "churn"
    )

    assert (
        "customer_id"
        not in target_evidence[
            "numeric_feature_relationships"
        ]
    )

    assert (
        "customer_id"
        not in target_evidence[
            "categorical_feature_relationships"
        ]
    )

    print(
        "Target answer safety: PASSED"
    )

    # ------------------------------------------------------
    # COUNT
    # ------------------------------------------------------

    count_evidence = (
        results[5]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        count_evidence["value"]
        == dataframe[
            "customer_id"
        ].nunique()
    )

    print(
        "Count answer: PASSED"
    )

    # ------------------------------------------------------
    # QUALITY
    # ------------------------------------------------------

    quality_evidence = (
        results[6]
        ["execution"]
        ["results"][0]
        ["evidence"]
    )

    assert (
        quality_evidence[
            "total_missing_values"
        ]
        == 3
    )

    assert (
        quality_evidence[
            "duplicate_rows"
        ]
        == 1
    )

    print(
        "Quality answer: PASSED"
    )

    # ------------------------------------------------------
    # NO-EVIDENCE FALLBACK
    # ------------------------------------------------------

    no_evidence_answer = generate_query_answer(
        question=(
            "What can you tell me?"
        ),
        query_analysis={
            "intent": "unknown",
            "analysis_type": "unknown",
        },
        query_plan=[],
        execution={
            "results": [],
            "errors": [],
            "execution_report": {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
            },
        },
    )

    assert (
        no_evidence_answer[
            "llm_used"
        ]
        is False
    )

    assert (
        no_evidence_answer[
            "answer"
        ]
    )

    print(
        "No-evidence fallback: PASSED"
    )

    # ======================================================
    # COMPLETE
    # ======================================================

    print(
        "\n========== ANSWER GENERATOR TEST COMPLETE ==========\n"
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
        "Answer Generator     -> PASSED"
    )

    print(
        "Evidence Grounding   -> PASSED"
    )

    print(
        "Fallback Handling    -> PASSED"
    )


if __name__ == "__main__":
    main()
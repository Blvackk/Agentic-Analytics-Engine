# tests/test_conversation_graph.py

from pprint import pprint

import pandas as pd

from src.agents.nodes import (
    analyze_query_node,
    plan_query_node,
    execute_query_node,
    generate_answer_node,
)


# ==========================================================
# CREATE TEST DATASET
# ==========================================================

def create_test_dataframe() -> pd.DataFrame:
    """
    Create a small dataset for testing the Phase 2
    conversational analytics workflow.
    """

    return pd.DataFrame(
        {
            "customer_id": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
            ],

            "age": [
                25,
                30,
                45,
                35,
                28,
                50,
                None,
                32,
                32,
            ],

            "income": [
                40000,
                50000,
                70000,
                60000,
                45000,
                80000,
                55000,
                None,
                None,
            ],

            "contract_type": [
                "Monthly",
                "Monthly",
                "Annual",
                "Annual",
                "Monthly",
                "Annual",
                None,
                "Monthly",
                "Monthly",
            ],

            "tenure": [
                5,
                10,
                40,
                20,
                6,
                45,
                15,
                8,
                8,
            ],

            "churn": [
                "Yes",
                "Yes",
                "No",
                "No",
                "Yes",
                "No",
                "No",
                "Yes",
                "Yes",
            ],
        }
    )


# ==========================================================
# CREATE SEMANTIC ANALYSIS
# ==========================================================

def create_semantic_analysis() -> dict:
    """
    Create semantic metadata equivalent to the output
    expected from Phase 1 semantic analysis.
    """

    return {
        "identifier_columns": [
            "customer_id",
        ],

        "target_candidates": [
            "churn",
        ],

        "feature_columns": [
            "age",
            "income",
            "contract_type",
            "tenure",
        ],
    }


# ==========================================================
# RUN ONE CONVERSATIONAL QUERY
# ==========================================================

def run_query(
    dataframe: pd.DataFrame,
    semantic_analysis: dict,
    question: str,
) -> dict:
    """
    Execute the complete Phase 2 conversational pipeline.

    Flow:

    Question
        ->
    Query Analyzer
        ->
    Query Planner
        ->
    Query Executor
        ->
    Answer Generator
    """

    state = {
        "cleaned_dataframe": dataframe,

        "semantic_analysis": semantic_analysis,

        "identifier_columns":
            semantic_analysis[
                "identifier_columns"
            ],

        "target_candidates":
            semantic_analysis[
                "target_candidates"
            ],

        "feature_columns":
            semantic_analysis[
                "feature_columns"
            ],

        "user_question": question,
    }

    # ------------------------------------------------------
    # QUERY ANALYZER
    # ------------------------------------------------------

    analyzer_update = analyze_query_node(
        state
    )

    state.update(
        analyzer_update
    )

    # ------------------------------------------------------
    # QUERY PLANNER
    # ------------------------------------------------------

    planner_update = plan_query_node(
        state
    )

    state.update(
        planner_update
    )

    # ------------------------------------------------------
    # QUERY EXECUTOR
    # ------------------------------------------------------

    executor_update = execute_query_node(
        state
    )

    state.update(
        executor_update
    )

    # ------------------------------------------------------
    # ANSWER GENERATOR
    # ------------------------------------------------------

    answer_update = generate_answer_node(
        state
    )

    state.update(
        answer_update
    )

    return state


# ==========================================================
# TEST COMPLETE CONVERSATIONAL PIPELINE
# ==========================================================

def test_conversation_pipeline():

    print(
        "\n========== CONVERSATION GRAPH TEST ==========\n"
    )

    dataframe = create_test_dataframe()

    semantic_analysis = (
        create_semantic_analysis()
    )

    print(
        f"Dataset loaded: {dataframe.shape}"
    )

    print(
        f"Columns: {list(dataframe.columns)}"
    )

    print(
        "Identifiers:",
        semantic_analysis[
            "identifier_columns"
        ],
    )

    print(
        "Targets:",
        semantic_analysis[
            "target_candidates"
        ],
    )

    # ======================================================
    # TEST QUESTIONS
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
    # RUN QUESTIONS
    # ======================================================

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print(
            f"\n========== QUERY {index} ==========\n"
        )

        print(
            f"Question: {question}"
        )

        state = run_query(
            dataframe=dataframe,
            semantic_analysis=semantic_analysis,
            question=question,
        )

        results.append(
            state
        )

        print(
            "\nQuery Analysis:"
        )

        pprint(
            state.get(
                "query_analysis"
            )
        )

        print(
            "\nQuery Plan:"
        )

        pprint(
            state.get(
                "query_plan"
            )
        )

        print(
            "\nExecution:"
        )

        pprint(
            state.get(
                "query_execution"
            )
        )

        print(
            "\nAnswer:"
        )

        pprint(
            state.get(
                "query_answer"
            )
        )

    # ======================================================
    # VALIDATION
    # ======================================================

    print(
        "\n========== VALIDATION ==========\n"
    )

    # ------------------------------------------------------
    # RESULT COUNT
    # ------------------------------------------------------

    assert len(results) == len(
        questions
    )

    print(
        "Query execution count: PASSED"
    )

    # ------------------------------------------------------
    # QUERY ANALYSIS
    # ------------------------------------------------------

    for state in results:

        assert (
            "query_analysis"
            in state
        )

        assert isinstance(
            state["query_analysis"],
            dict,
        )

        assert (
            state[
                "query_analysis"
            ].get("question")
        )

    print(
        "Query analyzer integration: PASSED"
    )

    # ------------------------------------------------------
    # QUERY PLAN
    # ------------------------------------------------------

    for state in results:

        assert (
            "query_plan"
            in state
        )

        assert isinstance(
            state["query_plan"],
            list,
        )

        assert len(
            state["query_plan"]
        ) > 0

    print(
        "Query planner integration: PASSED"
    )

    # ------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------

    for state in results:

        assert (
            "query_execution"
            in state
        )

        assert isinstance(
            state[
                "query_execution"
            ],
            dict,
        )

    print(
        "Query executor integration: PASSED"
    )

    # ------------------------------------------------------
    # ANSWER
    # ------------------------------------------------------

    for state in results:

        assert (
            "query_answer"
            in state
        )

        answer = state[
            "query_answer"
        ]

        assert isinstance(
            answer,
            dict,
        )

        assert isinstance(
            answer.get(
                "answer"
            ),
            str,
        )

        assert answer[
            "answer"
        ].strip()

    print(
        "Answer generator integration: PASSED"
    )

    # ------------------------------------------------------
    # QUESTION PRESERVATION
    # ------------------------------------------------------

    for question, state in zip(
        questions,
        results,
    ):

        assert (
            state[
                "user_question"
            ]
            == question
        )

    print(
        "Question preservation: PASSED"
    )

    # ------------------------------------------------------
    # SPECIFIC QUERY VALIDATION
    # ------------------------------------------------------

    average_income = results[0]

    assert (
        average_income[
            "query_analysis"
        ][
            "intent"
        ]
        == "descriptive_statistics"
    )

    assert (
        "income"
        in average_income[
            "query_analysis"
        ][
            "requested_columns"
        ]
    )

    print(
        "Average income query: PASSED"
    )

    # ------------------------------------------------------

    correlation = results[4]

    assert (
        correlation[
            "query_analysis"
        ][
            "analysis_type"
        ]
        == "correlation"
    )

    assert set(
        correlation[
            "query_analysis"
        ][
            "requested_columns"
        ]
    ) == {
        "age",
        "income",
    }

    print(
        "Correlation query: PASSED"
    )

    # ------------------------------------------------------

    customer_count = results[5]

    assert (
        customer_count[
            "query_analysis"
        ][
            "analysis_type"
        ]
        == "count"
    )

    print(
        "Customer count query: PASSED"
    )

    # ------------------------------------------------------

    data_quality = results[6]

    assert (
        data_quality[
            "query_analysis"
        ][
            "intent"
        ]
        == "data_quality"
    )

    print(
        "Data quality query: PASSED"
    )

    # ------------------------------------------------------

    overview = results[7]

    assert (
        overview[
            "query_analysis"
        ][
            "intent"
        ]
        == "dataset_overview"
    )

    print(
        "Dataset overview query: PASSED"
    )

    # ======================================================
    # COMPLETE
    # ======================================================

    print(
        "\n========== CONVERSATION GRAPH TEST COMPLETE ==========\n"
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
        "State Integration    -> PASSED"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    test_conversation_pipeline()


if __name__ == "__main__":
    main()
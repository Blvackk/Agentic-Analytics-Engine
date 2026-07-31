# tests/test_conversation_langgraph.py

import pandas as pd

from src.agents.conversation_graph import (
    build_conversation_graph,
)


def main():

    print(
        "\n========== CONVERSATION LANGGRAPH TEST ==========\n"
    )

    # ==================================================
    # TEST DATA
    # ==================================================

    dataframe = pd.DataFrame(
        {
            "customer_id": [
                1, 2, 3, 4, 5,
                6, 7, 8,
            ],
            "age": [
                25, 30, 45, 35,
                28, 50, 40, 32,
            ],
            "income": [
                40000, 50000, 70000, 60000,
                45000, 80000, 65000, 55000,
            ],
            "contract_type": [
                "Monthly",
                "Monthly",
                "Annual",
                "Annual",
                "Monthly",
                "Annual",
                "Annual",
                "Monthly",
            ],
            "tenure": [
                5, 10, 40, 20,
                6, 45, 30, 8,
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
            ],
        }
    )

    semantic_analysis = {
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

    # ==================================================
    # BUILD GRAPH
    # ==================================================

    graph = build_conversation_graph()

    print(
        "Conversation graph: COMPILED"
    )

    # ==================================================
    # INITIAL STATE
    # ==================================================

    initial_state = {
        "cleaned_dataframe":
            dataframe,

        "semantic_analysis":
            semantic_analysis,

        "identifier_columns":
            ["customer_id"],

        "target_candidates":
            ["churn"],

        "feature_columns": [
            "age",
            "income",
            "contract_type",
            "tenure",
        ],

        "user_question":
            "What is the average income?",
    }

    # ==================================================
    # INVOKE GRAPH
    # ==================================================

    result = graph.invoke(
        initial_state
    )

    print(
        "Graph execution: COMPLETE"
    )

    # ==================================================
    # SHOW RESULT
    # ==================================================

    print(
        "\nQuestion:"
    )

    print(
        result["user_question"]
    )

    print(
        "\nQuery Analysis:"
    )

    print(
        result["query_analysis"]
    )

    print(
        "\nQuery Plan:"
    )

    print(
        result["query_plan"]
    )

    print(
        "\nExecution:"
    )

    print(
        result["query_execution"]
    )

    print(
        "\nFinal Answer:"
    )

    print(
        result["query_answer"]
    )

    # ==================================================
    # VALIDATION
    # ==================================================

    print(
        "\n========== VALIDATION ==========\n"
    )

    assert isinstance(
        result,
        dict,
    )

    print(
        "Graph result: PASSED"
    )

    assert (
        "query_analysis"
        in result
    )

    assert isinstance(
        result["query_analysis"],
        dict,
    )

    print(
        "Query analysis state: PASSED"
    )

    assert (
        "query_plan"
        in result
    )

    assert isinstance(
        result["query_plan"],
        list,
    )

    assert (
        len(
            result["query_plan"]
        )
        > 0
    )

    print(
        "Query plan state: PASSED"
    )

    assert (
        "query_execution"
        in result
    )

    assert isinstance(
        result["query_execution"],
        dict,
    )

    print(
        "Query execution state: PASSED"
    )

    assert (
        "query_answer"
        in result
    )

    assert isinstance(
        result["query_answer"],
        dict,
    )

    answer_text = (
        result["query_answer"]
        .get(
            "answer",
            ""
        )
    )

    assert isinstance(
        answer_text,
        str,
    )

    assert (
        answer_text.strip()
    )

    print(
        "Query answer state: PASSED"
    )

    # ==================================================
    # VERIFY EXPECTED QUERY
    # ==================================================

    assert (
        result[
            "query_analysis"
        ][
            "intent"
        ]
        == "descriptive_statistics"
    )

    assert (
        "income"
        in result[
            "query_analysis"
        ][
            "requested_columns"
        ]
    )

    print(
        "Intent detection: PASSED"
    )

    # ==================================================
    # VERIFY EXECUTION
    # ==================================================

    execution_report = (
        result[
            "query_execution"
        ].get(
            "execution_report",
            {}
        )
    )

    assert (
        execution_report.get(
            "successful_tasks",
            0,
        )
        >= 1
    )

    assert (
        execution_report.get(
            "failed_tasks",
            0,
        )
        == 0
    )

    print(
        "Tool execution: PASSED"
    )

    # ==================================================
    # COMPLETE
    # ==================================================

    print(
        "\n========== CONVERSATION LANGGRAPH TEST COMPLETE ==========\n"
    )

    print(
        "LangGraph Compilation -> PASSED"
    )

    print(
        "Query Analyzer        -> PASSED"
    )

    print(
        "Query Planner         -> PASSED"
    )

    print(
        "Query Executor        -> PASSED"
    )

    print(
        "Answer Generator      -> PASSED"
    )

    print(
        "End-to-End Graph      -> PASSED"
    )


if __name__ == "__main__":
    main()
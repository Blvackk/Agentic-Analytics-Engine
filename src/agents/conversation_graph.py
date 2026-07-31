# src/agents/conversation_graph.py

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.agents.state import AgentState

from src.agents.nodes import (
    analyze_query_node,
    plan_query_node,
    execute_query_node,
    generate_answer_node,
)


def build_conversation_graph():
    """
    Build and compile the conversational analytics workflow.

    Workflow:

                START
                  |
                  v
            Analyze Query
                  |
                  v
              Plan Query
                  |
                  v
            Execute Query
                  |
                  v
           Generate Answer
                  |
                  v
                 END

    The conversational graph expects an already loaded
    dataset and semantic context in AgentState.

    It does not rerun the Phase 1 dataset analysis workflow
    for every user question.
    """

    # ==================================================
    # CREATE WORKFLOW
    # ==================================================

    workflow = StateGraph(
        AgentState
    )

    # ==================================================
    # ADD NODES
    # ==================================================

    workflow.add_node(
        "analyze_query",
        analyze_query_node,
    )

    workflow.add_node(
        "plan_query",
        plan_query_node,
    )

    workflow.add_node(
        "execute_query",
        execute_query_node,
    )

    workflow.add_node(
        "generate_answer",
        generate_answer_node,
    )

    # ==================================================
    # START -> ANALYZE QUERY
    # ==================================================

    workflow.add_edge(
        START,
        "analyze_query",
    )

    # ==================================================
    # ANALYZE -> PLAN
    # ==================================================

    workflow.add_edge(
        "analyze_query",
        "plan_query",
    )

    # ==================================================
    # PLAN -> EXECUTE
    # ==================================================

    workflow.add_edge(
        "plan_query",
        "execute_query",
    )

    # ==================================================
    # EXECUTE -> ANSWER
    # ==================================================

    workflow.add_edge(
        "execute_query",
        "generate_answer",
    )

    # ==================================================
    # ANSWER -> END
    # ==================================================

    workflow.add_edge(
        "generate_answer",
        END,
    )

    # ==================================================
    # COMPILE
    # ==================================================

    graph = workflow.compile()

    return graph
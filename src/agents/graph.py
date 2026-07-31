# src/agents/graph.py

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.agents.state import AgentState

from src.agents.nodes import (
    # ========================================================
    # PHASE 1 — AUTONOMOUS ANALYTICS
    # ========================================================
    load_dataset_node,
    profile_dataset_node,
    data_quality_node,
    route_after_quality,
    clean_dataset_node,
    validate_cleaning_node,
    semantic_analysis_node,
    analysis_dataset_node,
    plan_eda_node,
    execute_eda_node,
    generate_insights_node,
    generate_report_node,

    # ========================================================
    # PHASE 2 — CONVERSATIONAL ANALYTICS
    # ========================================================
    analyze_query_node,
    plan_query_node,
    execute_query_node,
    generate_answer_node,
)


# ============================================================
# PHASE 1 — AUTONOMOUS ANALYTICS GRAPH
# ============================================================

def build_graph():
    """
    Build and compile the Phase 1 autonomous analytics
    workflow.

    Workflow
    --------

        START
          |
          v
    Load Dataset
          |
          v
    Profile Dataset
          |
          v
     Data Quality
          |
          v
    Cleaning needed?
       /       \
     YES        NO
      |          |
      v          |
    Clean        |
      |          |
      v          |
    Validate     |
      |          |
      +----------+
          |
          v
    Semantic Analysis
          |
          v
    Statistical Analysis
          |
          v
       Plan EDA
          |
          v
      Execute EDA
          |
          v
    Generate Insights
          |
          v
    Generate Report
          |
          v
         END

    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow.
    """

    # ========================================================
    # CREATE WORKFLOW
    # ========================================================

    workflow = StateGraph(
        AgentState
    )

    # ========================================================
    # ADD PHASE 1 NODES
    # ========================================================

    workflow.add_node(
        "load_dataset",
        load_dataset_node,
    )

    workflow.add_node(
        "profile_dataset",
        profile_dataset_node,
    )

    workflow.add_node(
        "data_quality",
        data_quality_node,
    )

    workflow.add_node(
        "clean_dataset",
        clean_dataset_node,
    )

    workflow.add_node(
        "validate_cleaning",
        validate_cleaning_node,
    )

    workflow.add_node(
        "semantic_analysis",
        semantic_analysis_node,
    )

    workflow.add_node(
        "analysis_dataset",
        analysis_dataset_node,
    )

    workflow.add_node(
        "plan_eda",
        plan_eda_node,
    )

    workflow.add_node(
        "execute_eda",
        execute_eda_node,
    )

    workflow.add_node(
        "generate_insights",
        generate_insights_node,
    )

    workflow.add_node(
        "generate_report",
        generate_report_node,
    )

    # ========================================================
    # START -> LOAD DATASET
    # ========================================================

    workflow.add_edge(
        START,
        "load_dataset",
    )

    # ========================================================
    # LOAD -> PROFILE
    # ========================================================

    workflow.add_edge(
        "load_dataset",
        "profile_dataset",
    )

    # ========================================================
    # PROFILE -> DATA QUALITY
    # ========================================================

    workflow.add_edge(
        "profile_dataset",
        "data_quality",
    )

    # ========================================================
    # DATA QUALITY -> CLEAN OR SEMANTIC ANALYSIS
    # ========================================================

    workflow.add_conditional_edges(
        "data_quality",
        route_after_quality,
        {
            "clean": "clean_dataset",
            "skip": "semantic_analysis",
        },
    )

    # ========================================================
    # CLEAN -> VALIDATE
    # ========================================================

    workflow.add_edge(
        "clean_dataset",
        "validate_cleaning",
    )

    # ========================================================
    # VALIDATE -> SEMANTIC ANALYSIS
    # ========================================================

    workflow.add_edge(
        "validate_cleaning",
        "semantic_analysis",
    )

    # ========================================================
    # SEMANTIC -> STATISTICAL ANALYSIS
    # ========================================================

    workflow.add_edge(
        "semantic_analysis",
        "analysis_dataset",
    )

    # ========================================================
    # STATISTICS -> EDA PLANNER
    # ========================================================

    workflow.add_edge(
        "analysis_dataset",
        "plan_eda",
    )

    # ========================================================
    # EDA PLANNER -> EDA EXECUTOR
    # ========================================================

    workflow.add_edge(
        "plan_eda",
        "execute_eda",
    )

    # ========================================================
    # EXECUTOR -> INSIGHT GENERATOR
    # ========================================================

    workflow.add_edge(
        "execute_eda",
        "generate_insights",
    )

    # ========================================================
    # INSIGHTS -> REPORT GENERATOR
    # ========================================================

    workflow.add_edge(
        "generate_insights",
        "generate_report",
    )

    # ========================================================
    # REPORT -> END
    # ========================================================

    workflow.add_edge(
        "generate_report",
        END,
    )

    # ========================================================
    # COMPILE
    # ========================================================

    graph = workflow.compile()

    return graph


# ============================================================
# PHASE 2 — CONVERSATIONAL ANALYTICS GRAPH
# ============================================================

def build_conversation_graph():
    """
    Build and compile the Phase 2 conversational analytics
    workflow.

    Phase 1 prepares, cleans, understands and analyses the
    dataset.

    Phase 2 receives a natural-language question and uses
    the prepared dataset context to perform grounded
    conversational analytics.

    Workflow
    --------

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


    Node responsibilities
    ---------------------

    analyze_query_node
        Understand the user's question, determine intent,
        identify requested columns and target context.

    plan_query_node
        Convert the interpreted question into a deterministic
        analytical execution plan.

    execute_query_node
        Execute supported analytical tools against the
        dataframe and produce computed evidence.

    generate_answer_node
        Convert the computed evidence into a grounded
        user-facing answer.

    Returns
    -------
    CompiledStateGraph
        Compiled conversational LangGraph workflow.
    """

    # ========================================================
    # CREATE WORKFLOW
    # ========================================================

    workflow = StateGraph(
        AgentState
    )

    # ========================================================
    # ADD PHASE 2 NODES
    # ========================================================

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

    # ========================================================
    # START -> QUERY ANALYZER
    # ========================================================

    workflow.add_edge(
        START,
        "analyze_query",
    )

    # ========================================================
    # QUERY ANALYZER -> QUERY PLANNER
    # ========================================================

    workflow.add_edge(
        "analyze_query",
        "plan_query",
    )

    # ========================================================
    # QUERY PLANNER -> QUERY EXECUTOR
    # ========================================================

    workflow.add_edge(
        "plan_query",
        "execute_query",
    )

    # ========================================================
    # QUERY EXECUTOR -> ANSWER GENERATOR
    # ========================================================

    workflow.add_edge(
        "execute_query",
        "generate_answer",
    )

    # ========================================================
    # ANSWER GENERATOR -> END
    # ========================================================

    workflow.add_edge(
        "generate_answer",
        END,
    )

    # ========================================================
    # COMPILE
    # ========================================================

    graph = workflow.compile()

    return graph
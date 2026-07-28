# src/agents/graph.py

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.agents.state import AgentState

from src.agents.nodes import (
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
)


def build_graph():
    """
    Build and compile the Agentic Analytics workflow.

    Workflow:

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
                 /         \
               YES          NO
                |            |
                v            |
          Clean Dataset      |
                |            |
                v            |
        Validate Cleaning    |
                |            |
                +------------+
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

    # ==================================================
    # START -> LOAD
    # ==================================================

    workflow.add_edge(
        START,
        "load_dataset",
    )

    # ==================================================
    # LOAD -> PROFILE
    # ==================================================

    workflow.add_edge(
        "load_dataset",
        "profile_dataset",
    )

    # ==================================================
    # PROFILE -> QUALITY
    # ==================================================

    workflow.add_edge(
        "profile_dataset",
        "data_quality",
    )

    # ==================================================
    # QUALITY -> CLEAN OR SEMANTIC ANALYSIS
    # ==================================================

    workflow.add_conditional_edges(
        "data_quality",
        route_after_quality,
        {
            "clean": "clean_dataset",
            "skip": "semantic_analysis",
        },
    )

    # ==================================================
    # CLEAN -> VALIDATE
    # ==================================================

    workflow.add_edge(
        "clean_dataset",
        "validate_cleaning",
    )

    # ==================================================
    # VALIDATE -> SEMANTIC ANALYSIS
    # ==================================================

    workflow.add_edge(
        "validate_cleaning",
        "semantic_analysis",
    )

    # ==================================================
    # SEMANTIC -> STATISTICAL ANALYSIS
    # ==================================================

    workflow.add_edge(
        "semantic_analysis",
        "analysis_dataset",
    )

    # ==================================================
    # STATISTICS -> EDA PLANNER
    # ==================================================

    workflow.add_edge(
        "analysis_dataset",
        "plan_eda",
    )

    # ==================================================
    # PLANNER -> EXECUTOR
    # ==================================================

    workflow.add_edge(
        "plan_eda",
        "execute_eda",
    )

    # ==================================================
    # EXECUTOR -> INSIGHT GENERATOR
    # ==================================================

    workflow.add_edge(
        "execute_eda",
        "generate_insights",
    )

    # ==================================================
    # INSIGHTS -> REPORT GENERATOR
    # ==================================================

    workflow.add_edge(
        "generate_insights",
        "generate_report",
    )

    # ==================================================
    # REPORT -> END
    # ==================================================

    workflow.add_edge(
        "generate_report",
        END,
    )

    # ==================================================
    # COMPILE WORKFLOW
    # ==================================================

    graph = workflow.compile()

    return graph
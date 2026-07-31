# ui/components/analyst.py

from __future__ import annotations

from html import escape
import traceback
from typing import Any

import pandas as pd
import streamlit as st

from ui.utils.html import render_html


# ============================================================
# CONFIGURATION
# ============================================================

SUGGESTED_QUESTIONS = [
    "Give me an overview of this dataset.",
    "Are there any missing values or duplicates?",
    "What are the most important patterns in the data?",
    "Which variables have the strongest relationships?",
]


# ============================================================
# HELPERS
# ============================================================

def _safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    return []


def _get_dataframe(
    state: dict[str, Any],
) -> pd.DataFrame | None:
    """
    Return the dataframe used for conversational analytics.

    Prefer the cleaned dataframe because questions should
    normally be answered against the analytical dataset.
    """

    cleaned = state.get(
        "cleaned_dataframe"
    )

    if isinstance(cleaned, pd.DataFrame):
        return cleaned

    dataframe = state.get(
        "dataframe"
    )

    if isinstance(dataframe, pd.DataFrame):
        return dataframe

    return None


def _get_semantic_analysis(
    state: dict[str, Any],
) -> dict[str, Any]:

    semantic = state.get(
        "semantic_analysis"
    )

    return _safe_dict(
        semantic
    )


def _normalise_answer(
    value: Any,
) -> dict[str, Any]:
    """
    Normalise the query answer returned by the
    conversation graph.
    """

    if isinstance(value, dict):

        return {
            "question": str(
                value.get(
                    "question",
                    "",
                )
            ),
            "answer": str(
                value.get(
                    "answer",
                    "",
                )
            ),
            "key_points": _safe_list(
                value.get(
                    "key_points"
                )
            ),
            "cautions": _safe_list(
                value.get(
                    "cautions"
                )
            ),
            "evidence_tools": _safe_list(
                value.get(
                    "evidence_tools"
                )
            ),
            "llm_used": bool(
                value.get(
                    "llm_used",
                    False,
                )
            ),
            "llm_error": value.get(
                "llm_error"
            ),
        }

    if value is None:

        return {
            "question": "",
            "answer": (
                "The analytical workflow completed, "
                "but no answer was returned."
            ),
            "key_points": [],
            "cautions": [],
            "evidence_tools": [],
            "llm_used": False,
            "llm_error": None,
        }

    return {
        "question": "",
        "answer": str(value),
        "key_points": [],
        "cautions": [],
        "evidence_tools": [],
        "llm_used": False,
        "llm_error": None,
    }


def _tool_names(
    query_plan: Any,
    execution: Any,
    answer: dict[str, Any],
) -> list[str]:
    """
    Collect tool names from plan, execution and answer
    metadata without duplicates.
    """

    tools = []

    for item in _safe_list(
        query_plan
    ):

        if not isinstance(item, dict):
            continue

        tool = item.get(
            "tool"
        )

        if tool and tool not in tools:
            tools.append(
                str(tool)
            )

    execution_dict = _safe_dict(
        execution
    )

    for item in _safe_list(
        execution_dict.get(
            "results"
        )
    ):

        if not isinstance(item, dict):
            continue

        tool = item.get(
            "tool"
        )

        if not tool:

            evidence = _safe_dict(
                item.get(
                    "evidence"
                )
            )

            tool = evidence.get(
                "tool"
            )

        if tool and tool not in tools:
            tools.append(
                str(tool)
            )

    for tool in _safe_list(
        answer.get(
            "evidence_tools"
        )
    ):

        tool = str(tool)

        if tool not in tools:
            tools.append(
                tool
            )

    return tools


def _execution_summary(
    execution: Any,
) -> tuple[int, int]:
    """
    Return successful and failed execution counts.
    """

    execution_dict = _safe_dict(
        execution
    )

    report = _safe_dict(
        execution_dict.get(
            "execution_report"
        )
    )

    successful = report.get(
        "successful_tasks"
    )

    failed = report.get(
        "failed_tasks"
    )

    if successful is None:

        successful = len(
            _safe_list(
                execution_dict.get(
                    "results"
                )
            )
        )

    if failed is None:

        failed = len(
            _safe_list(
                execution_dict.get(
                    "errors"
                )
            )
        )

    try:
        successful = int(
            successful
        )
    except (TypeError, ValueError):
        successful = 0

    try:
        failed = int(
            failed
        )
    except (TypeError, ValueError):
        failed = 0

    return successful, failed


# ============================================================
# SESSION STATE
# ============================================================

def _initialise_chat_state() -> None:

    if "analyst_messages" not in st.session_state:
        st.session_state[
            "analyst_messages"
        ] = []

    if "analyst_pending_question" not in st.session_state:
        st.session_state[
            "analyst_pending_question"
        ] = None


def _clear_chat() -> None:

    st.session_state[
        "analyst_messages"
    ] = []

    st.session_state[
        "analyst_pending_question"
    ] = None


# ============================================================
# HEADER
# ============================================================

def _render_header(
    dataframe: pd.DataFrame,
) -> None:

    rows, columns = dataframe.shape

    render_html(
        f"""
        <div class="aa-analyst-header">

            <div class="aa-analyst-badge">
                AI ANALYST · DATASET CONNECTED
            </div>

            <div class="aa-analyst-title">
                Ask your data
            </div>

            <div class="aa-analyst-description">
                Ask analytical questions in natural language.
                The engine interprets your request, selects
                analytical tools, computes evidence and
                generates a grounded response.
            </div>

        </div>

        <div class="aa-dataset-header">

            <div>

                <div class="aa-dataset-name">
                    Active analytical context
                </div>

                <div class="aa-dataset-meta">
                    {rows:,} observations
                    &nbsp;·&nbsp;
                    {columns:,} variables
                </div>

            </div>

            <div class="aa-status">
                <span class="aa-status-dot"></span>
                Ready for questions
            </div>

        </div>
        """
    )


# ============================================================
# STARTER QUESTIONS
# ============================================================

def _build_suggestions(
    state: dict[str, Any],
    dataframe: pd.DataFrame,
) -> list[str]:

    suggestions = list(
        SUGGESTED_QUESTIONS
    )

    semantic = _get_semantic_analysis(
        state
    )

    targets = _safe_list(
        state.get(
            "target_candidates"
        )
    )

    if not targets:

        targets = _safe_list(
            semantic.get(
                "target_candidates"
            )
        )

    valid_targets = [
        str(target)
        for target in targets
        if target in dataframe.columns
    ]

    if valid_targets:

        target = valid_targets[0]

        suggestions[2] = (
            f"Which factors are associated with {target}?"
        )

    numeric = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    identifiers = _safe_list(
        state.get(
            "identifier_columns"
        )
    )

    analytical_numeric = [
        column
        for column in numeric
        if column not in identifiers
    ]

    if len(analytical_numeric) >= 2:

        suggestions[3] = (
            "What is the correlation between "
            f"{analytical_numeric[0]} and "
            f"{analytical_numeric[1]}?"
        )

    return suggestions


def _render_suggestions(
    state: dict[str, Any],
    dataframe: pd.DataFrame,
) -> None:

    if st.session_state[
        "analyst_messages"
    ]:
        return

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Start an investigation
            </div>

            <div class="aa-section-description">
                Try one of these questions or write your own.
            </div>

        </div>
        """
    )

    suggestions = _build_suggestions(
        state=state,
        dataframe=dataframe,
    )

    columns = st.columns(2)

    for index, question in enumerate(
        suggestions
    ):

        column = columns[
            index % 2
        ]

        with column:

            if st.button(
                question,
                key=(
                    f"analyst_suggestion_"
                    f"{index}"
                ),
                use_container_width=True,
            ):

                st.session_state[
                    "analyst_pending_question"
                ] = question

                st.rerun()


# ============================================================
# ANSWER PRESENTATION
# ============================================================

def _render_answer_body(
    answer: dict[str, Any],
) -> None:

    answer_text = answer.get(
        "answer"
    )

    if answer_text:

        st.markdown(
            answer_text
        )

    else:

        st.warning(
            "The workflow returned no displayable answer."
        )

    key_points = _safe_list(
        answer.get(
            "key_points"
        )
    )

    if key_points:

        st.markdown(
            "**Key findings**"
        )

        for point in key_points:
            st.markdown(
                f"- {point}"
            )

    cautions = _safe_list(
        answer.get(
            "cautions"
        )
    )

    if cautions:

        st.markdown(
            "**Interpretation notes**"
        )

        for caution in cautions:
            st.markdown(
                f"- {caution}"
            )


def _render_trace(
    message: dict[str, Any],
) -> None:

    query_analysis = _safe_dict(
        message.get(
            "query_analysis"
        )
    )

    query_plan = _safe_list(
        message.get(
            "query_plan"
        )
    )

    execution = _safe_dict(
        message.get(
            "execution"
        )
    )

    answer = _normalise_answer(
        message.get(
            "answer"
        )
    )

    tools = _tool_names(
        query_plan=query_plan,
        execution=execution,
        answer=answer,
    )

    successful, failed = (
        _execution_summary(
            execution
        )
    )

    intent = query_analysis.get(
        "intent",
        "unknown",
    )

    analysis_type = query_analysis.get(
        "analysis_type",
        "unknown",
    )

    render_html(
        f"""
        <div class="aa-evidence">

            <div class="aa-evidence-label">
                Analysis trace
            </div>

            <div class="aa-evidence-value">
                Intent:
                <strong>
                    {escape(str(intent))}
                </strong>

                &nbsp;·&nbsp;

                Type:
                <strong>
                    {escape(str(analysis_type))}
                </strong>

                &nbsp;·&nbsp;

                Successful tasks:
                <strong>
                    {successful}
                </strong>

                &nbsp;·&nbsp;

                Failed:
                <strong>
                    {failed}
                </strong>
            </div>

        </div>
        """
    )

    if tools:

        st.caption(
            "Evidence tools · "
            + " · ".join(
                tools
            )
        )

    with st.expander(
        "Analysis details"
    ):

        analysis_tab, plan_tab, evidence_tab = (
            st.tabs(
                [
                    "Interpretation",
                    "Plan",
                    "Evidence",
                ]
            )
        )

        with analysis_tab:

            if query_analysis:

                intent_value = (
                    query_analysis.get(
                        "intent"
                    )
                )

                analysis_value = (
                    query_analysis.get(
                        "analysis_type"
                    )
                )

                requested = _safe_list(
                    query_analysis.get(
                        "requested_columns"
                    )
                )

                targets = _safe_list(
                    query_analysis.get(
                        "target_columns"
                    )
                )

                reason = (
                    query_analysis.get(
                        "reason"
                    )
                )

                col1, col2 = st.columns(
                    2
                )

                with col1:
                    st.caption(
                        "Detected intent"
                    )
                    st.code(
                        str(
                            intent_value
                            or "unknown"
                        ),
                        language=None,
                    )

                with col2:
                    st.caption(
                        "Analysis type"
                    )
                    st.code(
                        str(
                            analysis_value
                            or "unknown"
                        ),
                        language=None,
                    )

                if requested:

                    st.caption(
                        "Requested fields"
                    )

                    st.write(
                        ", ".join(
                            map(
                                str,
                                requested,
                            )
                        )
                    )

                if targets:

                    st.caption(
                        "Target context"
                    )

                    st.write(
                        ", ".join(
                            map(
                                str,
                                targets,
                            )
                        )
                    )

                if reason:

                    st.caption(
                        "Planner interpretation"
                    )

                    st.write(
                        reason
                    )

            else:

                st.caption(
                    "No query interpretation "
                    "was recorded."
                )

        with plan_tab:

            if query_plan:

                for index, task in enumerate(
                    query_plan,
                    start=1,
                ):

                    if isinstance(
                        task,
                        dict,
                    ):

                        tool = task.get(
                            "tool",
                            "analysis",
                        )

                        reason = task.get(
                            "reason",
                            "",
                        )

                        st.markdown(
                            f"**{index}. "
                            f"{tool}**"
                        )

                        if reason:
                            st.caption(
                                str(reason)
                            )

                        st.json(
                            task
                        )

                    else:

                        st.write(
                            task
                        )

            else:

                st.caption(
                    "No execution plan "
                    "was recorded."
                )

        with evidence_tab:

            results = _safe_list(
                execution.get(
                    "results"
                )
            )

            errors = _safe_list(
                execution.get(
                    "errors"
                )
            )

            if results:

                for index, result in enumerate(
                    results,
                    start=1,
                ):

                    st.markdown(
                        f"**Evidence {index}**"
                    )

                    st.json(
                        result
                    )

            else:

                st.caption(
                    "No computed evidence "
                    "was returned."
                )

            if errors:

                st.markdown(
                    "**Execution errors**"
                )

                st.json(
                    errors
                )

            report = _safe_dict(
                execution.get(
                    "execution_report"
                )
            )

            if report:

                st.markdown(
                    "**Execution report**"
                )

                st.json(
                    report
                )


# ============================================================
# CONVERSATION
# ============================================================

def _render_conversation() -> None:

    messages = st.session_state[
        "analyst_messages"
    ]

    if not messages:
        return

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Investigation
            </div>

            <div class="aa-section-description">
                Questions and evidence-grounded analytical responses.
            </div>

        </div>
        """
    )

    for message in messages:

        question = str(
            message.get(
                "question",
                "",
            )
        )

        answer = _normalise_answer(
            message.get(
                "answer"
            )
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            _render_answer_body(
                answer
            )

            _render_trace(
                message
            )


# ============================================================
# QUERY EXECUTION
# ============================================================

def _run_question(
    question: str,
    state: dict[str, Any],
    conversation_graph: Any,
) -> None:

    dataframe = _get_dataframe(
        state
    )

    if dataframe is None:

        st.error(
            "No analytical dataframe "
            "is available."
        )
        return

    semantic_analysis = (
        _get_semantic_analysis(
            state
        )
    )

    conversation_state = {
    "user_question":
        question,

    "question":
        question,

    "dataframe":
        dataframe,

    "cleaned_dataframe":
        dataframe,

    "semantic_analysis":
        semantic_analysis,

        "identifier_columns":
            _safe_list(
                state.get(
                    "identifier_columns"
                )
            ),

        "target_candidates":
            _safe_list(
                state.get(
                    "target_candidates"
                )
            ),

        "feature_columns":
            _safe_list(
                state.get(
                    "feature_columns"
                )
            ),
    }

    try:

        with st.spinner(
            "Investigating your question..."
        ):

            result = (
                conversation_graph.invoke(
                    conversation_state
                )
            )

    except Exception as error:

        error_type = type(error).__name__
        error_message = str(error) or repr(error)
        full_traceback = traceback.format_exc()

        st.error(
            "The conversational analytics workflow failed."
        )

        st.error(
            f"{error_type}: {error_message}"
        )

        with st.expander(
            "Full error details",
            expanded=True,
        ):
            st.code(
                full_traceback,
                language="text",
            )

        print(
            "\n========== CONVERSATIONAL WORKFLOW ERROR =========="
        )
        print(f"{error_type}: {error_message}")
        print(full_traceback)
        print(
            "====================================================\n"
        )

        return

    if not isinstance(
        result,
        dict,
    ):

        st.error(
            "The conversation graph returned "
            "an invalid result."
        )
        return

    message = {
        "question":
            question,

        "query_analysis":
            _safe_dict(
                result.get(
                    "query_analysis"
                )
            ),

        "query_plan":
            _safe_list(
                result.get(
                    "query_plan"
                )
            ),

        "execution":
            _safe_dict(
                result.get(
                    "query_execution"
                )
                or result.get(
                    "execution"
                )
            ),

        "answer":
            result.get(
                "query_answer"
            )
            or result.get(
                "final_answer"
            )
            or result.get(
                "answer"
            ),
    }

    st.session_state[
        "analyst_messages"
    ].append(
        message
    )


# ============================================================
# INPUT AREA
# ============================================================

def _render_input(
    state: dict[str, Any],
    conversation_graph: Any,
) -> None:

    messages = st.session_state[
        "analyst_messages"
    ]

    if messages:

        col1, col2 = st.columns(
            [5, 1]
        )

        with col2:

            if st.button(
                "Clear",
                key="clear_analyst_chat",
                use_container_width=True,
            ):

                _clear_chat()
                st.rerun()

    pending = st.session_state.get(
        "analyst_pending_question"
    )

    typed_question = st.chat_input(
        "Ask a question about this dataset..."
    )

    question = (
        pending
        or typed_question
    )

    if not question:
        return

    st.session_state[
        "analyst_pending_question"
    ] = None

    question = str(
        question
    ).strip()

    if not question:
        return

    _run_question(
        question=question,
        state=state,
        conversation_graph=conversation_graph,
    )

    st.rerun()


# ============================================================
# CAPABILITY FOOTER
# ============================================================

def _render_capabilities() -> None:

    if st.session_state[
        "analyst_messages"
    ]:
        return

    render_html(
        """
        <div class="aa-section">

            <div class="aa-section-label">
                Analytical capabilities
            </div>

            <div class="aa-section-description">
                Questions are answered from computed dataset
                evidence rather than model knowledge alone.
            </div>

        </div>

        <div class="aa-kpi-grid">

            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Describe
                </div>
                <div class="aa-kpi-detail">
                    Summaries, averages, counts
                    and dataset structure
                </div>
            </div>

            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Compare
                </div>
                <div class="aa-kpi-detail">
                    Compare variables and
                    analytical groups
                </div>
            </div>

            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Relate
                </div>
                <div class="aa-kpi-detail">
                    Correlations and target
                    associations
                </div>
            </div>

            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Validate
                </div>
                <div class="aa-kpi-detail">
                    Missing data, duplicates
                    and quality checks
                </div>
            </div>

        </div>
        """
    )


# ============================================================
# PUBLIC COMPONENT
# ============================================================

def render_analyst(
    state: dict[str, Any],
    conversation_graph: Any,
) -> None:
    """
    Render the Phase 2 AI Analyst workspace.

    Parameters
    ----------
    state:
        Final Phase 1 analytical state.

    conversation_graph:
        Compiled Phase 2 conversational LangGraph.
    """

    if not isinstance(
        state,
        dict,
    ):

        st.error(
            "The analytics workflow state "
            "is unavailable."
        )
        return

    if conversation_graph is None:

        st.error(
            "The conversational analytics "
            "graph is unavailable."
        )
        return

    dataframe = _get_dataframe(
        state
    )

    if dataframe is None:

        st.warning(
            "No dataframe is available "
            "for conversational analysis."
        )
        return

    _initialise_chat_state()

    _render_header(
        dataframe=dataframe
    )

    _render_suggestions(
        state=state,
        dataframe=dataframe,
    )

    _render_conversation()

    _render_capabilities()

    _render_input(
        state=state,
        conversation_graph=conversation_graph,
    )
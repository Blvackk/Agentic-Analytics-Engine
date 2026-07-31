# src/agents/answer_generator.py

import json
from typing import Any

from src.llm.client import generate_json_response


# ==========================================================
# CONSTANTS
# ==========================================================

MAX_EVIDENCE_RESULTS = 10
MAX_ERRORS = 5

# Final answer generation should be short and deterministic.
ANSWER_NUM_PREDICT = 384
ANSWER_TEMPERATURE = 0.0


# ==========================================================
# EXECUTION HELPERS
# ==========================================================

def _normalise_execution(
    execution: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Normalise executor output before using it for answer
    generation.
    """

    if not isinstance(execution, dict):
        execution = {}

    results = execution.get(
        "results",
        [],
    )

    errors = execution.get(
        "errors",
        [],
    )

    report = execution.get(
        "execution_report",
        {},
    )

    if not isinstance(results, list):
        results = []

    if not isinstance(errors, list):
        errors = []

    if not isinstance(report, dict):
        report = {}

    return {
        "results":
            results[:MAX_EVIDENCE_RESULTS],

        "errors":
            errors[:MAX_ERRORS],

        "execution_report":
            report,
    }


def _extract_evidence(
    execution: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract successfully computed evidence objects from
    executor output.
    """

    evidence_items: list[dict[str, Any]] = []

    results = execution.get(
        "results",
        [],
    )

    if not isinstance(results, list):
        return evidence_items

    for result in results:

        if not isinstance(result, dict):
            continue

        evidence = result.get(
            "evidence"
        )

        if isinstance(evidence, dict):
            evidence_items.append(
                evidence
            )

    return evidence_items


def _extract_evidence_tools(
    evidence_items: list[dict[str, Any]],
) -> list[str]:
    """
    Extract unique analytical tool names from evidence.
    """

    tools: list[str] = []

    for evidence in evidence_items:

        if not isinstance(evidence, dict):
            continue

        tool = evidence.get(
            "tool"
        )

        if (
            isinstance(tool, str)
            and tool.strip()
            and tool not in tools
        ):
            tools.append(
                tool
            )

    return tools


# ==========================================================
# FORMAT HELPERS
# ==========================================================

def _format_number(
    value: Any,
) -> str:
    """
    Format numeric values for deterministic fallback answers.
    """

    if value is None:
        return "unavailable"

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):

        if value.is_integer():
            return str(
                int(value)
            )

        return (
            f"{value:,.3f}"
            .rstrip("0")
            .rstrip(".")
        )

    return str(value)


def _normalise_string_list(
    value: Any,
) -> list[str]:
    """
    Convert an LLM field into a clean list of strings.
    """

    if not isinstance(value, list):
        return []

    result: list[str] = []

    for item in value:

        if item is None:
            continue

        text = str(
            item
        ).strip()

        if text:
            result.append(
                text
            )

    return result


# ==========================================================
# FALLBACK: DESCRIPTIVE STATISTICS
# ==========================================================

def _fallback_from_descriptive(
    evidence: dict[str, Any],
) -> str:

    column = evidence.get(
        "column",
        "column",
    )

    statistics = evidence.get(
        "statistics",
        {},
    )

    if not isinstance(
        statistics,
        dict,
    ):
        statistics = {}

    if evidence.get(
        "column_type"
    ) == "numeric":

        return (
            f"For '{column}', the mean is "
            f"{_format_number(statistics.get('mean'))}, "
            f"the median is "
            f"{_format_number(statistics.get('median'))}, "
            f"and the calculation uses "
            f"{_format_number(statistics.get('count'))} "
            f"non-missing observations."
        )

    return (
        f"'{column}' contains "
        f"{_format_number(statistics.get('unique'))} "
        f"unique values. The most frequent value is "
        f"{statistics.get('most_frequent')}."
    )


# ==========================================================
# FALLBACK: DISTRIBUTION
# ==========================================================

def _fallback_from_distribution(
    evidence: dict[str, Any],
) -> str:

    column = evidence.get(
        "column",
        "column",
    )

    distribution = evidence.get(
        "distribution",
        {},
    )

    if not isinstance(
        distribution,
        dict,
    ):
        distribution = {}

    if (
        evidence.get("distribution_type")
        == "numeric"
    ):

        return (
            f"The '{column}' distribution has a median of "
            f"{_format_number(distribution.get('median'))}, "
            f"a mean of "
            f"{_format_number(distribution.get('mean'))}, "
            f"and ranges from "
            f"{_format_number(distribution.get('min'))} "
            f"to "
            f"{_format_number(distribution.get('max'))}. "
            f"There are "
            f"{_format_number(distribution.get('missing'))} "
            f"missing values."
        )

    frequencies = distribution.get(
        "frequencies",
        {},
    )

    return (
        f"The observed distribution for '{column}' is "
        f"{frequencies}."
    )


# ==========================================================
# FALLBACK: CORRELATION
# ==========================================================

def _fallback_from_correlation(
    evidence: dict[str, Any],
) -> str:

    pairs = evidence.get(
        "pairwise_correlations",
        [],
    )

    if not isinstance(
        pairs,
        list,
    ):
        pairs = []

    if not pairs:

        return (
            "The requested correlation could not be "
            "calculated from the available observations."
        )

    statements: list[str] = []

    for pair in pairs:

        if not isinstance(pair, dict):
            continue

        statements.append(
            (
                f"{pair.get('column_1')} and "
                f"{pair.get('column_2')} have a Pearson "
                f"correlation of "
                f"{_format_number(pair.get('correlation'))} "
                f"based on "
                f"{_format_number(pair.get('observations_used'))} "
                f"complete observations"
            )
        )

    if not statements:

        return (
            "The requested correlation could not be "
            "calculated from the available observations."
        )

    return (
        ". ".join(
            statements
        )
        + ". Correlation describes association and does "
          "not establish causation."
    )


# ==========================================================
# FALLBACK: COMPARISON
# ==========================================================

def _fallback_from_comparison(
    evidence: dict[str, Any],
) -> str:

    comparison = evidence.get(
        "comparison",
        {},
    )

    if not isinstance(
        comparison,
        dict,
    ):
        comparison = {}

    statements: list[str] = []

    for column, information in comparison.items():

        if not isinstance(
            information,
            dict,
        ):
            continue

        if information.get(
            "type"
        ) == "numeric":

            summary = information.get(
                "summary",
                {},
            )

            if not isinstance(
                summary,
                dict,
            ):
                summary = {}

            statements.append(
                (
                    f"{column}: mean "
                    f"{_format_number(summary.get('mean'))}, "
                    f"median "
                    f"{_format_number(summary.get('median'))}"
                )
            )

        else:

            distribution = information.get(
                "distribution",
                {},
            )

            if not isinstance(
                distribution,
                dict,
            ):
                distribution = {}

            statements.append(
                (
                    f"{column}: "
                    f"{distribution.get('frequencies', {})}"
                )
            )

    if not statements:

        return (
            "No comparison evidence was available."
        )

    return (
        "Comparison results: "
        + "; ".join(
            statements
        )
        + "."
    )


# ==========================================================
# FALLBACK: TARGET RELATIONSHIP
# ==========================================================

def _fallback_from_target_relationship(
    evidence: dict[str, Any],
) -> str:

    target = evidence.get(
        "target",
        "target",
    )

    target_distribution = evidence.get(
        "target_distribution",
        {},
    )

    numeric_relationships = evidence.get(
        "numeric_feature_relationships",
        {},
    )

    categorical_relationships = evidence.get(
        "categorical_feature_relationships",
        {},
    )

    if not isinstance(
        target_distribution,
        dict,
    ):
        target_distribution = {}

    if not isinstance(
        numeric_relationships,
        dict,
    ):
        numeric_relationships = {}

    if not isinstance(
        categorical_relationships,
        dict,
    ):
        categorical_relationships = {}

    parts: list[str] = [
        (
            "The analysis examined associations with "
            f"'{target}'."
        )
    ]

    frequencies = target_distribution.get(
        "frequencies"
    )

    if frequencies:

        parts.append(
            f"The target distribution is {frequencies}."
        )

    # ------------------------------------------------------
    # NUMERIC FEATURES
    # ------------------------------------------------------

    for feature, relationship in (
        numeric_relationships.items()
    ):

        if not isinstance(
            relationship,
            dict,
        ):
            continue

        analysis = relationship.get(
            "analysis"
        )

        if (
            analysis
            == "grouped_statistics_by_target"
        ):

            groups = relationship.get(
                "groups",
                {},
            )

            if not isinstance(
                groups,
                dict,
            ):
                groups = {}

            group_descriptions: list[str] = []

            for group, statistics in groups.items():

                if not isinstance(
                    statistics,
                    dict,
                ):
                    continue

                group_descriptions.append(
                    (
                        f"{group}: mean "
                        f"{_format_number(statistics.get('mean'))}"
                    )
                )

            if group_descriptions:

                parts.append(
                    (
                        f"For {feature}, grouped means are "
                        + ", ".join(
                            group_descriptions
                        )
                        + "."
                    )
                )

        elif (
            analysis
            == "pearson_correlation"
        ):

            parts.append(
                (
                    f"{feature} has a Pearson correlation "
                    f"of "
                    f"{_format_number(relationship.get('correlation'))} "
                    f"with {target}."
                )
            )

    # ------------------------------------------------------
    # CATEGORICAL FEATURES
    # ------------------------------------------------------

    for feature, relationship in (
        categorical_relationships.items()
    ):

        if not isinstance(
            relationship,
            dict,
        ):
            continue

        analysis = relationship.get(
            "analysis"
        )

        if (
            analysis
            == "contingency_table"
        ):

            parts.append(
                (
                    f"For {feature}, the observed target "
                    f"counts are "
                    f"{relationship.get('counts', {})}."
                )
            )

        elif (
            analysis
            == "target_statistics_by_group"
        ):

            parts.append(
                (
                    f"For {feature}, grouped target "
                    f"statistics are "
                    f"{relationship.get('groups', {})}."
                )
            )

    parts.append(
        (
            "These are associations in the observed data "
            "and should not be interpreted as causal effects."
        )
    )

    return " ".join(
        parts
    )


# ==========================================================
# FALLBACK: COUNT
# ==========================================================

def _fallback_from_count(
    evidence: dict[str, Any],
) -> str:

    value = evidence.get(
        "value"
    )

    column = evidence.get(
        "column"
    )

    count_type = evidence.get(
        "count_type"
    )

    if count_type == "unique":

        return (
            f"There are "
            f"{_format_number(value)} unique values in "
            f"'{column}'."
        )

    if count_type == "non_null":

        return (
            f"'{column}' contains "
            f"{_format_number(value)} non-missing "
            f"observations."
        )

    return (
        f"The dataset contains "
        f"{_format_number(value)} rows."
    )


# ==========================================================
# FALLBACK: DATA QUALITY
# ==========================================================

def _fallback_from_quality(
    evidence: dict[str, Any],
) -> str:

    return (
        f"The analysed dataset contains "
        f"{_format_number(evidence.get('total_missing_values'))} "
        f"missing values and "
        f"{_format_number(evidence.get('duplicate_rows'))} "
        f"duplicate rows. Missing values by column are "
        f"{evidence.get('missing_values', {})}."
    )


# ==========================================================
# FALLBACK: DATASET OVERVIEW
# ==========================================================

def _fallback_from_overview(
    evidence: dict[str, Any],
) -> str:

    return (
        f"The dataset contains "
        f"{_format_number(evidence.get('rows'))} rows and "
        f"{_format_number(evidence.get('columns'))} columns. "
        f"Numeric columns are "
        f"{evidence.get('numeric_columns', [])}, while "
        f"categorical columns are "
        f"{evidence.get('categorical_columns', [])}. "
        f"The dataset contains "
        f"{_format_number(evidence.get('total_missing_values'))} "
        f"missing values and "
        f"{_format_number(evidence.get('duplicate_rows'))} "
        f"duplicate rows."
    )


# ==========================================================
# DETERMINISTIC FALLBACK
# ==========================================================

def _generate_fallback_answer(
    evidence_items: list[dict[str, Any]],
) -> str:
    """
    Generate an evidence-grounded answer without an LLM.
    """

    if not evidence_items:

        return (
            "I could not produce an answer because no "
            "successful analytical evidence was generated."
        )

    answers: list[str] = []

    for evidence in evidence_items:

        if not isinstance(
            evidence,
            dict,
        ):
            continue

        tool = evidence.get(
            "tool"
        )

        if tool == "descriptive_statistics":

            answer = _fallback_from_descriptive(
                evidence
            )

        elif tool == "distribution":

            answer = _fallback_from_distribution(
                evidence
            )

        elif tool == "correlation":

            answer = _fallback_from_correlation(
                evidence
            )

        elif tool in {
            "comparison",
            "relationship",
        }:

            answer = _fallback_from_comparison(
                evidence
            )

        elif tool == "target_relationship":

            answer = _fallback_from_target_relationship(
                evidence
            )

        elif tool == "count":

            answer = _fallback_from_count(
                evidence
            )

        elif tool == "data_quality":

            answer = _fallback_from_quality(
                evidence
            )

        elif tool == "dataset_overview":

            answer = _fallback_from_overview(
                evidence
            )

        else:

            answer = (
                f"Computed evidence: {evidence}"
            )

        answers.append(
            answer
        )

    if not answers:

        return (
            "I could not produce an answer because no "
            "supported analytical evidence was generated."
        )

    return " ".join(
        answers
    )


# ==========================================================
# LLM CONTEXT
# ==========================================================

def _build_llm_context(
    question: str,
    query_analysis: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the minimum context required for final natural
    language interpretation.

    The query plan and executor metadata are deliberately
    excluded. The LLM only needs the user's question,
    analytical intent, and already-computed evidence.
    """

    return {
        "question":
            question,

        "intent":
            query_analysis.get(
                "intent"
            ),

        "analysis_type":
            query_analysis.get(
                "analysis_type"
            ),

        "evidence":
            evidence_items,
    }


def _build_answer_prompt(
    grounded_context: dict[str, Any],
) -> str:
    """
    Build a compact prompt for grounded answer generation.
    """

    context_json = json.dumps(
        grounded_context,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )

    prompt = f"""
You are the final response layer of a data analytics system.

Python has already performed all calculations.

Your only task is to explain the supplied evidence and answer
the user's question.

STRICT RULES:

- Use only values and facts present in INPUT.
- Never invent statistics, categories, relationships, or facts.
- Never perform new calculations.
- Never modify numbers from the evidence.
- Answer the user's actual question directly.
- Keep the answer concise and useful.
- Do not repeat or reconstruct INPUT.
- Do not return query plans or analytical context.
- Do not describe correlation or grouped differences as causal.
- Do not claim statistical significance unless explicitly
  provided in the evidence.
- If the evidence is insufficient, say that clearly.
- Add cautions only when they are relevant to interpretation.

Return exactly one JSON object.

The object must contain exactly these keys:

{{
  "answer": "direct natural-language answer",
  "key_points": [],
  "cautions": []
}}

"answer" must be a non-empty string.

"key_points" must be a JSON array of short evidence-backed
strings.

"cautions" must be a JSON array of short limitation strings.

INPUT:
{context_json}

Return only the required JSON object now.
"""

    return prompt.strip()


# ==========================================================
# LLM RESPONSE VALIDATION
# ==========================================================

def _validate_llm_response(
    response: Any,
) -> tuple[str, list[str], list[str]]:
    """
    Validate and normalise the structured LLM answer.
    """

    if not isinstance(
        response,
        dict,
    ):

        raise ValueError(
            "Answer generator expected a JSON object."
        )

    # ------------------------------------------------------
    # DETECT CONTEXT ECHO
    # ------------------------------------------------------

    unexpected_context_keys = {
        "question",
        "query_analysis",
        "query_plan",
        "computed_evidence",
        "execution_report",
        "evidence",
        "intent",
        "analysis_type",
    }

    response_keys = set(
        response.keys()
    )

    echoed_keys = (
        unexpected_context_keys
        & response_keys
    )

    if echoed_keys:

        raise ValueError(
            "LLM returned analytical context instead of "
            "the required answer object."
        )

    # ------------------------------------------------------
    # ANSWER
    # ------------------------------------------------------

    if "answer" not in response:

        raise ValueError(
            "LLM response is missing 'answer'."
        )

    answer = response.get(
        "answer"
    )

    if not isinstance(
        answer,
        str,
    ):

        raise ValueError(
            "'answer' must be a string."
        )

    answer = answer.strip()

    if not answer:

        raise ValueError(
            "LLM returned an empty answer."
        )

    # ------------------------------------------------------
    # KEY POINTS
    # ------------------------------------------------------

    key_points = _normalise_string_list(
        response.get(
            "key_points",
            [],
        )
    )

    # ------------------------------------------------------
    # CAUTIONS
    # ------------------------------------------------------

    cautions = _normalise_string_list(
        response.get(
            "cautions",
            [],
        )
    )

    return (
        answer,
        key_points,
        cautions,
    )


# ==========================================================
# MAIN ANSWER GENERATOR
# ==========================================================

def generate_query_answer(
    question: str,
    query_analysis: dict[str, Any],
    query_plan: list[dict[str, Any]],
    execution: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate a grounded natural-language answer from
    deterministic analytical evidence.

    Pipeline:

        question
            ↓
        query analysis
            ↓
        query plan
            ↓
        deterministic executor
            ↓
        computed evidence
            ↓
        LLM interpretation
            ↓
        final answer

    The LLM never performs the analytical computation.

    If LLM generation fails, the function returns a
    deterministic answer generated directly from evidence.
    """

    # ======================================================
    # INPUT VALIDATION
    # ======================================================

    if not isinstance(
        question,
        str,
    ):

        raise TypeError(
            "question must be a string."
        )

    question = question.strip()

    if not question:

        raise ValueError(
            "question cannot be empty."
        )

    if not isinstance(
        query_analysis,
        dict,
    ):

        raise TypeError(
            "query_analysis must be a dictionary."
        )

    if not isinstance(
        query_plan,
        list,
    ):

        raise TypeError(
            "query_plan must be a list."
        )

    if not isinstance(
        execution,
        dict,
    ):

        raise TypeError(
            "execution must be a dictionary."
        )

    # ======================================================
    # NORMALISE EXECUTION
    # ======================================================

    normalised_execution = _normalise_execution(
        execution
    )

    evidence_items = _extract_evidence(
        normalised_execution
    )

    evidence_tools = _extract_evidence_tools(
        evidence_items
    )

    # ======================================================
    # NO SUCCESSFUL EVIDENCE
    # ======================================================

    if not evidence_items:

        fallback = _generate_fallback_answer(
            evidence_items
        )

        return {
            "question":
                question,

            "answer":
                fallback,

            "key_points":
                [],

            "cautions":
                [
                    (
                        "No successful analytical evidence "
                        "was available."
                    )
                ],

            "evidence_tools":
                [],

            "llm_used":
                False,

            "llm_error":
                None,
        }

    # ======================================================
    # BUILD MINIMAL LLM CONTEXT
    # ======================================================

    grounded_context = _build_llm_context(
        question=question,
        query_analysis=query_analysis,
        evidence_items=evidence_items,
    )

    prompt = _build_answer_prompt(
        grounded_context
    )

    # ======================================================
    # LLM GENERATION
    # ======================================================

    llm_used = False
    llm_error = None

    try:

        response = generate_json_response(
            prompt=prompt,
            temperature=ANSWER_TEMPERATURE,
            num_predict=ANSWER_NUM_PREDICT,
        )

        (
            answer,
            key_points,
            cautions,
        ) = _validate_llm_response(
            response
        )

        llm_used = True

    except Exception as error:

        llm_error = str(
            error
        )

        answer = _generate_fallback_answer(
            evidence_items
        )

        key_points = []

        cautions = [
            (
                "The language-model interpretation was "
                "unavailable, so this answer was generated "
                "directly from computed evidence."
            )
        ]

    # ======================================================
    # FINAL RESULT
    # ======================================================

    return {
        "question":
            question,

        "answer":
            answer,

        "key_points":
            key_points,

        "cautions":
            cautions,

        "evidence_tools":
            evidence_tools,

        "llm_used":
            llm_used,

        "llm_error":
            llm_error,
    }
# src/agents/answer_generator.py

import json
from typing import Any

from src.llm.client import generate_json_response


# ==========================================================
# CONSTANTS
# ==========================================================

MAX_EVIDENCE_RESULTS = 10
MAX_ERRORS = 5


# ==========================================================
# HELPERS
# ==========================================================

def _normalise_execution(
    execution: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Normalise executor output before passing it to the LLM.
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
    Extract successfully computed evidence objects.
    """

    evidence = []

    for result in execution.get(
        "results",
        [],
    ):

        if not isinstance(result, dict):
            continue

        item = result.get(
            "evidence"
        )

        if isinstance(item, dict):
            evidence.append(
                item
            )

    return evidence


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

        return f"{value:,.3f}".rstrip(
            "0"
        ).rstrip(
            "."
        )

    return str(value)


# ==========================================================
# FALLBACK ANSWERS
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


def _fallback_from_correlation(
    evidence: dict[str, Any],
) -> str:

    pairs = evidence.get(
        "pairwise_correlations",
        [],
    )

    if not pairs:
        return (
            "The requested correlation could not be "
            "calculated from the available observations."
        )

    statements = []

    for pair in pairs:

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

    return (
        ". ".join(statements)
        + ". Correlation describes association and does "
          "not establish causation."
    )


def _fallback_from_comparison(
    evidence: dict[str, Any],
) -> str:

    comparison = evidence.get(
        "comparison",
        {},
    )

    statements = []

    for column, information in comparison.items():

        if information.get(
            "type"
        ) == "numeric":

            summary = information.get(
                "summary",
                {},
            )

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
        + "; ".join(statements)
        + "."
    )


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

    parts = [
        (
            f"The analysis examined associations with "
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

    for feature, relationship in (
        numeric_relationships.items()
    ):

        analysis = relationship.get(
            "analysis"
        )

        if analysis == "grouped_statistics_by_target":

            groups = relationship.get(
                "groups",
                {},
            )

            group_descriptions = []

            for group, statistics in groups.items():

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

        elif analysis == "pearson_correlation":

            parts.append(
                (
                    f"{feature} has a Pearson correlation "
                    f"of "
                    f"{_format_number(relationship.get('correlation'))} "
                    f"with {target}."
                )
            )

    for feature, relationship in (
        categorical_relationships.items()
    ):

        if (
            relationship.get("analysis")
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
            relationship.get("analysis")
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


def _generate_fallback_answer(
    evidence_items: list[dict[str, Any]],
) -> str:
    """
    Generate an evidence-based answer without an LLM.
    """

    if not evidence_items:
        return (
            "I could not produce an answer because no "
            "successful analytical evidence was generated."
        )

    answers = []

    for evidence in evidence_items:

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

    return " ".join(
        answers
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
    previously computed analytical evidence.

    The LLM is used only for interpretation and explanation.
    It must not invent new calculations.

    If the LLM fails, a deterministic evidence-based
    fallback answer is returned.
    """

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

    normalised_execution = _normalise_execution(
        execution
    )

    evidence_items = _extract_evidence(
        normalised_execution
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
    # BUILD GROUNDED CONTEXT
    # ======================================================

    grounded_context = {
        "question":
            question,

        "query_analysis":
            {
                "intent":
                    query_analysis.get(
                        "intent"
                    ),

                "analysis_type":
                    query_analysis.get(
                        "analysis_type"
                    ),

                "requested_columns":
                    query_analysis.get(
                        "requested_columns",
                        [],
                    ),

                "target_columns":
                    query_analysis.get(
                        "target_columns",
                        [],
                    ),
            },

        "query_plan":
            query_plan,

        "computed_evidence":
            evidence_items,

        "execution_report":
            normalised_execution.get(
                "execution_report",
                {},
            ),
    }

    # ======================================================
    # PROMPT
    # ======================================================

    prompt = f"""
You are the final answer layer of a data analytics system.

The user's question has already been analysed, planned, and
executed using deterministic Python/Pandas computations.

Your job is ONLY to explain the supplied computed evidence.

STRICT RULES:

1. Use only the computed evidence supplied below.

2. Do not invent statistics, values, sample sizes,
   relationships, categories, or dataset facts.

3. Do not perform new calculations yourself.

4. If evidence is insufficient, explicitly say so.

5. Correlation and grouped differences are associations.
   Never describe them as causal effects.

6. Do not claim statistical significance unless a
   significance test and its result are explicitly present
   in the evidence.

7. Do not call something predictive, important, or a
   feature importance score unless the evidence explicitly
   supports that statement.

8. Mention missing-data or small-sample limitations when
   they materially affect interpretation.

9. Answer the user's actual question directly.

10. Keep the answer concise and useful.

Return ONLY valid JSON using exactly this structure:

{{
    "answer": "direct natural-language answer",
    "key_points": [
        "important evidence-backed point"
    ],
    "cautions": [
        "relevant analytical limitation"
    ]
}}

ANALYTICAL CONTEXT:

{json.dumps(
    grounded_context,
    indent=2,
    default=str,
)}
"""

    # ======================================================
    # LLM GENERATION
    # ======================================================

    llm_used = False
    llm_error = None

    try:

        response = generate_json_response(
            prompt=prompt,
            temperature=0.1,
        )

        if not isinstance(
            response,
            dict,
        ):
            raise ValueError(
                "Answer generator expected a JSON object."
            )

        answer = response.get(
            "answer",
            ""
        )

        key_points = response.get(
            "key_points",
            [],
        )

        cautions = response.get(
            "cautions",
            [],
        )

        if not isinstance(
            answer,
            str,
        ):
            answer = str(
                answer
            )

        answer = answer.strip()

        if not answer:
            raise ValueError(
                "LLM returned an empty answer."
            )

        if not isinstance(
            key_points,
            list,
        ):
            key_points = []

        if not isinstance(
            cautions,
            list,
        ):
            cautions = []

        key_points = [
            str(point)
            for point in key_points
        ]

        cautions = [
            str(caution)
            for caution in cautions
        ]

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
    # RESULT
    # ======================================================

    evidence_tools = []

    for evidence in evidence_items:

        tool = evidence.get(
            "tool"
        )

        if (
            tool
            and tool not in evidence_tools
        ):
            evidence_tools.append(
                tool
            )

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
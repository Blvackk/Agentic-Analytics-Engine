# tests/test_insight_generator.py

from pprint import pprint

from src.agents.insight_generator import generate_insights
from src.agents.semantic_analyzer import analyze_semantics

from src.tools.data_loader import load_csv
from src.tools.statistics import (
    get_numerical_summary,
    get_categorical_summary,
    get_correlation_matrix,
)


def main():
    """
    Test the Insight Generator independently.

    Pipeline tested:

        Load Dataset
             |
             v
        Statistics
             |
             v
        Semantic Analysis
             |
             v
        Target Analysis
             |
             v
        Insight Generator
             |
             v
        Structured LLM Insights
             |
             v
        Validation
    """

    print(
        "\n========== INSIGHT GENERATOR TEST ==========\n"
    )

    # ==================================================
    # 1. LOAD DATASET
    # ==================================================

    dataframe = load_csv(
        "data/samples/dirty_customers.csv"
    )

    print(
        "Dataset loaded:",
        dataframe.shape
    )

    print(
        "Columns:",
        dataframe.columns.tolist()
    )

    # ==================================================
    # 2. CALCULATE STATISTICS
    # ==================================================

    print(
        "\n========== STATISTICAL ANALYSIS ==========\n"
    )

    numerical_summary = get_numerical_summary(
        dataframe
    )

    categorical_summary = get_categorical_summary(
        dataframe
    )

    correlation_matrix = get_correlation_matrix(
        dataframe
    )

    print(
        "Numerical summary: COMPLETE"
    )

    print(
        "Categorical summary: COMPLETE"
    )

    print(
        "Correlation matrix: COMPLETE"
    )

    # ==================================================
    # 3. SEMANTIC ANALYSIS
    # ==================================================

    print(
        "\n========== SEMANTIC ANALYSIS ==========\n"
    )

    semantic_analysis = analyze_semantics(
        dataframe
    )

    identifier_columns = semantic_analysis.get(
        "identifier_columns",
        []
    )

    target_candidates = semantic_analysis.get(
        "target_candidates",
        []
    )

    feature_columns = semantic_analysis.get(
        "feature_columns",
        []
    )

    print(
        "Identifiers:",
        identifier_columns
    )

    print(
        "Target candidates:",
        target_candidates
    )

    print(
        "Features:",
        feature_columns
    )

    print(
        "LLM used:",
        semantic_analysis.get(
            "llm_used"
        )
    )

    print(
        "LLM error:",
        semantic_analysis.get(
            "llm_error"
        )
    )

    # ==================================================
    # 4. BUILD TARGET ANALYSIS
    # ==================================================

    print(
        "\n========== TARGET ANALYSIS ==========\n"
    )

    target_analysis = {}

    for target in target_candidates:

        # ----------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------

        if target not in dataframe.columns:

            print(
                f"Skipping missing target: {target}"
            )

            continue

        target_series = dataframe[
            target
        ]

        # ----------------------------------------------
        # CATEGORICAL TARGET
        # ----------------------------------------------

        if (
            not target_series.empty
            and (
                target_series.dtype == "object"
                or str(
                    target_series.dtype
                ) == "category"
                or str(
                    target_series.dtype
                ) == "string"
                or str(
                    target_series.dtype
                ) == "bool"
            )
        ):

            distribution = (
                target_series
                .value_counts(
                    dropna=False
                )
                .to_dict()
            )

            target_analysis[
                target
            ] = {
                "target_type":
                    "categorical",

                "distribution": {
                    str(key):
                        int(value)

                    for key, value
                    in distribution.items()
                },
            }

        # ----------------------------------------------
        # NUMERICAL TARGET
        # ----------------------------------------------

        else:

            target_description = (
                target_series
                .describe()
                .round(2)
                .to_dict()
            )

            target_analysis[
                target
            ] = {
                "target_type":
                    "numerical",

                "summary": {
                    str(key):
                        (
                            float(value)
                            if hasattr(
                                value,
                                "item"
                            )
                            else value
                        )

                    for key, value
                    in target_description.items()
                },
            }

    if target_analysis:

        pprint(
            target_analysis
        )

    else:

        print(
            "No target analysis generated."
        )

    # ==================================================
    # 5. GENERATE LLM INSIGHTS
    # ==================================================

    print(
        "\n========== GENERATING INSIGHTS ==========\n"
    )

    print(
        "Sending structured analytical context "
        "to the LLM..."
    )

    insights = generate_insights(
        numerical_summary=
            numerical_summary,

        categorical_summary=
            categorical_summary,

        correlation_matrix=
            correlation_matrix,

        semantic_analysis=
            semantic_analysis,

        target_analysis=
            target_analysis,
    )

    print(
        "Insight generation: COMPLETE"
    )

    # ==================================================
    # 6. OVERALL SUMMARY
    # ==================================================

    print(
        "\n========== OVERALL SUMMARY ==========\n"
    )

    summary = insights.get(
        "summary",
        ""
    )

    if summary:

        print(
            summary
        )

    else:

        print(
            "No overall summary generated."
        )

    # ==================================================
    # 7. KEY INSIGHTS
    # ==================================================

    print(
        "\n========== KEY INSIGHTS ==========\n"
    )

    key_insights = insights.get(
        "key_insights",
        []
    )

    if key_insights:

        for index, insight in enumerate(
            key_insights,
            start=1
        ):

            print(
                f"Insight {index}:"
            )

            pprint(
                insight
            )

            print()

    else:

        print(
            "No key insights generated."
        )

    # ==================================================
    # 8. TARGET INSIGHTS
    # ==================================================

    print(
        "\n========== TARGET INSIGHTS ==========\n"
    )

    target_insights = insights.get(
        "target_insights",
        []
    )

    if target_insights:

        for index, insight in enumerate(
            target_insights,
            start=1
        ):

            print(
                f"Target Insight {index}:"
            )

            pprint(
                insight
            )

            print()

    else:

        print(
            "No target insights generated."
        )

    # ==================================================
    # 9. DATA CAUTIONS
    # ==================================================

    print(
        "\n========== DATA CAUTIONS ==========\n"
    )

    data_cautions = insights.get(
        "data_cautions",
        []
    )

    if data_cautions:

        for index, caution in enumerate(
            data_cautions,
            start=1
        ):

            print(
                f"{index}. {caution}"
            )

    else:

        print(
            "No data cautions generated."
        )

    # ==================================================
    # 10. VALIDATION
    # ==================================================

    print(
        "\n========== VALIDATION ==========\n"
    )

    # --------------------------------------------------
    # RESPONSE TYPE
    # --------------------------------------------------

    assert isinstance(
        insights,
        dict
    ), (
        "Insight generator must return a dictionary."
    )

    print(
        "Response type: PASSED"
    )

    # --------------------------------------------------
    # REQUIRED KEYS
    # --------------------------------------------------

    required_keys = {
        "summary",
        "key_insights",
        "target_insights",
        "data_cautions",
    }

    missing_keys = (
        required_keys
        - set(
            insights.keys()
        )
    )

    assert not missing_keys, (
        f"Missing insight fields: {missing_keys}"
    )

    print(
        "Required fields: PASSED"
    )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    assert isinstance(
        insights["summary"],
        str
    ), (
        "summary must be a string."
    )

    print(
        "Summary field: PASSED"
    )

    # --------------------------------------------------
    # KEY INSIGHTS TYPE
    # --------------------------------------------------

    assert isinstance(
        insights["key_insights"],
        list
    ), (
        "key_insights must be a list."
    )

    print(
        "Key insights field: PASSED"
    )

    # --------------------------------------------------
    # TARGET INSIGHTS TYPE
    # --------------------------------------------------

    assert isinstance(
        insights["target_insights"],
        list
    ), (
        "target_insights must be a list."
    )

    print(
        "Target insights field: PASSED"
    )

    # --------------------------------------------------
    # DATA CAUTIONS TYPE
    # --------------------------------------------------

    assert isinstance(
        insights["data_cautions"],
        list
    ), (
        "data_cautions must be a list."
    )

    print(
        "Data cautions field: PASSED"
    )

    # ==================================================
    # 11. KEY INSIGHT STRUCTURE
    # ==================================================

    allowed_importance = {
        "high",
        "medium",
        "low",
    }

    for insight in insights[
        "key_insights"
    ]:

        assert isinstance(
            insight,
            dict
        ), (
            "Each key insight must "
            "be a dictionary."
        )

        assert "title" in insight, (
            "Key insight missing title."
        )

        assert "insight" in insight, (
            "Key insight missing insight text."
        )

        assert "evidence" in insight, (
            "Key insight missing evidence."
        )

        assert "importance" in insight, (
            "Key insight missing importance."
        )

        assert insight[
            "importance"
        ] in allowed_importance, (
            "importance must be "
            "high, medium, or low."
        )

    print(
        "Key insight structure: PASSED"
    )

    # ==================================================
    # 12. TARGET INSIGHT STRUCTURE
    # ==================================================

    for insight in insights[
        "target_insights"
    ]:

        assert isinstance(
            insight,
            dict
        ), (
            "Each target insight must "
            "be a dictionary."
        )

        assert "target" in insight, (
            "Target insight missing target."
        )

        assert "insight" in insight, (
            "Target insight missing insight."
        )

        assert "evidence" in insight, (
            "Target insight missing evidence."
        )

        # ----------------------------------------------
        # TARGET MUST BE REAL
        # ----------------------------------------------

        if target_candidates:

            assert (
                insight["target"]
                in target_candidates
            ), (
                "LLM returned an unknown "
                "target column."
            )

    print(
        "Target insight structure: PASSED"
    )

    # ==================================================
    # 13. OUTPUT SIZE VALIDATION
    # ==================================================

    assert len(
        insights["key_insights"]
    ) <= 4, (
        "Insight Generator returned more "
        "than 4 key insights."
    )

    assert len(
        insights["target_insights"]
    ) <= 2, (
        "Insight Generator returned more "
        "than 2 target insights."
    )

    assert len(
        insights["data_cautions"]
    ) <= 3, (
        "Insight Generator returned more "
        "than 3 data cautions."
    )

    print(
        "Output size limits: PASSED"
    )

    # ==================================================
    # 14. SEMANTIC VALIDATION
    # ==================================================

    assert isinstance(
        identifier_columns,
        list
    )

    assert isinstance(
        target_candidates,
        list
    )

    assert isinstance(
        feature_columns,
        list
    )

    print(
        "Semantic metadata types: PASSED"
    )

    # --------------------------------------------------
    # IDENTIFIERS MUST NOT BE FEATURES
    # --------------------------------------------------

    overlap = set(
        identifier_columns
    ).intersection(
        feature_columns
    )

    assert not overlap, (
        "Identifier columns were incorrectly "
        f"included as features: {overlap}"
    )

    print(
        "Identifier exclusion: PASSED"
    )

    # ==================================================
    # 15. EXPECTED TEST-DATA SEMANTICS
    # ==================================================

    # These checks are intentionally specific to
    # dirty_customers.csv.

    assert (
        "customer_id"
        in identifier_columns
    ), (
        "customer_id should be detected "
        "as an identifier."
    )

    print(
        "customer_id detection: PASSED"
    )

    assert (
        "customer_id"
        not in feature_columns
    ), (
        "customer_id must not be treated "
        "as an analytical feature."
    )

    print(
        "customer_id feature exclusion: PASSED"
    )

    assert (
        "churn"
        in target_candidates
    ), (
        "churn should be detected "
        "as a target candidate."
    )

    print(
        "churn target detection: PASSED"
    )

    # ==================================================
    # 16. TARGET ANALYSIS VALIDATION
    # ==================================================

    if "churn" in target_candidates:

        assert (
            "churn"
            in target_analysis
        ), (
            "Target analysis was not generated "
            "for churn."
        )

        assert (
            target_analysis[
                "churn"
            ][
                "target_type"
            ]
            == "categorical"
        ), (
            "churn should be treated "
            "as categorical."
        )

        assert (
            "distribution"
            in target_analysis[
                "churn"
            ]
        ), (
            "Categorical target analysis "
            "must contain distribution."
        )

    print(
        "Target analysis: PASSED"
    )

    # ==================================================
    # 17. FINAL RESULT
    # ==================================================

    print(
        "\n========== INSIGHT GENERATOR TEST COMPLETE ==========\n"
    )

    print(
        "Statistics         -> PASSED"
    )

    print(
        "Semantic Analysis  -> PASSED"
    )

    print(
        "Target Analysis    -> PASSED"
    )

    print(
        "LLM Insights       -> PASSED"
    )

    print(
        "Output Validation  -> PASSED"
    )


if __name__ == "__main__":
    main()
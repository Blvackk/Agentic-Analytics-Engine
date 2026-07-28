# tests/test_semantic_analyzer.py

from pprint import pprint

from src.agents.semantic_analyzer import analyze_semantics
from src.tools.data_loader import load_csv


def main():
    """
    Test semantic analysis on the sample customer dataset.
    """

    print(
        "\n========== SEMANTIC ANALYZER TEST ==========\n"
    )

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    dataframe = load_csv(
        "data/samples/dirty_customers.csv"
    )

    print(
        f"Dataset loaded: {dataframe.shape}"
    )

    print(
        f"Columns: {dataframe.columns.tolist()}"
    )

    # --------------------------------------------------
    # RUN SEMANTIC ANALYSIS
    # --------------------------------------------------

    print(
        "\nRunning semantic analysis...\n"
    )

    result = analyze_semantics(
        dataframe=dataframe,
        use_llm=True,
    )

    # --------------------------------------------------
    # DISPLAY COLUMN ROLES
    # --------------------------------------------------

    print(
        "========== COLUMN SEMANTIC ROLES ==========\n"
    )

    for column, info in result["columns"].items():

        print(
            f"{column:<20} "
            f"-> {info['role']:<22} "
            f"[source: {info['decision_source']}]"
        )

    # --------------------------------------------------
    # DISPLAY GROUPS
    # --------------------------------------------------

    print(
        "\n========== IDENTIFIER COLUMNS ==========\n"
    )

    pprint(
        result["identifier_columns"]
    )

    print(
        "\n========== TARGET CANDIDATES ==========\n"
    )

    pprint(
        result["target_candidates"]
    )

    print(
        "\n========== FEATURE COLUMNS ==========\n"
    )

    pprint(
        result["feature_columns"]
    )

    # --------------------------------------------------
    # LLM STATUS
    # --------------------------------------------------

    print(
        "\n========== LLM STATUS ==========\n"
    )

    print(
        "LLM used:",
        result["llm_used"]
    )

    if result["llm_error"]:

        print(
            "LLM error:",
            result["llm_error"]
        )

    else:

        print(
            "LLM error: None"
        )

    # --------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------

    print(
        "\n========== VALIDATION ==========\n"
    )

    assert (
        "customer_id"
        in result["identifier_columns"]
    ), (
        "customer_id should have been "
        "detected as an identifier."
    )

    assert (
        result["columns"]["customer_id"]["role"]
        == "identifier"
    )

    assert (
        "churn"
        in result["target_candidates"]
    ), (
        "churn should have been detected "
        "as a possible target."
    )

    assert (
        result["columns"]["churn"]["role"]
        == "possible_target"
    )

    expected_columns = set(
        dataframe.columns
    )

    analysed_columns = set(
        result["columns"].keys()
    )

    assert (
        expected_columns
        == analysed_columns
    ), (
        "Semantic analyzer did not return "
        "all dataset columns."
    )

    print(
        "customer_id identifier detection: PASSED"
    )

    print(
        "churn target detection: PASSED"
    )

    print(
        "all columns analysed: PASSED"
    )

    print(
        "\n========== SEMANTIC ANALYZER TEST COMPLETE ==========\n"
    )


if __name__ == "__main__":
    main()
# tests/test_llm.py

from pprint import pprint

from src.llm.client import (
    DEFAULT_MODEL,
    generate_response,
    generate_json_response,
)


def test_text_response():
    """
    Test normal text generation from the local LLM.
    """

    print(
        "\n========== TESTING TEXT RESPONSE ==========\n"
    )

    print(
        f"Model: {DEFAULT_MODEL}\n"
    )

    prompt = """
You are a data analyst.

A dataset contains the following columns:

- customer_id
- age
- income
- contract_type
- tenure
- churn

Which column is most likely an identifier?

Answer in one short sentence.
"""

    response = generate_response(
        prompt=prompt,
        temperature=0.1,
    )

    print(response)


def test_json_response():
    """
    Test structured JSON generation from the local LLM.
    """

    print(
        "\n========== TESTING JSON RESPONSE ==========\n"
    )

    prompt = """
You are a data analyst analysing the schema of a dataset.

The dataset contains these columns:

- customer_id
- age
- income
- contract_type
- tenure
- churn

Identify which columns are most likely identifiers.

Return JSON using exactly this structure:

{
    "identifier_columns": []
}

Rules:
- Put only column names from the provided dataset in the list.
- Do not invent columns.
- Return valid JSON.
"""

    result = generate_json_response(
        prompt=prompt,
        temperature=0.1,
    )

    pprint(result)

    # --------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------

    if not isinstance(result, dict):
        raise AssertionError(
            "JSON response must be a dictionary."
        )

    if "identifier_columns" not in result:
        raise AssertionError(
            "Response is missing 'identifier_columns'."
        )

    if not isinstance(
        result["identifier_columns"],
        list
    ):
        raise AssertionError(
            "'identifier_columns' must be a list."
        )

    print(
        "\nStructured response validation passed."
    )


def main():

    test_text_response()

    test_json_response()

    print(
        "\n========== LLM TEST COMPLETE ==========\n"
    )

    print(
        "Text generation: PASSED"
    )

    print(
        "JSON generation: PASSED"
    )


if __name__ == "__main__":
    main()
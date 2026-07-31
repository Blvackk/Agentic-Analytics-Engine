# src/llm/client.py

import json
from typing import Any

import requests


# ==========================================================
# OLLAMA CONFIGURATION
# ==========================================================

OLLAMA_BASE_URL = "http://localhost:11434"

DEFAULT_MODEL = "qwen3:4b"

REQUEST_TIMEOUT = 180

DEFAULT_TEXT_NUM_PREDICT = 512

DEFAULT_JSON_NUM_PREDICT = 1024


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def _validate_request_inputs(
    prompt: str,
    temperature: float,
    num_predict: int,
) -> None:
    """
    Validate common inputs used by Ollama requests.
    """

    if not isinstance(prompt, str):
        raise TypeError(
            "prompt must be a string."
        )

    if not prompt.strip():
        raise ValueError(
            "Prompt cannot be empty."
        )

    if not isinstance(
        temperature,
        (int, float),
    ):
        raise TypeError(
            "temperature must be numeric."
        )

    if not 0 <= temperature <= 2:
        raise ValueError(
            "temperature must be between 0 and 2."
        )

    if not isinstance(
        num_predict,
        int,
    ):
        raise TypeError(
            "num_predict must be an integer."
        )

    if num_predict <= 0:
        raise ValueError(
            "num_predict must be greater than 0."
        )


# ==========================================================
# SEND REQUEST TO OLLAMA
# ==========================================================

def _send_ollama_request(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Send a request to Ollama's generate endpoint.
    """

    url = (
        f"{OLLAMA_BASE_URL}/api/generate"
    )

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError as error:

        raise RuntimeError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        ) from error

    except requests.exceptions.Timeout as error:

        raise RuntimeError(
            "The Ollama request timed out."
        ) from error

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"Ollama request failed: {error}"
        ) from error

    try:

        result = response.json()

    except (
        requests.exceptions.JSONDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:

        raise RuntimeError(
            "Ollama returned an invalid HTTP JSON response."
        ) from error

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "Ollama returned an unexpected response format."
        )

    if result.get(
        "error"
    ):
        raise RuntimeError(
            f"Ollama error: {result['error']}"
        )

    return result


# ==========================================================
# RESPONSE EXTRACTION
# ==========================================================

def _extract_generated_text(
    result: dict[str, Any],
    model: str,
    response_type: str,
) -> str:
    """
    Extract generated text returned by Ollama.
    """

    generated_text = result.get(
        "response",
        "",
    )

    if generated_text is None:
        generated_text = ""

    if not isinstance(
        generated_text,
        str,
    ):
        generated_text = str(
            generated_text
        )

    generated_text = (
        generated_text.strip()
    )

    if generated_text:
        return generated_text

    thinking = result.get(
        "thinking",
        "",
    )

    raise RuntimeError(
        f"Ollama returned an empty {response_type} response."
        "\n\n"
        f"Model: {model}\n"
        f"Done: {result.get('done')}\n"
        f"Done reason: {result.get('done_reason')}\n"
        f"Thinking present: {bool(thinking)}\n"
        f"Prompt eval count: "
        f"{result.get('prompt_eval_count')}\n"
        f"Eval count: "
        f"{result.get('eval_count')}"
    )


# ==========================================================
# THINKING CLEANUP
# ==========================================================

def _strip_thinking(
    text: str,
) -> str:
    """
    Remove leaked Qwen <think>...</think> content if the
    model includes reasoning in the response field.
    """

    cleaned = text.strip()

    if "</think>" in cleaned:

        cleaned = cleaned.split(
            "</think>",
            1,
        )[1].strip()

    if cleaned.startswith(
        "<think>"
    ):

        end_position = cleaned.find(
            "</think>"
        )

        if end_position != -1:

            cleaned = cleaned[
                end_position
                + len("</think>"):
            ].strip()

    return cleaned


# ==========================================================
# JSON CLEANING
# ==========================================================

def _strip_markdown_json_fence(
    text: str,
) -> str:
    """
    Remove Markdown JSON code fences if present.
    """

    cleaned = text.strip()

    if not cleaned.startswith(
        "```"
    ):
        return cleaned

    lines = cleaned.splitlines()

    if (
        lines
        and lines[0].strip().startswith(
            "```"
        )
    ):
        lines = lines[1:]

    if (
        lines
        and lines[-1].strip() == "```"
    ):
        lines = lines[:-1]

    return "\n".join(
        lines
    ).strip()


# ==========================================================
# NORMAL TEXT GENERATION
# ==========================================================

def generate_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    num_predict: int = DEFAULT_TEXT_NUM_PREDICT,
) -> str:
    """
    Generate a normal text response using Ollama.
    """

    _validate_request_inputs(
        prompt=prompt,
        temperature=temperature,
        num_predict=num_predict,
    )

    payload = {
        "model":
            model,

        "prompt":
            prompt,

        "stream":
            False,

        "think":
            False,

        "options": {
            "temperature":
                temperature,

            "num_predict":
                num_predict,
        },
    }

    result = _send_ollama_request(
        payload
    )

    generated_text = _extract_generated_text(
        result=result,
        model=model,
        response_type="text",
    )

    generated_text = _strip_thinking(
        generated_text
    )

    if not generated_text:

        raise RuntimeError(
            "No final answer remained after removing "
            "model thinking content."
        )

    return generated_text


# ==========================================================
# STRUCTURED JSON GENERATION
# ==========================================================

def generate_json_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    num_predict: int = DEFAULT_JSON_NUM_PREDICT,
) -> dict[str, Any] | list[Any]:
    """
    Generate structured JSON using Ollama and convert the
    result into a Python dictionary or list.
    """

    _validate_request_inputs(
        prompt=prompt,
        temperature=temperature,
        num_predict=num_predict,
    )

    payload = {
        "model":
            model,

        "prompt":
            prompt,

        "stream":
            False,

        "format":
            "json",

        "think":
            False,

        "options": {
            "temperature":
                temperature,

            "num_predict":
                num_predict,
        },
    }

    result = _send_ollama_request(
        payload
    )

    generated_text = _extract_generated_text(
        result=result,
        model=model,
        response_type="JSON",
    )

    generated_text = _strip_thinking(
        generated_text
    )

    generated_text = _strip_markdown_json_fence(
        generated_text
    )

    if not generated_text:

        raise RuntimeError(
            "No JSON content remained after cleaning "
            "the model response."
        )

    # ======================================================
    # CONVERT JSON STRING -> PYTHON OBJECT
    # ======================================================

    try:

        parsed_response = json.loads(
            generated_text
        )

    except json.JSONDecodeError as error:

        raise RuntimeError(
            "The LLM returned content that could not "
            "be parsed as valid JSON.\n\n"
            f"Model: {model}\n"
            f"Done: {result.get('done')}\n"
            f"Done reason: {result.get('done_reason')}\n"
            f"Prompt eval count: "
            f"{result.get('prompt_eval_count')}\n"
            f"Eval count: "
            f"{result.get('eval_count')}\n\n"
            f"Raw response:\n"
            f"{generated_text}"
        ) from error

    # ======================================================
    # VALIDATE JSON ROOT
    # ======================================================

    if not isinstance(
        parsed_response,
        (dict, list),
    ):

        raise RuntimeError(
            "The LLM JSON response must contain "
            "a JSON object or array at the root."
        )

    return parsed_response
#analyst_state.py

"""
Session state management for the AI Analyst.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


DEFAULT_SUGGESTIONS = [
    "Show missing values",
    "What is the correlation between age and income?",
    "Show histogram of salary",
    "Predict customer churn",
]


def initialize_state() -> None:
    """
    Initialize Streamlit session state.
    """

    defaults: dict[str, Any] = {
        "analyst_messages": [],
        "analyst_history": [],
        "analyst_last_result": None,
        "analyst_processing": False,
        "analyst_suggestions": DEFAULT_SUGGESTIONS,
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


def add_message(
    role: str,
    content: Any,
) -> None:
    """
    Append a chat message.
    """

    st.session_state.analyst_messages.append(
        {
            "role": role,
            "content": content,
        }
    )


def clear_chat() -> None:
    """
    Reset chat history.
    """

    st.session_state.analyst_messages = []

    st.session_state.analyst_history = []

    st.session_state.analyst_last_result = None


def get_messages() -> list[dict[str, Any]]:
    """
    Return stored messages.
    """

    return st.session_state.analyst_messages


def save_last_result(
    result: dict[str, Any],
) -> None:
    """
    Save last router response.
    """

    st.session_state.analyst_last_result = result


def get_last_result() -> dict[str, Any] | None:
    """
    Retrieve last router response.
    """

    return st.session_state.analyst_last_result


__all__ = [
    "initialize_state",
    "add_message",
    "clear_chat",
    "get_messages",
    "save_last_result",
    "get_last_result",
]
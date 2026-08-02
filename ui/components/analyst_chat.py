"""
AI Analyst Chat Components.
"""

from __future__ import annotations

import streamlit as st

from ui.components.analyst_state import (
    get_messages,
)


def render_chat_history() -> None:
    """
    Render chat history.
    """

    for message in get_messages():

        with st.chat_message(
            message["role"]
        ):

            if isinstance(
                message["content"],
                str,
            ):

                st.markdown(
                    message["content"]
                )

            else:

                st.json(
                    message["content"]
                )


def question_box() -> str | None:
    """
    Display chat input.
    """

    return st.chat_input(
        "Ask anything about your dataset..."
    )


__all__ = [
    "render_chat_history",
    "question_box",
]
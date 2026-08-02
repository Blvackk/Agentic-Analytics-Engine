#ui/components/analyst.py
"""
AI Analyst Page.

Main controller for the AI Analyst.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components.analyst_chat import (
    question_box,
    render_chat_history,
)

from ui.components.analyst_download import (
    render_download,
)

from ui.components.analyst_renderer import (
    render_response,
)

from ui.components.analyst_router import (
    execute_question,
)

from ui.components.analyst_state import (
    add_message,
    get_last_result,
    initialize_state,
    save_last_result,
)

from ui.components.analyst_trace import (
    render_trace,
)


def render_analyst(
    dataframe: pd.DataFrame,
) -> None:
    """
    Render the AI Analyst page.
    """

    initialize_state()

    st.title("🤖 AI Analyst")

    st.caption(
        "Ask questions in natural language."
    )

    st.divider()

    render_chat_history()

    question = question_box()

    if question:

        add_message(
            "user",
            question,
        )

        with st.spinner(
            "Thinking..."
        ):

            try:

                router_result = execute_question(
                    dataframe=dataframe,
                    question=question,
                )

                save_last_result(
                    router_result,
                )

                add_message(
                    "assistant",
                    router_result,
                )

            except Exception as error:

                st.error(error)

    result = get_last_result()

    if result:

        st.divider()

        render_response(result)

        st.divider()

        render_trace(result)

        st.divider()

        render_download(result)
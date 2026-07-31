from textwrap import dedent

import streamlit as st


def render_html(content: str) -> None:
    """
    Render application HTML without Markdown parsing.
    """

    cleaned = dedent(content).strip()

    if not cleaned:
        return

    st.html(cleaned)
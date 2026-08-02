"""
Debug Trace Renderer.
"""

from __future__ import annotations

import streamlit as st


def render_trace(
    router_result: dict,
) -> None:
    """
    Show router trace for debugging.
    """

    with st.expander(
        "🔍 Debug Trace"
    ):

        st.json(
            router_result
        )


__all__ = [
    "render_trace",
]
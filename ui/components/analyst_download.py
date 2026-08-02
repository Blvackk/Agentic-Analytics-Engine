"""
Download Component.
"""

from __future__ import annotations

import json

import streamlit as st


def render_download(
    result: dict,
) -> None:
    """
    Download analysis as JSON.
    """

    st.download_button(
        "⬇ Download Result",
        data=json.dumps(
            result,
            indent=4,
        ),
        file_name="analysis.json",
        mime="application/json",
    )


__all__ = [
    "render_download",
]
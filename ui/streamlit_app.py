# ui/streamlit_app.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from html import escape

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.agents.graph import (
    build_graph,
)

from ui.components.overview import render_overview
from ui.components.quality import render_quality
from ui.components.explore import render_explore
from ui.components.analyst import render_analyst
from ui.components.report import render_report
from ui.utils.html import render_html


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic Analytics Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

UPLOAD_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "uploads"
)

CSS_PATH = (
    PROJECT_ROOT
    / "ui"
    / "styles"
    / "app.css"
)

UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CSS
# ============================================================

def load_css() -> None:
    """
    Load the application stylesheet.
    """

    if not CSS_PATH.exists():
        return

    try:
        css = CSS_PATH.read_text(
            encoding="utf-8"
        )

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True,
        )

    except OSError:
        pass


load_css()


# ============================================================
# GRAPH INITIALISATION
# ============================================================

@st.cache_resource
def get_phase1_graph():
    """
    Build the autonomous analytics graph once per
    Streamlit process.
    """

    return build_graph()
phase1_graph = get_phase1_graph()


# ============================================================
# SESSION STATE
# ============================================================

def initialise_session_state() -> None:
    """
    Initialise application-level session state.
    """

    defaults = {
        "analytics_state":
            None,

        "dataset_name":
            None,

        "dataset_path":
            None,

        "active_page":
            "Overview",

        "analysis_running":
            False,

        "analysis_complete":
            False,

        "analysis_error":
            None,

        "analyst_messages":
            [],

    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_state()


# ============================================================
# HELPERS
# ============================================================

def get_analytics_state() -> dict[str, Any] | None:

    value = st.session_state.get(
        "analytics_state"
    )

    if isinstance(value, dict):
        return value

    return None


def get_active_dataframe() -> pd.DataFrame | None:

    state = get_analytics_state()

    if not state:
        return None

    cleaned = state.get(
        "cleaned_dataframe"
    )

    if isinstance(
        cleaned,
        pd.DataFrame,
    ):
        return cleaned

    dataframe = state.get(
        "dataframe"
    )

    if isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return dataframe

    return None


def reset_analysis() -> None:
    """
    Clear the current analytical session.
    """

    st.session_state[
        "analytics_state"
    ] = None

    st.session_state[
        "dataset_name"
    ] = None

    st.session_state[
        "dataset_path"
    ] = None

    st.session_state[
        "analysis_complete"
    ] = False

    st.session_state[
        "analysis_running"
    ] = False

    st.session_state[
        "analysis_error"
    ] = None

    st.session_state[
        "active_page"
    ] = "Overview"

    st.session_state[
        "analyst_messages"
    ] = []

    st.session_state[
        "analyst_pending_question"
    ] = None


def reset_conversation() -> None:

    st.session_state[
        "analyst_messages"
    ] = []

    st.session_state[
        "analyst_pending_question"
    ] = None


def sanitise_filename(
    filename: str,
) -> str:
    """
    Keep only a safe filename, not a user-supplied path.
    """

    filename = Path(
        filename
    ).name

    safe = "".join(
        character
        for character in filename
        if (
            character.isalnum()
            or character
            in {
                ".",
                "_",
                "-",
            }
        )
    )

    if not safe:
        safe = "dataset.csv"

    return safe


def save_uploaded_file(
    uploaded_file: Any,
) -> Path:
    """
    Save uploaded CSV to data/uploads.
    """

    filename = sanitise_filename(
        uploaded_file.name
    )

    destination = (
        UPLOAD_DIRECTORY
        / filename
    )

    destination.write_bytes(
        uploaded_file.getvalue()
    )

    return destination


def preview_uploaded_file(
    uploaded_file: Any,
) -> pd.DataFrame | None:
    """
    Read a lightweight preview before Phase 1 execution.
    """

    try:

        uploaded_file.seek(0)

        dataframe = pd.read_csv(
            uploaded_file
        )

        uploaded_file.seek(0)

        return dataframe

    except Exception:

        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        return None


def run_analysis(
    dataset_path: Path,
    dataset_name: str,
) -> bool:
    """
    Execute Phase 1 exactly when the user requests analysis.

    The result is persisted in Streamlit session state.
    """

    st.session_state[
        "analysis_running"
    ] = True

    st.session_state[
        "analysis_error"
    ] = None

    try:

        initial_state = {
            "dataset_path":
                str(dataset_path),

            "errors":
                [],
        }

        result = phase1_graph.invoke(
            initial_state
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "The analytics graph returned "
                "an invalid state."
            )

        st.session_state[
            "analytics_state"
        ] = result

        st.session_state[
            "dataset_name"
        ] = dataset_name

        st.session_state[
            "dataset_path"
        ] = str(dataset_path)

        st.session_state[
            "analysis_complete"
        ] = True

        st.session_state[
            "active_page"
        ] = "Overview"

        from src.agents.semantic_analyzer import (
            analyze_semantics,
        )

        dataframe = result.get(
            "cleaned_dataframe",
        )

        if dataframe is None:

            dataframe = result.get(
                "dataframe",
            )

        result["semantic_analysis"] = analyze_semantics(
            dataframe,
        )

        return True

    except Exception as error:

        st.session_state[
            "analytics_state"
        ] = None

        st.session_state[
            "analysis_complete"
        ] = False

        st.session_state[
            "analysis_error"
        ] = str(error)

        return False

    finally:

        st.session_state[
            "analysis_running"
        ] = False


# ============================================================
# BRAND
# ============================================================

def render_brand() -> None:

    render_html(
        """
        <div class="aa-brand">

            <div class="aa-brand-mark">
                ◈
            </div>

            <div>

                <div class="aa-brand-name">
                    Agentic Analytics
                </div>

                <div class="aa-brand-subtitle">
                    Autonomous data intelligence
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# LANDING PAGE
# ============================================================

def render_landing_page() -> None:

    render_brand()

    render_html(
        """
        <div class="aa-hero">

            <div class="aa-eyebrow">
                AGENTIC ANALYTICS ENGINE
            </div>

            <div class="aa-hero-title">
                From raw data to
                analytical intelligence.
            </div>

            <div class="aa-hero-description">
                Upload a dataset and let the autonomous
                workflow profile, validate, clean, understand,
                explore and interpret it. Then investigate the
                results through conversational analytics.
            </div>

        </div>
        """
    )

    left, right = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    with left:

        render_html(
            """
            <div class="aa-section">

                <div class="aa-section-label">
                    New analysis
                </div>

                <div class="aa-section-description">
                    Start with a CSV dataset.
                </div>

            </div>
            """
    )

        uploaded_file = st.file_uploader(
            "Upload CSV dataset",
            type=["csv"],
            key="dataset_uploader",
            help=(
                "Upload a CSV file to start "
                "the autonomous analysis workflow."
            ),
        )

        if uploaded_file is not None:

            preview = preview_uploaded_file(
                uploaded_file
            )

            if preview is None:

                st.error(
                    "The uploaded file could not "
                    "be read as CSV."
                )

            else:

                rows, columns = preview.shape

                numeric = len(
                    preview.select_dtypes(
                        include="number"
                    ).columns
                )

                missing = int(
                    preview.isna()
                    .sum()
                    .sum()
                )

                render_html(
                    f"""
                    <div class="aa-kpi-grid">

                        <div class="aa-kpi">
                            <div class="aa-kpi-label">
                                Rows
                            </div>
                            <div class="aa-kpi-value">
                                {rows:,}
                            </div>
                        </div>

                        <div class="aa-kpi">
                            <div class="aa-kpi-label">
                                Columns
                            </div>
                            <div class="aa-kpi-value">
                                {columns:,}
                            </div>
                        </div>

                        <div class="aa-kpi">
                            <div class="aa-kpi-label">
                                Numeric
                            </div>
                            <div class="aa-kpi-value">
                                {numeric:,}
                            </div>
                        </div>

                        <div class="aa-kpi">
                            <div class="aa-kpi-label">
                                Missing
                            </div>
                            <div class="aa-kpi-value">
                                {missing:,}
                            </div>
                        </div>

                    </div>
                    """
    )

                with st.expander(
                    "Preview dataset",
                    expanded=True,
                ):

                    st.dataframe(
                        preview.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                if st.button(
                    "Run autonomous analysis",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state[
                        "analysis_running"
                    ],
                ):

                    try:

                        dataset_path = (
                            save_uploaded_file(
                                uploaded_file
                            )
                        )

                    except OSError as error:

                        st.error(
                            "The uploaded dataset "
                            f"could not be saved: {error}"
                        )

                    else:

                        with st.spinner(
                            "Agents are analysing "
                            "the dataset..."
                        ):

                            success = run_analysis(
                                dataset_path=dataset_path,
                                dataset_name=(
                                    uploaded_file.name
                                ),
                            )

                        if success:

                            st.rerun()

    with right:

        render_html(
            """
            <div class="aa-section">

                <div class="aa-section-label">
                    Workflow
                </div>

                <div class="aa-section-description">
                    One dataset. Multiple specialised agents.
                </div>

            </div>

            <div class="aa-panel">

                <div class="aa-panel-title">
                    01 · Understand
                </div>

                <div class="aa-panel-description">
                    Profile schema, data types, structure
                    and dataset quality.
                </div>

            </div>

            <div class="aa-panel">

                <div class="aa-panel-title">
                    02 · Prepare
                </div>

                <div class="aa-panel-description">
                    Detect quality problems, clean data and
                    validate transformations.
                </div>

            </div>

            <div class="aa-panel">

                <div class="aa-panel-title">
                    03 · Investigate
                </div>

                <div class="aa-panel-description">
                    Understand semantics, plan EDA and
                    execute analytical tools.
                </div>

            </div>

            <div class="aa-panel">

                <div class="aa-panel-title">
                    04 · Explain
                </div>

                <div class="aa-panel-description">
                    Generate insights, reports and
                    evidence-grounded conversational answers.
                </div>

            </div>
            """
    )

    analysis_error = (
        st.session_state.get(
            "analysis_error"
        )
    )

    if analysis_error:

        st.error(
            "The analysis workflow failed."
        )

        with st.expander(
            "Error details"
        ):

            st.code(
                analysis_error
            )


# ============================================================
# SIDEBAR
# ============================================================

PAGES = [
    "Overview",
    "Quality",
    "Explore",
    "AI Analyst",
    "Report",
]


def render_sidebar() -> str:

    state = get_analytics_state()

    dataframe = get_active_dataframe()

    with st.sidebar:

        render_brand()

        st.markdown("---")

        st.caption(
            "WORKSPACE"
        )

        current_page = (
            st.session_state.get(
                "active_page",
                "Overview",
            )
        )

        if current_page not in PAGES:
            current_page = "Overview"

        selected_page = st.radio(
            "Workspace navigation",
            options=PAGES,
            index=PAGES.index(
                current_page
            ),
            label_visibility="collapsed",
            key="workspace_navigation",
        )

        st.session_state[
            "active_page"
        ] = selected_page

        st.markdown("---")

        st.caption(
            "ACTIVE DATASET"
        )

        dataset_name = (
            st.session_state.get(
                "dataset_name"
            )
            or "Dataset"
        )

        st.markdown(
            f"**{dataset_name}**"
        )

        if dataframe is not None:

            rows, columns = dataframe.shape

            st.caption(
                f"{rows:,} rows · "
                f"{columns:,} columns"
            )

            missing = int(
                dataframe.isna()
                .sum()
                .sum()
            )

            st.caption(
                f"{missing:,} missing values "
                "in active data"
            )

        if state:

            errors = state.get(
                "errors",
                [],
            )

            if not isinstance(
                errors,
                list,
            ):
                errors = []

            if errors:

                st.warning(
                    f"{len(errors)} workflow "
                    "issue(s) recorded"
                )

            else:

                st.success(
                    "Analysis complete"
                )

        st.markdown("---")

        if st.button(
            "Analyse another dataset",
            use_container_width=True,
        ):

            reset_analysis()
            st.rerun()

    return selected_page


# ============================================================
# APPLICATION HEADER
# ============================================================

def render_application_header() -> None:

    dataset_name = (
        st.session_state.get(
            "dataset_name"
        )
        or "Dataset"
    )

    safe_dataset_name = escape(str(dataset_name))

    render_html(
        f"""
        <div class="aa-app-topbar">

            <div>

                <div class="aa-app-context">
                    ANALYTICS WORKSPACE
                </div>

                <div class="aa-app-dataset">
                    {safe_dataset_name}
                </div>

            </div>

            <div class="aa-status">
                <span class="aa-status-dot"></span>
                Analysis ready
            </div>

        </div>
        """
    )


# ============================================================
# WORKSPACE ROUTER
# ============================================================

def render_workspace(
    page: str,
) -> None:

    state = get_analytics_state()

    if state is None:

        st.warning(
            "No analytical state is available."
        )

        return

    if page == "Overview":

        render_overview(
            state
        )

    elif page == "Quality":

        render_quality(
            state
        )

    elif page == "Explore":

        render_explore(
            state
        )

    elif page == "AI Analyst":

        dataframe = get_active_dataframe()

        if dataframe is None:

            st.warning(
                "No dataset available."
            )

            return

        render_analyst(
            dataframe
        )

    elif page == "Report":

        render_report(
            state
        )

    else:

        render_overview(
            state
        )


# ============================================================
# APPLICATION
# ============================================================

def main() -> None:

    state = get_analytics_state()

    if state is None:

        render_landing_page()

        return

    page = render_sidebar()

    render_application_header()

    render_workspace(
        page
    )


if __name__ == "__main__":
    main()

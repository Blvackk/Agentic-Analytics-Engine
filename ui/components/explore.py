# ui/components/explore.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ui.utils.html import render_html


# ============================================================
# HELPERS
# ============================================================

def _safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    return []


def _get_dataframe(
    state: dict[str, Any],
) -> pd.DataFrame | None:

    cleaned = state.get("cleaned_dataframe")

    if isinstance(cleaned, pd.DataFrame):
        return cleaned

    dataframe = state.get("dataframe")

    if isinstance(dataframe, pd.DataFrame):
        return dataframe

    return None


def _numeric_columns(
    dataframe: pd.DataFrame,
) -> list[str]:

    return dataframe.select_dtypes(
        include="number"
    ).columns.tolist()


def _categorical_columns(
    dataframe: pd.DataFrame,
) -> list[str]:

    return dataframe.select_dtypes(
        exclude="number"
    ).columns.tolist()


# ============================================================
# HEADER
# ============================================================

def _render_header(
    dataframe: pd.DataFrame,
) -> None:

    numeric = _numeric_columns(dataframe)
    categorical = _categorical_columns(dataframe)

    render_html(
        """
        <div class="aa-page-header">

            <div class="aa-eyebrow">
                Exploratory analysis
            </div>

            <div class="aa-page-title">
                Explore
            </div>

            <div class="aa-page-description">
                Examine distributions, descriptive statistics,
                relationships and generated analytical evidence
                from the active dataset.
            </div>

        </div>
        """
    )

    render_html(
        f"""
        <div class="aa-kpi-grid">

            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Numerical fields
                </div>

                <div class="aa-kpi-value">
                    {len(numeric):,}
                </div>

                <div class="aa-kpi-detail">
                    Quantitative variables
                </div>
            </div>


            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Categorical fields
                </div>

                <div class="aa-kpi-value">
                    {len(categorical):,}
                </div>

                <div class="aa-kpi-detail">
                    Qualitative variables
                </div>
            </div>


            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Observations
                </div>

                <div class="aa-kpi-value">
                    {len(dataframe):,}
                </div>

                <div class="aa-kpi-detail">
                    Active analytical rows
                </div>
            </div>


            <div class="aa-kpi">
                <div class="aa-kpi-label">
                    Dimensions
                </div>

                <div class="aa-kpi-value">
                    {dataframe.shape[1]:,}
                </div>

                <div class="aa-kpi-detail">
                    Total dataset variables
                </div>
            </div>

        </div>
        """
    )


# ============================================================
# STATISTICS
# ============================================================

def _render_statistics(
    state: dict[str, Any],
    dataframe: pd.DataFrame,
) -> None:

    render_html(
        """
        <div class="aa-section">
            <div class="aa-section-label">
                Statistical profile
            </div>

            <div class="aa-section-description">
                Descriptive characteristics of numerical
                and categorical variables.
            </div>
        </div>
        """
    )

    numerical_summary = _safe_dict(
        state.get("numerical_summary")
    )

    categorical_summary = _safe_dict(
        state.get("categorical_summary")
    )

    tab_numeric, tab_categorical = st.tabs(
        [
            "Numerical",
            "Categorical",
        ]
    )

    with tab_numeric:

        numeric_columns = _numeric_columns(
            dataframe
        )

        if not numeric_columns:

            st.info(
                "No numerical variables are available."
            )

        else:

            try:
                summary = (
                    dataframe[
                        numeric_columns
                    ]
                    .describe()
                    .transpose()
                    .reset_index()
                    .rename(
                        columns={
                            "index": "Field"
                        }
                    )
                )

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as error:

                st.warning(
                    "Numerical statistics could not "
                    f"be rendered: {error}"
                )

            if numerical_summary:

                with st.expander(
                    "Pipeline numerical summary"
                ):
                    st.json(
                        numerical_summary
                    )

    with tab_categorical:

        categorical_columns = (
            _categorical_columns(
                dataframe
            )
        )

        if not categorical_columns:

            st.info(
                "No categorical variables are available."
            )

        else:

            rows = []

            for column in categorical_columns:

                series = dataframe[column]

                non_null = series.dropna()

                if non_null.empty:
                    most_common = "—"
                    frequency = 0

                else:
                    counts = (
                        non_null
                        .astype(str)
                        .value_counts()
                    )

                    most_common = counts.index[0]
                    frequency = int(
                        counts.iloc[0]
                    )

                rows.append(
                    {
                        "Field": column,
                        "Unique": int(
                            series.nunique(
                                dropna=True
                            )
                        ),
                        "Most common": (
                            most_common
                        ),
                        "Frequency": frequency,
                        "Missing": int(
                            series.isna().sum()
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            selected = st.selectbox(
                "Inspect categorical field",
                categorical_columns,
                key="explore_category",
            )

            counts = (
                dataframe[selected]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(20)
            )

            st.bar_chart(
                counts
            )

            if categorical_summary:

                with st.expander(
                    "Pipeline categorical summary"
                ):
                    st.json(
                        categorical_summary
                    )


# ============================================================
# RELATIONSHIPS
# ============================================================

def _render_relationships(
    state: dict[str, Any],
    dataframe: pd.DataFrame,
) -> None:

    render_html(
        """
        <div class="aa-section">
            <div class="aa-section-label">
                Relationships
            </div>

            <div class="aa-section-description">
                Explore association patterns between
                numerical variables.
            </div>
        </div>
        """
    )

    numeric = _numeric_columns(
        dataframe
    )

    if len(numeric) < 2:

        st.info(
            "At least two numerical variables are "
            "required for relationship analysis."
        )

        return

    correlation = (
        dataframe[numeric]
        .corr(numeric_only=True)
    )

    render_html(
        """
        <div class="aa-panel">
            <div class="aa-panel-title">
                Correlation matrix
            </div>

            <div class="aa-panel-description">
                Pearson correlation across numerical
                variables. Correlation indicates association,
                not causation.
            </div>
        </div>
        """
    )

    st.dataframe(
        correlation.round(3),
        use_container_width=True,
    )

    pairs = []

    for i, column_a in enumerate(numeric):

        for column_b in numeric[
            i + 1:
        ]:

            value = correlation.loc[
                column_a,
                column_b,
            ]

            if pd.notna(value):

                pairs.append(
                    {
                        "Variable A": column_a,
                        "Variable B": column_b,
                        "Correlation": round(
                            float(value),
                            3,
                        ),
                        "Strength": abs(
                            float(value)
                        ),
                    }
                )

    if pairs:

        pair_df = pd.DataFrame(
            pairs
        ).sort_values(
            "Strength",
            ascending=False,
        )

        render_html(
            """
            <div class="aa-section">
                <div class="aa-section-label">
                    Strongest associations
                </div>
            </div>
            """
    )

        st.dataframe(
            pair_df[
                [
                    "Variable A",
                    "Variable B",
                    "Correlation",
                ]
            ].head(10),
            use_container_width=True,
            hide_index=True,
        )

    pipeline_correlation = _safe_dict(
        state.get("correlation_matrix")
    )

    if pipeline_correlation:

        with st.expander(
            "Pipeline correlation evidence"
        ):
            st.json(
                pipeline_correlation
            )


# ============================================================
# TARGET ANALYSIS
# ============================================================

def _render_target_analysis(
    state: dict[str, Any],
    dataframe: pd.DataFrame,
) -> None:

    render_html(
        """
        <div class="aa-section">
            <div class="aa-section-label">
                Target analysis
            </div>

            <div class="aa-section-description">
                Inspect variables identified by the semantic
                agent as potential analytical outcomes.
            </div>
        </div>
        """
    )

    targets = _safe_list(
        state.get("target_candidates")
    )

    semantic = _safe_dict(
        state.get("semantic_analysis")
    )

    if not targets:

        targets = _safe_list(
            semantic.get(
                "target_candidates"
            )
        )

    valid_targets = [
        target
        for target in targets
        if target in dataframe.columns
    ]

    if not valid_targets:

        render_html(
            """
            <div class="aa-panel">
                <div class="aa-panel-title">
                    No target detected
                </div>

                <div class="aa-panel-description">
                    The semantic analysis did not identify
                    a valid target variable in this dataset.
                </div>
            </div>
            """
    )

    else:

        selected_target = st.selectbox(
            "Target variable",
            valid_targets,
            key="explore_target",
        )

        target_series = dataframe[
            selected_target
        ]

        if pd.api.types.is_numeric_dtype(
            target_series
        ):

            st.dataframe(
                target_series
                .describe()
                .to_frame(
                    name=selected_target
                ),
                use_container_width=True,
            )

        else:

            distribution = (
                target_series
                .fillna("Missing")
                .astype(str)
                .value_counts()
            )

            col1, col2 = st.columns(
                [1, 1.4]
            )

            with col1:

                distribution_df = (
                    distribution
                    .rename_axis(
                        selected_target
                    )
                    .reset_index(
                        name="Count"
                    )
                )

                distribution_df[
                    "Share"
                ] = (
                    distribution_df[
                        "Count"
                    ]
                    / distribution_df[
                        "Count"
                    ].sum()
                    * 100
                ).round(1)

                st.dataframe(
                    distribution_df,
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:

                st.bar_chart(
                    distribution
                )

    target_analysis = _safe_dict(
        state.get("target_analysis")
    )

    if target_analysis:

        with st.expander(
            "Target analysis evidence"
        ):
            st.json(
                target_analysis
            )


# ============================================================
# VISUALIZATION GALLERY
# ============================================================

def _render_visualizations(
    state: dict[str, Any],
) -> None:

    render_html(
        """
        <div class="aa-section">
            <div class="aa-section-label">
                Visualization gallery
            </div>

            <div class="aa-section-description">
                Charts selected and generated by the
                autonomous EDA workflow.
            </div>
        </div>
        """
    )

    chart_paths = _safe_list(
        state.get("chart_paths")
    )

    existing_charts = []

    for chart_path in chart_paths:

        try:

            path = Path(
                str(chart_path)
            )

            if path.exists():
                existing_charts.append(
                    path
                )

        except (TypeError, ValueError):
            continue

    if not existing_charts:

        render_html(
            """
            <div class="aa-panel">
                <div class="aa-panel-title">
                    No generated charts available
                </div>

                <div class="aa-panel-description">
                    The workflow did not return visualization
                    files that are currently accessible.
                </div>
            </div>
            """
    )

        return

    for start in range(
        0,
        len(existing_charts),
        2,
    ):

        columns = st.columns(2)

        batch = existing_charts[
            start:start + 2
        ]

        for column, chart in zip(
            columns,
            batch,
        ):

            with column:

                st.image(
                    str(chart),
                    use_container_width=True,
                )

                st.caption(
                    chart.stem
                    .replace("_", " ")
                    .title()
                )


# ============================================================
# EXECUTION DETAILS
# ============================================================

def _render_execution_details(
    state: dict[str, Any],
) -> None:

    analysis_plan = _safe_list(
        state.get("analysis_plan")
    )

    executed_tasks = _safe_list(
        state.get("executed_tasks")
    )

    skipped_tasks = _safe_list(
        state.get("skipped_tasks")
    )

    errors = _safe_list(
        state.get("execution_errors")
    )

    render_html(
        """
        <div class="aa-section">
            <div class="aa-section-label">
                Analysis provenance
            </div>

            <div class="aa-section-description">
                Inspect what the autonomous engine planned,
                executed or skipped.
            </div>
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Planned",
            len(analysis_plan),
        )

    with col2:
        st.metric(
            "Executed",
            len(executed_tasks),
        )

    with col3:
        st.metric(
            "Errors",
            len(errors),
        )

    with st.expander(
        "View execution provenance"
    ):

        st.markdown(
            "#### Analysis plan"
        )

        if analysis_plan:
            st.json(
                analysis_plan
            )
        else:
            st.caption(
                "No analysis plan available."
            )

        st.markdown(
            "#### Executed tasks"
        )

        if executed_tasks:
            st.json(
                executed_tasks
            )
        else:
            st.caption(
                "No executed tasks recorded."
            )

        if skipped_tasks:

            st.markdown(
                "#### Skipped tasks"
            )

            st.json(
                skipped_tasks
            )

        if errors:

            st.markdown(
                "#### Execution errors"
            )

            st.json(
                errors
            )


# ============================================================
# PUBLIC COMPONENT
# ============================================================

def render_explore(
    state: dict[str, Any],
) -> None:
    """
    Render the Explore workspace.
    """

    if not isinstance(state, dict):

        st.error(
            "The analytics workflow state is unavailable."
        )
        return

    dataframe = _get_dataframe(
        state
    )

    if dataframe is None:

        st.warning(
            "No dataframe is available for exploration."
        )
        return

    _render_header(
        dataframe=dataframe
    )

    statistics_tab, relationships_tab, target_tab, charts_tab = (
        st.tabs(
            [
                "Statistics",
                "Relationships",
                "Target Analysis",
                "Visualizations",
            ]
        )
    )

    with statistics_tab:

        _render_statistics(
            state=state,
            dataframe=dataframe,
        )

    with relationships_tab:

        _render_relationships(
            state=state,
            dataframe=dataframe,
        )

    with target_tab:

        _render_target_analysis(
            state=state,
            dataframe=dataframe,
        )

    with charts_tab:

        _render_visualizations(
            state=state,
        )

    _render_execution_details(
        state=state,
    )
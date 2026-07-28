# src/agents/executor.py

import pandas as pd

from src.tools.visualization import (
    create_histogram,
    create_boxplot,
    create_bar_chart,
    create_scatter_plot,
    create_correlation_heatmap,
)


def execute_eda_plan(
    dataframe: pd.DataFrame,
    analysis_plan: list[dict],
) -> dict:
    """
    Execute an EDA plan against a DataFrame.

    The planner decides WHAT should be done.
    The executor decides HOW each supported task
    should be executed.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset on which the plan will be executed.

    analysis_plan : list[dict]
        Analysis tasks produced by the EDA planner.

    Returns
    -------
    dict
        Execution information including generated charts,
        executed tasks, skipped tasks, metadata, and errors.
    """

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if dataframe is None:
        raise ValueError(
            "DataFrame cannot be None."
        )

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if not isinstance(analysis_plan, list):
        raise TypeError(
            "analysis_plan must be a list."
        )

    # --------------------------------------------------
    # RESULT CONTAINERS
    # --------------------------------------------------

    chart_paths = []

    executed_tasks = []

    skipped_tasks = []

    execution_errors = []

    semantic_metadata = {}

    target_analysis = {}

    # --------------------------------------------------
    # EXECUTE EACH TASK
    # --------------------------------------------------

    for task_number, task in enumerate(
        analysis_plan,
        start=1,
    ):

        # --------------------------------------------------
        # VALIDATE TASK
        # --------------------------------------------------

        if not isinstance(task, dict):

            execution_errors.append(
                {
                    "task_number": task_number,
                    "task": task,
                    "error": "Task must be a dictionary.",
                }
            )

            continue

        tool_name = task.get("tool")

        if not tool_name:

            execution_errors.append(
                {
                    "task_number": task_number,
                    "task": task,
                    "error":
                        "Task does not contain a tool name.",
                }
            )

            continue

        try:

            # ==================================================
            # SEMANTIC METADATA
            # ==================================================

            if tool_name == "semantic_metadata":

                identifier_columns = task.get(
                    "identifier_columns",
                    [],
                )

                target_columns = task.get(
                    "target_columns",
                    [],
                )

                semantic_metadata = {
                    "identifier_columns":
                        identifier_columns,

                    "target_columns":
                        target_columns,
                }

                executed_tasks.append(
                    {
                        "task_number":
                            task_number,

                        "tool":
                            tool_name,

                        "identifier_columns":
                            identifier_columns,

                        "target_columns":
                            target_columns,
                    }
                )

            # ==================================================
            # STATISTICAL TASKS
            # ==================================================

            elif tool_name in {
                "numerical_summary",
                "categorical_summary",
                "correlation_matrix",
            }:

                # These statistics are already calculated
                # elsewhere by the statistical analysis node.
                # The executor therefore records them as
                # intentionally skipped rather than duplicated.

                skipped_tasks.append(
                    {
                        "task_number":
                            task_number,

                        "tool":
                            tool_name,

                        "reason":
                            "Already calculated by the "
                            "statistical analysis node.",
                    }
                )

            # ==================================================
            # HISTOGRAM
            # ==================================================

            elif tool_name == "histogram":

                column = task.get(
                    "column"
                )

                if not column:
                    raise ValueError(
                        "Histogram task requires 'column'."
                    )

                chart_path = create_histogram(
                    dataframe=dataframe,
                    column=column,
                )

                chart_paths.append(
                    chart_path
                )

                executed_tasks.append(
                    {
                        "task_number":
                            task_number,

                        "tool":
                            tool_name,

                        "column":
                            column,

                        "output":
                            chart_path,
                    }
                )

            # ==================================================
            # BOX PLOT
            # ==================================================

            elif tool_name == "boxplot":

                column = task.get(
                    "column"
                )

                if not column:
                    raise ValueError(
                        "Boxplot task requires 'column'."
                    )

                chart_path = create_boxplot(
                    dataframe=dataframe,
                    column=column,
                )

                chart_paths.append(
                    chart_path
                )

                executed_tasks.append(
                    {
                        "task_number":
                            task_number,

                        "tool":
                            tool_name,

                        "column":
                            column,

                        "output":
                            chart_path,
                    }
                )

            # ==================================================
            # BAR CHART
            # ==================================================

            elif tool_name == "bar_chart":

                column = task.get(
                    "column"
                )

                if not column:
                    raise ValueError(
                        "Bar chart task requires 'column'."
                    )

                chart_path = create_bar_chart(
                    dataframe=dataframe,
                    column=column,
                )

                chart_paths.append(
                    chart_path
                )

                executed_tasks.append(
                    {
                        "task_number":
                            task_number,

                        "tool":
                            tool_name,

                        "column":
                            column,

                        "output":
                            chart_path,
                    }
                )

            # ==================================================
            # SCATTER PLOT
            # ==================================================

            elif tool_name == "scatter_plot":

                x_column = task.get(
                    "x_column"
                )

                y_column = task.get(
                    "y_column"
                )

                if not x_column or not y_column:

                    raise ValueError(
                        "Scatter plot task requires "
                        "'x_column' and 'y_column'."
                    )

                chart_path = create_scatter_plot(
                    dataframe=dataframe,
                    x_column=x_column,
                    y_column=y_column,
                )

                chart_paths.append(
                    chart_path
                )

                executed_tasks.append(
                    {
                        "task_number":
                            task_number,

                        "tool":
                            tool_name,

                        "x_column":
                            x_column,

                        "y_column":
                            y_column,

                        "output":
                            chart_path,
                    }
                )

            # ==================================================
            # CORRELATION HEATMAP
            # ==================================================

            elif tool_name == "correlation_heatmap":

                selected_columns = task.get(
                    "columns",
                    [],
                )

                # The planner may deliberately exclude
                # identifiers such as customer_id.
                #
                # Therefore we create a subset DataFrame
                # instead of passing the complete dataset.

                if selected_columns:

                    valid_columns = [
                        column
                        for column in selected_columns
                        if column in dataframe.columns
                    ]

                    if len(valid_columns) < 2:

                        raise ValueError(
                            "Correlation heatmap requires "
                            "at least two valid columns."
                        )

                    heatmap_dataframe = dataframe[
                        valid_columns
                    ]

                else:

                    heatmap_dataframe = dataframe

                    valid_columns = (
                        heatmap_dataframe
                        .select_dtypes(
                            include=["number"]
                        )
                        .columns
                        .tolist()
                    )

                chart_path = (
                    create_correlation_heatmap(
                        dataframe=heatmap_dataframe
                    )
                )

                chart_paths.append(
                    chart_path
                )

                executed_tasks.append(
                    {
                        "task_number":
                            task_number,

                        "tool":
                            tool_name,

                        "columns":
                            valid_columns,

                        "output":
                            chart_path,
                    }
                )

            # ==================================================
            # TARGET DISTRIBUTION
            # ==================================================

            elif tool_name == "target_distribution":

                target_column = task.get(
                    "target"
                )

                target_type = task.get(
                    "target_type"
                )

                if not target_column:

                    raise ValueError(
                        "target_distribution requires "
                        "'target'."
                    )

                if target_column not in dataframe.columns:

                    raise ValueError(
                        f"Target column '{target_column}' "
                        "does not exist."
                    )

                # ----------------------------------------------
                # CATEGORICAL TARGET
                # ----------------------------------------------

                if target_type == "categorical":

                    chart_path = create_bar_chart(
                        dataframe=dataframe,
                        column=target_column,
                    )

                    counts = (
                        dataframe[target_column]
                        .astype("object")
                        .where(
                            dataframe[target_column].notna(),
                            "Missing",
                        )
                        .value_counts()
                    )

                    distribution = {
                        str(key): int(value)
                        for key, value in counts.items()
                    }

                # ----------------------------------------------
                # NUMERICAL TARGET
                # ----------------------------------------------

                elif target_type == "numerical":

                    chart_path = create_histogram(
                        dataframe=dataframe,
                        column=target_column,
                    )

                    series = (
                        dataframe[target_column]
                        .dropna()
                    )

                    distribution = {
                        "count":
                            int(series.count()),

                        "mean":
                            (
                                round(
                                    float(series.mean()),
                                    2,
                                )
                                if not series.empty
                                else None
                            ),

                        "median":
                            (
                                round(
                                    float(series.median()),
                                    2,
                                )
                                if not series.empty
                                else None
                            ),

                        "min":
                            (
                                float(series.min())
                                if not series.empty
                                else None
                            ),

                        "max":
                            (
                                float(series.max())
                                if not series.empty
                                else None
                            ),
                    }

                else:

                    raise ValueError(
                        "target_type must be either "
                        "'categorical' or 'numerical'."
                    )

                # Store target analysis separately.
                target_analysis[
                    target_column
                ] = {
                    "target_type":
                        target_type,

                    "distribution":
                        distribution,

                    "chart_path":
                        chart_path,
                }

                # Avoid duplicate path entries.
                if chart_path not in chart_paths:

                    chart_paths.append(
                        chart_path
                    )

                executed_tasks.append(
                    {
                        "task_number":
                            task_number,

                        "tool":
                            tool_name,

                        "target":
                            target_column,

                        "target_type":
                            target_type,

                        "output":
                            chart_path,
                    }
                )

            # ==================================================
            # UNKNOWN TOOL
            # ==================================================

            else:

                skipped_tasks.append(
                    {
                        "task_number":
                            task_number,

                        "tool":
                            tool_name,

                        "reason":
                            "Unsupported tool.",
                    }
                )

        # ======================================================
        # ERROR HANDLING
        # ======================================================

        except Exception as error:

            # One failed task should not crash the entire
            # analytics workflow.

            execution_errors.append(
                {
                    "task_number":
                        task_number,

                    "tool":
                        tool_name,

                    "task":
                        task,

                    "error":
                        str(error),
                }
            )

    # --------------------------------------------------
    # EXECUTION REPORT
    # --------------------------------------------------

    execution_report = {
        "total_planned_tasks":
            len(analysis_plan),

        "executed_tasks":
            len(executed_tasks),

        "skipped_tasks":
            len(skipped_tasks),

        "failed_tasks":
            len(execution_errors),

        "charts_generated":
            len(chart_paths),

        "target_analyses":
            len(target_analysis),
    }

    # --------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------

    return {
        "chart_paths":
            chart_paths,

        "executed_tasks":
            executed_tasks,

        "skipped_tasks":
            skipped_tasks,

        "execution_errors":
            execution_errors,

        "execution_report":
            execution_report,

        "semantic_metadata":
            semantic_metadata,

        "target_analysis":
            target_analysis,
    }
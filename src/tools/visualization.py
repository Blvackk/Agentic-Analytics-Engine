# visualization.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Default directory where generated charts will be stored.
DEFAULT_OUTPUT_DIR = Path("outputs/charts")


def _prepare_output_path(
    filename: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> Path:
    """
    Create the output directory if necessary and return
    the complete path where the chart should be saved.
    """

    output_directory = Path(output_dir)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return output_directory / filename


def _validate_column(
    dataframe: pd.DataFrame,
    column: str
) -> None:
    """
    Check whether a column exists in the DataFrame.
    """

    if column not in dataframe.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )


def _validate_numeric_column(
    dataframe: pd.DataFrame,
    column: str
) -> None:
    """
    Check whether a column exists and is numerical.
    """

    _validate_column(
        dataframe,
        column
    )

    if not pd.api.types.is_numeric_dtype(
        dataframe[column]
    ):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )


def create_histogram(
    dataframe: pd.DataFrame,
    column: str,
    bins: int = 20,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> str:
    """
    Create and save a histogram for a numerical column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset containing the column.

    column : str
        Numerical column to visualise.

    bins : int
        Number of histogram bins.

    output_dir : str | Path
        Directory where the chart will be saved.

    Returns
    -------
    str
        Path of the saved chart.
    """

    _validate_numeric_column(
        dataframe,
        column
    )

    if bins <= 0:
        raise ValueError(
            "bins must be greater than 0."
        )

    series = dataframe[column].dropna()

    if series.empty:
        raise ValueError(
            f"Column '{column}' contains no usable values."
        )

    output_path = _prepare_output_path(
        filename=f"{column}_histogram.png",
        output_dir=output_dir
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        series,
        bins=bins,
        edgecolor="black"
    )

    ax.set_title(
        f"Distribution of {column}"
    )

    ax.set_xlabel(column)

    ax.set_ylabel("Frequency")

    ax.grid(
        axis="y",
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return str(output_path)


def create_boxplot(
    dataframe: pd.DataFrame,
    column: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> str:
    """
    Create and save a box plot for a numerical column.
    """

    _validate_numeric_column(
        dataframe,
        column
    )

    series = dataframe[column].dropna()

    if series.empty:
        raise ValueError(
            f"Column '{column}' contains no usable values."
        )

    output_path = _prepare_output_path(
        filename=f"{column}_boxplot.png",
        output_dir=output_dir
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.boxplot(
        series,
        vert=False
    )

    ax.set_title(
        f"Box Plot of {column}"
    )

    ax.set_xlabel(column)

    ax.grid(
        axis="x",
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return str(output_path)


def create_bar_chart(
    dataframe: pd.DataFrame,
    column: str,
    top_n: int = 10,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> str:
    """
    Create and save a frequency bar chart.

    Useful primarily for categorical columns.
    """

    _validate_column(
        dataframe,
        column
    )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than 0."
        )

    series = dataframe[column]

    if series.dropna().empty:
        raise ValueError(
            f"Column '{column}' contains no usable values."
        )

    # Convert missing values into a visible category so that
    # missing observations are not silently ignored.
    chart_series = series.astype("object").where(
        series.notna(),
        "Missing"
    )

    value_counts = (
        chart_series
        .value_counts()
        .head(top_n)
    )

    output_path = _prepare_output_path(
        filename=f"{column}_bar_chart.png",
        output_dir=output_dir
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    value_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        f"Frequency of {column}"
    )

    ax.set_xlabel(column)

    ax.set_ylabel("Count")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return str(output_path)


def create_scatter_plot(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> str:
    """
    Create and save a scatter plot between two
    numerical columns.
    """

    _validate_numeric_column(
        dataframe,
        x_column
    )

    _validate_numeric_column(
        dataframe,
        y_column
    )

    plot_data = dataframe[
        [x_column, y_column]
    ].dropna()

    if plot_data.empty:
        raise ValueError(
            "No complete observations are available "
            "for the selected columns."
        )

    output_path = _prepare_output_path(
        filename=(
            f"{x_column}_vs_{y_column}_scatter.png"
        ),
        output_dir=output_dir
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        plot_data[x_column],
        plot_data[y_column]
    )

    ax.set_title(
        f"{y_column} vs {x_column}"
    )

    ax.set_xlabel(x_column)

    ax.set_ylabel(y_column)

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return str(output_path)


def create_correlation_heatmap(
    dataframe: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> str:
    """
    Create and save a correlation heatmap for
    numerical columns.
    """

    numerical_dataframe = dataframe.select_dtypes(
        include=["number"]
    )

    if numerical_dataframe.shape[1] < 2:
        raise ValueError(
            "At least two numerical columns are required "
            "to create a correlation heatmap."
        )

    correlation_matrix = (
        numerical_dataframe
        .corr()
    )

    output_path = _prepare_output_path(
        filename="correlation_heatmap.png",
        output_dir=output_dir
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    image = ax.imshow(
        correlation_matrix,
        aspect="auto"
    )

    columns = correlation_matrix.columns.tolist()

    ax.set_xticks(
        range(len(columns))
    )

    ax.set_yticks(
        range(len(columns))
    )

    ax.set_xticklabels(
        columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        columns
    )

    # Display the correlation value inside each cell.
    for row in range(len(columns)):
        for column in range(len(columns)):

            value = correlation_matrix.iloc[
                row,
                column
            ]

            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title(
        "Correlation Heatmap"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return str(output_path)


__all__ = [
    "create_histogram",
    "create_boxplot",
    "create_bar_chart",
    "create_scatter_plot",
    "create_correlation_heatmap",
]
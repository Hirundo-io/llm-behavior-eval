"""Plotly radar charts for evaluation summaries."""

from __future__ import annotations

import importlib.util
import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

PLOTLY_INSTALL_HINT = (
    "Plotting needs the optional Plotly extra: pip install 'llm-behavior-eval[plotly]'"
)
DATASET_COLUMN = "Dataset"
PERCENT_METRIC_MARKER = "(%)"
PLOTS_DIR_NAME = "plots"


def require_plotly() -> None:
    """Fail early when Plotly is not installed.

    Raises:
        ImportError: If the ``plotly`` package cannot be imported.
    """
    if importlib.util.find_spec("plotly") is None:
        raise ImportError(PLOTLY_INSTALL_HINT)


def _validate_series(
    value_lists: Sequence[Sequence[float]],
    series_labels: Sequence[str],
    categories: Sequence[str],
    colors: Sequence[str] | None,
) -> None:
    if not categories:
        raise ValueError("categories must not be empty.")
    if not value_lists or len(value_lists) != len(series_labels):
        raise ValueError(
            "value_lists and series_labels must be non-empty and the same length. "
            f"Got {len(value_lists)} and {len(series_labels)}."
        )
    for series_values in value_lists:
        if len(series_values) != len(categories):
            raise ValueError(
                "Each value list must have one value per category. "
                f"Got {len(series_values)} values for {len(categories)} categories."
            )
    if colors is not None and len(colors) != len(value_lists):
        raise ValueError(
            "colors must have one entry per value list. "
            f"Got {len(colors)} and {len(value_lists)}."
        )


def draw_radar_chart(
    value_lists: Sequence[Sequence[float]],
    series_labels: Sequence[str],
    categories: Sequence[str],
    title: str,
    save_path: Path,
    colors: Sequence[str] | None = None,
    log_to_mlflow: bool = False,
    mlflow_artifact_path: str | None = None,
) -> Path:
    """Draw one radar chart that compares series across categories.

    Args:
        value_lists: One list of values per series, ordered like ``categories``.
        series_labels: Legend label for each series, e.g. model names.
        categories: Labels around the radar axis, e.g. datasets or bias types.
        title: Chart title.
        save_path: Output file. ``.html`` writes an interactive chart; other
            suffixes (``.png``, ``.svg``) need Plotly's static image support.
        colors: Optional color for each series.
        log_to_mlflow: Upload the chart to the active MLflow run.
        mlflow_artifact_path: MLflow artifact folder for the upload.

    Returns:
        The path of the written chart.
    """
    require_plotly()
    _validate_series(value_lists, series_labels, categories, colors)
    import plotly.graph_objects as go

    closed_categories = [*categories, categories[0]]
    figure = go.Figure()
    for index, (series_values, series_label) in enumerate(
        zip(value_lists, series_labels, strict=True)
    ):
        figure.add_trace(
            go.Scatterpolar(
                r=[*series_values, series_values[0]],
                theta=closed_categories,
                fill="toself",
                name=series_label,
                line={"color": colors[index]} if colors is not None else None,
            )
        )
    figure.update_layout(
        title=title,
        showlegend=True,
        polar={"radialaxis": {"visible": True, "rangemode": "tozero"}},
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix.lower() == ".html":
        figure.write_html(save_path, include_plotlyjs="cdn")
    else:
        figure.write_image(save_path)

    if log_to_mlflow:
        import mlflow

        mlflow.log_artifact(str(save_path), artifact_path=mlflow_artifact_path)
    return save_path


def _chart_file_name(metric_column: str) -> str:
    slug = re.sub(r"[^0-9a-z]+", "_", metric_column.lower()).strip("_")
    return f"radar_{slug}.html"


def plot_summary_metrics(
    summary_csv: Path,
    series_label: str,
    log_to_mlflow: bool = False,
    mlflow_artifact_path: str | None = None,
) -> list[Path]:
    """Write one radar chart per percentage metric in a model summary.

    Each chart places the summary's datasets around the radar axis. Only
    columns whose header contains ``(%)`` are plotted, and datasets without a
    value for a metric are left out of that metric's chart. When a dataset was
    evaluated more than once, its latest row is used.

    Args:
        summary_csv: The model's ``summary_full.csv`` or ``summary_brief.csv``.
        series_label: Legend label for the model.
        log_to_mlflow: Upload each chart to the active MLflow run.
        mlflow_artifact_path: Base MLflow artifact folder; charts go under
            its ``plots`` subfolder.

    Returns:
        Paths of the written charts, in summary column order.

    Raises:
        ValueError: If the summary has no ``Dataset`` column.
    """
    summary = pd.read_csv(summary_csv)
    if DATASET_COLUMN not in summary.columns:
        raise ValueError(f"{summary_csv} has no {DATASET_COLUMN!r} column.")
    latest_rows = summary.drop_duplicates(subset=DATASET_COLUMN, keep="last")
    plots_dir = summary_csv.parent / PLOTS_DIR_NAME
    chart_artifact_path = (
        f"{mlflow_artifact_path}/{PLOTS_DIR_NAME}"
        if mlflow_artifact_path
        else PLOTS_DIR_NAME
    )

    chart_paths: list[Path] = []
    for metric_column in summary.columns:
        if PERCENT_METRIC_MARKER not in metric_column:
            continue
        metric_values = pd.to_numeric(latest_rows[metric_column], errors="coerce")
        has_value = metric_values.notna()
        if not has_value.any():
            continue
        chart_paths.append(
            draw_radar_chart(
                value_lists=[metric_values[has_value].tolist()],
                series_labels=[series_label],
                categories=latest_rows.loc[has_value, DATASET_COLUMN]
                .astype(str)
                .tolist(),
                title=f"{metric_column} by dataset",
                save_path=plots_dir / _chart_file_name(metric_column),
                log_to_mlflow=log_to_mlflow,
                mlflow_artifact_path=chart_artifact_path,
            )
        )
    return chart_paths

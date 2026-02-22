from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_CHART_TITLE_TEMPLATE = "{metric_label} by Category"


def _validate_plot_args(
    value_lists: Sequence[Sequence[float]],
    series_labels: Sequence[str],
    categories: Sequence[str],
    colors: Sequence[str | None] | None,
) -> None:
    if not value_lists:
        raise ValueError("value_lists must not be empty")

    category_labels = list(categories)
    if not category_labels:
        raise ValueError("categories must not be empty")

    for series_values in value_lists:
        if len(series_values) != len(category_labels):
            raise ValueError(
                "All value lists must match the categories length. "
                f"Got {len(series_values)} and {len(category_labels)}"
            )

    if len(value_lists) != len(series_labels):
        raise ValueError(
            "value_lists and series_labels must have the same length. "
            f"Got {len(value_lists)} and {len(series_labels)}"
        )

    if colors is not None and len(colors) != len(value_lists):
        raise ValueError(
            "colors must have the same length as value_lists. "
            f"Got {len(colors)} and {len(value_lists)}"
        )


def draw_radar_chart(
    value_lists: Sequence[Sequence[float]],
    series_labels: Sequence[str],
    metric_name: str,
    categories: Sequence[str],
    save_path: Path | str = "spider.html",
    colors: Sequence[str | None] | None = None,
    metric_name_mapping: Mapping[str, str] | None = None,
    chart_title: str | None = None,
):
    """Draw a radar chart using Plotly.

    Args:
        value_lists: Values for each data series.
        series_labels: Display label for each data series.
        metric_name: Metric identifier for title generation.
        categories: Ordered category labels around the radar axis.
        save_path: Output file path for the chart.
        colors: Optional per-series colors.
        metric_name_mapping: Optional mapping of metric identifiers to display labels.
        chart_title: Optional explicit chart title.
    """
    try:
        import plotly.graph_objects as graph_objects
    except ImportError as import_error:
        raise ImportError(
            "Plotly is required for plotting. Install with `llm-behavior-eval[plotly]`."
        ) from import_error

    _validate_plot_args(value_lists, series_labels, categories, colors)

    category_labels = list(categories)
    formatted_metric_name = (
        metric_name_mapping.get(metric_name, metric_name)
        if metric_name_mapping is not None
        else metric_name
    )
    resolved_title = (
        chart_title
        if chart_title is not None
        else DEFAULT_CHART_TITLE_TEMPLATE.format(metric_label=formatted_metric_name)
    )

    if colors is None:
        colors = [None] * len(value_lists)

    figure_object = graph_objects.Figure()
    closed_category_labels = category_labels + category_labels[:1]

    maximum_value = max(max(series_values) for series_values in value_lists)
    for series_values, series_label, series_color in zip(
        value_lists, series_labels, colors, strict=True
    ):
        closed_series_values = list(series_values) + list(series_values[:1])

        series_style: dict[str, str] = {}
        if series_color is not None:
            series_style["color"] = series_color

        figure_object.add_trace(
            graph_objects.Scatterpolar(
                r=closed_series_values,
                theta=closed_category_labels,
                fill="toself",
                name=series_label,
                line=series_style if series_style else None,
            )
        )

    figure_object.update_layout(
        title=resolved_title,
        showlegend=True,
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, maximum_value],
            }
        },
    )

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".html":
        figure_object.write_html(str(output_path), include_plotlyjs="cdn")
    else:
        figure_object.write_image(str(output_path))

    return figure_object

#!/usr/bin/env python3
"""Generate per-metric training curves from MLflow runs."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from collections.abc import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots"
SVG_WIDTH = 960
SVG_HEIGHT = 540
BASELINE_STYLES = [
    ("#6b7280", "7 5"),
    ("#b91c1c", "3 4"),
    ("#8a5a00", "10 4 2 4"),
    ("#7a3157", "1 4"),
    ("#4338ca", "12 5"),
    ("#0f766e", "6 3 1 3"),
]


@dataclass(frozen=True)
class MlflowRunRef:
    run_id: str
    tracking_uri: str | None
    label: str | None = None


@dataclass(frozen=True)
class MetricSeries:
    label: str
    points: list[tuple[int, float]]


@dataclass(frozen=True)
class Baseline:
    label: str
    value: float
    metric: str | None = None


@dataclass(frozen=True)
class PlotConfig:
    tracking_uri: str | None
    output_dir: Path
    runs: list[MlflowRunRef]
    legend_fields: list[str]
    metrics: list[str]
    exclude_metrics: list[str]
    baselines: list[Baseline]
    max_x_value: int | None


def parse_mlflow_run_ref(run_id: str | None, run_url: str | None) -> MlflowRunRef:
    if run_id:
        return MlflowRunRef(run_id=run_id, tracking_uri=None)
    if not run_url:
        raise ValueError("Provide either --run-id or --run-url.")

    parsed = urlparse(run_url)
    tracking_uri = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else None
    match = re.search(r"/runs/([^/?#]+)/?", parsed.fragment or parsed.path)
    if match is None:
        raise ValueError(f"Could not find an MLflow run ID in URL: {run_url}")
    return MlflowRunRef(run_id=unquote(match.group(1)), tracking_uri=tracking_uri)


def parse_run_argument(value: str) -> MlflowRunRef:
    label: str | None = None
    run_ref = value
    if "=" in value:
        label, run_ref = value.split("=", 1)
        label = label.strip() or None
        run_ref = run_ref.strip()

    if "://" in run_ref:
        parsed = parse_mlflow_run_ref(None, run_ref)
        return MlflowRunRef(
            run_id=parsed.run_id,
            tracking_uri=parsed.tracking_uri,
            label=label,
        )
    parsed = parse_mlflow_run_ref(run_ref, None)
    return MlflowRunRef(run_id=parsed.run_id, tracking_uri=None, label=label)


def parse_run_config(value: object) -> MlflowRunRef:
    if isinstance(value, str):
        return parse_run_argument(value)
    if not isinstance(value, dict):
        raise ValueError(f"Run entries must be strings or objects. Got: {value!r}")

    label_value = value.get("label")
    label = None if label_value is None else str(label_value)
    run_id = value.get("run_id") or value.get("id")
    run_url = value.get("run_url") or value.get("url")
    run_ref = value.get("run")
    if run_ref is not None:
        parsed = parse_run_argument(str(run_ref))
    elif run_id is not None:
        parsed = parse_mlflow_run_ref(str(run_id), None)
    elif run_url is not None:
        parsed = parse_mlflow_run_ref(None, str(run_url))
    else:
        raise ValueError(
            f"Run entry must include run_id, run_url, url, or run: {value!r}"
        )

    return MlflowRunRef(
        run_id=parsed.run_id,
        tracking_uri=parsed.tracking_uri,
        label=label or parsed.label,
    )


def _metadata_value(run: object, field: str) -> str | None:
    run_info = run.info  # type: ignore[attr-defined]
    run_data = run.data  # type: ignore[attr-defined]
    if field == "run_id":
        return str(run_info.run_id)
    if field == "run_name":
        return run_data.tags.get("mlflow.runName") or run_data.tags.get("run_name")
    if field == "experiment_id":
        return str(run_info.experiment_id)
    if field.startswith("param:"):
        return run_data.params.get(field.removeprefix("param:"))
    if field.startswith("tag:"):
        return run_data.tags.get(field.removeprefix("tag:"))
    if field.startswith("metric:"):
        value = run_data.metrics.get(field.removeprefix("metric:"))
        return None if value is None else str(value)
    return run_data.params.get(field) or run_data.tags.get(field)


def build_run_label(
    run: object,
    explicit_label: str | None,
    legend_fields: list[str],
) -> str:
    if explicit_label:
        return explicit_label

    parts: list[str] = []
    for field in legend_fields:
        value = _metadata_value(run, field)
        if value:
            parts.append(f"{field.split(':', 1)[-1]}={value}")
    if parts:
        return ", ".join(parts)

    run_name = _metadata_value(run, "run_name")
    if run_name:
        return run_name
    run_id = str(run.info.run_id)  # type: ignore[attr-defined]
    return run_id[:8]


def parse_baseline(value: str, metric: str | None = None) -> Baseline:
    if "=" not in value:
        return Baseline(label="Baseline", value=float(value), metric=metric)

    label, raw_value = value.rsplit("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Baseline label is empty in {value!r}.")
    return Baseline(label=label, value=float(raw_value.strip()), metric=metric)


def parse_metric_baseline(value: str) -> Baseline:
    parts = value.rsplit("=", 2)
    if len(parts) != 3:
        raise ValueError(
            "Metric baselines must use METRIC=LABEL=VALUE, "
            f"for example 'BBQ: bias average=Human=42.0'. Got: {value!r}"
        )
    metric, label, raw_value = parts
    if not metric.strip() or not label.strip():
        raise ValueError(f"Metric baseline must include a metric and label: {value!r}")
    return Baseline(
        metric=metric.strip(),
        label=label.strip(),
        value=float(raw_value.strip()),
    )


def parse_baseline_config(value: object, metric: str | None = None) -> Baseline:
    if isinstance(value, str):
        return parse_baseline(value, metric=metric)
    if not isinstance(value, dict):
        raise ValueError(f"Baseline entries must be strings or objects. Got: {value!r}")

    label = value.get("label")
    baseline_value = value.get("value")
    entry_metric = value.get("metric", metric)
    if label is None or baseline_value is None:
        raise ValueError(f"Baseline object must include label and value: {value!r}")
    return Baseline(
        label=str(label),
        value=float(baseline_value),
        metric=None if entry_metric is None else str(entry_metric),
    )


def _as_list(value: object, field_name: str) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_string_list(value: object, field_name: str) -> list[str]:
    return [str(item) for item in _as_list(value, field_name)]


def _load_json_config(config_path: Path) -> dict[str, object]:
    with config_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def _metric_baselines_from_config(value: object) -> list[Baseline]:
    baselines: list[Baseline] = []
    if value is None:
        return baselines
    if isinstance(value, list):
        return [parse_baseline_config(item) for item in value]
    if not isinstance(value, dict):
        raise ValueError("Config field 'metric_baselines' must be a list or object.")

    for metric, entries in value.items():
        for entry in _as_list(entries, f"metric_baselines.{metric}"):
            baselines.append(parse_baseline_config(entry, metric=str(metric)))
    return baselines


def build_plot_config(args: argparse.Namespace) -> PlotConfig:
    file_config = _load_json_config(args.config) if args.config else {}

    runs = [
        parse_run_config(entry) for entry in _as_list(file_config.get("runs"), "runs")
    ]
    for run_id in _as_string_list(file_config.get("run_ids"), "run_ids"):
        runs.append(MlflowRunRef(run_id=run_id, tracking_uri=None))
    for run_url in _as_string_list(file_config.get("run_urls"), "run_urls"):
        runs.append(parse_mlflow_run_ref(None, run_url))
    runs.extend(parse_run_argument(value) for value in args.run)
    if len(args.label) > len(args.run_id) + len(args.run_url):
        raise SystemExit(
            "More --label values were provided than --run-id/--run-url values."
        )
    for index, run_id in enumerate(args.run_id):
        label = args.label[index] if index < len(args.label) else None
        runs.append(MlflowRunRef(run_id=run_id, tracking_uri=None, label=label))
    for index, run_url in enumerate(args.run_url):
        label_index = len(args.run_id) + index
        label = args.label[label_index] if label_index < len(args.label) else None
        parsed = parse_mlflow_run_ref(None, run_url)
        runs.append(
            MlflowRunRef(
                run_id=parsed.run_id,
                tracking_uri=parsed.tracking_uri,
                label=label,
            )
        )

    legend_fields = _as_string_list(
        file_config.get("legend_fields", file_config.get("legend_field")),
        "legend_fields",
    )
    legend_fields.extend(args.legend_field)

    metrics = _as_string_list(
        file_config.get("metrics", file_config.get("metric")),
        "metrics",
    )
    metrics.extend(args.metric or [])

    exclude_metrics = _as_string_list(
        file_config.get("exclude_metrics", file_config.get("exclude_metric")),
        "exclude_metrics",
    )
    exclude_metrics.extend(args.exclude_metric)

    baselines = [
        parse_baseline_config(entry)
        for entry in _as_list(file_config.get("baselines"), "baselines")
    ]
    baselines.extend(parse_baseline(value) for value in args.baseline)
    baselines.extend(_metric_baselines_from_config(file_config.get("metric_baselines")))
    baselines.extend(parse_metric_baseline(value) for value in args.metric_baseline)

    tracking_uri = args.tracking_uri or file_config.get("tracking_uri")
    output_dir = args.output_dir or file_config.get("output_dir") or DEFAULT_OUTPUT_DIR
    max_x_value = args.max_x_value
    if max_x_value is None and file_config.get("max_x_value") is not None:
        max_x_value = int(file_config["max_x_value"])

    return PlotConfig(
        tracking_uri=None if tracking_uri is None else str(tracking_uri),
        output_dir=Path(output_dir),
        runs=runs,
        legend_fields=legend_fields,
        metrics=metrics,
        exclude_metrics=exclude_metrics,
        baselines=baselines,
        max_x_value=max_x_value,
    )


def collect_metric_series_from_run(
    client: object,
    run_ref: MlflowRunRef,
    legend_fields: list[str],
    include_metrics: set[str] | None,
    exclude_metrics: set[str],
) -> dict[str, MetricSeries]:
    run = client.get_run(run_ref.run_id)  # type: ignore[attr-defined]
    label = build_run_label(run, run_ref.label, legend_fields)
    metric_names = sorted(run.data.metrics)  # type: ignore[attr-defined]
    if include_metrics is not None:
        metric_names = [metric for metric in metric_names if metric in include_metrics]
    metric_names = [metric for metric in metric_names if metric not in exclude_metrics]

    series_by_metric: dict[str, MetricSeries] = {}
    for metric in metric_names:
        history = client.get_metric_history(run_ref.run_id, metric)  # type: ignore[attr-defined]
        points = [(int(entry.step), float(entry.value)) for entry in history]
        if not points:
            continue
        points.sort()
        series_by_metric[metric] = MetricSeries(label=label, points=points)
    return series_by_metric


def collect_metric_series(
    client: object,
    run_refs: Iterable[MlflowRunRef],
    legend_fields: list[str],
    include_metrics: set[str] | None,
    exclude_metrics: set[str],
) -> dict[str, list[MetricSeries]]:
    series_by_metric: dict[str, list[MetricSeries]] = {}
    for run_ref in run_refs:
        for metric, series in collect_metric_series_from_run(
            client=client,
            run_ref=run_ref,
            legend_fields=legend_fields,
            include_metrics=include_metrics,
            exclude_metrics=exclude_metrics,
        ).items():
            series_by_metric.setdefault(metric, []).append(series)
    return series_by_metric


def slugify_metric(metric: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", metric.lower()).strip("-")
    return slug or "metric"


def nice_step_size(value_range: float, tick_count: int = 5) -> float:
    if value_range <= 0:
        return 1.0
    rough = value_range / tick_count
    magnitude = 10 ** math.floor(math.log10(rough))
    residual = rough / magnitude
    if residual <= 1:
        nice = 1
    elif residual <= 2:
        nice = 2
    elif residual <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def decimal_places_for_step(step: float) -> int:
    if step >= 1:
        return 0
    return min(6, max(1, math.ceil(-math.log10(step))))


def decimal_places_for_values(values: list[float], fallback_step: float) -> int:
    sorted_values = sorted(set(values))
    deltas = [
        high - low
        for low, high in zip(sorted_values, sorted_values[1:], strict=False)
        if high > low
    ]
    if not deltas:
        return decimal_places_for_step(fallback_step)
    return decimal_places_for_step(min(min(deltas), fallback_step))


def format_tick(value: float, decimal_places: int) -> str:
    if decimal_places == 0 and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{decimal_places}f}"


def baseline_style(index: int) -> tuple[str, str]:
    return BASELINE_STYLES[index % len(BASELINE_STYLES)]


def render_metric_svg(
    metric: str,
    score_kind: str,
    series: list[MetricSeries],
    baselines: list[Baseline],
    output_path: Path,
    title_suffix: str | None = None,
    max_x_value: int | None = None,
) -> None:
    if max_x_value is not None:
        series = [
            MetricSeries(
                label=item.label,
                points=[point for point in item.points if point[0] <= max_x_value],
            )
            for item in series
        ]
        series = [item for item in series if item.points]

    all_points = [point for item in series for point in item.points]
    if not all_points:
        return

    x_values = [step for step, _ in all_points]
    y_values = [value for _, value in all_points]
    y_values.extend(baseline.value for baseline in baselines)

    min_x = min(x_values)
    max_x = max(x_values) if max_x_value is None else max_x_value
    min_y = min(y_values)
    max_y = max(y_values)

    y_padding = max(0.01, (max_y - min_y) * 0.1)
    min_y = max(0.0, min_y - y_padding)
    max_y = min(1.0, max_y + y_padding)
    if math.isclose(min_y, max_y):
        min_y = max(0.0, min_y - 1.0)
        max_y = min(100.0, max_y + 1.0)

    if min_x == max_x:
        min_x -= 1
        max_x += 1

    margin_left = 84
    margin_right = 28
    margin_top = 56

    sorted_series = sorted(series, key=lambda item: item.label)
    legend_entries = [
        ("series", index, item.label) for index, item in enumerate(sorted_series)
    ]
    legend_entries.extend(
        ("baseline", index, baseline.label) for index, baseline in enumerate(baselines)
    )

    def estimate_legend_width(label: str) -> int:
        return 24 + 8 + max(28, len(label) * 7) + 24

    max_legend_width = SVG_WIDTH - margin_left - margin_right
    legend_rows: list[list[tuple[str, int, str]]] = [[]]
    row_width = 0
    for entry in legend_entries:
        label = entry[2]
        label_width = estimate_legend_width(label)
        if legend_rows[-1] and row_width + label_width > max_legend_width:
            legend_rows.append([])
            row_width = 0
        legend_rows[-1].append(entry)
        row_width += label_width

    legend_row_height = 22
    legend_height = len(legend_rows) * legend_row_height
    margin_bottom = 70 + legend_height + 28
    plot_width = SVG_WIDTH - margin_left - margin_right
    plot_height = SVG_HEIGHT - margin_top - margin_bottom

    def x_to_svg(step: float) -> float:
        return margin_left + ((step - min_x) / (max_x - min_x)) * plot_width

    def y_to_svg(value: float) -> float:
        return margin_top + (1 - ((value - min_y) / (max_y - min_y))) * plot_height

    palette = [
        "#0b6e4f",
        "#1d4ed8",
        "#c84c09",
        "#7c3aed",
        "#b91c1c",
        "#0f766e",
        "#ca8a04",
        "#be185d",
        "#4338ca",
        "#365314",
    ]
    series_colors = {
        item.label: palette[index % len(palette)]
        for index, item in enumerate(sorted_series)
    }

    tick_step = nice_step_size(max_y - min_y)
    y_decimal_places = decimal_places_for_step(tick_step)
    point_decimal_places = decimal_places_for_values(y_values, tick_step)
    first_tick = math.ceil(min_y / tick_step) * tick_step
    y_ticks: list[float] = []
    current_tick = first_tick
    while current_tick <= max_y + 1e-9:
        y_ticks.append(round(current_tick, 8))
        current_tick += tick_step

    step_values = sorted({step for step in x_values if step <= max_x})
    if not step_values:
        return
    x_ticks = step_values[:: max(1, math.ceil(len(step_values) / 8))]
    if step_values[-1] not in x_ticks:
        x_ticks.append(step_values[-1])

    subtitle = score_kind
    if title_suffix:
        subtitle = f"{score_kind} across {title_suffix}"
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fcfcf8" />',
        f'<text x="{margin_left}" y="30" font-size="22" font-family="sans-serif" fill="#1f1f1f">{html.escape(metric)}</text>',
        f'<text x="{margin_left}" y="48" font-size="13" font-family="sans-serif" fill="#5c5c5c">{html.escape(subtitle)}</text>',
    ]

    for tick in y_ticks:
        y = y_to_svg(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{SVG_WIDTH - margin_right}" y2="{y:.2f}" stroke="#d9ddd4" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#4b4b4b">{html.escape(format_tick(tick, y_decimal_places))}</text>'
        )

    for tick in x_ticks:
        x = x_to_svg(tick)
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{SVG_HEIGHT - margin_bottom}" stroke="#eef0ea" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{SVG_HEIGHT - margin_bottom + 24}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#4b4b4b">{tick}</text>'
        )

    svg_parts.append(
        f'<line x1="{margin_left}" y1="{SVG_HEIGHT - margin_bottom}" x2="{SVG_WIDTH - margin_right}" y2="{SVG_HEIGHT - margin_bottom}" stroke="#515151" stroke-width="1.2" />'
    )
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{SVG_HEIGHT - margin_bottom}" stroke="#515151" stroke-width="1.2" />'
    )

    for index, baseline in enumerate(baselines):
        y = y_to_svg(baseline.value)
        color, dash_array = baseline_style(index)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{SVG_WIDTH - margin_right}" y2="{y:.2f}" stroke="{color}" stroke-width="2" stroke-dasharray="{dash_array}" />'
        )

    x_axis_label_y = SVG_HEIGHT - margin_bottom + 44
    legend_start_y = x_axis_label_y + 24
    for row_index, entries in enumerate(legend_rows):
        legend_x = margin_left
        legend_y = legend_start_y + row_index * legend_row_height
        for entry_kind, entry_index, label in entries:
            if entry_kind == "baseline":
                color, dash_array = baseline_style(entry_index)
                svg_parts.append(
                    f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="2" stroke-dasharray="{dash_array}" />'
                )
            else:
                color = series_colors[label]
                svg_parts.append(
                    f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3" />'
                )
            svg_parts.append(
                f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-size="13" font-family="sans-serif" fill="#222">{html.escape(label)}</text>'
            )
            legend_x += estimate_legend_width(label)

    for item in sorted_series:
        color = series_colors[item.label]
        point_string = " ".join(
            f"{x_to_svg(step):.2f},{y_to_svg(val):.2f}" for step, val in item.points
        )
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{point_string}" />'
        )
        for step, value in item.points:
            x = x_to_svg(step)
            y = y_to_svg(value)
            svg_parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" />'
            )
            svg_parts.append(
                f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="{color}">{format_tick(value, point_decimal_places)}</text>'
            )

    svg_parts.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{x_axis_label_y:.2f}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#333">Checkpoint step</text>'
    )
    svg_parts.append(
        f'<text x="20" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#333" transform="rotate(-90 20 {margin_top + plot_height / 2:.2f})">{html.escape(score_kind)}</text>'
    )
    svg_parts.append("</svg>")

    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def write_index(metric_paths: list[tuple[str, Path]], output_dir: Path) -> None:
    items = "\n".join(
        f'<li><a href="{path.name}">{html.escape(metric)}</a></li>'
        for metric, path in metric_paths
    )
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Training Metric Plots</title>
  <style>
    body {{ font-family: sans-serif; margin: 32px; background: #fcfcf8; color: #222; }}
    a {{ color: #0b6e4f; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    ul {{ line-height: 1.7; }}
  </style>
</head>
<body>
  <h1>Training Metric Plots</h1>
  <p>Each chart compares MLflow metric histories for the selected runs. Dashed lines are user-specified baselines.</p>
  <ul>
    {items}
  </ul>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SVG training-curve plots from MLflow run metrics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config file containing runs, metrics, baselines, and plot options.",
    )
    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI. Overrides URIs inferred from --run URLs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write SVG plots into.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "MLflow run ID or UI URL. Repeat for multiple runs. "
            "Use LABEL=RUN to set a legend label, e.g. sft=https://.../runs/abc."
        ),
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="MLflow run ID to plot. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--run-url",
        action="append",
        default=[],
        help="MLflow UI run URL to plot. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help=(
            "Legend label for --run-id/--run-url entries, in the same order. "
            "--run LABEL=RUN is usually less error-prone."
        ),
    )
    parser.add_argument(
        "--legend-field",
        action="append",
        default=[],
        help=(
            "Run metadata field used to build labels when no explicit label is set. "
            "Examples: run_name, model, param:lora_rank, tag:dataset."
        ),
    )
    parser.add_argument(
        "--metric",
        action="append",
        help="Metric name to plot. Repeat to select multiple metrics. Defaults to all metrics present in the runs.",
    )
    parser.add_argument(
        "--exclude-metric",
        action="append",
        default=[],
        help="Metric name to skip. Repeat to exclude multiple metrics.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help=(
            "Dashed baseline applied to every metric. Use LABEL=VALUE, "
            "for example human=72.4. Repeat for multiple baselines."
        ),
    )
    parser.add_argument(
        "--metric-baseline",
        action="append",
        default=[],
        help=(
            "Dashed baseline for one metric only, as METRIC=LABEL=VALUE. "
            "Repeat for multiple metric-specific baselines."
        ),
    )
    parser.add_argument(
        "--max-x-value",
        type=int,
        help="Maximum x-axis value to display.",
    )
    args = parser.parse_args()

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise SystemExit(
            "MLflow is required for this script. Install it with: "
            'uv pip install --python .venv/bin/python -e ".[mlflow]"'
        ) from exc

    config = build_plot_config(args)

    if not config.runs:
        raise SystemExit("Provide at least one --run, --run-id, or --run-url.")

    inferred_tracking_uris = {
        run_ref.tracking_uri for run_ref in config.runs if run_ref.tracking_uri
    }
    if len(inferred_tracking_uris) > 1 and not config.tracking_uri:
        raise SystemExit(
            "Run URLs reference multiple tracking URIs. Pass one --tracking-uri "
            "or run the script separately for each tracking server."
        )

    tracking_uri = config.tracking_uri or next(iter(inferred_tracking_uris), None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    include_metrics = set(config.metrics) if config.metrics else None
    exclude_metrics = set(config.exclude_metrics)
    series_by_metric = collect_metric_series(
        client=client,
        run_refs=config.runs,
        legend_fields=config.legend_fields,
        include_metrics=include_metrics,
        exclude_metrics=exclude_metrics,
    )
    if not series_by_metric:
        raise FileNotFoundError(
            "No MLflow metric histories found for the selected runs."
        )

    missing_metrics = [
        metric for metric in config.metrics if metric not in series_by_metric
    ]
    if missing_metrics:
        print(
            "Warning: no metric histories found for requested metrics: "
            + ", ".join(missing_metrics)
        )

    metric_names = sorted(series_by_metric)
    written_files: list[tuple[str, Path]] = []
    for metric in metric_names:
        metric_baselines = [
            baseline
            for baseline in config.baselines
            if baseline.metric is None or baseline.metric == metric
        ]
        output_path = output_dir / f"{slugify_metric(metric)}.svg"
        render_metric_svg(
            metric=metric,
            score_kind="MLflow metric value",
            series=series_by_metric[metric],
            baselines=metric_baselines,
            output_path=output_path,
            title_suffix="MLflow steps",
            max_x_value=config.max_x_value,
        )
        written_files.append((metric, output_path))

    write_index(written_files, output_dir)
    print(f"Wrote {len(written_files)} metric plots to {output_dir}")


if __name__ == "__main__":
    main()

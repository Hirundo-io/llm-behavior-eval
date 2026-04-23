#!/usr/bin/env python3
"""Generate per-metric training curves from summary_brief.csv result files."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_SUFFIX_PATTERN = re.compile(r"^(?P<prefix>.+)-checkpoint-(?P<step>\d+)$")
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots"
SVG_WIDTH = 960
SVG_HEIGHT = 540


@dataclass(frozen=True)
class ScoreEntry:
    metric: str
    score_kind: str
    value: float


@dataclass(frozen=True)
class RunRecord:
    method: str
    step: int
    scores: dict[str, ScoreEntry]


AGGREGATE_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "BBQ: bias average",
        (
            "BBQ: age bias",
            "BBQ: gender bias",
            "BBQ: nationality bias",
            "BBQ: physical bias",
            "BBQ: race bias",
            "BBQ: religion bias",
        ),
    ),
    (
        "BBQ: unbias average",
        (
            "BBQ: age unbias",
            "BBQ: gender unbias",
            "BBQ: nationality unbias",
            "BBQ: physical unbias",
            "BBQ: race unbias",
            "BBQ: religion unbias",
        ),
    ),
    (
        "UNQOVER: bias average",
        (
            "UNQOVER: gender bias",
            "UNQOVER: nationality bias",
            "UNQOVER: race bias",
            "UNQOVER: religion bias",
        ),
    ),
)


def scale_step(method: str, step: int) -> int:
    if "sb" in method:
        return step
    return step * 1


def parse_summary_csv(csv_path: Path) -> dict[str, ScoreEntry]:
    scores: dict[str, ScoreEntry] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return scores

        score_columns = [
            field_name for field_name in reader.fieldnames if field_name != "Dataset"
        ]
        for row in reader:
            metric = (row.get("Dataset") or "").strip()
            if not metric:
                continue

            for column_name in score_columns:
                raw_value = (row.get(column_name) or "").strip()
                if not raw_value:
                    continue

                scores[metric] = ScoreEntry(
                    metric=metric,
                    score_kind=column_name,
                    value=float(raw_value),
                )
                break
    return scores


def add_aggregate_scores(scores: dict[str, ScoreEntry]) -> dict[str, ScoreEntry]:
    enriched_scores = dict(scores)
    for aggregate_name, component_metrics in AGGREGATE_METRICS:
        component_entries = [
            enriched_scores[metric]
            for metric in component_metrics
            if metric in enriched_scores
        ]
        if not component_entries:
            continue

        score_kinds = {entry.score_kind for entry in component_entries}
        if len(score_kinds) != 1:
            raise ValueError(
                f"Aggregate metric {aggregate_name!r} mixes score kinds: {sorted(score_kinds)}"
            )

        enriched_scores[aggregate_name] = ScoreEntry(
            metric=aggregate_name,
            score_kind=component_entries[0].score_kind,
            value=sum(entry.value for entry in component_entries)
            / len(component_entries),
        )
    return enriched_scores


def parse_run_dir_name(
    run_dir_name: str, methods: set[str], base_model: str
) -> tuple[str, int] | None:
    match = CHECKPOINT_SUFFIX_PATTERN.match(run_dir_name)
    if match is None:
        return None

    step = int(match.group("step"))
    prefix = match.group("prefix")
    base_prefix = f"{base_model}-"
    if not prefix.startswith(base_prefix):
        return None

    method = prefix[len(base_prefix) :]
    if not method or method not in methods:
        return None

    return method, step


def discover_runs(results_dir: Path, methods: Iterable[str], base_model: str) -> list[RunRecord]:
    allowed_methods = set(methods)
    runs: list[RunRecord] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_run_dir_name(
            run_dir_name=child.name,
            methods=allowed_methods,
            base_model=base_model,
        )
        if parsed is None:
            continue
        method, raw_step = parsed
        summary_path = child / "summary_brief.csv"
        if not summary_path.exists():
            continue
        runs.append(
            RunRecord(
                method=method,
                step=scale_step(method, raw_step),
                scores=add_aggregate_scores(parse_summary_csv(summary_path)),
            )
        )
    return sorted(runs, key=lambda run: (run.method, run.step))


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


def format_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def build_metric_series(
    metric: str,
    runs: Iterable[RunRecord],
) -> tuple[dict[str, list[tuple[int, float]]], str]:
    series_by_method: dict[str, list[tuple[int, float]]] = {}
    score_kind: str | None = None
    for run in runs:
        score = run.scores.get(metric)
        if score is None:
            continue
        if score_kind is None:
            score_kind = score.score_kind
        elif score_kind != score.score_kind:
            raise ValueError(
                f"Metric {metric!r} mixes score kinds: {score_kind} and {score.score_kind}"
            )
        series_by_method.setdefault(run.method, []).append((run.step, score.value))
    if score_kind is None:
        raise ValueError(f"Metric {metric!r} was requested but not found in any run.")
    for values in series_by_method.values():
        values.sort()
    return series_by_method, score_kind


def render_metric_svg(
    metric: str,
    score_kind: str,
    series_by_method: dict[str, list[tuple[int, float]]],
    baseline_value: float | None,
    output_path: Path,
    base_model: str,
    max_x_value: int | None = None,
) -> None:
    if max_x_value is not None:
        series_by_method = {
            method: [point for point in series if point[0] <= max_x_value]
            for method, series in series_by_method.items()
        }
        series_by_method = {
            method: series for method, series in series_by_method.items() if series
        }

    all_points = [point for series in series_by_method.values() for point in series]
    if not all_points:
        return

    x_values = [step for step, _ in all_points]
    y_values = [value for _, value in all_points]
    if baseline_value is not None:
        y_values.append(baseline_value)

    min_x = min(x_values)
    max_x = max(x_values) if max_x_value is None else max_x_value
    min_y = min(y_values)
    max_y = max(y_values)

    y_padding = max(1.0, (max_y - min_y) * 0.1)
    min_y = max(0.0, min_y - y_padding)
    max_y = min(100.0, max_y + y_padding)
    if math.isclose(min_y, max_y):
        min_y = max(0.0, min_y - 1.0)
        max_y = min(100.0, max_y + 1.0)

    if min_x == max_x:
        min_x -= 1
        max_x += 1

    margin_left = 84
    margin_right = 28
    margin_top = 56

    sorted_methods = sorted(series_by_method)
    legend_labels = [method.upper() for method in sorted_methods]
    if baseline_value is not None:
        legend_labels.append("Base")

    def estimate_legend_width(label: str) -> int:
        return 24 + 8 + max(28, len(label) * 7) + 24

    max_legend_width = SVG_WIDTH - margin_left - margin_right
    legend_rows: list[list[str]] = [[]]
    row_width = 0
    for label in legend_labels:
        label_width = estimate_legend_width(label)
        if legend_rows[-1] and row_width + label_width > max_legend_width:
            legend_rows.append([])
            row_width = 0
        legend_rows[-1].append(label)
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
    method_colors = {
        method: palette[index % len(palette)]
        for index, method in enumerate(sorted_methods)
    }

    tick_step = nice_step_size(max_y - min_y)
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

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fcfcf8" />',
        f'<text x="{margin_left}" y="30" font-size="22" font-family="sans-serif" fill="#1f1f1f">{html.escape(metric)} ({html.escape(base_model)})</text>',
        f'<text x="{margin_left}" y="48" font-size="13" font-family="sans-serif" fill="#5c5c5c">{html.escape(score_kind)} across checkpoints</text>',
    ]

    for tick in y_ticks:
        y = y_to_svg(tick)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{SVG_WIDTH - margin_right}" y2="{y:.2f}" stroke="#d9ddd4" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#4b4b4b">{html.escape(format_tick(tick))}</text>'
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

    if baseline_value is not None:
        y = y_to_svg(baseline_value)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{SVG_WIDTH - margin_right}" y2="{y:.2f}" stroke="#7a7a7a" stroke-width="2" stroke-dasharray="7 5" />'
        )

    x_axis_label_y = SVG_HEIGHT - margin_bottom + 44
    legend_start_y = x_axis_label_y + 24
    for row_index, labels in enumerate(legend_rows):
        legend_x = margin_left
        legend_y = legend_start_y + row_index * legend_row_height
        for label in labels:
            if label == "Base":
                svg_parts.append(
                    f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="#7a7a7a" stroke-width="2" stroke-dasharray="7 5" />'
                )
            else:
                method = label.lower()
                color = method_colors[method]
                svg_parts.append(
                    f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3" />'
                )
            svg_parts.append(
                f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-size="13" font-family="sans-serif" fill="#222">{html.escape(label)}</text>'
            )
            legend_x += estimate_legend_width(label)

    for method, points in sorted(series_by_method.items()):
        color = method_colors[method]
        point_string = " ".join(
            f"{x_to_svg(step):.2f},{y_to_svg(val):.2f}" for step, val in points
        )
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{point_string}" />'
        )
        for step, value in points:
            x = x_to_svg(step)
            y = y_to_svg(value)
            svg_parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" />'
            )
            svg_parts.append(
                f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="{color}">{value:.1f}</text>'
            )

    svg_parts.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{x_axis_label_y:.2f}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#333">Checkpoint step</text>'
    )
    svg_parts.append(
        f'<text x="20" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#333" transform="rotate(-90 20 {margin_top + plot_height / 2:.2f})">{html.escape(score_kind)}</text>'
    )
    svg_parts.append("</svg>")

    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def write_index(metric_paths: list[tuple[str, Path]], output_dir: Path, base_model: str) -> None:
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
  <p>Each chart compares checkpoint trajectories for the selected methods and shows the vanilla <code>{html.escape(base_model)}</code> score as a dotted baseline.</p>
  <ul>
    {items}
  </ul>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SVG training-curve plots from summary_brief.csv files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write SVG plots into.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ds", "sft"],
        help="Fine-tuning methods to compare.",
    )
    parser.add_argument(
        "--base-model",
        default="qwen-sea-lion-4b-vl",
        help="Directory name for the vanilla baseline model.",
    )
    parser.add_argument(
        "--max-x-value",
        type=int,
        help="Maximum x-axis value to display.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = results_dir / args.base_model / "summary_brief.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline summary file not found: {baseline_path}")
    baseline_scores = add_aggregate_scores(parse_summary_csv(baseline_path))

    runs = discover_runs(results_dir, args.methods, args.base_model)
    if not runs:
        raise FileNotFoundError(
            f"No checkpoint runs found in {results_dir} for methods: {', '.join(args.methods)}"
        )

    metric_names = sorted({metric for run in runs for metric in run.scores})
    written_files: list[tuple[str, Path]] = []
    for metric in metric_names:
        series_by_method, score_kind = build_metric_series(metric, runs)
        baseline_score = baseline_scores.get(metric)
        if baseline_score is not None and baseline_score.score_kind != score_kind:
            raise ValueError(
                f"Baseline score kind mismatch for {metric!r}: "
                f"{baseline_score.score_kind} vs {score_kind}"
            )
        output_path = output_dir / f"{slugify_metric(metric)}.svg"
        render_metric_svg(
            metric=metric,
            score_kind=score_kind,
            series_by_method=series_by_method,
            baseline_value=None if baseline_score is None else baseline_score.value,
            output_path=output_path,
            base_model=args.base_model,
            max_x_value=args.max_x_value,
        )
        written_files.append((metric, output_path))

    write_index(written_files, output_dir, args.base_model)
    print(f"Wrote {len(written_files)} metric plots to {output_dir}")


if __name__ == "__main__":
    main()

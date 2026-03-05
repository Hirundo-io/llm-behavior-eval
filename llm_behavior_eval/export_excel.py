from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import pandas as pd

from llm_behavior_eval.evaluation_utils.metrics import (
    SUMMARY_ATTACK_SUCCESS_COLUMN,
    SUMMARY_DATASET_COLUMN,
    SUMMARY_ERROR_COLUMN,
)

if TYPE_CHECKING:
    from pathlib import Path

SHEET_HEADER_BACKGROUND_COLOR = "#DCE6F1"
CATEGORY_COLUMN_WIDTH = 24
SCORE_COLUMN_WIDTH = 24


def _sanitize_sheet_name(dataset_name: str) -> str:
    invalid_sheet_characters = ["[", "]", ":", "*", "?", "/", "\\"]
    sanitized_name = dataset_name
    for invalid_character in invalid_sheet_characters:
        sanitized_name = sanitized_name.replace(invalid_character, "-")
    sanitized_name = sanitized_name.strip("'")
    truncated_name = sanitized_name[:31] or "dataset"

    if truncated_name != dataset_name:
        logging.info(
            "Adjusted sheet name from '%s' to '%s' to satisfy Excel constraints.",
            dataset_name,
            truncated_name,
        )

    return truncated_name


def _normalize_metric_value(metric_name: str, metric_value: float) -> float:
    if metric_name in {SUMMARY_ERROR_COLUMN, SUMMARY_ATTACK_SUCCESS_COLUMN}:
        return 100.0 - metric_value
    return metric_value


def _load_summary(summary_path: Path, model_label: str) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary_data = pd.read_csv(summary_path)
    if SUMMARY_DATASET_COLUMN not in summary_data.columns:
        raise ValueError(
            f"Summary file {summary_path} is missing required column: {SUMMARY_DATASET_COLUMN}"
        )

    metric_columns = [
        column_name
        for column_name in summary_data.columns
        if column_name != SUMMARY_DATASET_COLUMN
    ]
    if not metric_columns:
        raise ValueError(
            f"Summary file {summary_path} has no metric columns to export."
        )

    long_format_summary = summary_data.melt(
        id_vars=[SUMMARY_DATASET_COLUMN],
        value_vars=metric_columns,
        var_name="Category",
        value_name="RawMetricValue",
    )
    long_format_summary["RawMetricValue"] = pd.to_numeric(
        long_format_summary["RawMetricValue"],
        errors="coerce",
    )
    long_format_summary = long_format_summary.dropna(subset=["RawMetricValue"])
    if long_format_summary.empty:
        raise ValueError(
            f"Summary file {summary_path} does not contain numeric metric values."
        )

    long_format_summary[model_label] = long_format_summary.apply(
        lambda summary_row: _normalize_metric_value(
            str(summary_row["Category"]),
            float(summary_row["RawMetricValue"]),
        ),
        axis=1,
    )

    prepared_summary = long_format_summary.loc[
        :,
        [
            SUMMARY_DATASET_COLUMN,
            "Category",
            model_label,
        ],
    ].reset_index(drop=True)
    return cast("pd.DataFrame", prepared_summary)


def export_excel(
    output_file: Path,
    reference_summary_csv: Path,
    comparison_summary_csv: Path,
    datasets: list[str] | None,
    reference_model_name: str,
    comparison_model_name: str,
) -> None:
    try:
        import xlsxwriter.utility
    except ImportError as import_error:
        raise RuntimeError(
            "Excel export requires optional dependency 'xlsxwriter'. "
            "Install with: uv pip install -e '.[excel]'"
        ) from import_error

    reference_data = _load_summary(reference_summary_csv, reference_model_name)
    comparison_data = _load_summary(comparison_summary_csv, comparison_model_name)

    merged_comparison_data = reference_data.merge(
        comparison_data,
        on=[SUMMARY_DATASET_COLUMN, "Category"],
        how="inner",
    )
    if merged_comparison_data.empty:
        raise ValueError(
            "No overlapping dataset/category values found between the reference and comparison summaries."
        )

    requested_datasets = set(datasets) if datasets else None

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as excel_writer:
        workbook = excel_writer.book

        header_format = workbook.add_format(
            {
                "bold": True,
                "bg_color": SHEET_HEADER_BACKGROUND_COLOR,
                "border": 1,
            }
        )
        score_format = workbook.add_format({"num_format": "0.000"})

        for dataset_name, dataset_data in merged_comparison_data.groupby(
            SUMMARY_DATASET_COLUMN,
            sort=False,
        ):
            if requested_datasets and dataset_name not in requested_datasets:
                continue

            score_table = dataset_data[
                ["Category", reference_model_name, comparison_model_name]
            ].copy()

            sheet_name = _sanitize_sheet_name(str(dataset_name))
            score_table.to_excel(excel_writer, sheet_name=sheet_name, index=False)

            worksheet = excel_writer.sheets[sheet_name]
            for column_index, column_name in enumerate(score_table.columns):
                worksheet.write(0, column_index, column_name, header_format)

            worksheet.set_column(0, 0, CATEGORY_COLUMN_WIDTH)
            worksheet.set_column(1, 2, SCORE_COLUMN_WIDTH, score_format)

            category_row_count = len(score_table)
            chart = workbook.add_chart({"type": "bar"})
            category_range = f"='{sheet_name}'!$A$2:$A${category_row_count + 1}"

            for series_column_index, series_name in [
                (1, reference_model_name),
                (2, comparison_model_name),
            ]:
                column_letter = xlsxwriter.utility.xl_col_to_name(series_column_index)
                value_range = f"='{sheet_name}'!${column_letter}$2:${column_letter}${category_row_count + 1}"
                chart.add_series(
                    {
                        "name": series_name,
                        "categories": category_range,
                        "values": value_range,
                    }
                )

            chart.set_title({"name": f"{dataset_name} Comparison"})
            chart.set_x_axis({"name": "Score (%)", "min": 0, "max": 100})
            chart.set_y_axis({"name": "Category"})
            chart.set_legend({"position": "bottom"})

            worksheet.insert_chart("E2", chart)

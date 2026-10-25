"""Excel workbooks that compare two models' ``summary_brief.csv`` results."""

from __future__ import annotations

import importlib.util
import logging
import re
from typing import TYPE_CHECKING

import pandas as pd

from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    CCPC_SUMMARY_RATE_HEADER,
)
from llm_behavior_eval.evaluation_utils.summary_columns import (
    ACCURACY_COLUMN,
    ATTACK_SUCCESS_RATE_COLUMN,
    DATASET_COLUMN,
    ERROR_COLUMN,
)

if TYPE_CHECKING:
    from pathlib import Path

EXCEL_INSTALL_HINT = (
    "Excel export needs the optional Excel extra: "
    "pip install 'llm-behavior-eval[excel]'"
)
DEFAULT_PRETRAINED_NAME = "Pretrained model"
DEFAULT_UNLEARNED_NAME = "Unlearned model"
SUMMARY_FILE_NAME = "summary_brief.csv"
# Each summary row reports one headline metric; refusal rows report two rates
# and no headline, so they are left out.
HEADLINE_METRIC_COLUMNS = (
    ERROR_COLUMN,
    ATTACK_SUCCESS_RATE_COLUMN,
    CCPC_SUMMARY_RATE_HEADER,
    ACCURACY_COLUMN,
)
CATEGORY_HEADER = "Category"
METRIC_HEADER = "Metric"
SCORE_COLUMN = "Score"
SCORE_SUFFIXES = ("_pretrained", "_unlearned")
MAX_SHEET_NAME_LENGTH = 31
INVALID_SHEET_CHARACTERS = re.compile(r"[\[\]:*?/\\]")


def _resolve_summary_path(summary: Path) -> Path:
    summary_path = summary / SUMMARY_FILE_NAME if summary.is_dir() else summary
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    return summary_path


def _split_dataset_label(label: str) -> tuple[str, str]:
    """Split ``"BBQ: age bias"`` into dataset ``"BBQ"`` and category ``"age bias"``."""
    dataset, separator, category = label.partition(":")
    if not separator:
        return label, label
    return dataset.strip(), category.strip()


def load_headline_scores(summary: Path) -> pd.DataFrame:
    """Read one model's headline score per summary row.

    Args:
        summary: A ``summary_brief.csv`` file or the model results folder that
            contains it.

    Returns:
        One row per summary label with ``Dataset``, ``Category``, ``Metric`` and
        ``Score`` columns. When a label appears more than once, the latest row wins.

    Raises:
        ValueError: If the summary has no ``Dataset`` column.
    """
    summary_path = _resolve_summary_path(summary)
    summary_rows = pd.read_csv(summary_path)
    if DATASET_COLUMN not in summary_rows.columns:
        raise ValueError(f"{summary_path} has no {DATASET_COLUMN!r} column.")
    latest_rows = summary_rows.drop_duplicates(subset=DATASET_COLUMN, keep="last")
    metric_columns = [
        column for column in HEADLINE_METRIC_COLUMNS if column in latest_rows.columns
    ]

    scores: list[dict[str, str | float]] = []
    for _, row in latest_rows.iterrows():
        label = str(row[DATASET_COLUMN])
        metric = next(
            (column for column in metric_columns if pd.notna(row[column])), None
        )
        if metric is None:
            logging.info("Skipping %r in %s: no headline metric.", label, summary_path)
            continue
        dataset, category = _split_dataset_label(label)
        scores.append(
            {
                DATASET_COLUMN: dataset,
                CATEGORY_HEADER: category,
                METRIC_HEADER: metric,
                SCORE_COLUMN: float(row[metric]),
            }
        )
    return pd.DataFrame(
        scores, columns=[DATASET_COLUMN, CATEGORY_HEADER, METRIC_HEADER, SCORE_COLUMN]
    )


def _sheet_name(dataset: str, used_names: set[str]) -> str:
    base_name = INVALID_SHEET_CHARACTERS.sub("-", dataset).strip("'") or "Dataset"
    sheet_name = base_name[:MAX_SHEET_NAME_LENGTH]
    suffix_number = 2
    while sheet_name.lower() in used_names:
        suffix = f" ({suffix_number})"
        sheet_name = base_name[: MAX_SHEET_NAME_LENGTH - len(suffix)] + suffix
        suffix_number += 1
    used_names.add(sheet_name.lower())
    return sheet_name


def export_excel(
    pretrained_summary: Path,
    unlearned_summary: Path,
    output_file: Path,
    datasets: list[str] | None = None,
    pretrained_name: str = DEFAULT_PRETRAINED_NAME,
    unlearned_name: str = DEFAULT_UNLEARNED_NAME,
) -> list[str]:
    """Write a workbook with one comparison sheet and bar chart per dataset.

    A dataset is the part of a summary label before ``:`` (``BBQ`` in
    ``BBQ: age bias``) and its categories are the parts after it. Labels without
    ``:`` (for example ``halueval``) form their own dataset.

    Args:
        pretrained_summary: The first model's ``summary_brief.csv`` or results folder.
        unlearned_summary: The second model's ``summary_brief.csv`` or results folder.
        output_file: The ``.xlsx`` file to write.
        datasets: Datasets to export, in sheet order. Defaults to every dataset
            in either summary, in summary order.
        pretrained_name: Column and legend name for the first model.
        unlearned_name: Column and legend name for the second model.

    Returns:
        The written sheet names.

    Raises:
        ImportError: If XlsxWriter is not installed.
        ValueError: If the model names match, a requested dataset is missing,
            or there is nothing to export.
    """
    if importlib.util.find_spec("xlsxwriter") is None:
        raise ImportError(EXCEL_INSTALL_HINT)
    if pretrained_name == unlearned_name:
        raise ValueError("The two model names must differ.")

    merge_keys = [DATASET_COLUMN, CATEGORY_HEADER, METRIC_HEADER]
    # Fixed internal score columns; the model names only label the workbook,
    # so a name like "Dataset" cannot collide with a merge key.
    comparison = pd.merge(
        load_headline_scores(pretrained_summary),
        load_headline_scores(unlearned_summary),
        on=merge_keys,
        how="outer",
        sort=False,
        suffixes=SCORE_SUFFIXES,
    )
    score_columns = [f"{SCORE_COLUMN}{suffix}" for suffix in SCORE_SUFFIXES]
    available_datasets = list(dict.fromkeys(comparison[DATASET_COLUMN]))
    selected_datasets = datasets or available_datasets
    missing_datasets = sorted(set(selected_datasets) - set(available_datasets))
    if missing_datasets:
        raise ValueError(
            f"Datasets not found in either summary: {missing_datasets}. "
            f"Available: {available_datasets}"
        )
    if not selected_datasets:
        raise ValueError("Neither summary has a row with a headline metric.")

    import xlsxwriter

    output_file.parent.mkdir(parents=True, exist_ok=True)
    sheet_names: list[str] = []
    used_names: set[str] = set()
    with xlsxwriter.Workbook(output_file) as workbook:
        header_format = workbook.add_format({"bold": True, "bottom": 1})
        score_format = workbook.add_format({"num_format": "0.0"})
        for dataset in selected_datasets:
            sheet_name = _sheet_name(dataset, used_names)
            sheet_names.append(sheet_name)
            rows = comparison[comparison[DATASET_COLUMN] == dataset]
            worksheet = workbook.add_worksheet(sheet_name)
            headers = [CATEGORY_HEADER, METRIC_HEADER, pretrained_name, unlearned_name]
            worksheet.write_row(0, 0, headers, header_format)
            for row_index, (_, row) in enumerate(rows.iterrows(), start=1):
                worksheet.write_string(row_index, 0, row[CATEGORY_HEADER])
                worksheet.write_string(row_index, 1, row[METRIC_HEADER])
                scores = row[score_columns]
                for column_index, score in enumerate(scores, start=2):
                    if pd.notna(score):
                        worksheet.write_number(
                            row_index, column_index, score, score_format
                        )
            worksheet.set_column(0, 1, 28)
            worksheet.set_column(2, 3, 18)

            last_row = len(rows)
            chart = workbook.add_chart({"type": "bar"})
            if chart is None:
                raise RuntimeError("XlsxWriter could not create a bar chart.")
            for column_index in (2, 3):
                chart.add_series(
                    {
                        "name": [sheet_name, 0, column_index],
                        "categories": [sheet_name, 1, 0, last_row, 0],
                        "values": [sheet_name, 1, column_index, last_row, column_index],
                    }
                )
            chart.set_title(
                {"name": f"{dataset}: {pretrained_name} vs {unlearned_name}"}
            )
            chart.set_x_axis({"name": "Score (%)"})
            chart.set_y_axis({"reverse": True})
            chart.set_legend({"position": "bottom"})
            chart.set_size({"width": 720, "height": 120 + 32 * last_row})
            worksheet.insert_chart(1, 5, chart)
    return sheet_names

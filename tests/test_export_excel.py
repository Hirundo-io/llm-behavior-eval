from __future__ import annotations

import importlib.util
import logging
import zipfile
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

module_spec = importlib.util.spec_from_file_location(
    "export_excel_module",
    "llm_behavior_eval/export_excel.py",
)
assert module_spec is not None
assert module_spec.loader is not None
export_excel_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(export_excel_module)

_sanitize_sheet_name = export_excel_module._sanitize_sheet_name
export_excel = export_excel_module.export_excel


def _write_summary(summary_path: Path, summary_dataframe: pd.DataFrame) -> None:
    summary_dataframe.to_csv(summary_path, index=False)


def test_sanitize_sheet_name_removes_invalid_characters_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        sanitized_name = _sanitize_sheet_name("BBQ: gender/bias?*")

    assert sanitized_name == "BBQ- gender-bias--"
    assert "Adjusted sheet name" in caplog.text


def test_export_excel_writes_requested_datasets(tmp_path: Path) -> None:
    reference_summary_path = tmp_path / "reference_summary_brief.csv"
    comparison_summary_path = tmp_path / "comparison_summary_brief.csv"
    output_file_path = tmp_path / "comparison.xlsx"

    _write_summary(
        reference_summary_path,
        pd.DataFrame(
            {
                "Dataset": ["BBQ: gender bias", "UNQOVER: race bias"],
                "Accuracy (%) ⬆️": [82.5, 74.1],
                "Stereotype Bias (%)": [12.0, 18.5],
            }
        ),
    )
    _write_summary(
        comparison_summary_path,
        pd.DataFrame(
            {
                "Dataset": ["BBQ: gender bias", "UNQOVER: race bias"],
                "Accuracy (%) ⬆️": [88.0, 79.0],
                "Stereotype Bias (%)": [8.0, 15.0],
            }
        ),
    )

    export_excel(
        output_file=output_file_path,
        reference_summary_csv=reference_summary_path,
        comparison_summary_csv=comparison_summary_path,
        datasets=["BBQ: gender bias"],
        reference_model_name="Pretrained model",
        comparison_model_name="Unlearned model",
    )

    assert output_file_path.exists()

    with zipfile.ZipFile(output_file_path, "r") as workbook_zip:
        workbook_xml = workbook_zip.read("xl/workbook.xml").decode("utf-8")
        sheet_xml = workbook_zip.read("xl/worksheets/sheet1.xml").decode("utf-8")

    assert "BBQ- gender bias" in workbook_xml
    assert "82.5" in sheet_xml
    assert "88" in sheet_xml
    assert "12" in sheet_xml
    assert "8" in sheet_xml


def test_export_excel_requires_overlap(tmp_path: Path) -> None:
    reference_summary_path = tmp_path / "reference_summary_brief.csv"
    comparison_summary_path = tmp_path / "comparison_summary_brief.csv"
    output_file_path = tmp_path / "comparison.xlsx"

    _write_summary(
        reference_summary_path,
        pd.DataFrame({"Dataset": ["BBQ: age bias"], "Accuracy (%) ⬆️": [70.0]}),
    )
    _write_summary(
        comparison_summary_path,
        pd.DataFrame(
            {
                "Dataset": ["UNQOVER: race bias"],
                "Accuracy (%) ⬆️": [80.0],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="No overlapping dataset/category values",
    ):
        export_excel(
            output_file=output_file_path,
            reference_summary_csv=reference_summary_path,
            comparison_summary_csv=comparison_summary_path,
            datasets=None,
            reference_model_name="Pretrained model",
            comparison_model_name="Unlearned model",
        )

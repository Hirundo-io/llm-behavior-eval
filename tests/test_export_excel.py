from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING, Any

import pytest

from llm_behavior_eval.export_excel import export_excel, load_headline_scores

if TYPE_CHECKING:
    from pathlib import Path

SUMMARY_HEADER = (
    "Dataset,Thinking,Accuracy (%) ⬆️,Error (%) ⬇️,Attack success rate (%) ⬇️,"
    "Safe refusal rate (%) ⬇️,Incomplete response rate (%) ⬇️\n"
)


def _write_summary(model_dir: Path, rows: str) -> Path:
    model_dir.mkdir(parents=True)
    summary_path = model_dir / "summary_brief.csv"
    summary_path.write_text(SUMMARY_HEADER + rows, encoding="utf-8")
    return summary_path


@pytest.fixture
def summaries(tmp_path: Path) -> tuple[Path, Path]:
    pretrained = _write_summary(
        tmp_path / "pretrained",
        "BBQ: age bias,off,,41.0,,,0.0\n"
        "BBQ: race bias,off,,30.0,,,0.0\n"
        "halueval,off,72.0,,,,0.0\n"
        "XSTest,off,,,,8.0,0.0\n"
        "BBQ: age bias,off,,40.0,,,0.0\n",
    )
    unlearned = _write_summary(
        tmp_path / "unlearned",
        "BBQ: age bias,off,,12.5,,,0.0\n"
        "UNQOVER: race bias,off,,20.0,,,0.0\n"
        "halueval,off,70.0,,,,0.0\n",
    )
    return pretrained, unlearned


def test_load_headline_scores_splits_labels_and_keeps_latest_row(
    summaries: tuple[Path, Path],
) -> None:
    pretrained, _ = summaries

    scores = load_headline_scores(pretrained.parent)

    assert scores.to_dict("records") == [
        {
            "Dataset": "BBQ",
            "Category": "race bias",
            "Metric": "Error (%) ⬇️",
            "Score": 30.0,
        },
        {
            "Dataset": "halueval",
            "Category": "halueval",
            "Metric": "Accuracy (%) ⬆️",
            "Score": 72.0,
        },
        {
            "Dataset": "BBQ",
            "Category": "age bias",
            "Metric": "Error (%) ⬇️",
            "Score": 40.0,
        },
    ]


def _workbook_parts(workbook_path: Path) -> tuple[str, str, int]:
    with zipfile.ZipFile(workbook_path) as workbook:
        names = workbook.namelist()
        return (
            workbook.read("xl/workbook.xml").decode("utf-8"),
            workbook.read("xl/sharedStrings.xml").decode("utf-8"),
            sum(name.startswith("xl/charts/chart") for name in names),
        )


def test_export_excel_writes_a_sheet_and_chart_per_dataset(
    summaries: tuple[Path, Path], tmp_path: Path
) -> None:
    pretrained, unlearned = summaries
    output_file = tmp_path / "out" / "comparison.xlsx"

    sheet_names = export_excel(pretrained, unlearned, output_file)

    workbook_xml, shared_strings, chart_count = _workbook_parts(output_file)
    assert sheet_names == ["BBQ", "UNQOVER", "halueval"]
    assert chart_count == 3
    for sheet_name in sheet_names:
        assert f'name="{sheet_name}"' in workbook_xml
    for text in ("Pretrained model", "Unlearned model", "age bias", "race bias"):
        assert text in shared_strings


def test_export_excel_uses_requested_datasets_and_model_names(
    summaries: tuple[Path, Path], tmp_path: Path
) -> None:
    pretrained, unlearned = summaries
    output_file = tmp_path / "comparison.xlsx"

    sheet_names = export_excel(
        pretrained,
        unlearned,
        output_file,
        datasets=["halueval"],
        pretrained_name="Base",
        unlearned_name="Unlearned v2",
    )

    _, shared_strings, chart_count = _workbook_parts(output_file)
    assert sheet_names == ["halueval"]
    assert chart_count == 1
    assert "Base" in shared_strings
    assert "Unlearned v2" in shared_strings
    assert "age bias" not in shared_strings


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"datasets": ["CBBQ"]}, "CBBQ", id="unknown-dataset"),
        pytest.param(
            {"pretrained_name": "Same", "unlearned_name": "Same"},
            "must differ",
            id="same-names",
        ),
    ],
)
def test_export_excel_rejects_bad_requests(
    summaries: tuple[Path, Path],
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    pretrained, unlearned = summaries

    with pytest.raises(ValueError, match=message):
        export_excel(pretrained, unlearned, tmp_path / "out.xlsx", **kwargs)


@pytest.mark.parametrize("model_name", ["Dataset", "Category", "Metric", "Score"])
def test_export_excel_accepts_model_names_matching_column_names(
    summaries: tuple[Path, Path], tmp_path: Path, model_name: str
) -> None:
    pretrained, unlearned = summaries
    output_file = tmp_path / "comparison.xlsx"

    sheet_names = export_excel(
        pretrained, unlearned, output_file, pretrained_name=model_name
    )

    _, shared_strings, _ = _workbook_parts(output_file)
    assert sheet_names == ["BBQ", "UNQOVER", "halueval"]
    assert f"<t>{model_name}</t>" in shared_strings


def test_export_excel_reports_missing_summary(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="summary_brief.csv"):
        export_excel(tmp_path, tmp_path, tmp_path / "out.xlsx")

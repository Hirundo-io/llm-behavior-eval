from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from llm_behavior_eval.evaluation_utils import plotting

if TYPE_CHECKING:
    from pathlib import Path


def test_draw_radar_chart_writes_html_with_every_series(tmp_path: Path) -> None:
    chart_path = plotting.draw_radar_chart(
        value_lists=[[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]],
        series_labels=["Pretrained model", "Unlearned model"],
        categories=["age", "gender", "race"],
        title="Error by bias type",
        save_path=tmp_path / "charts" / "error.html",
        colors=["#8D97EC", "#4A3DF4"],
    )

    html = chart_path.read_text(encoding="utf-8")
    assert chart_path == tmp_path / "charts" / "error.html"
    assert "Pretrained model" in html
    assert "Unlearned model" in html
    assert "Error by bias type" in html


@pytest.mark.parametrize(
    ("value_lists", "series_labels", "categories", "colors", "message"),
    [
        pytest.param([[1.0]], ["model"], [], None, "categories", id="no-categories"),
        pytest.param([], [], ["a"], None, "series_labels", id="no-series"),
        pytest.param([[1.0]], ["a", "b"], ["a"], None, "series_labels", id="labels"),
        pytest.param([[1.0, 2.0]], ["model"], ["a"], None, "per category", id="len"),
        pytest.param([[1.0]], ["model"], ["a"], ["red", "blue"], "colors", id="colors"),
    ],
)
def test_draw_radar_chart_rejects_mismatched_inputs(
    tmp_path: Path,
    value_lists: list[list[float]],
    series_labels: list[str],
    categories: list[str],
    colors: list[str] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plotting.draw_radar_chart(
            value_lists=value_lists,
            series_labels=series_labels,
            categories=categories,
            title="chart",
            save_path=tmp_path / "chart.html",
            colors=colors,
        )


def test_require_plotly_explains_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plotting.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(ImportError, match=r"llm-behavior-eval\[plotly\]"):
        plotting.require_plotly()


def test_plot_summary_metrics_charts_each_percentage_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_csv = tmp_path / "model" / "summary_full.csv"
    summary_csv.parent.mkdir()
    summary_csv.write_text(
        "Model,Dataset,Thinking,Accuracy (%) ⬆️,Error (%) ⬇️,Empty Responses\n"
        "model,bbq-age-bias,off,,50.000,0\n"
        "model,bbq-race-bias,off,,40.000,1\n"
        "model,halueval,off,70.000,,0\n"
        "model,bbq-age-bias,off,,45.000,0\n",
        encoding="utf-8",
    )
    logged: list[tuple[str, str | None]] = []
    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(
            log_artifact=lambda path, artifact_path=None: logged.append(
                (path, artifact_path)
            )
        ),
    )

    chart_paths = plotting.plot_summary_metrics(
        summary_csv,
        series_label="model",
        log_to_mlflow=True,
        mlflow_artifact_path="llm-behavior-eval/run",
    )

    plots_dir = summary_csv.parent / "plots"
    assert chart_paths == [
        plots_dir / "radar_accuracy.html",
        plots_dir / "radar_error.html",
    ]
    error_html = chart_paths[1].read_text(encoding="utf-8")
    assert "bbq-race-bias" in error_html
    assert "halueval" not in error_html
    assert "45.0" in error_html
    assert "50.0" not in error_html
    assert logged == [
        (str(path), "llm-behavior-eval/run/plots") for path in chart_paths
    ]


def test_plot_summary_metrics_requires_dataset_column(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary_full.csv"
    summary_csv.write_text("Model,Error (%) ⬇️\nmodel,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset"):
        plotting.plot_summary_metrics(summary_csv, series_label="model")

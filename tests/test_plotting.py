import builtins
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "llm_behavior_eval"
    / "evaluation_utils"
    / "plotting.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("plotting_for_tests", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
plotting = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(plotting)


class ScatterPolarStub:
    def __init__(self, r, theta, fill, name, line=None) -> None:
        self.r = r
        self.theta = theta
        self.fill = fill
        self.name = name
        self.line = line


class FigureStub:
    def __init__(self) -> None:
        self.data: list[ScatterPolarStub] = []
        self.layout = _LayoutStub()
        self.showlegend = False
        self.polar: dict[str, Any] = {}

    def add_trace(self, trace: ScatterPolarStub) -> None:
        self.data.append(trace)

    def update_layout(self, title, showlegend, polar) -> None:
        self.layout.title.text = title
        self.showlegend = showlegend
        self.polar = polar

    def write_html(self, file_path: str, include_plotlyjs: str) -> None:
        Path(file_path).write_text(f"html:{include_plotlyjs}")

    def write_image(self, file_path: str) -> None:
        Path(file_path).write_text("image")


class GraphObjectsStub:
    Figure = FigureStub
    Scatterpolar = ScatterPolarStub


class _TitleStub:
    def __init__(self) -> None:
        self.text = ""


class _LayoutStub:
    def __init__(self) -> None:
        self.title = _TitleStub()


def _install_plotly_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    plotly_module: Any = types.ModuleType("plotly")
    graph_objects_module: Any = types.ModuleType("plotly.graph_objects")
    graph_objects_module.Figure = FigureStub
    graph_objects_module.Scatterpolar = ScatterPolarStub
    plotly_module.graph_objects = graph_objects_module
    monkeypatch.setitem(sys.modules, "plotly", plotly_module)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", graph_objects_module)


def test_validate_plot_args_validates_lengths() -> None:
    with pytest.raises(ValueError, match="categories length"):
        plotting._validate_plot_args(
            value_lists=[[0.2]],
            series_labels=["baseline"],
            categories=["first", "second"],
            colors=None,
        )


def test_draw_radar_chart_writes_html_and_closes_polygon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_plotly_stubs(monkeypatch)
    output_path = tmp_path / "radar.html"

    figure_object = plotting.draw_radar_chart(
        value_lists=[[0.2, 0.5, 0.4], [0.1, 0.3, 0.6]],
        series_labels=["baseline", "candidate"],
        metric_name="accuracy",
        categories=["alpha", "beta", "gamma"],
        save_path=output_path,
    )

    assert output_path.exists()
    first_trace = figure_object.data[0]
    assert first_trace.r[0] == first_trace.r[-1]
    assert first_trace.theta[0] == first_trace.theta[-1]


def test_draw_radar_chart_uses_metric_name_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_plotly_stubs(monkeypatch)
    output_path = tmp_path / "mapped_title.html"

    figure_object = plotting.draw_radar_chart(
        value_lists=[[0.4, 0.2]],
        series_labels=["series"],
        metric_name="custom_metric",
        categories=["first", "second"],
        save_path=output_path,
        metric_name_mapping={"custom_metric": "Readable Metric"},
    )

    assert figure_object.layout.title.text == "Readable Metric by Category"


def test_draw_radar_chart_raises_clear_error_when_plotly_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_import = builtins.__import__

    def import_without_plotly(name, *args, **kwargs):
        if name == "plotly" or name.startswith("plotly."):
            raise ImportError("No module named 'plotly'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_plotly)

    with pytest.raises(ImportError, match="Plotly is required"):
        plotting.draw_radar_chart(
            value_lists=[[0.2, 0.5]],
            series_labels=["baseline"],
            metric_name="accuracy",
            categories=["alpha", "beta"],
            save_path=tmp_path / "missing_plotly.html",
        )

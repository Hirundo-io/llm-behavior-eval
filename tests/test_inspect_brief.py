"""Offline contract tests against real Inspect logs and the installed exporter."""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from llm_behavior_eval import _registry
from llm_behavior_eval.target_metrics import target_metrics, target_metrics_path
from tests.native_fixtures import local_task_dataset


def run_task(name, tmp_path, monkeypatch, *, kind="bias", judge=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    kwargs = {"kind": kind} if name in ("bbq", "bloom", "unqover") else {}
    if name != "inspect_template":
        kwargs["dataset_path"] = str(local_task_dataset(name, tmp_path, monkeypatch))
    text = (
        judge
        or {
            "bbq": "gt_agree: yes candidate_uncertain: no",
            "bloom": "gt_agree: yes candidate_uncertain: no",
            "unqover": "gt_agree: yes candidate_uncertain: no",
            "halueval": "A",
            "medhallu": "A",
            "ccpc": "True",
            "xstest": "2_full_refusal",
            "orbench": "[[direct_refusal]]",
            "inspect_template": "SCORE: 1",
            "prompt_injection": "Yes",
        }[name]
    )
    calls = 0

    def answer(*args):
        nonlocal calls
        calls += 1
        output = ModelOutput.from_content("mockllm/model", "answer")
        if calls % 2 == 0:
            output.choices[0].stop_reason = "max_tokens"
        return output

    logs = inspect_eval(
        f"llm_behavior_eval/{name}",
        task_args=kwargs,
        limit=2,
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={
            "grader": get_model(
                "mockllm/grader",
                custom_outputs=lambda *a: ModelOutput.from_content(
                    "mockllm/grader", text
                ),
            )
        },
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].status == "success", logs[0].error
    return logs[0]


pytest.importorskip("inspect_brief")


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("INSPECT_BRIEF_ENABLED", "true")
    monkeypatch.setenv("INSPECT_BRIEF_SKIP_EXISTING", "false")
    monkeypatch.setenv("INSPECT_BRIEF_TARGET_METRICS", str(target_metrics_path()))
    monkeypatch.delenv("INSPECT_BRIEF_TASKS", raising=False)
    monkeypatch.delenv("INSPECT_BRIEF_CSV_PATH", raising=False)


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def cli(log_dir, csv_path, config=None):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_brief",
            "--log-dir",
            str(log_dir),
            "--target-metrics",
            str(config or target_metrics_path()),
            "--csv-path",
            str(csv_path),
            "--skip-existing",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return rows(csv_path)


def test_registered_coverage():
    assert {
        f"llm_behavior_eval/{n}" for n in _registry.__all__
    } <= target_metrics().keys()


@pytest.mark.parametrize("name", _registry.__all__)
def test_real_task_hook_and_cli(name, tmp_path, monkeypatch):
    output = tmp_path / "hook.csv"
    monkeypatch.setenv("INSPECT_BRIEF_CSV_PATH", str(output))
    monkeypatch.setenv("INSPECT_BRIEF_TASKS", f"llm_behavior_eval/{name}")
    log = run_task(name, tmp_path / "logs", monkeypatch)
    assert log.results
    targets = target_metrics()[log.eval.task]
    emitted = {k: m.value for s in log.results.scores for k, m in s.metrics.items()}
    assert {m["name"] for m in targets} <= emitted.keys()
    exported = rows(output)
    assert len(exported) == len(targets)
    for row, metric in zip(exported, targets, strict=True):
        expected = emitted[metric["name"]]
        if metric["is_percentage"] and metric["is_normalized"]:
            expected *= 100
        assert float(row["Score"]) == pytest.approx(expected)
        assert row["Metric"] == metric["name"] + (
            " (%)" if metric["is_percentage"] else ""
        ) + (" ⬆️" if metric["is_higher_better"] else " ⬇️")
    assert not any("heuristic" in r["Metric"] for r in exported)
    assert cli(tmp_path / "logs", tmp_path / "cli.csv") == exported
    assert cli(tmp_path / "logs", tmp_path / "cli.csv") == exported
    run_task(name, tmp_path / "logs", monkeypatch)
    assert len(rows(output)) == 2 * len(targets)


@pytest.mark.parametrize(
    "name,kind,primary",
    [
        ("bbq", "bias", "error"),
        ("bbq", "unbias", "accuracy"),
        ("bloom", "ambiguous", "error"),
        ("bloom", "unbias", "accuracy"),
        ("unqover", "bias", "error"),
    ],
)
def test_kind_config(name, kind, primary, tmp_path, monkeypatch):
    config = target_metrics(bias_kind=kind)
    selected = config[f"llm_behavior_eval/{name}"]
    assert {m["name"] for m in selected} & {"accuracy", "error"} == {primary}
    config_path = tmp_path / "metrics.json"
    config_path.write_text(json.dumps(config))
    run_task(name, tmp_path / "logs", monkeypatch, kind=kind)
    exported = cli(tmp_path / "logs", tmp_path / "brief.csv", config_path)
    assert len(exported) == 3
    assert exported[0]["Metric"].startswith(primary)


def test_ccpc_undefined_remains_nan(tmp_path, monkeypatch):
    output = tmp_path / "hook.csv"
    monkeypatch.setenv("INSPECT_BRIEF_CSV_PATH", str(output))
    run_task("ccpc", tmp_path / "logs", monkeypatch, judge="unparseable")
    assert math.isnan(float(rows(output)[0]["Score"]))
    assert math.isnan(float(cli(tmp_path / "logs", tmp_path / "cli.csv")[0]["Score"]))


def test_prompt_injection_native_percentage(tmp_path, monkeypatch):
    log = run_task("prompt_injection", tmp_path / "logs", monkeypatch)
    assert log.results
    emitted = {k: m.value for s in log.results.scores for k, m in s.metrics.items()}
    exported = cli(tmp_path / "logs", tmp_path / "brief.csv")
    assert float(exported[0]["Score"]) == emitted["attack_success_rate"]
    assert float(exported[1]["Score"]) == 50


def test_accessor_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "llm_behavior_eval.target_metrics", "--path"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(Path(result.stdout.strip()).read_text()) == target_metrics()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_behavior_eval.target_metrics",
            "--bias-kind",
            "unbias",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == target_metrics(bias_kind="unbias")


def test_registered_cli_hook(tmp_path, monkeypatch):
    """Exercise installed entry-point naming in a fresh interpreter."""
    output = tmp_path / "hook.csv"
    monkeypatch.setenv("INSPECT_BRIEF_CSV_PATH", str(output))
    monkeypatch.setenv("INSPECT_BRIEF_TASKS", "llm_behavior_eval/inspect_template")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "llm_behavior_eval/inspect_template",
            "--model",
            "mockllm/model",
            "--display",
            "none",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    exported = rows(output)
    assert len(exported) == 2
    assert {r["Benchmark"] for r in exported} == {"llm_behavior_eval/inspect_template"}
    assert cli(tmp_path / "logs", tmp_path / "cli.csv") == exported


def test_scale_and_direction_contract():
    config = target_metrics()
    for task, metrics in config.items():
        for metric in metrics:
            name = metric["name"]
            already_percentage = (
                task.endswith(("/halueval", "/medhallu")) and name == "accuracy"
            ) or name == "attack_success_rate"
            assert metric["is_normalized"] is (
                not already_percentage and name != "stereotype_bias"
            )
            assert metric["is_percentage"] is (name != "stereotype_bias")
            assert metric["is_higher_better"] is (
                name in ("accuracy", "unsafe_refusal_rate", "field_rate")
            )

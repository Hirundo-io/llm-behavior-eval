"""inspect-brief on real mock-model ``.eval`` logs of every registered task.

The merged per-package target metrics (``common.target_metrics``) are handed to
inspect-brief as a file or as inline JSON, exactly as users do. Both the hook
(``INSPECT_BRIEF_*`` environment) and the ``inspect-brief`` CLI must report
every target with the values in the log.
"""

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from llm_behavior_eval import _registry
from llm_behavior_eval.common.target_metrics import (
    load_target_metrics,
    missing_target_metrics,
    write_target_metrics,
)
from tests.native_fixtures import GRADER_VERDICTS, local_task_dataset, require_fixture

pytest.importorskip("inspect_brief")

REPO = Path(__file__).parents[1]
TEMPLATE = "inspect_template"


@pytest.fixture
def targets_file(tmp_path, monkeypatch) -> Path:
    """The merged target metrics, written once and configured for the hook."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED", raising=False)
    monkeypatch.delenv("INSPECT_BRIEF_TASKS", raising=False)
    monkeypatch.delenv("INSPECT_BRIEF_CSV_PATH", raising=False)
    monkeypatch.setenv("INSPECT_BRIEF_ENABLED", "true")
    monkeypatch.setenv("INSPECT_BRIEF_SKIP_EXISTING", "false")
    path = write_target_metrics(tmp_path / "target_metrics.json")
    monkeypatch.setenv("INSPECT_BRIEF_TARGET_METRICS", str(path))
    return path


def mock_output(model: str, content: str) -> ModelOutput:
    """A mock reply with explicit usage, so MockLLM never fetches a tokenizer."""
    output = ModelOutput.from_content(model, content)
    output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    return output


def run_task(
    name: str,
    log_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset_dir: Path | None = None,
) -> EvalLog:
    """Run a registered task on two samples; every other answer hits max_tokens."""
    log_dir.mkdir(parents=True, exist_ok=True)
    args: dict[str, str] = {}
    if name != TEMPLATE:
        args["dataset_path"] = str(
            local_task_dataset(name, dataset_dir or log_dir, monkeypatch)
        )
    calls = 0

    def answer(*_):
        nonlocal calls
        calls += 1
        output = mock_output("mockllm/model", "answer")
        if calls % 2 == 0:
            output.choices[0].stop_reason = "max_tokens"
        return output

    grader = get_model(
        "mockllm/grader",
        custom_outputs=lambda *_: mock_output("mockllm/grader", GRADER_VERDICTS[name]),
    )
    [log] = inspect_eval(
        f"llm_behavior_eval/{name}",
        task_args=args,
        limit=2,
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={"grader": grader},
        log_dir=str(log_dir),
        display="none",
    )
    assert log.status == "success", log.error
    return log


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def brief_cli(log_dir: Path, csv_path: Path, targets: str) -> list[dict[str, str]]:
    executable = shutil.which("inspect-brief", path=str(Path(sys.executable).parent))
    assert executable, "the [brief] extra installs the inspect-brief command"
    result = subprocess.run(
        [
            executable,
            "--log-dir",
            str(log_dir),
            "--target-metrics",
            targets,
            "--csv-path",
            str(csv_path),
            "--skip-existing",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return rows(csv_path)


def expected_metric_column(metric: dict[str, object]) -> str:
    name = str(metric["name"])
    if metric["is_percentage"]:
        name += " (%)"
    return name + (" ⬆️" if metric["is_higher_better"] else " ⬇️")


@pytest.mark.parametrize("name", _registry.__all__)
def test_hook_and_cli_report_every_target(name, tmp_path, monkeypatch, targets_file):
    require_fixture(name)
    hook_csv = tmp_path / "hook.csv"
    monkeypatch.setenv("INSPECT_BRIEF_CSV_PATH", str(hook_csv))
    monkeypatch.setenv("INSPECT_BRIEF_TASKS", f"llm_behavior_eval/{name}")
    log = run_task(name, tmp_path / "logs", monkeypatch)
    assert log.results
    assert missing_target_metrics(log.location) == []
    targets = load_target_metrics()[log.eval.task]
    emitted = {k: m.value for s in log.results.scores for k, m in s.metrics.items()}

    exported = rows(hook_csv)
    assert [row["Metric"] for row in exported] == [
        expected_metric_column(m) for m in targets
    ]
    for row, metric in zip(exported, targets, strict=True):
        assert row["Benchmark"] == f"llm_behavior_eval/{name}"
        expected = emitted[metric["name"]]
        if metric["is_percentage"] and metric["is_normalized"]:
            expected *= 100
        assert float(row["Score"]) == pytest.approx(expected, nan_ok=True)

    # The CLI reads the same logs into the same rows; --skip-existing makes a
    # second export a no-op, and a new run appends one more set.
    cli_csv = tmp_path / "cli.csv"
    assert brief_cli(tmp_path / "logs", cli_csv, str(targets_file)) == exported
    assert brief_cli(tmp_path / "logs", cli_csv, str(targets_file)) == exported
    run_task(name, tmp_path / "logs", monkeypatch, dataset_dir=tmp_path / "rerun")
    assert len(rows(hook_csv)) == 2 * len(targets)


def test_module_cli_writes_targets_outside_the_checkout(
    tmp_path, monkeypatch, targets_file
):
    """``python -m llm_behavior_eval.common.target_metrics --output`` feeds the CLI.

    Pass a file, not inline JSON: inspect-brief 0.1.1 first treats the value as
    a path, and on Python 3.12/3.13 a value longer than a file name raises
    ``ENAMETOOLONG``, which the merged mapping exceeds.
    """
    log = run_task("halueval", tmp_path / "logs", monkeypatch)
    written = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_behavior_eval.common.target_metrics",
            "--output",
            "targets.json",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert json.loads((tmp_path / written).read_text()) == load_target_metrics()
    exported = brief_cli(
        tmp_path / "logs", tmp_path / "out.csv", str(tmp_path / written)
    )
    assert [row["Metric"] for row in exported] == [
        expected_metric_column(m) for m in load_target_metrics()[log.eval.task]
    ]


@pytest.mark.parametrize("name", ["bbq", "bloom", "unqover"])
def test_stereotype_bias_is_not_found_without_a_stereotype_column(
    name, tmp_path, monkeypatch, targets_file
):
    """Legacy behavior: no ``stereotyped_answer`` column, no stereotype_bias."""
    from datasets import Dataset

    source = tmp_path / "source"
    Dataset.from_list([{"question": "q", "answer": "a"}] * 2).save_to_disk(source)
    hook_csv = tmp_path / "hook.csv"
    monkeypatch.setenv("INSPECT_BRIEF_CSV_PATH", str(hook_csv))
    [log] = inspect_eval(
        f"llm_behavior_eval/{name}",
        task_args={"dataset_path": str(source)},
        model=get_model(
            "mockllm/model",
            custom_outputs=lambda *_: mock_output("mockllm/model", "answer"),
        ),
        model_roles={
            "grader": get_model(
                "mockllm/grader",
                custom_outputs=lambda *_: mock_output(
                    "mockllm/grader", GRADER_VERDICTS[name]
                ),
            )
        },
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert missing_target_metrics(log.location) == ["stereotype_bias"]
    exported = {row["Metric"].split(" ")[0]: row["Score"] for row in rows(hook_csv)}
    assert exported["stereotype_bias"] == "Metric 'stereotype_bias' not found"
    for metric in ("error", "accuracy", "incomplete_response_rate"):
        assert not math.isnan(float(exported[metric]))


def run_template_cli(command: list[str], cwd: Path, tmp_path: Path) -> list[dict]:
    """Run the template through an Inspect CLI in ``cwd``; return hook rows."""
    csv_path = tmp_path / f"{Path(command[-1]).name}-{cwd.name}.csv"
    result = subprocess.run(
        [
            *command,
            "eval",
            f"llm_behavior_eval/{TEMPLATE}",
            "--model",
            "mockllm/model",
            "--display",
            "none",
            "--log-dir",
            str(tmp_path / "logs" / csv_path.stem),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "INSPECT_BRIEF_CSV_PATH": str(csv_path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return rows(csv_path)


def test_inspect_command_applies_targets_from_any_directory(tmp_path, targets_file):
    inspect = shutil.which("inspect", path=str(Path(sys.executable).parent))
    assert inspect
    for cwd in (REPO, tmp_path):
        exported = run_template_cli([inspect], cwd, tmp_path)
        assert {row["Benchmark"] for row in exported} == {
            f"llm_behavior_eval/{TEMPLATE}"
        }
        assert [row["Metric"] for row in exported] == [
            expected_metric_column(m)
            for m in load_target_metrics()[f"llm_behavior_eval/{TEMPLATE}"]
        ]


def test_python_m_inspect_ai_from_the_repo_root_drops_the_task_prefix(
    tmp_path, targets_file
):
    """Documented caveat: the source tree's egg-info hides the install metadata.

    The log's task is then ``inspect_template``, which is not a target key, so
    inspect-brief writes raw metric values without the target conventions.
    """
    if not any(REPO.glob("*.egg-info")):
        pytest.skip("no source-tree egg-info in this checkout")
    exported = run_template_cli([sys.executable, "-m", "inspect_ai"], REPO, tmp_path)
    assert {row["Benchmark"] for row in exported} == {TEMPLATE}
    assert "field_rate" in {row["Metric"] for row in exported}

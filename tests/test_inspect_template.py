import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalSpec, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig

from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)
from llm_behavior_eval.inspect_template.task import inspect_template

TEMPLATE = "llm_behavior_eval.inspect_template"


def test_template_sets_no_inference_settings():
    task = inspect_template()
    assert task.config == GenerateConfig()


def test_shipped_parity_config_holds_only_inference_settings():
    path = parity_config_path(TEMPLATE)
    assert path.parent == Path(__file__).parents[1] / "llm_behavior_eval" / (
        "inspect_template"
    )
    config = yaml.safe_load(path.read_text())
    assert set(config) <= {"generate_config", "eval_config", "model_roles"}
    GenerateConfig.model_validate(config["generate_config"])
    grader = config["model_roles"]["grader"]
    GenerateConfig.model_validate(grader["config"])
    # HF graders need greedy decoding requested explicitly.
    assert grader["model"].startswith("hf/")
    assert grader["args"] == {"do_sample": False}


@pytest.fixture
def mock_parity_config(tmp_path) -> Path:
    """The shipped parity config with only the grader model swapped for a mock."""
    config = yaml.safe_load(parity_config_path(TEMPLATE).read_text())
    config["model_roles"]["grader"]["model"] = "mockllm/model"
    config["model_roles"]["grader"]["args"] = {}
    path = tmp_path / "parity.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def assert_parity_recorded(spec: EvalSpec) -> None:
    generate = spec.model_generate_config
    assert (generate.temperature, generate.max_tokens) == (0.0, 128)
    assert spec.config.limit == 500
    assert spec.model_roles
    grader = spec.model_roles["grader"]
    assert not isinstance(grader, list)
    assert (grader.config.temperature, grader.config.max_tokens) == (0.0, 32)


def test_parity_config_cli(tmp_path, mock_parity_config):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "llm_behavior_eval/inspect_template",
            "--model",
            "mockllm/model",
            "--run-config",
            str(mock_parity_config),
            "--log-dir",
            str(tmp_path / "logs"),
            "--display",
            "none",
        ],
        capture_output=True,
        text=True,
        # Not the repository root: `python -m` puts the working directory first
        # on sys.path, where the source-tree egg-info hides the install metadata
        # Inspect needs to name the task llm_behavior_eval/inspect_template.
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    logs = list_eval_logs(str(tmp_path / "logs"))
    assert len(logs) == 1
    log = read_eval_log(logs[0])
    assert_parity_recorded(log.eval)
    assert missing_target_metrics(log) == []


def test_parity_config_python(tmp_path, mock_parity_config):
    logs = inspect_eval(
        inspect_template(),
        model="mockllm/model",
        log_dir=str(tmp_path),
        display="none",
        **read_parity_config(mock_parity_config),
    )
    assert logs[0].status == "success"
    assert_parity_recorded(logs[0].eval)


@pytest.mark.parametrize(
    "extra",
    [{"model": "hf/org/repo"}, {"task": "x"}, {"solver": "generate"}],
)
def test_parity_config_rejects_non_settings(tmp_path, extra: dict[str, Any]):
    path = tmp_path / "parity.yaml"
    path.write_text(yaml.safe_dump({"generate_config": {"temperature": 0.0}} | extra))
    with pytest.raises(ValueError, match="only inference settings"):
        read_parity_config(path)


def test_missing_parity_config():
    with pytest.raises(FileNotFoundError):
        parity_config_path("llm_behavior_eval.common")


def test_template_target_metrics_match_eval_metric_names(tmp_path):
    path = parity_config_path(TEMPLATE).with_name(TARGET_METRICS_FILE)
    targets = read_target_metrics_file(path)
    assert list(targets) == ["llm_behavior_eval/inspect_template"]
    assert (
        load_target_metrics()["llm_behavior_eval/inspect_template"]
        == (targets["llm_behavior_eval/inspect_template"])
    )
    logs = inspect_eval(
        inspect_template(), model="mockllm/model", log_dir=str(tmp_path), display="none"
    )
    assert logs[0].status == "success", logs[0].error
    # Every target name is found in the written .eval log as inspect-brief
    # reads it: by log.eval.task, then as a key of a scorer's metrics.
    assert missing_target_metrics(logs[0].location) == []

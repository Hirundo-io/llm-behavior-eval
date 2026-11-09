"""The retired legacy CLI and the native preset example."""

import subprocess
import sys

import pytest
from inspect_ai.model import GenerateConfig
from typer.testing import CliRunner

from llm_behavior_eval import get_dataset_preset
from llm_behavior_eval.evaluate import app, preset_commands


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["org/model", "hallu"],
        ["org/model", "hallu", "--use-mlflow"],
        ["--unknown-legacy-flag"],
        ["--model-token"],
    ],
)
def test_retired_cli_never_executes(args):
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "no evaluation ran" in result.output
    assert "inspect eval llm_behavior_eval/halueval" in result.output
    assert "--model-role grader=" in result.output
    assert "parity.yaml" in result.output
    assert "INSPECT_MIGRATION.md" in result.output


def test_retired_cli_lists_the_preset_tasks_without_echoing_arguments():
    secret = "hf_secret_token_value"
    result = CliRunner().invoke(
        app, ["/models/private", " BLOOM : BIAS : ALL ", "--model-token", secret]
    )
    assert result.exit_code == 2
    assert secret not in result.output
    assert "/models/private" not in result.output
    for bias_type in ("age", "gender", "race"):
        assert (
            f"inspect eval llm_behavior_eval/bloom -T bias_type={bias_type} -T kind=bias"
            in result.output
        )


def test_preset_commands_carry_no_inference_settings():
    [command] = preset_commands(get_dataset_preset("refusal:xstest"))
    assert command == (
        "inspect eval llm_behavior_eval/xstest "
        "--model <provider/model> --model-role grader=<provider/model>"
    )
    for flag in ("--temperature", "--max-tokens", "--generate-config", "-M"):
        assert flag not in command


def test_help_explains_retirement():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Retired" in result.output


@pytest.mark.parametrize("module", ["llm_behavior_eval", "llm_behavior_eval.evaluate"])
def test_module_entrypoints_are_retired(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "org/model", "hallu"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "no evaluation ran" in result.stderr
    assert "inspect eval llm_behavior_eval/halueval --model" in result.stderr


def test_example_runs_preset_tasks_with_per_role_settings(monkeypatch):
    from examples import presets_customization

    models = []
    calls = []

    def model(name, **kwargs):
        models.append((name, kwargs))
        return f"model:{name}"

    monkeypatch.setattr(presets_customization, "get_model", model)
    monkeypatch.setattr(
        presets_customization, "inspect_eval", lambda *a, **kw: calls.append((a, kw))
    )
    presets_customization.main(
        ["--preset", "bias:gender", "--preset", "hallu-med", "--limit", "3"]
    )
    [((tasks,), options)] = calls
    assert [task.name for task in tasks] == [
        "llm_behavior_eval/bbq",
        "llm_behavior_eval/medhallu",
    ]
    # The tasks themselves carry no generation settings.
    assert all(task.config == GenerateConfig() for task in tasks)
    assert all(not task.model_roles for task in tasks)
    assert options["model"] == "hf/meta-llama/Llama-3.1-8B-Instruct"
    assert options["model_args"] == {"do_sample": False}
    assert (options["temperature"], options["max_tokens"]) == (0.0, 512)
    assert options["limit"] == 3
    assert options["model_roles"] == {"grader": "model:hf/google/gemma-4-12B-it"}
    [(grader, grader_kwargs)] = models
    assert grader == "hf/google/gemma-4-12B-it"
    assert grader_kwargs["config"] == GenerateConfig(temperature=0.0, max_tokens=64)
    assert grader_kwargs["do_sample"] is False


def test_example_prints_equivalent_commands(capsys):
    from examples import presets_customization

    presets_customization.main(["--preset", "refusal:all", "--print-commands"])
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[2] for line in lines] == [
        "llm_behavior_eval/xstest",
        "llm_behavior_eval/orbench",
    ]
    assert all("--model-role" in line and "do_sample: false" in line for line in lines)

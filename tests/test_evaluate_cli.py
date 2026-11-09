import pytest
from typer.testing import CliRunner

from llm_behavior_eval.evaluate import app


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
    assert "hf/google/gemma-4-12B-it" in result.output
    assert "INSPECT_MIGRATION.md" in result.output


def test_help_explains_retirement():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Retired" in result.output


def test_native_example_selects_tasks_and_independent_grader(monkeypatch):
    from importlib import import_module

    from inspect_ai import Task

    from examples import presets_customization

    selected = []
    for name in ("bbq", "medhallu"):

        def capture(_name=name, **kwargs):
            selected.append((f"llm_behavior_eval/{_name}", kwargs))
            return Task()

        monkeypatch.setattr(
            import_module(f"llm_behavior_eval.{name}.task"), name, capture
        )
    calls = []
    models = []

    def model(name, **kwargs):
        models.append((name, kwargs))
        return name

    monkeypatch.setattr(presets_customization, "get_model", model)
    monkeypatch.setattr(
        presets_customization, "inspect_eval", lambda *a, **kw: calls.append((a, kw))
    )
    presets_customization.main()
    (tasks,), options = calls[0]
    assert len(tasks) == 2
    assert [task.name for task in tasks] == [
        "llm_behavior_eval/bbq",
        "llm_behavior_eval/medhallu",
    ]
    assert selected == [
        (
            "llm_behavior_eval/bbq",
            {
                "bias_type": "gender",
                "kind": "bias",
                "max_samples": 200,
                "compatibility_tokenizer": "/models/answer",
            },
        ),
        (
            "llm_behavior_eval/medhallu",
            {"max_samples": 200, "compatibility_tokenizer": "/models/answer"},
        ),
    ]
    assert options["model_roles"] == {"grader": "hf/local-judge"}
    assert options["model"] != options["model_roles"]["grader"]
    assert models == [
        (
            options["model"],
            {
                "enable_thinking": False,
                "do_sample": False,
                "model_path": "/models/answer",
                "tokenizer_call_args": {
                    "truncation": True,
                    "max_length": 1024,
                    "add_special_tokens": True,
                },
            },
        ),
        (
            options["model_roles"]["grader"],
            {
                "enable_thinking": False,
                "do_sample": False,
                "model_path": "/models/judge",
            },
        ),
    ]


@pytest.mark.parametrize("module", ["llm_behavior_eval", "llm_behavior_eval.evaluate"])
def test_module_entrypoints_are_retired(module):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", module, "org/model", "hallu"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "no evaluation ran" in result.stderr

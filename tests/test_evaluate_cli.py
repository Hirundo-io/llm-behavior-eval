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
    assert "INSPECT_MIGRATION.md" in result.output


def test_help_explains_retirement():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Retired" in result.output


def test_native_example_selects_tasks_and_independent_grader(monkeypatch):
    from importlib import import_module

    from examples import presets_customization

    selected = []
    for name in ("bbq", "medhallu"):

        def capture(_name=name, **kwargs):
            selected.append((f"llm_behavior_eval/{_name}", kwargs))
            return object()

        monkeypatch.setattr(
            import_module(f"llm_behavior_eval.{name}.task"), name, capture
        )
    calls = []
    monkeypatch.setattr(
        presets_customization, "inspect_eval", lambda *a, **kw: calls.append((a, kw))
    )
    presets_customization.main()
    (tasks,), options = calls[0]
    assert len(tasks) == 2
    assert selected == [
        (
            "llm_behavior_eval/bbq",
            {"bias_type": "gender", "kind": "bias", "max_samples": 200},
        ),
        ("llm_behavior_eval/medhallu", {"max_samples": 200}),
    ]
    assert options["model_roles"] == {"grader": "hf/google/gemma-3-12b-it"}
    assert options["model"] != options["model_roles"]["grader"]


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

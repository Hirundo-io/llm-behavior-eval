"""Retirement notice for the legacy CLI. Execution belongs to native Inspect.

``llm-behavior-eval <model> <preset>`` no longer evaluates anything. It prints
how to run the same benchmarks with ``inspect eval`` and exits with status 2.
When an argument names a known preset, the notice lists that preset's task
names and ``-T`` arguments. Nothing else from the command line is echoed:
legacy invocations may contain model paths or credentials.

Presets select tasks and task arguments only. Inference settings (temperature,
token budgets, thinking) are never translated from legacy flags; users set them
through Inspect for each role, or pass a benchmark's ``parity.yaml``.
"""

import shlex

import typer

from .presets import DatasetPreset, get_dataset_preset

MIGRATION = """The legacy llm-behavior-eval CLI has been retired; no evaluation ran.
Run the benchmarks with native Inspect (Python 3.11+), for example:
  inspect eval llm_behavior_eval/halueval --model hf/org/model \\
    --model-role grader=hf/google/gemma-4-12B-it
Choose any Inspect provider for either role (hf/, vllm/, openai-api/, ...).
The grader is never implied: always pass --model-role grader=...
Presets map to task names and -T arguments: see list_dataset_presets()[i].inspect_tasks.
--max-samples -> --limit N; output directories -> --log-dir.
Inference settings are not translated. Set them per role through Inspect
(--temperature, --max-tokens, -M ..., and the grader's role config), or pass a
benchmark's parity.yaml with --run-config to reproduce legacy settings.
MLflow: set LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED=1; see MLFLOW_INTEGRATION.md.
Summary CSVs: use inspect-brief; see README.md.
See INSPECT_MIGRATION.md for the complete flag mapping."""

app = typer.Typer(add_completion=False)


def _known_preset(argument: str) -> DatasetPreset | None:
    try:
        return get_dataset_preset(argument)
    except ValueError:
        return None


def preset_commands(preset: DatasetPreset) -> list[str]:
    """``inspect eval`` commands for a preset, with placeholder model names."""
    return [
        shlex.join(["inspect", "eval", mapping.task_name, *mapping.cli_args])
        + " --model <provider/model> --model-role grader=<provider/model>"
        for mapping in preset.inspect_tasks
    ]


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main(ctx: typer.Context) -> None:
    """Retired: use inspect eval; see INSPECT_MIGRATION.md for preset/flag mapping."""
    typer.echo(MIGRATION, err=True)
    presets = [p for p in map(_known_preset, ctx.args) if p is not None]
    for preset in presets:
        typer.echo(f"\nPreset {preset.name}:", err=True)
        for command in preset_commands(preset):
            typer.echo(f"  {command}", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()

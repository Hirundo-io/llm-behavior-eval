"""Retirement notice for the old CLI. Execution belongs to native Inspect."""

import typer

MIGRATION = """The legacy llm-behavior-eval CLI has been retired; no evaluation ran.
Use native Inspect (Python 3.12–3.13), for example:
  inspect eval llm_behavior_eval/halueval --model hf/org/model --model-role grader=hf/org/judge
Use list_dataset_presets()[...].inspect_tasks for exact task names and -T arguments.
--max-samples -> -T max_samples=N; output directory -> --log-dir.
Select providers independently with --model and --model-role grader=PROVIDER/MODEL.
Configure native provider arguments with -M or grader role YAML, not legacy flags.
Use environment credentials; never put tokens in task arguments.
Thinking flags differ by benchmark/provider; see INSPECT_MIGRATION.md.
For opt-in MLflow hook settings, see MLFLOW_INTEGRATION.md; legacy flags are not forwarded.
See INSPECT_MIGRATION.md for the complete migration and output schema changes."""

app = typer.Typer(add_completion=False)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main() -> None:
    """Retired: use inspect eval; see INSPECT_MIGRATION.md for preset/flag mapping."""
    # Never echo arguments: obsolete invocations may contain credentials.
    typer.echo(MIGRATION, err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()

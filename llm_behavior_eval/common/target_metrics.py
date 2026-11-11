"""Merge the per-benchmark ``target_metrics.json`` files.

Each benchmark package may ship ``llm_behavior_eval/<benchmark>/target_metrics.json``
mapping its registered task names to the metrics a report should show, in the
format inspect-brief reads::

    {
      "llm_behavior_eval/<task>": [
        {"name": "<metric>", "is_percentage": true,
         "is_higher_better": false, "is_normalized": true}
      ]
    }

``load_target_metrics()`` reads every package's file (sorted by package name),
validates it, and merges the entries into one mapping with sorted keys. A task
key defined twice, within one file or across packages, is an error. Tools that
need a single file path call ``write_target_metrics(path)``, or run
``python -m llm_behavior_eval.common.target_metrics --output PATH``.

``missing_target_metrics(log)`` checks a finished ``.eval`` log the way
inspect-brief reads it; benchmark tests assert it returns ``[]``.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictBool,
    StrictStr,
    ValidationError,
)

from llm_behavior_eval.common.packages import ROOT_PACKAGE, package_dirs, subpackages

TARGET_METRICS_FILE = "target_metrics.json"

if TYPE_CHECKING:
    from inspect_ai.log import EvalLog

TargetMetrics = dict[str, list[dict[str, Any]]]
"""inspect-brief's wire format: task name to a list of metric objects."""


class TargetMetric(BaseModel):
    """One metric entry, with exactly the keys inspect-brief accepts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: StrictStr = Field(min_length=1)
    is_percentage: StrictBool
    is_higher_better: StrictBool
    is_normalized: StrictBool


class TargetMetricsFile(RootModel[dict[str, list[TargetMetric]]]):
    """A ``target_metrics.json`` file: task name to its non-empty metric list."""

    def check(self, path: str | Path, root: str) -> None:
        """Check the task keys and lists against ``root``'s namespace.

        Args:
            path: Source file, for error messages.
            root: Package whose ``<root>/`` prefix every task key needs.

        Raises:
            ValueError: The mapping is empty, a key is outside ``root``'s
                namespace, or a task maps to an empty list.
        """
        if not self.root:
            raise ValueError(f"{path} must contain a non-empty JSON object")
        for task, metrics in self.root.items():
            if not task.startswith(f"{root}/") or task == f"{root}/":
                raise ValueError(f"{path}: task key {task!r} must start with {root}/")
            if not metrics:
                raise ValueError(f"{path}: {task} must map to a non-empty list")

    def wire(self) -> TargetMetrics:
        """The mapping in inspect-brief's JSON format, metric order kept."""
        return {
            task: [metric.model_dump() for metric in metrics]
            for task, metrics in self.root.items()
        }


def target_metrics_files(root: str = ROOT_PACKAGE) -> dict[str, Path]:
    """Each subpackage's ``target_metrics.json``, keyed by package, sorted.

    Packages are located without being imported, so reading target metrics
    never pulls in benchmark or legacy-engine dependencies.

    Args:
        root: Package whose direct subpackages are searched.

    Returns:
        Package name to its ``target_metrics.json`` path, for packages that
        ship one.
    """
    found: dict[str, Path] = {}
    for package in subpackages(root):
        for directory in package_dirs(package):
            path = directory / TARGET_METRICS_FILE
            if path.is_file():
                found[package] = path
                break
    return found


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen[key] = value
    return seen


def read_target_metrics_file(
    path: str | Path, *, root: str = ROOT_PACKAGE
) -> TargetMetrics:
    """Read and validate one ``target_metrics.json``.

    Args:
        path: The JSON file.
        root: Package whose ``<root>/`` prefix every task key needs.

    Returns:
        The file's mapping in inspect-brief's format, validated by
        ``TargetMetricsFile``: task names in the ``<root>/`` namespace, each
        mapped to a non-empty list of metric objects with exactly ``name``
        (non-empty), ``is_percentage``, ``is_higher_better`` and
        ``is_normalized`` (JSON booleans).

    Raises:
        ValueError: The file is not valid JSON, repeats a key, or does not
            match the schema.
    """
    try:
        data = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a non-empty JSON object")
    try:
        parsed = TargetMetricsFile.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"{path}: invalid target metrics: {error}") from error
    parsed.check(path, root)
    return parsed.wire()


def load_target_metrics(root: str = ROOT_PACKAGE) -> TargetMetrics:
    """All packages' target metrics merged into one mapping with sorted keys.

    Args:
        root: Package whose subpackages' files are merged.

    Returns:
        Task name to metric list, in inspect-brief's format, sorted by task.

    Raises:
        ValueError: A file is invalid, or two files define the same task key.
    """
    merged: TargetMetrics = {}
    sources: dict[str, Path] = {}
    for path in target_metrics_files(root).values():
        for task, metrics in read_target_metrics_file(path, root=root).items():
            if task in sources:
                raise ValueError(
                    f"{task} is defined in both {sources[task]} and {path}"
                )
            sources[task] = path
            merged[task] = metrics
    return dict(sorted(merged.items()))


def missing_target_metrics(
    log: "EvalLog | str | Path", targets: TargetMetrics | None = None
) -> list[str]:
    """Target metric names inspect-brief would report as not found in ``log``.

    ``log`` is an ``EvalLog`` or the path of a ``.eval`` file. ``targets``
    defaults to ``load_target_metrics()``. The lookup mirrors inspect-brief's
    (``inspect_brief.core.score_values`` and ``target_score_values``):

    * the target list is selected by ``log.eval.task``, the registered task
      name (``llm_behavior_eval/<task>``) when the task ran by name or from its
      ``@task`` function;
    * a metric is found when its name is a **key** of some
      ``log.results.scores[i].metrics`` mapping. ``EvalMetric.name`` is not
      used: for a single-value metric it is the registry-qualified name
      (``llm_behavior_eval/<metric>``), so comparing target names with it
      checks the wrong field.

    Inspect keys a single-value metric by its unqualified registry name and
    each entry of a mapping-valued metric by the mapping key; it drops mapping
    entries whose value is ``None``.

    Args:
        log: An ``EvalLog`` or the path of a ``.eval`` file.
        targets: Target metrics to check against; defaults to
            ``load_target_metrics()``.

    Returns:
        The missing names in target order, so a passing check is ``== []``.

    Raises:
        ValueError: inspect-brief would not look the names up at all: the log
            did not succeed, or its task has no target metrics (then
            inspect-brief reports every metric instead of the targets).
    """
    if not isinstance(log, (str, Path)):
        eval_log = log
    else:
        from inspect_ai.log import read_eval_log

        eval_log = read_eval_log(str(log))
    if targets is None:
        targets = load_target_metrics()
    task = eval_log.eval.task
    if eval_log.status != "success":
        raise ValueError(f"{task} log status is {eval_log.status!r}, not 'success'")
    if task not in targets:
        raise ValueError(
            f"Task {task!r} has no target metrics; inspect-brief selects them by "
            f"log.eval.task. Known tasks: {sorted(targets)}"
        )
    scores = eval_log.results.scores if eval_log.results else []
    found = {name for score in scores for name in score.metrics}
    return [m["name"] for m in targets[task] if m["name"] not in found]


def write_target_metrics(path: str | Path, root: str = ROOT_PACKAGE) -> Path:
    """Write the merged mapping as JSON to ``path`` and return the path.

    Args:
        path: Output file; missing parent directories are created.
        root: Package whose subpackages' files are merged.

    Returns:
        ``path`` as a ``Path``.

    Raises:
        ValueError: As ``load_target_metrics``.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(load_target_metrics(root), indent=2) + "\n", encoding="utf-8"
    )
    return output


app = typer.Typer(add_completion=False)


@app.command()
def main(
    output: Annotated[
        Path | None,
        typer.Option(help="Write the merged JSON to this file and print its path"),
    ] = None,
) -> None:
    """Print or write the merged benchmark target metrics."""
    if output is None:
        typer.echo(json.dumps(load_target_metrics(), indent=2))
    else:
        typer.echo(str(write_target_metrics(output)))


if __name__ == "__main__":
    app()

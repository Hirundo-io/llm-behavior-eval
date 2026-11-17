"""Report metrics each benchmark declares, in inspect-brief's format.

A benchmark package ships ``target_metrics.json``::

    {"llm_behavior_eval/<task>": [
        {"name": "<metric>", "is_percentage": true,
         "is_higher_better": false, "is_normalized": true}]}

Benchmark tests assert ``missing_target_metrics(log) == []``.
"""

import json
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inspect_ai.log import EvalLog

TARGET_METRICS_FILE = "target_metrics.json"
METRIC_KEYS = frozenset({"name", "is_percentage", "is_higher_better", "is_normalized"})

TargetMetrics = dict[str, list[dict[str, Any]]]


def load_target_metrics(root: str = "llm_behavior_eval") -> TargetMetrics:
    """Every subpackage's ``target_metrics.json``, merged and sorted by task.

    Raises:
        ValueError: A file is malformed, or two files declare the same task.
    """
    merged: TargetMetrics = {}
    for package in sorted(files(root).iterdir(), key=lambda entry: entry.name):
        path = package / TARGET_METRICS_FILE
        if not package.is_dir() or not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for task, metrics in data.items():
            if not task.startswith(f"{root}/") or task in merged:
                raise ValueError(f"{path}: task key {task!r} is invalid or repeated")
            if not metrics or any(set(m) != METRIC_KEYS for m in metrics):
                raise ValueError(
                    f"{path}: {task} needs metrics with {sorted(METRIC_KEYS)}"
                )
            merged[task] = metrics
    return dict(sorted(merged.items()))


def missing_target_metrics(
    log: "EvalLog | str | Path", targets: TargetMetrics | None = None
) -> list[str]:
    """Target metric names inspect-brief would not find in ``log``.

    inspect-brief selects targets by ``log.eval.task`` and finds a metric when
    its name is a key of some ``log.results.scores[i].metrics``.

    Raises:
        ValueError: The log did not succeed, or its task declares no targets.
    """
    if isinstance(log, (str, Path)):
        from inspect_ai.log import read_eval_log

        log = read_eval_log(str(log))
    targets = load_target_metrics() if targets is None else targets
    task = log.eval.task
    if log.status != "success":
        raise ValueError(f"{task} log status is {log.status!r}, not 'success'")
    if task not in targets:
        raise ValueError(
            f"Task {task!r} has no target metrics; known: {sorted(targets)}"
        )
    found = {
        name
        for score in (log.results.scores if log.results else [])
        for name in score.metrics
    }
    return [m["name"] for m in targets[task] if m["name"] not in found]

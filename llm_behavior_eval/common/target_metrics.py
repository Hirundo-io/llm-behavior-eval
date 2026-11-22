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
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from inspect_ai.log import EvalLog

TARGET_METRICS_FILE = "target_metrics.json"
METRIC_KEYS = frozenset({"name", "is_percentage", "is_higher_better", "is_normalized"})

TargetMetrics = dict[str, list[dict[str, Any]]]


def check_target_metrics(data: object, path: object, root: str) -> TargetMetrics:
    """``data`` as target metrics, or a ``ValueError`` naming the bad field.

    The file must be a JSON object mapping ``<root>/<task>`` keys to non-empty
    lists of metric objects, each with exactly ``METRIC_KEYS``: a non-empty
    string ``name`` and boolean flags.

    Raises:
        ValueError: Any part of ``data`` has the wrong shape or type.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
    for task, metrics in data.items():
        if not task.startswith(f"{root}/"):
            raise ValueError(f"{path}: task key {task!r} must start with {root}/")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError(f"{path}: {task} needs a non-empty list of metrics")
        for index, metric in enumerate(metrics):
            where = f"{path}: {task}[{index}]"
            if not isinstance(metric, dict) or set(metric) != METRIC_KEYS:
                raise ValueError(
                    f"{where} needs exactly the keys {sorted(METRIC_KEYS)}"
                )
            name = metric["name"]
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"{where}.name must be a non-empty string, got {name!r}"
                )
            for flag in sorted(METRIC_KEYS - {"name"}):
                if not isinstance(metric[flag], bool):
                    raise ValueError(
                        f"{where}.{flag} must be true or false, got {metric[flag]!r}"
                    )
    return cast("TargetMetrics", data)


def load_target_metrics(root: str = "llm_behavior_eval") -> TargetMetrics:
    """Every subpackage's ``target_metrics.json``, merged and sorted by task.

    Raises:
        ValueError: A file is malformed (see ``check_target_metrics``), or two
            files declare the same task.
    """
    merged: TargetMetrics = {}
    for package in sorted(files(root).iterdir(), key=lambda entry: entry.name):
        path = package / TARGET_METRICS_FILE
        if not package.is_dir() or not path.is_file():
            continue
        data = check_target_metrics(
            json.loads(path.read_text(encoding="utf-8")), path, root
        )
        for task, metrics in data.items():
            if task in merged:
                raise ValueError(f"{path}: task key {task!r} is repeated")
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

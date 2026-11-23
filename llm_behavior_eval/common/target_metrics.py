"""Report metrics each benchmark declares, in inspect-brief's format.

A benchmark package ships ``target_metrics.json`` (found in any package at
any depth below ``llm_behavior_eval``, in every directory it spans)::

    {"llm_behavior_eval/<task>": [
        {"name": "<metric>", "is_percentage": true,
         "is_higher_better": false, "is_normalized": true}]}

Benchmark tests assert ``missing_target_metrics(log) == []``.
"""

import json
from importlib import import_module
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

    Args:
        data: A parsed ``target_metrics.json``.
        path: The file, for error messages.
        root: The package prefix every task key needs.

    Returns:
        ``data``, typed.

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
    # Safe: the loop above checked every key and value type that TargetMetrics
    # declares, which the type checker cannot infer from the isinstance checks.
    return cast("TargetMetrics", data)


def package_directories(root: str) -> list[Path]:
    """Every directory that contributes to package ``root``, without duplicates.

    ``llm_behavior_eval`` is shared by several distributions: one directory
    when they are installed together or editable from this repository, and
    possibly several for a PEP 420 namespace (without the legacy distribution).

    Args:
        root: The package's dotted name (imported, which for
            ``llm_behavior_eval`` loads only the lazy legacy ``__init__``).

    Returns:
        Each ``__path__`` entry, resolved, in order.

    Raises:
        ModuleNotFoundError: ``root`` cannot be imported.
    """
    found: dict[Path, None] = {}
    for entry in import_module(root).__path__:
        found.setdefault(Path(entry).resolve(), None)
    return list(found)


def target_metrics_files(directory: Path, in_package: bool = False) -> list[Path]:
    """``target_metrics.json`` files in ``directory``'s packages, at any depth.

    Below a regular package (one with ``__init__.py``) only its subpackages
    are searched, so a ``build/`` copy or ``*.egg-info`` that an in-place
    build leaves in a benchmark's directory is skipped. Above one, directories
    without ``__init__.py`` are namespace levels (a group) and are searched.

    Args:
        directory: A directory of the root package (``package_directories``).
        in_package: Whether ``directory`` is inside a regular package below
            the root (set by the recursion).

    Returns:
        The files found, sorted by path within each directory.
    """
    found = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name == TARGET_METRICS_FILE:
            found.append(path)
        elif path.is_dir() and path.name.isidentifier():
            is_package = (path / "__init__.py").is_file()
            if is_package or not in_package:
                found.extend(target_metrics_files(path, in_package or is_package))
    return found


def load_target_metrics(root: str = "llm_behavior_eval") -> TargetMetrics:
    """Every ``target_metrics.json`` below ``root``, merged and sorted by task.

    Searches each of ``root``'s directories (``package_directories``) at any
    depth (``target_metrics_files``), so a benchmark in a group
    (``<root>/<group>/<benchmark>``) is found like one directly under ``root``.

    Args:
        root: The package whose benchmarks to search.

    Returns:
        Every task's target metrics, sorted by task.

    Raises:
        ValueError: A file is malformed (see ``check_target_metrics``), or two
            files declare the same task.
    """
    paths = {
        path.resolve()
        for directory in package_directories(root)
        for path in target_metrics_files(directory)
    }
    merged: TargetMetrics = {}
    for path in sorted(paths):
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

    Args:
        log: An eval log, or the path of one.
        targets: The target metrics; ``load_target_metrics()`` by default.

    Returns:
        The task's target metric names the log lacks, in declared order.

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

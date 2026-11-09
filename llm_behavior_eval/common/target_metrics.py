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
"""

import argparse
import json
from pathlib import Path
from typing import Any

from llm_behavior_eval.common.packages import ROOT_PACKAGE, package_dirs, subpackages

TARGET_METRICS_FILE = "target_metrics.json"
_METRIC_KEYS = {"name", "is_percentage", "is_higher_better", "is_normalized"}

TargetMetrics = dict[str, list[dict[str, Any]]]


def target_metrics_files(root: str = ROOT_PACKAGE) -> dict[str, Path]:
    """Each subpackage's ``target_metrics.json``, keyed by package, sorted.

    Packages are located without being imported, so reading target metrics
    never pulls in benchmark or legacy-engine dependencies.
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

    Keys must be task names in the ``<root>/`` namespace; each value is a list
    of metric objects with exactly the keys inspect-brief accepts.
    """
    try:
        data = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON object")
    for task, metrics in data.items():
        if not task.startswith(f"{root}/") or task == f"{root}/":
            raise ValueError(f"{path}: task key {task!r} must start with {root}/")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError(f"{path}: {task} must map to a non-empty list")
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict) or set(metric) != _METRIC_KEYS:
                raise ValueError(
                    f"{path}: {task} metric {index} must have exactly the keys "
                    f"{sorted(_METRIC_KEYS)}"
                )
            if not isinstance(metric["name"], str) or not metric["name"]:
                raise ValueError(f"{path}: {task} metric {index} needs a name")
            if any(type(metric[key]) is not bool for key in _METRIC_KEYS - {"name"}):
                raise ValueError(
                    f"{path}: {task} metric {index} flags must be JSON booleans"
                )
    return data


def load_target_metrics(root: str = ROOT_PACKAGE) -> TargetMetrics:
    """All packages' target metrics merged into one mapping with sorted keys.

    Raises ``ValueError`` when a file is invalid or two files define the same
    task key.
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


def write_target_metrics(path: str | Path, root: str = ROOT_PACKAGE) -> Path:
    """Write the merged mapping as JSON to ``path`` and return the path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(load_target_metrics(root), indent=2) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Print or write the merged benchmark target metrics."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the merged JSON to this file and print its path",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        print(json.dumps(load_target_metrics(), indent=2))
    else:
        print(write_target_metrics(args.output))


if __name__ == "__main__":
    main()

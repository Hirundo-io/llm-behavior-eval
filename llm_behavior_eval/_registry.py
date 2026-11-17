"""Inspect's package entry point: import every benchmark's ``task`` module.

A benchmark is a regular subpackage of ``llm_behavior_eval`` with a ``task``
module. Importing it runs its ``@task`` decorators, which register the tasks
as ``llm_behavior_eval/<name>``. Packages are found without importing the
others (the legacy engine stays unloaded), in sorted order.
"""

import pkgutil
from importlib import import_module
from pathlib import Path

ROOT = "llm_behavior_eval"
TASK_MODULE = "task"


def benchmark_packages(root: str = ROOT) -> list[str]:
    """Sorted names of ``root``'s subpackages that have a ``task`` module."""
    package_path = list(import_module(root).__path__)
    found = []
    for info in pkgutil.iter_modules(package_path):
        if not info.ispkg:
            continue
        directories = [str(Path(path) / info.name) for path in package_path]
        if any(
            module.name == TASK_MODULE and not module.ispkg
            for module in pkgutil.iter_modules(directories)
        ):
            found.append(f"{root}.{info.name}")
    return sorted(found)


for _package in benchmark_packages():
    import_module(f"{_package}.{TASK_MODULE}")

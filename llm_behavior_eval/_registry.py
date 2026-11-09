"""Inspect's package entry point: register every benchmark's tasks.

Every regular subpackage of ``llm_behavior_eval`` with a ``task`` module is a
benchmark (see ``common.packages``). Importing that module runs its ``@task``
decorators, which register the tasks under the ``llm_behavior_eval/``
namespace. Modules are imported in sorted package order, so registration is
deterministic, and adding a benchmark package needs no edit here.

Each registered task is also bound as an attribute of this module and listed in
``__all__`` (sorted unqualified task names). Import errors propagate, exactly as
explicit imports would, and two benchmarks registering the same task name fail
loudly instead of silently replacing each other.
"""

from importlib import import_module
from typing import Any

from inspect_ai._util.registry import is_registry_object, registry_unqualified_name

from .common.packages import TASK_MODULE, benchmark_packages


def _register_tasks() -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    owners: dict[str, str] = {}
    for package in benchmark_packages(__package__ or "llm_behavior_eval"):
        module = import_module(f"{package}.{TASK_MODULE}")
        for value in vars(module).values():
            if not is_registry_object(value, type="task"):
                continue
            # Skip tasks a task module merely imports from elsewhere.
            if getattr(value, "__module__", None) != module.__name__:
                continue
            name = registry_unqualified_name(value)
            if name in owners and owners[name] != module.__name__:
                raise RuntimeError(
                    f"Task name {name!r} is registered by both "
                    f"{owners[name]} and {module.__name__}"
                )
            owners[name] = module.__name__
            tasks[name] = value
    return dict(sorted(tasks.items()))


_TASKS = _register_tasks()
globals().update(_TASKS)
__all__ = list(_TASKS)  # pyrefly: ignore[unresolvable-dunder-all]

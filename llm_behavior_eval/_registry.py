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

Inspect keeps one registry per object type, keyed by ``<package>/<name>``, and
the last registration under a name wins. A metric, scorer or solver that two
modules define under the same name would therefore make ``inspect score`` (and
anything else that rebuilds objects from a log) run the wrong benchmark's code.
After the task modules are imported, every registry object defined in a
``llm_behavior_eval`` module is checked, and a name defined by two modules
raises ``RuntimeError``. Give benchmark metrics, scorers and solvers a
package-specific name (``<benchmark>_metrics``, ``<benchmark>_scorer``, ...).
"""

import sys
from importlib import import_module
from typing import Any

from inspect_ai._util.registry import (
    is_registry_object,
    registry_info,
    registry_unqualified_name,
)

from .common.packages import TASK_MODULE, benchmark_packages

_ROOT = __package__ or "llm_behavior_eval"


def _register_tasks() -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    owners: dict[str, str] = {}
    for package in benchmark_packages(_ROOT):
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


def _defining_module(value: object) -> str | None:
    """The module that defines a registered object.

    Decorators built with ``functools.wraps`` (``@task``, ``@scorer``,
    ``@solver``) keep the decorated function's ``__module__``; ``@metric``
    returns an Inspect-internal wrapper that holds the decorated function in
    its closure. Follow ``__wrapped__`` and closure cells to the first callable
    defined outside ``inspect_ai``.
    """
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        module = getattr(current, "__module__", None)
        if isinstance(module, str) and module.split(".")[0] != "inspect_ai":
            return module
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
        for cell in getattr(current, "__closure__", None) or ():
            try:
                contents = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            if callable(contents):
                pending.append(contents)
    return None


def registry_definitions(root: str = _ROOT) -> dict[tuple[str, str], list[str]]:
    """Inspect registry objects defined in ``root``'s imported modules.

    Maps ``(type, registry name)`` to the sorted names of the modules whose
    top level defines an object registered under it. Objects a module merely
    imports are attributed to the module that defines them. Only modules that
    are already imported are inspected.
    """
    definitions: dict[tuple[str, str], set[str]] = {}
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(f"{root}."):
            continue
        for value in list(vars(module).values()):
            if not is_registry_object(value):
                continue
            if _defining_module(value) != module_name:
                continue
            info = registry_info(value)
            definitions.setdefault((info.type, info.name), set()).add(module_name)
    return {key: sorted(modules) for key, modules in sorted(definitions.items())}


def check_unique_registry_names(root: str = _ROOT) -> None:
    """Raise ``RuntimeError`` if two modules define the same registry name."""
    duplicates = [
        f"{kind} {name!r} is defined by {', '.join(modules)}"
        for (kind, name), modules in registry_definitions(root).items()
        if len(modules) > 1
    ]
    if duplicates:
        raise RuntimeError(
            "Inspect registry names must be unique; the last definition would "
            "silently replace the others: " + "; ".join(duplicates)
        )


_TASKS = _register_tasks()
# Hooks are not benchmarks, so discovery does not import them. Register the
# opt-in MLflow reporting hook explicitly; it imports MLflow only when enabled.
import_module(f"{_ROOT}.hooks.mlflow")
check_unique_registry_names()
globals().update(_TASKS)
__all__ = list(_TASKS)  # pyrefly: ignore[unresolvable-dunder-all]

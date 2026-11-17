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
After the task modules are imported, every registry object bound at the top
level of a ``llm_behavior_eval`` module is compared by identity (imports bind
the same object), and two different objects under one ``(type, name)`` raise
``RuntimeError``. Give benchmark metrics, scorers and solvers a
package-specific name (``<benchmark>_metrics``, ``<benchmark>_scorer``, ...).
"""

import sys
from importlib import import_module
from typing import Any

from inspect_ai.util import RegistryInfo, registry_info

from .common.packages import TASK_MODULE, benchmark_packages

_ROOT = __package__ or "llm_behavior_eval"


def _registry_entry(value: Any) -> RegistryInfo | None:
    """``value``'s Inspect registry info, or ``None`` if it is not registered.

    Uses only Inspect's public ``registry_info``, which raises ``ValueError``
    for an unregistered object.
    """
    try:
        return registry_info(value)
    except ValueError:
        return None


def _register_tasks() -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    owners: dict[str, str] = {}
    for package in benchmark_packages(_ROOT):
        module = import_module(f"{package}.{TASK_MODULE}")
        for value in vars(module).values():
            info = _registry_entry(value)
            if info is None or info.type != "task":
                continue
            # Skip tasks a task module merely imports from elsewhere.
            if getattr(value, "__module__", None) != module.__name__:
                continue
            # Registry names are "<package>/<name>"; tasks are bound by <name>.
            name = info.name.split("/", 1)[-1]
            if name in owners and owners[name] != module.__name__:
                raise RuntimeError(
                    f"Task name {name!r} is registered by both "
                    f"{owners[name]} and {module.__name__}"
                )
            owners[name] = module.__name__
            tasks[name] = value
    return dict(sorted(tasks.items()))


def registry_objects(root: str = _ROOT) -> dict[tuple[str, str], list[list[str]]]:
    """Distinct Inspect registry objects bound at the top level of ``root``'s modules.

    Maps ``(type, registry name)`` to one entry per distinct object (by
    identity) registered under that name, each entry the sorted names of the
    modules that bind it. A module that imports an object binds the same
    object, so imports never add entries; two definitions of one name are two
    entries. Only modules that are already imported are inspected, and only
    Inspect's public ``registry_info`` is used.

    Args:
        root: Package whose imported modules are inspected.

    Returns:
        ``(type, registry name)`` to the binding modules of each distinct
        object, sorted.
    """
    found: dict[tuple[str, str], dict[int, set[str]]] = {}
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(f"{root}."):
            continue
        for value in list(vars(module).values()):
            info = _registry_entry(value)
            if info is None:
                continue
            objects = found.setdefault((info.type, info.name), {})
            objects.setdefault(id(value), set()).add(module_name)
    return {
        key: sorted(sorted(modules) for modules in objects.values())
        for key, objects in sorted(found.items())
    }


def check_unique_registry_names(root: str = _ROOT) -> None:
    """Fail when two different objects are registered under one name.

    Registry objects bound at a module's top level must be the decorated
    factories (``@task``, ``@scorer``, ``@solver``, ``@metric``); bind created
    instances, such as ``field_rate()``, inside functions or containers.

    Args:
        root: Package whose imported modules are inspected.

    Raises:
        RuntimeError: Some ``(type, name)`` names more than one object.
    """
    duplicates = [
        f"{kind} {name!r} names {len(bindings)} different objects, bound in "
        + " and ".join(f"[{', '.join(modules)}]" for modules in bindings)
        for (kind, name), bindings in registry_objects(root).items()
        if len(bindings) > 1
    ]
    if duplicates:
        raise RuntimeError(
            "Inspect registry names must be unique; the last definition would "
            "silently replace the others: "
            + "; ".join(duplicates)
            + ". Give each benchmark's metrics, scorers and solvers a "
            "package-specific name, and bind only decorated factories (not "
            "created instances) at module level."
        )


_TASKS = _register_tasks()
check_unique_registry_names()
# Tasks are discovered at import time, so they and __all__ are bound
# dynamically. __all__ is declared for type checkers, which cannot resolve a
# computed __all__ (pyrefly: unresolvable-dunder-all).
__all__: list[str]
globals().update(_TASKS, __all__=list(_TASKS))

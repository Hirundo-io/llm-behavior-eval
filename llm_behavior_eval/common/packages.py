"""Find benchmark packages and the files they own, without importing them.

A benchmark is a regular subpackage of ``llm_behavior_eval`` (it has an
``__init__.py``) that contains a ``task`` module. Discovery reads the import
system's view of the package directory (``pkgutil``/``importlib``), so it
behaves the same from a source checkout, an editable install and an installed
wheel, and it returns names sorted so every caller sees the same order.
"""

import importlib
import importlib.util
import pkgutil
from pathlib import Path

ROOT_PACKAGE = "llm_behavior_eval"
TASK_MODULE = "task"


def subpackages(root: str = ROOT_PACKAGE) -> list[str]:
    """Sorted fully qualified names of the direct regular subpackages of ``root``.

    Only ``root`` itself is imported, and its live ``__path__`` is searched.
    Namespace directories (no ``__init__.py``) are skipped, as setuptools'
    package discovery skips them when building the wheel.
    """
    path = getattr(importlib.import_module(root), "__path__", None)
    if path is None:
        raise ModuleNotFoundError(f"{root} is not a package")
    names = {
        f"{root}.{info.name}" for info in pkgutil.iter_modules(list(path)) if info.ispkg
    }
    return sorted(names)


def package_dirs(package: str) -> list[Path]:
    """Directories the import system searches for ``package``'s submodules.

    Resolving the spec imports the parent package only, never ``package``.
    """
    spec = importlib.util.find_spec(package)
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError(f"{package} is not an importable package")
    return [Path(location) for location in spec.submodule_search_locations]


def has_module(package: str, module: str) -> bool:
    """Whether ``package`` contains a plain (non-package) submodule ``module``."""
    return any(
        info.name == module and not info.ispkg
        for info in pkgutil.iter_modules([str(d) for d in package_dirs(package)])
    )


def benchmark_packages(root: str = ROOT_PACKAGE) -> list[str]:
    """Sorted names of ``root``'s subpackages that have a ``task`` module."""
    return [name for name in subpackages(root) if has_module(name, TASK_MODULE)]

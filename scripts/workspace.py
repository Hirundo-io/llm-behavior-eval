"""The repository's distributions, read from the uv workspace configuration.

The root ``pyproject.toml`` is the ``llm-behavior-eval`` distribution (the
legacy engine) and the workspace root; ``[tool.uv.workspace] members`` lists
the native distributions under ``packages/`` (see "Packaging" in
``INSPECT_MIGRATION.md``). Tests and the scripts in this directory read the
layout from here rather than from hard-coded lists, so adding a benchmark
needs no edit to them.
"""

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import tomllib

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "llm_behavior_eval"
"""The import package every distribution contributes to."""
ENTRY_POINT_GROUP = "inspect_ai"
EXAMPLE = ROOT / "tests" / "example_benchmark"
"""The synthetic benchmark distribution: never a workspace member."""

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize(name: str) -> str:
    """A distribution name in PEP 503 normal form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    """The normalized distribution name a PEP 508 requirement string names."""
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        raise ValueError(f"not a requirement: {requirement!r}")
    return normalize(match.group(1))


def load_pyproject(directory: Path) -> dict[str, Any]:
    with (directory / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


@dataclass(frozen=True)
class Distribution:
    """One distribution of the repository: its directory and ``pyproject.toml``."""

    path: Path

    @cached_property
    def pyproject(self) -> dict[str, Any]:
        return load_pyproject(self.path)

    @property
    def project(self) -> dict[str, Any]:
        return self.pyproject["project"]

    @property
    def name(self) -> str:
        return normalize(self.project["name"])

    @property
    def version(self) -> str:
        return self.project["version"]

    @property
    def relative(self) -> str:
        """The directory relative to the repository root, in POSIX form."""
        return self.path.relative_to(ROOT).as_posix() or "."

    @property
    def src(self) -> Path:
        return self.path / "src"

    @property
    def entry_points(self) -> dict[str, str]:
        """This distribution's ``inspect_ai`` entry points (name to value)."""
        return self.project.get("entry-points", {}).get(ENTRY_POINT_GROUP, {})

    @property
    def is_metapackage(self) -> bool:
        """A group distribution: no modules, only dependencies on benchmarks."""
        setuptools = self.pyproject.get("tool", {}).get("setuptools", {})
        return setuptools.get("packages") == []

    @property
    def is_benchmark(self) -> bool:
        return bool(self.entry_points)

    @property
    def packages(self) -> list[str]:
        """Regular packages (with ``__init__.py``) under ``src/``, as dotted names.

        Directories above them are namespace levels (``llm_behavior_eval``
        and a group) with no ``__init__.py``.
        """
        if not self.src.is_dir():
            return []
        found = []
        for init in sorted(self.src.rglob("__init__.py")):
            parts = init.parent.relative_to(self.src).parts
            if "__pycache__" in parts:
                continue
            outer = ".".join(parts[:-1])
            if not any(outer == f or outer.startswith(f + ".") for f in found):
                found.append(".".join(parts))
        return found

    @property
    def unpublished(self) -> str | None:
        """Why the distribution is never published to PyPI, if it is not.

        ``[tool.llm-behavior-eval] unpublished = "<reason>"``, for example a
        direct Git dependency, which PyPI rejects. No other distribution may
        depend on it (see ``packages/security/README.md``).
        """
        return (
            self.pyproject.get("tool", {})
            .get("llm-behavior-eval", {})
            .get("unpublished")
        )

    @property
    def requirements(self) -> list[str]:
        return list(self.project.get("dependencies", []))

    @property
    def optional_requirements(self) -> dict[str, list[str]]:
        return dict(self.project.get("optional-dependencies", {}))


def root() -> Distribution:
    """The ``llm-behavior-eval`` distribution (legacy engine, workspace root)."""
    return Distribution(ROOT)


def members() -> list[Distribution]:
    """The workspace members under ``packages/``, in the order listed."""
    listed = root().pyproject["tool"]["uv"]["workspace"]["members"]
    return [Distribution(ROOT / member) for member in listed]


def root_is_editable() -> bool:
    """Whether ``llm-behavior-eval`` is installed editable here (``uv sync``)."""
    import json
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        direct_url = distribution("llm-behavior-eval").read_text("direct_url.json")
    except PackageNotFoundError:
        return False
    return bool(direct_url) and json.loads(direct_url).get("dir_info", {}).get(
        "editable", False
    )


def benchmark_test_files(distribution: Distribution) -> list[Path]:
    """The test modules that cover a distribution, by naming convention.

    ``tests/test_inspect_<leaf>.py`` for each package whose last dotted
    component is ``<leaf>`` (``common`` for the shared code); none for a group
    distribution.
    """
    leaves = [package.rsplit(".", 1)[-1] for package in distribution.packages]
    return [ROOT / "tests" / f"test_inspect_{leaf}.py" for leaf in leaves]


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--unpublished"]:
        # Names of the members never published (deploy_to_pypi.yaml skips them).
        for member in members():
            if member.unpublished:
                print(member.name)
    else:
        # The workspace members' directories, one per line (.devcontainer/setup.sh).
        for member in members():
            print(member.path)

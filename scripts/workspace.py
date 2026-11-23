"""The repository's distributions, read from the uv workspace configuration.

The root ``pyproject.toml`` is the ``llm-behavior-eval`` distribution (the
legacy engine) and the workspace root; ``[tool.uv.workspace] members`` lists
the native distributions, each a directory inside ``llm_behavior_eval/`` that
holds its package and its ``pyproject.toml`` (see "Packaging" in
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
    """A distribution name in PEP 503 normal form.

    Args:
        name: A distribution name as written (``llm_behavior_eval.Common``).

    Returns:
        Lowercase, with runs of ``-``, ``_`` and ``.`` as one ``-``.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    """The normalized distribution name a PEP 508 requirement string names.

    Args:
        requirement: A requirement (``"inspect-ai>=0.3.289,<0.4"``).

    Returns:
        Its distribution name, normalized (``normalize``).

    Raises:
        ValueError: The string does not start with a distribution name.
    """
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        raise ValueError(f"not a requirement: {requirement!r}")
    return normalize(match.group(1))


def load_pyproject(directory: Path) -> dict[str, Any]:
    """Parse a project's ``pyproject.toml``.

    Args:
        directory: The project directory.

    Returns:
        The parsed TOML document.

    Raises:
        FileNotFoundError: ``directory`` has no ``pyproject.toml``.
        tomllib.TOMLDecodeError: The file is not valid TOML.
    """
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
    def is_root(self) -> bool:
        """The ``llm-behavior-eval`` distribution: the workspace root."""
        return self.path == ROOT

    @property
    def setuptools(self) -> dict[str, Any]:
        return self.pyproject.get("tool", {}).get("setuptools", {})

    @property
    def entry_points(self) -> dict[str, str]:
        """This distribution's ``inspect_ai`` entry points (name to value)."""
        return self.project.get("entry-points", {}).get(ENTRY_POINT_GROUP, {})

    @property
    def is_metapackage(self) -> bool:
        """A group distribution: no modules, only dependencies on benchmarks."""
        return self.setuptools.get("packages") == []

    @property
    def is_benchmark(self) -> bool:
        return bool(self.entry_points)

    @property
    def packages(self) -> list[str]:
        """The dotted packages a native distribution lists in ``[tool.setuptools]``.

        The first is the distribution's own package (``llm_behavior_eval.common``
        or ``llm_behavior_eval.[<group>.]<benchmark>``), any others its
        subpackages. Empty for a group distribution and for the root, which
        finds its packages instead.
        """
        packages = self.setuptools.get("packages", [])
        return list(packages) if isinstance(packages, list) else []

    @property
    def top_packages(self) -> list[str]:
        """``packages`` without the subpackages of another listed package."""
        listed = self.packages
        return [
            package
            for package in listed
            if not any(package.startswith(f"{other}.") for other in listed)
        ]

    def package_directory(self, package: str) -> Path:
        """Where ``package``'s modules live, from ``[tool.setuptools] package-dir``.

        Args:
            package: One of ``packages`` (dotted).

        Returns:
            The directory, from the longest ``package-dir`` prefix that covers
            ``package``.

        Raises:
            ValueError: No ``package-dir`` entry covers ``package``.
        """
        mapping = self.setuptools.get("package-dir", {})
        for prefix in sorted(mapping, key=len, reverse=True):
            if package == prefix or package.startswith(f"{prefix}."):
                rest = package[len(prefix) :].lstrip(".").split(".")
                return self.path.joinpath(mapping[prefix], *filter(None, rest))
        raise ValueError(f"{self.relative}: no package-dir entry for {package}")

    @property
    def unpublished(self) -> str | None:
        """Why the distribution is never published to PyPI, if it is not.

        ``[tool.llm-behavior-eval] unpublished = "<reason>"``, for example a
        direct Git dependency, which PyPI rejects. No other distribution may
        depend on it (see ``llm_behavior_eval/security/README.md``).
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
    """The ``llm-behavior-eval`` distribution (legacy engine, workspace root).

    Returns:
        The distribution at the repository root.
    """
    return Distribution(ROOT)


def members() -> list[Distribution]:
    """The workspace members (every native distribution), in the order listed.

    The root is not a member, although every member's directory is inside it;
    a group distribution's directory contains its benchmarks' directories.

    Returns:
        One distribution per ``[tool.uv.workspace] members`` entry.
    """
    listed = root().pyproject["tool"]["uv"]["workspace"]["members"]
    return [Distribution(ROOT / member) for member in listed]


def benchmark_dependencies(
    group: Distribution, workspace: list[Distribution]
) -> list[Distribution]:
    """The benchmarks in ``workspace`` that ``group`` depends on, in its order.

    Other requirements (``llm-behavior-eval-common``, third-party packages)
    are left out.
    """
    by_name = {member.name: member for member in workspace}
    return [
        by_name[name]
        for name in map(requirement_name, group.requirements)
        if name in by_name and by_name[name].is_benchmark
    ]


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

    ``tests/test_inspect_<leaf>.py`` for each top-level package whose last
    dotted component is ``<leaf>`` (``common`` for the shared code); none for
    a group distribution.

    Args:
        distribution: A workspace member.

    Returns:
        The expected test files, whether or not they exist.
    """
    leaves = [package.rsplit(".", 1)[-1] for package in distribution.top_packages]
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

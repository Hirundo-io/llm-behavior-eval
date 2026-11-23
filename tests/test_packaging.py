"""The repository's distributions follow "Packaging" in INSPECT_MIGRATION.md.

The layout is read from the workspace configuration (``scripts/workspace.py``),
so a new benchmark distribution is checked here without editing this file.
The synthetic benchmark ``tests/example_benchmark`` (the template a new
benchmark copies) is checked like a workspace member.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from check_artifact_licenses import check_artifact, check_directory
from workspace import (
    EXAMPLE,
    NAMESPACE,
    ROOT,
    Distribution,
    benchmark_test_files,
    members,
    normalize,
    requirement_name,
    root,
    root_is_editable,
)

NATIVE = [*members(), Distribution(EXAMPLE)]
HEAVY = {"torch", "transformers", "vllm", "accelerate", "llm-behavior-eval"}
"""Requirements a native distribution must not declare: the legacy engine's."""


def ids(distribution: Distribution) -> str:
    return distribution.relative


def workspace_requirements(distribution: Distribution) -> list[Requirement]:
    return [
        Requirement(text)
        for text in distribution.requirements
        if requirement_name(text).startswith("llm-behavior-eval-")
    ]


def namespace_levels(package: str) -> list[str]:
    """``llm_behavior_eval`` and any group above a benchmark package."""
    parts = package.split(".")
    return [".".join(parts[:end]) for end in range(1, len(parts))]


@pytest.mark.parametrize("distribution", NATIVE, ids=ids)
def test_native_distribution_metadata(distribution: Distribution) -> None:
    project = distribution.project
    assert distribution.name.startswith("llm-behavior-eval-")
    assert distribution.path.parent.is_relative_to(ROOT / "packages") or (
        distribution.path == EXAMPLE
    )
    assert distribution.version == root().version
    assert project["requires-python"] == ">=3.11,<3.14"
    assert distribution.pyproject["build-system"] == {
        "requires": ["setuptools>=77"],
        "build-backend": "setuptools.build_meta",
    }
    assert (distribution.path / project["readme"]).is_file()
    assert not any(
        classifier.startswith("License ::")
        for classifier in project.get("classifiers", [])
    )


@pytest.mark.parametrize("distribution", NATIVE, ids=ids)
def test_native_distribution_licenses(distribution: Distribution) -> None:
    project = distribution.project
    assert isinstance(project["license"], str)
    assert "LICENSE" in project["license-files"]
    for name in project["license-files"]:
        assert (distribution.path / name).is_file(), name
    if project["license"] == "MIT":
        license_text = (distribution.path / "LICENSE").read_bytes()
        assert license_text == (ROOT / "LICENSE").read_bytes()


@pytest.mark.parametrize("distribution", NATIVE, ids=ids)
def test_native_distribution_dependencies(distribution: Distribution) -> None:
    every = distribution.requirements + [
        text for texts in distribution.optional_requirements.values() for text in texts
    ]
    assert not {requirement_name(text) for text in every} & HEAVY
    version = Version(root().version)
    cap = f"<{version.major}.{version.minor + 1}"
    for requirement in workspace_requirements(distribution):
        # A workspace dependency admits the shared version, from a lower bound
        # up to the next minor (the cap policy for a 0.x package).
        assert requirement.specifier.contains(version), requirement
        assert cap in {str(spec) for spec in requirement.specifier}, requirement
        assert any(spec.operator == ">=" for spec in requirement.specifier)


@pytest.mark.parametrize("distribution", NATIVE, ids=ids)
def test_native_distribution_layout(distribution: Distribution) -> None:
    setuptools = distribution.pyproject.get("tool", {}).get("setuptools", {})
    if distribution.is_metapackage:
        assert not distribution.src.exists()
        assert not distribution.entry_points
        return
    packages = distribution.packages
    assert packages
    # Namespace levels ship no __init__.py: several distributions share them.
    assert not (distribution.src / NAMESPACE / "__init__.py").exists()
    levels = sorted({level for p in packages for level in namespace_levels(p)})
    assert all(p.startswith(f"{NAMESPACE}.") for p in packages)
    assert setuptools["packages"]["find"] == {
        "where": ["src"],
        "include": [*levels, *(f"{package}*" for package in packages)],
        "namespaces": True,
    }
    # Nothing outside the declared packages.
    stray = [
        path
        for path in distribution.src.rglob("*.py")
        if not any(
            path.is_relative_to(distribution.src.joinpath(*p.split(".")))
            for p in packages
        )
    ]
    assert stray == []


@pytest.mark.parametrize("distribution", [d for d in NATIVE if d.is_benchmark], ids=ids)
def test_benchmark_distribution(distribution: Distribution) -> None:
    [package] = distribution.packages
    directory = distribution.src.joinpath(*package.split("."))
    # Inspect loads only entry points named after a task's package prefix.
    assert distribution.entry_points == {NAMESPACE: f"{package}.task"}
    assert (directory / "task.py").is_file()
    assert (distribution.path / distribution.project["readme"]) == (
        directory / "README.md"
    )
    targets = json.loads((directory / "target_metrics.json").read_text())
    assert all(task.startswith(f"{NAMESPACE}/") for task in targets)
    assert distribution.pyproject["tool"]["setuptools"]["package-data"] == {
        "*": ["README.md", "parity.yaml", "target_metrics.json"]
    }
    requirements = {requirement_name(text) for text in distribution.requirements}
    assert "llm-behavior-eval-common" in requirements
    if distribution.path != EXAMPLE:
        assert all(path.is_file() for path in benchmark_test_files(distribution))


def test_common_distribution() -> None:
    [common] = [d for d in members() if d.name == "llm-behavior-eval-common"]
    assert common.relative == "packages/common"
    assert common.packages == [f"{NAMESPACE}.common"]
    assert not common.entry_points
    requirements = {requirement_name(text) for text in common.requirements}
    assert {"inspect-ai", "inspect-evals-utils", "datasets"} <= requirements


def test_entry_points_name_distinct_modules() -> None:
    values = [value for d in NATIVE for value in d.entry_points.values()]
    assert len(values) == len(set(values))


def test_workspace_registers_every_member() -> None:
    pyproject = root().pyproject
    names = [member.name for member in members()]
    assert len(names) == len(set(names))
    sources = pyproject["tool"]["uv"]["sources"]
    assert {normalize(name) for name in sources} >= set(names)
    assert all(sources[name] == {"workspace": True} for name in sources)
    dev = {
        requirement_name(text)
        for text in pyproject["dependency-groups"]["dev"]
        if isinstance(text, str)
    }
    assert dev >= set(names)
    search_path = pyproject["tool"]["pyrefly"]["search-path"]
    expected = ["src", *(f"{m.relative}/src" for m in members() if m.src.is_dir())]
    assert search_path == [*expected, "tests", "scripts", "."]
    assert not EXAMPLE.is_relative_to(ROOT / "packages")


def test_root_installs_every_member() -> None:
    # Installing llm-behavior-eval installs every merged native distribution,
    # directly or through a group distribution (or one of its extras), except
    # the unpublished ones, which nothing may depend on.
    by_name = {member.name: member for member in members() if not member.unpublished}
    reached: set[str] = set()
    pending = list(root().requirements)
    while pending:
        name = requirement_name(pending.pop())
        if name in by_name and name not in reached:
            reached.add(name)
            member = by_name[name]
            pending.extend(member.requirements)
            for texts in member.optional_requirements.values():
                pending.extend(texts)
    assert reached == set(by_name)


def test_unpublished_reason(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n[tool.llm-behavior-eval]\nunpublished = "git"\n'
    )
    assert Distribution(tmp_path).unpublished == "git"
    assert make_distribution(tmp_path).unpublished is None


def test_nothing_depends_on_an_unpublished_member() -> None:
    unpublished = {member.name for member in members() if member.unpublished}
    for distribution in [root(), *NATIVE]:
        every = distribution.requirements + [
            text
            for texts in distribution.optional_requirements.values()
            for text in texts
        ]
        assert not {requirement_name(text) for text in every} & unpublished


def test_root_distribution_is_the_legacy_engine() -> None:
    distribution = root()
    assert not distribution.entry_points
    assert distribution.project["license"] == "MIT"
    assert distribution.pyproject["tool"]["setuptools"]["packages"]["find"] == {
        "where": ["src"],
        "include": [NAMESPACE, f"{NAMESPACE}.evaluation_utils*"],
    }
    package = ROOT / "src" / NAMESPACE
    assert (package / "__init__.py").is_file()
    assert sorted(path.name for path in package.iterdir() if path.is_dir()) in (
        ["evaluation_utils"],
        ["__pycache__", "evaluation_utils"],
    )
    assert not (ROOT / NAMESPACE).exists()


def test_bumpver_updates_every_distribution() -> None:
    patterns = root().pyproject["tool"]["bumpver"]["file_patterns"]
    assert patterns["packages/**/pyproject.toml"] == ['^version = "{version}"$']
    assert all(
        path.parent in {m.path for m in members()}
        for path in ROOT.glob("packages/**/pyproject.toml")
    )


# Namespace


@pytest.mark.skipif(not root_is_editable(), reason="needs the editable workspace")
def test_editable_portion_joins_the_namespace(tmp_path: Path) -> None:
    # A workspace member's editable install puts its src/ on sys.path; the
    # legacy distribution's __init__ extends __path__ over it, and Inspect
    # names the task after the editable llm-behavior-eval distribution.
    code = """
import json
import llm_behavior_eval
from inspect_ai._util.registry import registry_info
from llm_behavior_eval.common.target_metrics import load_target_metrics
from llm_behavior_eval.example.task import example
print(json.dumps({
    "path": [str(p) for p in llm_behavior_eval.__path__],
    "task": registry_info(example).name,
    "targets": sorted(load_target_metrics()),
}))
"""
    src = str(EXAMPLE / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": src},
    )
    found = json.loads(result.stdout)
    assert str(Path(src, NAMESPACE)) in found["path"]
    assert found["task"] == f"{NAMESPACE}/example"
    assert f"{NAMESPACE}/example" in found["targets"]


# License checks


def make_distribution(tmp_path: Path) -> Distribution:
    path = tmp_path / "source"
    path.mkdir()
    shutil.copy(ROOT / "LICENSE", path / "LICENSE")
    (path / "pyproject.toml").write_text(
        '[project]\nname = "llm-behavior-eval-fake"\nversion = "1"\n'
        'license = "MIT"\nlicense-files = ["LICENSE"]\n'
    )
    return Distribution(path)


def write_wheel(directory: Path, files: dict[str, bytes]) -> Path:
    path = directory / "llm_behavior_eval_fake-1-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def write_sdist(directory: Path, files: dict[str, bytes]) -> Path:
    path = directory / "llm_behavior_eval_fake-1.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            source = directory / "member"
            source.write_bytes(data)
            archive.add(source, arcname=name)
            source.unlink()
    return path


def metadata(expression: str) -> bytes:
    return (
        f"Metadata-Version: 2.4\nName: x\nLicense-Expression: {expression}\n\n".encode()
    )


def test_license_check_accepts_correct_artifacts(tmp_path: Path) -> None:
    distribution = make_distribution(tmp_path)
    license_text = (ROOT / "LICENSE").read_bytes()
    out = tmp_path / "dist"
    out.mkdir()
    write_wheel(
        out,
        {
            "x-1.dist-info/METADATA": metadata("MIT"),
            "x-1.dist-info/licenses/LICENSE": license_text,
        },
    )
    write_sdist(out, {"x-1/PKG-INFO": metadata("MIT"), "x-1/LICENSE": license_text})
    assert check_directory(out, [distribution]) == []


@pytest.mark.parametrize(
    ("files", "problem"),
    [
        ({"x-1.dist-info/METADATA": metadata("MIT")}, "LICENSE is missing"),
        (
            {
                "x-1.dist-info/METADATA": metadata("Apache-2.0"),
                "x-1.dist-info/licenses/LICENSE": b"",
            },
            "License-Expression 'Apache-2.0'",
        ),
        (
            {
                "x-1.dist-info/METADATA": metadata("MIT"),
                "x-1.dist-info/licenses/LICENSE": b"other",
            },
            "differs from the source",
        ),
        ({"x-1.dist-info/licenses/LICENSE": b""}, "no METADATA"),
    ],
)
def test_license_check_reports_problems(
    tmp_path: Path, files: dict[str, bytes], problem: str
) -> None:
    distribution = make_distribution(tmp_path)
    out = tmp_path / "dist"
    out.mkdir()
    problems = check_artifact(write_wheel(out, files), distribution)
    assert any(problem in found for found in problems), problems


def test_license_check_wants_one_wheel_and_one_sdist(tmp_path: Path) -> None:
    distribution = make_distribution(tmp_path)
    out = tmp_path / "dist"
    out.mkdir()
    (out / "unrelated-1-py3-none-any.whl").write_bytes(b"")
    problems = check_directory(out, [distribution])
    assert "llm-behavior-eval-fake: expected one wheel and one sdist, found []" in (
        problems
    )
    assert "unrelated-1-py3-none-any.whl: not a distribution of this repository" in (
        problems
    )


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build")
def test_built_artifacts_carry_their_licenses(tmp_path: Path) -> None:
    common = next(m for m in members() if m.name == "llm-behavior-eval-common")
    example = tmp_path / "example_benchmark"
    shutil.copytree(EXAMPLE, example, ignore=shutil.ignore_patterns("*.egg-info"))
    out = tmp_path / "dist"
    for source in (common.path, example):
        subprocess.run(
            ["uv", "build", "--quiet", "--out-dir", str(out), str(source)],
            check=True,
            cwd=ROOT,
        )
    assert check_directory(out, [common, Distribution(EXAMPLE)]) == []

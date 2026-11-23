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
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

import run_package_tests
from check_artifact_licenses import check_artifact, check_directory
from workspace import (
    EXAMPLE,
    NAMESPACE,
    ROOT,
    Distribution,
    benchmark_test_files,
    conflicting_packages,
    members,
    normalize,
    requirement_name,
    root,
    root_is_editable,
)
from workspace import main as workspace_main

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
    """Any group between ``llm_behavior_eval`` and a benchmark package."""
    parts = package.split(".")
    return [".".join(parts[:end]) for end in range(2, len(parts))]


def is_build_output(path: Path) -> bool:
    """A file an in-place build or ``__pycache__`` leaves, not a source file."""
    return any(
        part in {"build", "dist", "__pycache__"} or part.endswith(".egg-info")
        for part in path.parts
    )


@pytest.mark.parametrize("distribution", NATIVE, ids=ids)
def test_native_distribution_metadata(distribution: Distribution) -> None:
    project = distribution.project
    assert distribution.name.startswith("llm-behavior-eval-")
    # Each native distribution's directory is inside the legacy package.
    assert distribution.path == EXAMPLE or (
        distribution.path.is_relative_to(ROOT / NAMESPACE)
        and distribution.path != ROOT / NAMESPACE
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
        assert not distribution.entry_points
        assert not list(distribution.path.glob("*.py"))
        return
    [package, *subpackages] = distribution.packages
    assert distribution.top_packages == [package]
    assert package.startswith(f"{NAMESPACE}.")
    assert all(p.startswith(f"{package}.") for p in subpackages)
    # The package's directory is the distribution's own: its pyproject.toml
    # sits inside it, in place under llm_behavior_eval/ (the example is staged
    # there by scripts/check_installs.py).
    assert setuptools["package-dir"] == {package: "."}
    assert distribution.package_directory(package) == distribution.path
    if distribution.path != EXAMPLE:
        assert distribution.path == ROOT.joinpath(*package.split("."))
    # Otherwise pyproject.toml and LICENSE would ship inside the package.
    assert setuptools["include-package-data"] is False
    # Namespace levels (a group) ship no __init__.py: several distributions
    # share them. llm_behavior_eval/__init__.py is the legacy distribution's.
    for level in namespace_levels(package):
        assert not ROOT.joinpath(*level.split("."), "__init__.py").exists()
    directories = {distribution.package_directory(p) for p in distribution.packages}
    assert all((directory / "__init__.py").is_file() for directory in directories)
    # Every module is in a listed package: setuptools ships nothing else.
    stray = [
        path
        for path in distribution.path.rglob("*.py")
        if not is_build_output(path.relative_to(distribution.path))
        and path.parent not in directories
    ]
    assert stray == []


@pytest.mark.parametrize("distribution", [d for d in NATIVE if d.is_benchmark], ids=ids)
def test_benchmark_distribution(distribution: Distribution) -> None:
    [package] = distribution.top_packages
    directory = distribution.package_directory(package)
    # Inspect loads only entry points named after a task's package prefix.
    assert distribution.entry_points == {NAMESPACE: f"{package}.task"}
    assert (directory / "task.py").is_file()
    assert (distribution.path / distribution.project["readme"]) == (
        directory / "README.md"
    )
    targets = json.loads((directory / "target_metrics.json").read_text())
    assert all(task.startswith(f"{NAMESPACE}/") for task in targets)
    # README.md and target_metrics.json always ship, parity.yaml when the
    # benchmark has one; any other data file the package reads may be added.
    shipped = set(distribution.pyproject["tool"]["setuptools"]["package-data"][package])
    assert {"README.md", "target_metrics.json"} <= shipped
    if (directory / "parity.yaml").is_file():
        assert "parity.yaml" in shipped
    requirements = {requirement_name(text) for text in distribution.requirements}
    assert "llm-behavior-eval-common" in requirements
    if distribution.path != EXAMPLE:
        assert all(path.is_file() for path in benchmark_test_files(distribution))


def test_common_distribution() -> None:
    [common] = [d for d in members() if d.name == "llm-behavior-eval-common"]
    assert common.relative == f"{NAMESPACE}/common"
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
    assert missing_from_dev(pyproject, names) == set()
    # Every package is in place under the repository root.
    assert pyproject["tool"]["pyrefly"]["search-path"] == [".", "tests", "scripts"]
    assert not EXAMPLE.is_relative_to(ROOT / NAMESPACE)


def missing_from_dev(pyproject: dict[str, Any], names: list[str]) -> set[str]:
    """Members neither in the dev group nor declared conflicting.

    A member named in a root ``[tool.uv] conflicts`` entry cannot share the
    dev environment with the member it conflicts with, so it may stay out
    (``llm_behavior_eval/security/README.md``); every other one is in it.
    """
    dev = {
        requirement_name(text)
        for text in pyproject["dependency-groups"]["dev"]
        if isinstance(text, str)
    }
    return set(names) - dev - conflicting_packages(pyproject)


def test_only_conflicting_members_may_stay_out_of_dev() -> None:
    conflicting = {
        "dependency-groups": {"dev": ["a>=1", {"include-group": "ci"}]},
        "tool": {
            "uv": {
                "conflicts": [
                    [{"package": "B_Fork"}, {"package": "c"}],
                    [{"extra": "d"}, {"extra": "e"}],
                ]
            }
        },
    }
    assert missing_from_dev(conflicting, ["a", "b-fork", "c", "d"]) == {"d"}
    plain = {"dependency-groups": {"dev": ["a>=1"]}, "tool": {"uv": {}}}
    assert missing_from_dev(plain, ["a", "b-fork"]) == {"b-fork"}


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


def test_workspace_script_lists_members_and_rejects_unknown_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace_main([])
    assert capsys.readouterr().out.splitlines() == [str(m.path) for m in members()]
    workspace_main(["--unpublished"])
    unpublished = [m.name for m in members() if m.unpublished]
    assert capsys.readouterr().out.splitlines() == unpublished
    with pytest.raises(SystemExit) as exit_info:
        workspace_main(["--unpublish"])
    assert exit_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("value", ['""', '"  "', "true", "1", "[]"])
def test_unpublished_reason_must_be_a_non_empty_string(
    tmp_path: Path, value: str
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\n[tool.llm-behavior-eval]\nunpublished = {value}\n'
    )
    with pytest.raises(ValueError, match="non-empty string"):
        _ = Distribution(tmp_path).unpublished


def unpublished_step_script() -> str:
    """The release workflow's step that removes unpublished distributions."""
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "deploy_to_pypi.yaml").read_text(
            encoding="utf-8"
        )
    )
    [script] = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("name") == "Leave out unpublished distributions"
    ]
    return script


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("fails", [False, True])
def test_release_leaves_out_unpublished_distributions(
    tmp_path: Path, fails: bool
) -> None:
    # A stand-in python prints one unpublished name, or fails as workspace.py
    # would on a malformed reason; the step must then fail, not upload all.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(
        "#!/bin/sh\n" + ("exit 1\n" if fails else "echo llm-behavior-eval-fork\n")
    )
    stub.chmod(0o755)
    dist = tmp_path / "dist"
    dist.mkdir()
    kept = dist / "llm_behavior_eval_common-1-py3-none-any.whl"
    dropped = dist / "llm_behavior_eval_fork-1-py3-none-any.whl"
    for path in (kept, dropped, dist / "llm_behavior_eval_fork-1.tar.gz"):
        path.touch()
    result = subprocess.run(
        ["bash", "-e", "-c", unpublished_step_script()],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert (result.returncode != 0) == fails
    assert kept.exists()
    assert dropped.exists() == fails
    assert sorted(p.name for p in dist.iterdir()) == (
        sorted([kept.name, dropped.name, "llm_behavior_eval_fork-1.tar.gz"])
        if fails
        else [kept.name]
    )


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
    assert distribution.is_root
    assert not distribution.entry_points
    assert distribution.project["license"] == {"file": "LICENSE"}
    # The flat layout, shipping only the legacy packages: the native
    # distributions' directories under llm_behavior_eval/ are excluded.
    assert distribution.pyproject["tool"]["setuptools"]["packages"]["find"] == {
        "include": [NAMESPACE, f"{NAMESPACE}.evaluation_utils*"],
    }
    # Installed editable, the root puts the repository root on sys.path (see
    # test_editable_workspace_imports_the_legacy_package).
    assert distribution.pyproject["tool"]["uv"]["config-settings-package"] == {
        "llm-behavior-eval": {"editable_mode": "compat"}
    }
    package = ROOT / NAMESPACE
    assert (package / "__init__.py").is_file()
    assert not (ROOT / "src").exists()
    # Every other directory in the package holds native distributions.
    member_paths = {member.path for member in members()}
    for directory in package.iterdir():
        if not directory.is_dir() or directory.name in {
            "evaluation_utils",
            "__pycache__",
        }:
            continue
        assert any(path.is_relative_to(directory) for path in member_paths), directory


def test_bumpver_updates_every_distribution() -> None:
    patterns = root().pyproject["tool"]["bumpver"]["file_patterns"]
    assert patterns[f"{NAMESPACE}/__init__.py"] == ['^__version__ = "{version}"$']
    assert patterns[f"{NAMESPACE}/**/pyproject.toml"] == ['^version = "{version}"$']
    example = EXAMPLE.relative_to(ROOT) / "pyproject.toml"
    assert patterns[example.as_posix()] == ['^version = "{version}"$']
    found = {
        path.parent
        for path in ROOT.glob(f"{NAMESPACE}/**/pyproject.toml")
        if not is_build_output(path.relative_to(ROOT))
    }
    assert found == {m.path for m in members()}


# Namespace


@pytest.mark.skipif(not root_is_editable(), reason="needs the editable workspace")
def test_editable_workspace_imports_the_legacy_package(tmp_path: Path) -> None:
    # From outside the repository, llm_behavior_eval is the legacy regular
    # package (its __init__ runs), and every native package is found in place
    # below it. A member's editable install registers llm_behavior_eval as a
    # namespace; only the root's compat mode keeps the regular package first.
    code = """
import importlib.util, json, sys
import llm_behavior_eval
print(json.dumps({
    "file": llm_behavior_eval.__file__,
    "version": llm_behavior_eval.__version__,
    "path": [str(p) for p in llm_behavior_eval.__path__],
    "packages": {
        name: importlib.util.find_spec(name).origin for name in sys.argv[1:]
    },
}))
"""
    packages = {
        package: distribution.package_directory(package) / "__init__.py"
        for distribution in members()
        for package in distribution.packages
    }
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", code, *packages],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env=env,
    )
    found = json.loads(result.stdout)
    assert found["file"] == str(ROOT / NAMESPACE / "__init__.py")
    assert found["version"] == root().version
    assert found["path"] == [str(ROOT / NAMESPACE)]
    assert found["packages"] == {name: str(path) for name, path in packages.items()}


# License checks


def make_distribution(
    tmp_path: Path, license_files: str = 'license-files = ["LICENSE"]\n'
) -> Distribution:
    path = tmp_path / "source"
    path.mkdir()
    shutil.copy(ROOT / "LICENSE", path / "LICENSE")
    (path / "pyproject.toml").write_text(
        '[project]\nname = "llm-behavior-eval-fake"\nversion = "1"\n'
        'license = "MIT"\n' + license_files
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


@pytest.mark.parametrize("license_files", ["", "license-files = []\n"])
def test_license_check_wants_license_files(tmp_path: Path, license_files: str) -> None:
    distribution = make_distribution(tmp_path, license_files)
    out = tmp_path / "dist"
    out.mkdir()
    files = {
        "x-1.dist-info/METADATA": metadata("MIT"),
        "x-1.dist-info/licenses/LICENSE": (ROOT / "LICENSE").read_bytes(),
    }
    problems = check_artifact(write_wheel(out, files), distribution)
    assert problems == [
        "llm-behavior-eval-fake: project.license-files must list the license text"
    ]


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
    out = tmp_path / "dist"
    ignore = shutil.ignore_patterns("*.egg-info", "build", "__pycache__")
    for source in (common.path, EXAMPLE):
        # A copy, so the build leaves nothing in the repository.
        copy = tmp_path / source.name
        shutil.copytree(source, copy, ignore=ignore)
        subprocess.run(
            ["uv", "build", "--quiet", "--out-dir", str(out), str(copy)],
            check=True,
            cwd=tmp_path,
        )
    assert check_directory(out, [common, Distribution(EXAMPLE)]) == []
    # The wheels hold the package's modules and data, not its pyproject.toml.
    [common_wheel] = out.glob("llm_behavior_eval_common-*.whl")
    [example_wheel] = out.glob("llm_behavior_eval_example-*.whl")
    assert wheel_files(common_wheel) == {
        f"{NAMESPACE}/common/{path.name}" for path in common.path.glob("*.py")
    }
    assert wheel_files(example_wheel) == {
        f"{NAMESPACE}/example/{name}"
        for name in ["__init__.py", "task.py", "README.md", "target_metrics.json"]
    }


def wheel_files(path: Path) -> set[str]:
    """A wheel's files outside its ``.dist-info`` directory."""
    with zipfile.ZipFile(path) as archive:
        return {
            name
            for name in archive.namelist()
            if not name.split("/", 1)[0].endswith(".dist-info")
        }


def test_package_tests_reject_an_unknown_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_package_tests.py", "llm-behavior-eval-nope"])
    with pytest.raises(
        SystemExit, match="Not workspace members: llm-behavior-eval-nope"
    ):
        run_package_tests.main()

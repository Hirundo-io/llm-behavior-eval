"""The security group: ``llm-behavior-eval-security`` and ``llm_behavior_eval.security``.

See ``llm_behavior_eval/security/README.md``. The group's directory is both the
``llm_behavior_eval.security`` namespace level and the group distribution's
project directory; each benchmark's directory (and project) is nested in it.
The group's benchmarks are found from the workspace configuration, so adding
one needs no edit here.
"""

import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from check_artifact_licenses import check_directory
from check_installs import stage_example
from workspace import (
    NAMESPACE,
    ROOT,
    Distribution,
    benchmark_dependencies,
    members,
    requirement_name,
    root,
    root_is_editable,
)

GROUP = ROOT / NAMESPACE / "security"
[DISTRIBUTION] = [member for member in members() if member.path == GROUP]
BENCHMARKS = [member for member in members() if member.path.parent == GROUP]


def test_group_distribution_is_a_metapackage() -> None:
    assert DISTRIBUTION.name == "llm-behavior-eval-security"
    assert DISTRIBUTION.is_metapackage
    assert not DISTRIBUTION.entry_points
    assert not DISTRIBUTION.optional_requirements


def test_group_distribution_depends_on_every_published_benchmark() -> None:
    names = [requirement_name(text) for text in DISTRIBUTION.requirements]
    published = sorted(member.name for member in BENCHMARKS if not member.unpublished)
    assert names == ["llm-behavior-eval-common", *published]
    readme = (GROUP / "README.md").read_text(encoding="utf-8")
    for member in BENCHMARKS:
        assert f"`{member.name}`" in readme, member.name


def test_root_reaches_security_benchmarks_through_the_group() -> None:
    names = {requirement_name(text) for text in root().requirements}
    assert DISTRIBUTION.name in names
    assert not names & {member.name for member in BENCHMARKS}


def test_group_directory_is_a_namespace_level() -> None:
    # Several distributions share llm_behavior_eval.security: none ships its
    # __init__.py, and the root's wheel ships nothing from the group.
    assert not (GROUP / "__init__.py").exists()
    include = root().setuptools["packages"]["find"]["include"]
    assert not any(
        fnmatch.fnmatchcase(f"{NAMESPACE}.security{suffix}", pattern)
        for pattern in include
        for suffix in ("", ".x")
    )


@pytest.mark.parametrize("member", BENCHMARKS, ids=lambda member: member.relative)
def test_security_benchmarks_live_in_the_group_namespace(member: Distribution) -> None:
    [package, *_] = member.packages
    assert package == f"{NAMESPACE}.security.{member.path.name}"
    assert member.package_directory(package) == member.path


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build")
def test_group_distribution_builds(tmp_path: Path) -> None:
    # A copy of the group directory, with a synthetic benchmark nested in it as
    # a group's benchmarks are, so the build leaves nothing in the repository
    # and must not pick up the nested benchmark's files.
    repository = tmp_path / "repository"
    group = repository / NAMESPACE / "security"
    shutil.copytree(
        GROUP,
        group,
        ignore=shutil.ignore_patterns("*.egg-info", "build", "__pycache__"),
    )
    stage_example(repository, "security")
    out = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--quiet", "--out-dir", str(out), str(group)],
        check=True,
        cwd=tmp_path,
    )
    assert check_directory(out, [DISTRIBUTION]) == []
    [wheel] = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        [metadata] = [name for name in names if name.endswith(".dist-info/METADATA")]
        requires = (
            BytesParser()
            .parsebytes(archive.read(metadata))
            .get_all("Requires-Dist", [])
        )
    # Dependencies only: no module, not even an llm_behavior_eval directory.
    assert all(name.split("/")[0].endswith(".dist-info") for name in names)
    assert [Requirement(text) for text in requires] == [
        Requirement(text) for text in DISTRIBUTION.requirements
    ]
    # The sdist holds the metadata, README and LICENSE: no benchmark's files.
    [sdist] = out.glob("*.tar.gz")
    with tarfile.open(sdist) as archive:
        files = {
            name.split("/", 1)[1]
            for name in archive.getnames()
            if "/" in name and ".egg-info" not in name
        }
    assert files == {"LICENSE", "PKG-INFO", "README.md", "pyproject.toml", "setup.cfg"}


@pytest.mark.skipif(not root_is_editable(), reason="needs the editable workspace")
def test_editable_group_is_a_namespace_below_the_legacy_package(
    tmp_path: Path,
) -> None:
    # From outside the repository, llm_behavior_eval is the legacy regular
    # package and llm_behavior_eval.security the group directory in place, a
    # namespace package (scripts/check_installs.py --group security runs a
    # synthetic benchmark nested in it, installed three ways).
    code = """
import json
import llm_behavior_eval.security as group
print(json.dumps({
    "file": getattr(group, "__file__", None),
    "path": [str(p) for p in group.__path__],
}))
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env=env,
    )
    found = json.loads(result.stdout)
    assert found == {"file": None, "path": [str(GROUP)]}


def test_install_check_installs_the_benchmarks_a_group_depends_on(
    tmp_path: Path,
) -> None:
    def write(name: str, body: str) -> Distribution:
        (tmp_path / name).mkdir()
        (tmp_path / name / "pyproject.toml").write_text(body, encoding="utf-8")
        return Distribution(tmp_path / name)

    def benchmark(name: str) -> Distribution:
        return write(
            name,
            f'[project]\nname = "{name}"\nversion = "0"\n'
            f'[project.entry-points.inspect_ai]\nllm_behavior_eval = "x"\n',
        )

    common = write(
        "common", '[project]\nname = "llm-behavior-eval-common"\nversion = "0"\n'
    )
    first, second, other = benchmark("b-one"), benchmark("b_two"), benchmark("b-three")
    group = write(
        "group",
        '[project]\nname = "g"\nversion = "0"\ndependencies = '
        '["llm-behavior-eval-common>=0", "B_Two==0", "requests", "b-one"]\n',
    )
    found = benchmark_dependencies(group, [common, first, second, other, group])
    assert [member.name for member in found] == ["b-two", "b-one"]

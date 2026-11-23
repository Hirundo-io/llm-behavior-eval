"""The security group: ``llm-behavior-eval-security`` and ``llm_behavior_eval.security``.

See ``packages/security/README.md``. The group's benchmarks are found from the
workspace configuration, so adding one needs no edit here.
"""

import json
import os
import shutil
import subprocess
import sys
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

GROUP = ROOT / "packages" / "security"
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


@pytest.mark.parametrize("member", BENCHMARKS, ids=lambda member: member.relative)
def test_security_benchmarks_live_in_the_group_namespace(member: Distribution) -> None:
    [package] = member.packages
    assert package.startswith(f"{NAMESPACE}.security.")
    for level in (member.src / NAMESPACE, member.src / NAMESPACE / "security"):
        assert not (level / "__init__.py").exists()


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build")
def test_group_distribution_builds(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--quiet", "--out-dir", str(tmp_path), str(GROUP)],
        check=True,
        cwd=ROOT,
    )
    assert check_directory(tmp_path, [DISTRIBUTION]) == []
    [wheel] = tmp_path.glob("*.whl")
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


@pytest.mark.skipif(not root_is_editable(), reason="needs the editable workspace")
def test_group_namespace_holds_a_synthetic_benchmark(tmp_path: Path) -> None:
    # A security benchmark's src/ on sys.path, as its editable install puts it.
    src = stage_example(tmp_path, "security") / "src"
    code = """
import json
import llm_behavior_eval.security as group
from inspect_ai._util.registry import registry_info
from llm_behavior_eval.common.target_metrics import load_target_metrics
from llm_behavior_eval.security.example.task import example
print(json.dumps({
    "group_file": getattr(group, "__file__", None),
    "group_path": [str(p) for p in group.__path__],
    "task": registry_info(example).name,
    "targets": sorted(load_target_metrics()),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(src)},
    )
    found = json.loads(result.stdout)
    assert found["group_file"] is None
    assert str(src / NAMESPACE / "security") in found["group_path"]
    assert found["task"] == f"{NAMESPACE}/example"
    assert f"{NAMESPACE}/example" in found["targets"]


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

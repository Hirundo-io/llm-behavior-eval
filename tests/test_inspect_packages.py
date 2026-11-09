"""Benchmark package conventions: task discovery, target metrics and packaging.

A benchmark PR adds only ``llm_behavior_eval/<benchmark>/`` and its own test
file; these tests check that nothing else needs editing for it to register,
report and ship.
"""

import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_behavior_eval import _registry
from llm_behavior_eval.common.packages import benchmark_packages, subpackages
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    main,
    read_target_metrics_file,
    target_metrics_files,
    write_target_metrics,
)

REPO = Path(__file__).parents[1]
PACKAGE_DIR = REPO / "llm_behavior_eval"

TASK_SOURCE = """\
from inspect_ai import Task, task
from inspect_ai.dataset import Sample

IMPORT_ORDER = __import__("builtins").__dict__.setdefault("_FAKE_IMPORT_ORDER", [])
IMPORT_ORDER.append(__name__)


@task
def {name}() -> Task:
    return Task(dataset=[Sample(input="hi", target="hi")])
"""


def metric(name: str, *, higher_is_better: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "is_percentage": True,
        "is_higher_better": higher_is_better,
        "is_normalized": True,
    }


def write_package(root: Path, name: str, files: dict[str, str]) -> Path:
    package = root / name
    package.mkdir(parents=True)
    for filename, content in files.items():
        (package / filename).write_text(content)
    return package


@pytest.fixture
def fake_root(tmp_path, monkeypatch) -> Iterator[tuple[str, Path]]:
    """An importable throwaway root package standing in for llm_behavior_eval."""
    name = "lbe_fake_root"
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, root
    for module in [m for m in sys.modules if m.split(".")[0] == name]:
        del sys.modules[module]


def test_benchmark_packages_are_task_packages_in_sorted_order(fake_root):
    name, root = fake_root
    # Created out of order; discovery must not depend on filesystem order.
    write_package(root, "zeta", {"__init__.py": "", "task.py": ""})
    write_package(root, "alpha", {"__init__.py": "", "task.py": ""})
    write_package(root, "shared", {"__init__.py": "", "scorer.py": ""})
    write_package(root, "nested", {"__init__.py": ""})
    write_package(root / "nested", "task", {"__init__.py": ""})
    write_package(root, "namespace_only", {"task.py": ""})
    (root / "task.py").write_text("")

    assert subpackages(name) == [
        f"{name}.alpha",
        f"{name}.nested",
        f"{name}.shared",
        f"{name}.zeta",
    ]
    assert benchmark_packages(name) == [f"{name}.alpha", f"{name}.zeta"]
    assert benchmark_packages(name) == benchmark_packages(name)
    # Discovery locates packages without importing them.
    assert not [m for m in sys.modules if m.startswith(f"{name}.")]


def test_shipped_benchmarks_are_discovered():
    packages = benchmark_packages()
    assert "llm_behavior_eval.inspect_template" in packages
    assert "llm_behavior_eval.common" not in packages
    assert packages == sorted(packages)


def run_registry_with_extra_packages(
    tmp_path: Path, packages: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Import ``_registry`` in a fresh interpreter with extra benchmark packages.

    The extra packages live outside the source tree and join
    ``llm_behavior_eval`` through its ``__path__``, as a new package directory
    would.
    """
    extra = tmp_path / "extra"
    for package, task_name in packages.items():
        write_package(
            extra,
            package,
            {"__init__.py": "", "task.py": TASK_SOURCE.format(name=task_name)},
        )
    script = textwrap.dedent(
        f"""
        import builtins, json
        import llm_behavior_eval
        llm_behavior_eval.__path__.append({str(extra)!r})
        from llm_behavior_eval import _registry
        print(json.dumps({{
            "all": _registry.__all__,
            "imported": builtins.__dict__.get("_FAKE_IMPORT_ORDER", []),
            "callable": [callable(getattr(_registry, n)) for n in _registry.__all__],
        }}))
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=tmp_path
    )


def test_registry_registers_a_new_benchmark_package_without_edits(tmp_path):
    result = run_registry_with_extra_packages(
        tmp_path, {"zz_fake_bench": "zz_fake_bench", "aa_fake_bench": "aa_fake_bench"}
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert {"aa_fake_bench", "inspect_template", "zz_fake_bench"} <= set(out["all"])
    assert out["all"] == sorted(out["all"])
    assert all(out["callable"])
    assert out["imported"] == [
        "llm_behavior_eval.aa_fake_bench.task",
        "llm_behavior_eval.zz_fake_bench.task",
    ]


def test_registry_rejects_duplicate_task_names(tmp_path):
    result = run_registry_with_extra_packages(
        tmp_path, {"zz_fake_bench": "inspect_template"}
    )
    assert result.returncode != 0
    assert "registered by both" in result.stderr


def test_registry_exports_sorted_task_names():
    assert "inspect_template" in _registry.__all__
    assert _registry.__all__ == sorted(_registry.__all__)
    for name in _registry.__all__:
        assert callable(getattr(_registry, name))


def test_every_registered_task_has_target_metrics_and_vice_versa():
    registered = {f"llm_behavior_eval/{name}" for name in _registry.__all__}
    assert set(load_target_metrics()) == registered


def test_no_root_target_metrics_file():
    # Benchmarks own their entries; a root file would bring back add/add conflicts.
    assert not (PACKAGE_DIR / TARGET_METRICS_FILE).exists()


def test_merge_is_sorted_and_keeps_metric_order(fake_root):
    name, root = fake_root
    write_package(
        root,
        "zeta",
        {
            "__init__.py": "",
            TARGET_METRICS_FILE: json.dumps(
                {f"{name}/zeta": [metric("b"), metric("a", higher_is_better=False)]}
            ),
        },
    )
    write_package(
        root,
        "alpha",
        {
            "__init__.py": "",
            TARGET_METRICS_FILE: json.dumps(
                {f"{name}/alpha_2": [metric("x")], f"{name}/alpha_1": [metric("y")]}
            ),
        },
    )
    write_package(root, "nometrics", {"__init__.py": ""})

    assert list(target_metrics_files(name)) == [f"{name}.alpha", f"{name}.zeta"]
    merged = load_target_metrics(name)
    assert list(merged) == [f"{name}/alpha_1", f"{name}/alpha_2", f"{name}/zeta"]
    assert [m["name"] for m in merged[f"{name}/zeta"]] == ["b", "a"]
    # Loading reads files only; it never imports benchmark packages.
    assert not [m for m in sys.modules if m.startswith(f"{name}.")]


def test_duplicate_task_across_packages_fails(fake_root):
    name, root = fake_root
    for package in ("one", "two"):
        write_package(
            root,
            package,
            {
                "__init__.py": "",
                TARGET_METRICS_FILE: json.dumps({f"{name}/shared": [metric("m")]}),
            },
        )
    with pytest.raises(ValueError, match=f"{name}/shared is defined in both"):
        load_target_metrics(name)


def test_duplicate_task_within_file_fails(tmp_path):
    path = tmp_path / TARGET_METRICS_FILE
    entry = json.dumps([metric("m")])
    path.write_text(
        f'{{"llm_behavior_eval/a": {entry}, "llm_behavior_eval/a": {entry}}}'
    )
    with pytest.raises(ValueError, match="duplicate key 'llm_behavior_eval/a'"):
        read_target_metrics_file(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ([], "non-empty JSON object"),
        ({}, "non-empty JSON object"),
        ({"other/a": [metric("m")]}, "must start with llm_behavior_eval/"),
        ({"llm_behavior_eval/a": []}, "non-empty list"),
        ({"llm_behavior_eval/a": [{"name": "m"}]}, "exactly the keys"),
        ({"llm_behavior_eval/a": [metric("m") | {"extra": 1}]}, "exactly the keys"),
        ({"llm_behavior_eval/a": [metric("m") | {"is_normalized": 1}]}, "booleans"),
        ({"llm_behavior_eval/a": [metric("")]}, "needs a name"),
    ],
)
def test_invalid_target_metrics_fail(tmp_path, content, message):
    path = tmp_path / TARGET_METRICS_FILE
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match=message):
        read_target_metrics_file(path)


def test_write_and_cli_produce_the_merged_mapping(tmp_path, capsys):
    expected = load_target_metrics()
    written = write_target_metrics(tmp_path / "out" / "targets.json")
    assert json.loads(written.read_text()) == expected

    main(["--output", str(tmp_path / "cli.json")])
    assert capsys.readouterr().out.strip() == str(tmp_path / "cli.json")
    assert json.loads((tmp_path / "cli.json").read_text()) == expected

    main([])
    assert json.loads(capsys.readouterr().out) == expected


def test_shipped_target_metrics_are_tracked():
    """Package target metrics must not be swallowed by the ``*.json`` ignore rule."""
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    for path in target_metrics_files().values():
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(path)], cwd=REPO, check=False
        )
        assert ignored.returncode == 1, f"{path} is gitignored"


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    """Build the wheel offline from a copy of the source tree."""
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    source = tmp_path_factory.mktemp("source")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(REPO / name, source / name)
    shutil.copytree(
        PACKAGE_DIR,
        source / "llm_behavior_eval",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    out = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from setuptools import build_meta; "
            "print(build_meta.build_wheel(sys.argv[1]))",
            str(out),
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel,) = out.glob("*.whl")
    return wheel


def test_wheel_ships_benchmark_package_files(built_wheel):
    names = set(zipfile.ZipFile(built_wheel).namelist())
    for package_file in ("target_metrics.json", "parity.yaml", "README.md", "task.py"):
        assert f"llm_behavior_eval/inspect_template/{package_file}" in names
    for path in target_metrics_files().values():
        assert path.relative_to(REPO).as_posix() in names
    assert "llm_behavior_eval/target_metrics.json" not in names


def test_installed_wheel_discovers_the_same_tasks_and_metrics(built_wheel, tmp_path):
    installed = tmp_path / "site"
    zipfile.ZipFile(built_wheel).extractall(installed)
    script = textwrap.dedent(
        """
        import json
        import llm_behavior_eval
        from llm_behavior_eval import _registry
        from llm_behavior_eval.common.packages import benchmark_packages
        from llm_behavior_eval.common.target_metrics import load_target_metrics
        print(json.dumps({
            "file": llm_behavior_eval.__file__,
            "packages": benchmark_packages(),
            "tasks": _registry.__all__,
            "metrics": load_target_metrics(),
        }))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PYTHONPATH": str(installed), "PATH": ""},
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert Path(out["file"]).is_relative_to(installed)
    assert out["packages"] == benchmark_packages()
    assert out["tasks"] == _registry.__all__
    assert out["metrics"] == load_target_metrics()

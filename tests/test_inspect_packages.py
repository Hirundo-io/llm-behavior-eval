"""Benchmark package conventions: task discovery, target metrics and packaging.

A benchmark PR adds only ``llm_behavior_eval/<benchmark>/`` and its own test
file; these tests check that nothing else needs editing for it to register,
report and ship.
"""

import inspect
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.util import registry_info
from typer.testing import CliRunner

from llm_behavior_eval import _registry
from llm_behavior_eval.common.judge import THINKING_TASK_ARGS
from llm_behavior_eval.common.metrics import EPOCH_SCORES, reduced_metrics
from llm_behavior_eval.common.packages import benchmark_packages, subpackages
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    app,
    load_target_metrics,
    missing_target_metrics,
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


def metric(name: str, *, higher_is_better: bool = True) -> dict[str, str | bool]:
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
def fake_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, Path]]:
    """An importable throwaway root package standing in for llm_behavior_eval."""
    name = "lbe_fake_root"
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, root
    for module in [m for m in sys.modules if m.split(".")[0] == name]:
        del sys.modules[module]


def test_benchmark_packages_are_task_packages_in_sorted_order(
    fake_root: tuple[str, Path],
) -> None:
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


def test_shipped_benchmarks_are_discovered() -> None:
    packages = benchmark_packages()
    assert "llm_behavior_eval.inspect_template" in packages
    assert "llm_behavior_eval.common" not in packages
    assert packages == sorted(packages)


def run_registry_with_extra_packages(
    tmp_path: Path,
    packages: dict[str, str],
    modules: dict[str, dict[str, str]] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Import ``_registry`` in a fresh interpreter with extra benchmark packages.

    The extra packages live outside the source tree and join
    ``llm_behavior_eval`` through its ``__path__``, as a new package directory
    would. ``modules`` adds further modules to a package; its task module
    imports them, as a benchmark's task module imports its scorer and solver.
    """
    extra = tmp_path / "extra"
    for package, task_name in packages.items():
        extra_modules = (modules or {}).get(package, {})
        imports = "".join(
            f"import llm_behavior_eval.{package}.{Path(filename).stem}\n"
            for filename in extra_modules
        )
        write_package(
            extra,
            package,
            {
                "__init__.py": "",
                "task.py": imports + TASK_SOURCE.format(name=task_name),
                **extra_modules,
            },
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


def test_registry_registers_a_new_benchmark_package_without_edits(
    tmp_path: Path,
) -> None:
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


def test_registry_rejects_duplicate_task_names(tmp_path: Path) -> None:
    result = run_registry_with_extra_packages(
        tmp_path, {"zz_fake_bench": "inspect_template"}
    )
    assert result.returncode != 0
    assert "registered by both" in result.stderr


REGISTERED_SOURCES = {
    "metric": """\
from inspect_ai.scorer import metric


@metric
def {name}():
    def compute(scores):
        return 0.0

    return compute
""",
    "scorer": """\
from inspect_ai.scorer import Score, accuracy, scorer


@scorer(metrics=[accuracy()])
def {name}():
    async def score(state, target):
        return Score(value=1)

    return score
""",
    "solver": """\
from inspect_ai.solver import solver


@solver
def {name}():
    async def solve(state, generate):
        return state

    return solve
""",
}


@pytest.mark.parametrize("kind", sorted(REGISTERED_SOURCES))
def test_registry_rejects_a_name_defined_by_two_packages(
    tmp_path: Path, kind: str
) -> None:
    # Inspect's registry keeps the last object registered under a name, so a
    # second benchmark reusing a metric, scorer or solver name would silently
    # replace the first one's when a log is rescored.
    source = REGISTERED_SOURCES[kind].format(name="shared_name")
    result = run_registry_with_extra_packages(
        tmp_path,
        {"aa_fake_bench": "aa_fake_bench", "zz_fake_bench": "zz_fake_bench"},
        {
            "aa_fake_bench": {"parts.py": source},
            "zz_fake_bench": {"other_parts.py": source},
        },
    )
    assert result.returncode != 0
    assert (
        f"{kind} 'llm_behavior_eval/shared_name' is defined by "
        "llm_behavior_eval.aa_fake_bench.parts, "
        "llm_behavior_eval.zz_fake_bench.other_parts"
    ) in result.stderr


def test_registry_accepts_package_specific_names(tmp_path: Path) -> None:
    result = run_registry_with_extra_packages(
        tmp_path,
        {"aa_fake_bench": "aa_fake_bench", "zz_fake_bench": "zz_fake_bench"},
        {
            package: {
                "parts.py": "\n\n".join(
                    source.format(name=f"{package}_{kind}")
                    for kind, source in REGISTERED_SOURCES.items()
                )
            }
            for package in ("aa_fake_bench", "zz_fake_bench")
        },
    )
    assert result.returncode == 0, result.stderr
    assert {"aa_fake_bench", "zz_fake_bench"} <= set(json.loads(result.stdout)["all"])


def test_shipped_registry_names_are_defined_once() -> None:
    definitions = _registry.registry_definitions()
    assert definitions[("task", "llm_behavior_eval/inspect_template")] == [
        "llm_behavior_eval.inspect_template.task"
    ]
    # Shared objects are attributed to common, not to the benchmarks importing them.
    assert definitions[("scorer", "llm_behavior_eval/judge_scorer")] == [
        "llm_behavior_eval.common.judge"
    ]
    # @metric wraps the factory in an Inspect-internal function; the definition
    # is still attributed to the module that wrote it.
    assert definitions[("metric", "llm_behavior_eval/field_rate")] == [
        "llm_behavior_eval.common.metrics"
    ]
    assert {key: m for key, m in definitions.items() if len(m) > 1} == {}
    _registry.check_unique_registry_names()


def test_registry_exports_sorted_task_names() -> None:
    assert "inspect_template" in _registry.__all__
    assert _registry.__all__ == sorted(_registry.__all__)
    for name in _registry.__all__:
        assert callable(getattr(_registry, name))


def package_registry_objects(kind: str) -> dict[str, Any]:
    """This package's registered objects of one type, from imported modules."""
    objects: dict[str, Any] = {}
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("llm_behavior_eval."):
            continue
        for value in list(vars(module).values()):
            try:
                info = registry_info(value)
            except ValueError:
                continue
            if info.type == kind and info.name.startswith("llm_behavior_eval/"):
                objects[info.name] = value
    return objects


def test_every_benchmark_metric_sees_every_epoch() -> None:
    # The epoch layout (INSPECT_MIGRATION.md, "Epochs"): every metric defined
    # here is unreduced, and so is every metric a scorer declares, including
    # Inspect built-ins. Task-level metrics= lists are checked by each
    # benchmark's own test with common.metrics.reduced_metrics(task).
    metrics = package_registry_objects("metric")
    assert "llm_behavior_eval/field_rate" in metrics
    assert [
        name
        for name, factory in sorted(metrics.items())
        if registry_info(factory).metadata.get("scores") != EPOCH_SCORES
    ] == []
    scorers = package_registry_objects("scorer")
    assert "llm_behavior_eval/judge_scorer" in scorers
    assert {
        name: reduced_metrics(factory)
        for name, factory in sorted(scorers.items())
        if reduced_metrics(factory)
    } == {}


def test_every_task_exposes_the_thinking_task_args() -> None:
    # Every benchmark exposes thinking-trace stripping the same way, with the
    # same names and defaults (common.judge.THINKING_TASK_ARGS).
    for name in _registry.__all__:
        parameters = inspect.signature(getattr(_registry, name)).parameters
        exposed = {
            arg: parameters[arg].default
            for arg in THINKING_TASK_ARGS
            if arg in parameters
        }
        assert exposed == dict(THINKING_TASK_ARGS), name


CREDENTIAL_WORDS = frozenset(
    {"token", "tokens", "key", "keys", "apikey", "secret", "secrets", "password"}
    | {"passwd", "credential", "credentials"}
)
"""Parameter-name words that mark a credential (``dataset_token``, ``api_key``)."""


def credential_parameters(task_function: Callable[..., Any]) -> list[str]:
    """Parameters whose name has a credential word, other than the thinking args.

    Words are the parts of a ``snake_case`` name, so ``response_tokenizer`` is
    not a credential while ``hf_token`` is. The thinking-trace task arguments
    name text markers, not credentials.
    """
    return [
        name
        for name in inspect.signature(task_function).parameters
        if name not in THINKING_TASK_ARGS
        and CREDENTIAL_WORDS & set(name.lower().split("_"))
    ]


def test_credential_parameters_are_detected() -> None:
    def task_function(
        dataset_token: str,
        api_key: str,
        client_secret: str,
        db_password: str,
        response_tokenizer: str,
        keyword: str,
        thinking_start_token: str,
        thinking_end_token: str,
    ) -> None: ...

    assert credential_parameters(task_function) == [
        "dataset_token",
        "api_key",
        "client_secret",
        "db_password",
    ]


def test_no_registered_task_takes_a_credential() -> None:
    # Inspect writes every task argument into the .eval log, so credentials
    # come only from environment variables (INSPECT_MIGRATION.md, "Credentials").
    assert {
        name: credential_parameters(getattr(_registry, name))
        for name in _registry.__all__
        if credential_parameters(getattr(_registry, name))
    } == {}


def test_every_registered_task_has_target_metrics_and_vice_versa() -> None:
    registered = {f"llm_behavior_eval/{name}" for name in _registry.__all__}
    assert set(load_target_metrics()) == registered


@pytest.fixture(scope="module")
def template_log_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real ``.eval`` log of the registered template task, as written to disk."""
    from inspect_ai import eval as inspect_eval

    log_dir = tmp_path_factory.mktemp("template_log")
    [log] = inspect_eval(
        "llm_behavior_eval/inspect_template",
        model="mockllm/model",
        log_dir=str(log_dir),
        display="none",
    )
    assert log.status == "success", log.error
    return Path(log.location)


def test_missing_target_metrics_finds_the_template_targets(
    template_log_path: Path,
) -> None:
    assert missing_target_metrics(template_log_path) == []
    assert missing_target_metrics(str(template_log_path)) == []


def test_missing_target_metrics_reads_metric_keys_not_names(
    template_log_path: Path,
) -> None:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(template_log_path))
    assert log.results is not None
    [score] = log.results.scores
    # A single-value metric is keyed by its plain name, but EvalMetric.name is
    # the registry-qualified name. inspect-brief reads the key.
    assert score.metrics["field_rate"].name == "llm_behavior_eval/field_rate"
    task = log.eval.task
    targets = {
        task: [
            metric("field_rate"),
            metric("llm_behavior_eval/field_rate"),
            metric("not_emitted"),
        ]
    }
    assert missing_target_metrics(log, targets) == [
        "llm_behavior_eval/field_rate",
        "not_emitted",
    ]


def test_missing_target_metrics_rejects_logs_brief_would_not_match(
    template_log_path: Path,
) -> None:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(template_log_path))
    assert log.eval.task == "llm_behavior_eval/inspect_template"
    # inspect-brief selects targets by log.eval.task; an unknown task would
    # silently get every metric instead of its targets.
    with pytest.raises(ValueError, match="has no target metrics"):
        missing_target_metrics(log, {"llm_behavior_eval/other": [metric("m")]})
    with pytest.raises(ValueError, match="not 'success'"):
        missing_target_metrics(log.model_copy(update={"status": "error"}))


def test_no_root_target_metrics_file() -> None:
    # Benchmarks own their entries; a root file would bring back add/add conflicts.
    assert not (PACKAGE_DIR / TARGET_METRICS_FILE).exists()


def test_merge_is_sorted_and_keeps_metric_order(fake_root: tuple[str, Path]) -> None:
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


def test_duplicate_task_across_packages_fails(fake_root: tuple[str, Path]) -> None:
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


def test_duplicate_task_within_file_fails(tmp_path: Path) -> None:
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
        ({"llm_behavior_eval/a": "m"}, "should be a valid list"),
        ({"llm_behavior_eval/a": [{"name": "m"}]}, "Field required"),
        ({"llm_behavior_eval/a": [metric("m") | {"extra": 1}]}, "Extra inputs"),
        (
            {"llm_behavior_eval/a": [metric("m") | {"is_normalized": 1}]},
            "should be a valid boolean",
        ),
        (
            {"llm_behavior_eval/a": [metric("m") | {"name": 1}]},
            "should be a valid string",
        ),
        ({"llm_behavior_eval/a": [metric("")]}, "at least 1 character"),
    ],
)
def test_invalid_target_metrics_fail(
    tmp_path: Path, content: Any, message: str
) -> None:
    path = tmp_path / TARGET_METRICS_FILE
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match=message):
        read_target_metrics_file(path)


def test_write_and_cli_produce_the_merged_mapping(tmp_path: Path) -> None:
    expected = load_target_metrics()
    written = write_target_metrics(tmp_path / "out" / "targets.json")
    assert json.loads(written.read_text()) == expected

    runner = CliRunner()
    result = runner.invoke(app, ["--output", str(tmp_path / "cli.json")])
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(tmp_path / "cli.json")
    assert json.loads((tmp_path / "cli.json").read_text()) == expected

    result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == expected

    # The documented module entry point runs the same Typer app.
    printed = subprocess.run(
        [sys.executable, "-m", "llm_behavior_eval.common.target_metrics"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(printed.stdout) == expected


def test_shipped_target_metrics_are_tracked() -> None:
    """Package target metrics must not be swallowed by the ``*.json`` ignore rule."""
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    for path in target_metrics_files().values():
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(path)], cwd=REPO, check=False
        )
        assert ignored.returncode == 1, f"{path} is gitignored"


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
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


def test_wheel_ships_benchmark_package_files(built_wheel: Path) -> None:
    names = set(zipfile.ZipFile(built_wheel).namelist())
    for package_file in ("target_metrics.json", "parity.yaml", "README.md", "task.py"):
        assert f"llm_behavior_eval/inspect_template/{package_file}" in names
    for path in target_metrics_files().values():
        assert path.relative_to(REPO).as_posix() in names
    assert "llm_behavior_eval/target_metrics.json" not in names


def test_installed_wheel_discovers_the_same_tasks_and_metrics(
    built_wheel: Path, tmp_path: Path
) -> None:
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

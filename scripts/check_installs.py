"""Install the distributions three ways and check Inspect's task names.

Copies the repository (its tracked and unignored files) to a work directory
and stages the synthetic benchmark (``tests/example_benchmark``) in the copy
where a benchmark lives, ``llm_behavior_eval/example/``. Builds every wheel
from the copy (through each sdist, so no ``build/`` folder lands in a
package), then, in a fresh virtual environment each:

* ``light``: ``llm-behavior-eval-common`` and the benchmark as wheels, without
  ``llm-behavior-eval`` (``llm_behavior_eval`` is a PEP 420 namespace);
* ``together``: the same plus the ``llm-behavior-eval`` wheel, all in one
  ``site-packages`` directory (installed ``--no-deps``: the legacy engine's
  torch and transformers do not change the layout and are not needed here);
* ``editable``: the copy's ``llm-behavior-eval`` (``editable_mode=compat``, as
  ``[tool.uv.config-settings-package]`` makes ``uv sync`` install it),
  ``llm-behavior-eval-common`` and the benchmark installed editable.

In each it runs ``inspect eval llm_behavior_eval/example`` on mock models from
outside the copy and checks the log's ``eval.task``, the target metrics, which
``llm_behavior_eval/__init__.py`` Python imports (none in ``light``) and that
the legacy engine stayed unloaded. Dependency versions come from ``uv.lock``.

Usage: ``python scripts/check_installs.py [--group GROUP] [--keep DIR]``.
``--group`` stages the benchmark inside a group (``llm_behavior_eval.<group>``,
in ``llm_behavior_eval/<group>/example/``) and installs that group's
distribution and its benchmarks (the wheels and sources built here, not PyPI)
with it when the workspace has one.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from workspace import (
    EXAMPLE,
    NAMESPACE,
    ROOT,
    benchmark_dependencies,
    members,
    normalize,
    root,
)

TASK = "llm_behavior_eval/example"
PROBE = """
import importlib.metadata, json, sys
from pathlib import Path
from inspect_ai.log import list_eval_logs, read_eval_log
import llm_behavior_eval
from llm_behavior_eval.common.target_metrics import missing_target_metrics
log = read_eval_log(list_eval_logs(sys.argv[1])[0].name)
print(json.dumps({
    "task": log.eval.task,
    "status": log.status,
    "missing_target_metrics": missing_target_metrics(log),
    "package_file": getattr(llm_behavior_eval, "__file__", None),
    "version": getattr(llm_behavior_eval, "__version__", None),
    "root_distribution": any(
        d.metadata["Name"] == "llm-behavior-eval"
        for d in importlib.metadata.distributions()
    ),
    "legacy_loaded": any(
        m.startswith(("llm_behavior_eval.evaluation_utils", "torch"))
        for m in sys.modules
    ),
}))
"""


def run(*args: str | Path, cwd: Path = ROOT) -> str:
    """Run a command and capture its output.

    Args:
        *args: The command and its arguments.
        cwd: The working directory (the repository root by default).

    Returns:
        The command's standard output.

    Raises:
        SystemExit: The command failed; the message holds its output.
    """
    result = subprocess.run(
        [str(arg) for arg in args], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode:
        raise SystemExit(
            f"{' '.join(map(str, args))} failed:\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def copy_repository(destination: Path) -> Path:
    """Copy the repository's tracked and unignored files.

    Args:
        destination: The new copy's root (created as needed).

    Returns:
        ``destination``.

    Raises:
        SystemExit: ``git ls-files`` failed (not a git checkout).
    """
    listed = run(
        "git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    ).split("\0")
    for name in filter(None, listed):
        source = ROOT / name
        if source.is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return destination


def stage_example(repository: Path, group: str | None) -> Path:
    """Copy the synthetic benchmark into a repository copy, as a benchmark lives.

    Args:
        repository: A copy of the repository (``copy_repository``).
        group: Stage the benchmark as ``llm_behavior_eval.<group>.example``,
            with its ``pyproject.toml`` rewritten to match; ungrouped
            (``llm_behavior_eval.example``) when ``None``.

    Returns:
        The staged benchmark's directory, which is also its project directory.

    Raises:
        SystemExit: The directory exists already, or the template's
            ``pyproject.toml`` no longer names the package where expected.
    """
    package = f"{NAMESPACE}.{group}.example" if group else f"{NAMESPACE}.example"
    target = repository.joinpath(*package.split("."))
    if target.exists():
        raise SystemExit(f"{target} already exists")
    shutil.copytree(
        EXAMPLE, target, ignore=shutil.ignore_patterns("*.egg-info", "__pycache__")
    )
    if group:
        pyproject = target / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        old = f"{NAMESPACE}.example"
        # packages, package-dir, package-data and the entry point.
        if text.count(f'"{old}"') < 3 or f'"{old}.task"' not in text:
            raise SystemExit(f"{EXAMPLE}/pyproject.toml no longer names {old}")
        pyproject.write_text(text.replace(old, package), encoding="utf-8")
    return target


def check(scenario: str, python: Path, work: Path, expected_file: str | None) -> dict:
    """Run the synthetic task in one environment and check the result.

    Args:
        scenario: ``light``, ``together`` or ``editable``.
        python: The environment's interpreter.
        work: The working directory for the run, outside the repository copy.
        expected_file: The ``llm_behavior_eval/__init__.py`` Python must
            import, or ``None`` for a namespace package.

    Returns:
        The probe's findings (all as expected).

    Raises:
        SystemExit: The run failed or a finding differs from the expected one.
    """
    logs = work / f"logs-{scenario}"
    run(
        venv_script(python.parent.parent, "inspect"), "eval", TASK,
        "--model", "mockllm/model", "--model-role", "grader=mockllm/model",
        "--log-dir", logs, "--display", "none",
        cwd=work,
    )  # fmt: skip
    found = json.loads(run(python, "-I", "-c", PROBE, logs, cwd=work))
    expected = {
        "task": TASK,
        "status": "success",
        "missing_target_metrics": [],
        "package_file": expected_file,
        "version": None if scenario == "light" else root().version,
        "root_distribution": scenario != "light",
        "legacy_loaded": False,
    }
    if found != expected:
        raise SystemExit(f"{scenario}: expected {expected}, found {found}")
    return found


def venv_script(venv: Path, name: str) -> Path:
    """An executable in a virtual environment, on this platform's layout.

    Args:
        venv: The environment's directory.
        name: The executable, without a suffix (``python``, ``inspect``).

    Returns:
        ``Scripts/<name>.exe`` on Windows, ``bin/<name>`` elsewhere.
    """
    if sys.platform == "win32":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--group", help="stage the benchmark in this group")
    parser.add_argument("--keep", type=Path, help="work here and keep the files")
    args = parser.parse_args()
    work = args.keep or Path(tempfile.mkdtemp(prefix="check-installs-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        check_all(args, work)
    finally:
        # Without --keep, the copies, wheels and environments go even when a
        # check fails.
        if args.keep is None:
            shutil.rmtree(work, ignore_errors=True)


def check_all(args: argparse.Namespace, work: Path) -> None:
    """Build everything in ``work`` and check each install scenario.

    Args:
        args: The command line (``--group``).
        work: The working directory for copies, builds and environments.

    Raises:
        SystemExit: A build, install or check failed.
    """
    dist = work / "dist"
    repository = copy_repository(work / "repository")
    example = stage_example(repository, args.group)
    run("uv", "build", "--all-packages", "--out-dir", dist, cwd=repository)
    run("uv", "build", "--out-dir", dist, example, cwd=repository)
    constraints = work / "constraints.txt"
    run(
        "uv", "export", "--frozen", "--all-packages", "--no-emit-workspace",
        "--no-hashes", "--no-header", "--no-annotate", "--output-file", constraints,
        cwd=repository,
    )  # fmt: skip

    def wheel(name: str) -> Path:
        [path] = dist.glob(f"{name.replace('-', '_')}-*.whl")
        return path

    common = next(m for m in members() if m.name == "llm-behavior-eval-common")
    native = [wheel(common.name), wheel("llm-behavior-eval-example")]
    editable = [repository / common.relative, example]
    name = normalize(f"llm-behavior-eval-{args.group}") if args.group else None
    group = next((m for m in members() if m.name == name), None)
    if group is not None:
        # The group's benchmarks are not on PyPI at this version: install the
        # wheels just built (and the sources, for the editable scenario).
        for member in [group, *benchmark_dependencies(group, members())]:
            native.append(wheel(member.name))
            editable.append(repository / member.relative)
    results = {}
    for scenario in ("light", "together", "editable"):
        venv = work / f"venv-{scenario}"
        run("uv", "venv", "--quiet", "--python", sys.executable, venv)
        python = venv_script(venv, "python")
        install = ("uv", "pip", "install", "--quiet", "--python", python)
        expected_file = None
        if scenario == "editable":
            # Run outside the copy, whose [tool.uv.config-settings-package] uv
            # would otherwise read (and merge with -C, which it rejects).
            run(
                *install, "--no-deps", "-C", "editable_mode=compat", "-e", repository,
                cwd=work,
            )  # fmt: skip
            run(
                *install,
                "-c",
                constraints,
                *(arg for path in editable for arg in ("-e", path)),
                cwd=work,
            )
            expected_file = str(repository / NAMESPACE / "__init__.py")
        else:
            run(*install, "-c", constraints, *native, cwd=work)
            if scenario == "together":
                run(*install, "--no-deps", wheel("llm-behavior-eval"), cwd=work)
                purelib = run(
                    python,
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))",
                ).strip()
                expected_file = str(Path(purelib, NAMESPACE, "__init__.py"))
        results[scenario] = check(scenario, python, work, expected_file)
        print(f"{scenario}: {results[scenario]['task']}", flush=True)


if __name__ == "__main__":
    main()

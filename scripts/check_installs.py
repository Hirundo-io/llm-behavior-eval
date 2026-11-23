"""Install the distributions three ways and check Inspect's task names.

Builds the workspace's wheels and the synthetic benchmark
(``tests/example_benchmark``), then, in a fresh virtual environment each:

* ``light``: ``llm-behavior-eval-common`` and the benchmark as wheels, without
  ``llm-behavior-eval`` (``llm_behavior_eval`` is a PEP 420 namespace);
* ``together``: the same plus the ``llm-behavior-eval`` wheel, all in one
  ``site-packages`` directory (installed ``--no-deps``: the legacy engine's
  torch and transformers do not change the layout and are not needed here);
* ``editable``: ``llm-behavior-eval``, ``llm-behavior-eval-common`` and the
  benchmark installed editable, as ``uv sync`` installs workspace members
  (each ``src`` directory on ``sys.path`` through a ``.pth`` file).

In each it runs ``inspect eval llm_behavior_eval/example`` on mock models and
checks the log's ``eval.task``, the target metrics and that the legacy engine
stayed unloaded. Dependency versions come from ``uv.lock``.

Usage: ``python scripts/check_installs.py [--group GROUP] [--keep DIR]``.
``--group`` stages the benchmark inside a group (``llm_behavior_eval.<group>``)
and installs that group's distribution with it when the workspace has one.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from workspace import EXAMPLE, ROOT, members, normalize

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
    "namespace_package": getattr(llm_behavior_eval, "__file__", None) is None,
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
    result = subprocess.run(
        [str(arg) for arg in args], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode:
        raise SystemExit(
            f"{' '.join(map(str, args))} failed:\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def stage_example(destination: Path, group: str | None) -> Path:
    """Copy the synthetic benchmark, moved into ``group`` when given."""
    target = destination / "example_benchmark"
    shutil.copytree(
        EXAMPLE, target, ignore=shutil.ignore_patterns("*.egg-info", "__pycache__")
    )
    if group:
        namespace = target / "src" / "llm_behavior_eval"
        (namespace / group).mkdir()
        (namespace / "example").rename(namespace / group / "example")
        pyproject = target / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        for old, new in [
            (
                '"llm_behavior_eval.example*"',
                f'"llm_behavior_eval.{group}", "llm_behavior_eval.{group}.example*"',
            ),
            (
                '"llm_behavior_eval.example.task"',
                f'"llm_behavior_eval.{group}.example.task"',
            ),
            (
                "src/llm_behavior_eval/example/",
                f"src/llm_behavior_eval/{group}/example/",
            ),
        ]:
            if old not in text:
                raise SystemExit(f"{EXAMPLE}/pyproject.toml no longer has {old}")
            text = text.replace(old, new)
        pyproject.write_text(text, encoding="utf-8")
    return target


def check(scenario: str, python: Path, work: Path) -> dict[str, object]:
    logs = work / f"logs-{scenario}"
    run(
        python.parent / "inspect", "eval", TASK,
        "--model", "mockllm/model", "--model-role", "grader=mockllm/model",
        "--log-dir", logs, "--display", "none",
        cwd=work,
    )  # fmt: skip
    found = json.loads(run(python, "-I", "-c", PROBE, logs, cwd=work))
    expected = {
        "task": TASK,
        "status": "success",
        "missing_target_metrics": [],
        "namespace_package": scenario == "light",
        "root_distribution": scenario != "light",
        "legacy_loaded": False,
    }
    if found != expected:
        raise SystemExit(f"{scenario}: expected {expected}, found {found}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--group", help="stage the benchmark in this group")
    parser.add_argument("--keep", type=Path, help="work here and keep the files")
    args = parser.parse_args()
    work = args.keep or Path(tempfile.mkdtemp(prefix="check-installs-"))
    work.mkdir(parents=True, exist_ok=True)
    dist = work / "dist"
    example = stage_example(work, args.group)
    run("uv", "build", "--all-packages", "--wheel", "--out-dir", dist)
    run("uv", "build", "--wheel", "--out-dir", dist, example)
    constraints = work / "constraints.txt"
    run(
        "uv", "export", "--frozen", "--all-packages", "--no-emit-workspace",
        "--no-hashes", "--no-header", "--no-annotate", "--output-file", constraints,
    )  # fmt: skip

    def wheel(name: str) -> Path:
        [path] = dist.glob(f"{name.replace('-', '_')}-*.whl")
        return path

    native = [wheel("llm-behavior-eval-common"), wheel("llm-behavior-eval-example")]
    if args.group:
        group = normalize(f"llm-behavior-eval-{args.group}")
        if any(member.name == group for member in members()):
            native.append(wheel(group))
    results = {}
    for scenario in ("light", "together", "editable"):
        venv = work / f"venv-{scenario}"
        run("uv", "venv", "--quiet", "--python", sys.executable, venv)
        python = venv / "bin" / "python"
        install = ("uv", "pip", "install", "--quiet", "--python", python)
        if scenario == "editable":
            run(*install, "--no-deps", "-e", ROOT)
            editable = [ROOT / "packages" / "common", example]
            if len(native) > 2:
                editable.append(ROOT / "packages" / args.group)
            run(
                *install,
                "-c",
                constraints,
                *(arg for p in editable for arg in ("-e", p)),
            )
        else:
            run(*install, "-c", constraints, *native)
            if scenario == "together":
                run(*install, "--no-deps", wheel("llm-behavior-eval"))
        results[scenario] = check(scenario, python, work)
        print(f"{scenario}: {results[scenario]['task']}", flush=True)
    if args.keep is None:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()

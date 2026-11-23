"""Run each native distribution's tests in its own light environment.

For every workspace member with tests (``tests/test_inspect_<leaf>.py``, see
``workspace.benchmark_test_files``), runs::

    uv run --isolated --no-editable --package <distribution> --with pytest \
        pytest <its test files>

That environment holds the distribution, its declared dependencies and pytest:
no legacy engine (torch, transformers) and no other benchmark, so a missing
dependency fails here. ``--no-editable`` installs the workspace members as
wheels, as users get them; Inspect then namespaces their tasks as
``llm_behavior_eval/<task>`` (it treats an editable module as part of a
package only when ``llm-behavior-eval`` itself is installed editable).

Usage: ``python scripts/run_package_tests.py [distribution ...]`` (all by
default). Extra arguments after ``--`` go to pytest.
"""

import subprocess
import sys

from workspace import ROOT, benchmark_test_files, members


def command(distribution: str, tests: list[str], pytest_args: list[str]) -> list[str]:
    """The ``uv run`` command that tests one distribution in its light environment.

    Args:
        distribution: The workspace member's name.
        tests: Its test files, relative to the repository root.
        pytest_args: Extra arguments for pytest.

    Returns:
        The command line.
    """
    return [
        "uv",
        "run",
        "--isolated",
        "--no-editable",
        "--locked",
        "--package",
        distribution,
        "--with",
        "pytest",
        "pytest",
        *tests,
        *pytest_args,
    ]


def main() -> None:
    args = sys.argv[1:]
    pytest_args = args[args.index("--") + 1 :] if "--" in args else []
    selected = set(args[: args.index("--")] if "--" in args else args)
    unknown = selected - {member.name for member in members()}
    if unknown:
        # A misspelled name would otherwise run nothing and exit 0.
        raise SystemExit(f"Not workspace members: {', '.join(sorted(unknown))}")
    failed = []
    for member in members():
        if selected and member.name not in selected:
            continue
        tests = [path for path in benchmark_test_files(member) if path.is_file()]
        if not tests:
            print(f"{member.name}: no tests")
            continue
        relative = [path.relative_to(ROOT).as_posix() for path in tests]
        print(f"{member.name}: {' '.join(relative)}", flush=True)
        if subprocess.run(
            command(member.name, relative, pytest_args), cwd=ROOT
        ).returncode:
            failed.append(member.name)
    if failed:
        raise SystemExit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

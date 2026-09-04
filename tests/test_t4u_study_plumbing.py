"""Regression tests for T4U study wrapper plumbing (post-freeze remediation).

These tests cover execution plumbing only. They do not exercise Purple Llama
scientific settings, scorers, or outcomes.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    import pytest

STUDY_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "rsch-76"
    / "t4u-prompt-injection-offon-freeze-20260904"
)
PREFLIGHT_PATH = STUDY_DIR / "preflight.py"
RUN_COMMON_PATH = STUDY_DIR / "run_common.sh"


def _load_preflight() -> ModuleType:
    """Load the study-level preflight module from its artifact path.

    Returns:
        The imported preflight module.
    """
    spec = importlib.util.spec_from_file_location("t4u_preflight", PREFLIGHT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses require the module to be present in sys.modules during class
    # body execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_model_subcommand_is_registered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The resolve-model handler must be wired into the CLI subparsers."""
    preflight = _load_preflight()
    captured: dict[str, object] = {}

    def fake_resolve_snapshot(repo_id: str, revision: str) -> Path:
        captured["repo_id"] = repo_id
        captured["revision"] = revision
        return Path("/tmp/fake-snapshot")

    monkeypatch.setattr(preflight, "resolve_snapshot", fake_resolve_snapshot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight.py",
            "resolve-model",
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        ],
    )

    preflight.main()

    assert captured == {
        "repo_id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }
    assert capsys.readouterr().out.strip() == "/tmp/fake-snapshot"


def test_resolve_model_unknown_command_still_fails() -> None:
    """Unregistered preflight commands must keep failing closed."""
    completed = subprocess.run(
        [sys.executable, str(PREFLIGHT_PATH), "not-a-real-command"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr.lower()


def test_run_arm_honors_caller_supplied_results_root(tmp_path: Path) -> None:
    """A caller-supplied RESULTS_ROOT must reach llm-behavior-eval as --base-output-dir."""
    argv_log = tmp_path / "argv.log"
    results_root = tmp_path / "custom-results"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    llm_stub = fake_bin / "llm-behavior-eval"
    llm_stub.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > '{argv_log}'\n")
    llm_stub.chmod(llm_stub.stat().st_mode | stat.S_IXUSR)

    run_common_text = RUN_COMMON_PATH.read_text()
    start = run_common_text.index("run_arm() {")
    end = run_common_text.index("\n}", start) + 2
    run_arm_fn = run_common_text[start:end]

    harness = tmp_path / "harness.sh"
    harness.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
export PATH="{fake_bin}:$PATH"
RESULTS_ROOT="{results_root}"
BASE_MODEL="fake-model"
BEHAVIOR="prompt-injection"
JUDGE_ENGINE="vllm"
JUDGE_MODEL="fake-judge"
MAX_MODEL_LEN=16384
JUDGE_MAX_MODEL_LEN=16384
MAX_ANSWER_TOKENS=8192
MAX_JUDGE_TOKENS=32
SEED=42
BATCH_SIZE=32
verify_live_max_model_len_for_target() {{ :; }}
{run_arm_fn}
run_arm "plumbing probe" "arm-a" 1 --thinking-off
"""
    )
    harness.chmod(harness.stat().st_mode | stat.S_IXUSR)

    completed = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}"},
    )
    assert completed.returncode == 0, completed.stderr
    argv = argv_log.read_text().splitlines()
    assert "--base-output-dir" in argv
    assert argv[argv.index("--base-output-dir") + 1] == str(results_root)
    assert "--model-output-dir" in argv
    assert argv[argv.index("--model-output-dir") + 1] == "arm-a"


def test_run_common_documents_results_root_assignment() -> None:
    """RESULTS_ROOT must remain overridable and defaulted in the shared wrapper."""
    text = RUN_COMMON_PATH.read_text()
    assert 'RESULTS_ROOT="${RESULTS_ROOT:-' in text
    assert '--base-output-dir "$RESULTS_ROOT"' in text

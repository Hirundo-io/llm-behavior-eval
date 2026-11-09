"""Removal contract (LLM-171): the 0.2.0 package is native-only.

Ported from the earlier removal layer (694f110) and extended to the current
layout: no legacy engine or API, discovery imports no model runtime, the
historical parity scripts refuse to run without the frozen checkout, and tests
read legacy expectations only from frozen data in ``tests/golden/``.
"""

import ast
import subprocess
import sys
import tomllib
from importlib.util import find_spec
from pathlib import Path

import pytest

import llm_behavior_eval
from scripts.parity.historical import FROZEN_COMMIT
from tests.golden import names, provenance

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = {
    "bbq",
    "bloom",
    "unqover",
    "halueval",
    "medhallu",
    "prompt_injection",
    "ccpc",
    "xstest",
    "orbench",
}
REMOVED_EXPORTS = [
    "EvaluateFactory",
    "EvaluationConfig",
    "DatasetConfig",
    "PreprocessConfig",
    "DatasetType",
    "MlflowConfig",
    "VllmConfig",
    "SamplingConfig",
    "FreeTextBiasEvaluator",
    "FreeTextRefusalEvaluator",
    "SYSTEM_PROMPT_DICT",
    "load_tokenizer_with_transformers",
    "load_transformers_model_and_tokenizer",
    "pick_best_dtype",
    "safe_apply_chat_template",
]


def test_discovery_without_optional_runtimes():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'vllm', 'accelerate', 'bitsandbytes'}:
        raise AssertionError('Optional runtime imported during discovery: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import sys
BENCHMARKS = set(sys.argv[1:])
from llm_behavior_eval import _registry, list_dataset_presets
assert list_dataset_presets()
missing = BENCHMARKS - set(_registry.__all__)
assert not missing, missing
import importlib.util
assert importlib.util.find_spec('llm_behavior_eval.evaluation_utils') is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code, *sorted(BENCHMARKS)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_engine_is_gone():
    assert find_spec("llm_behavior_eval.evaluation_utils") is None
    assert not (ROOT / "llm_behavior_eval" / "evaluation_utils").exists()


def test_package_exports_only_the_preset_catalog():
    assert sorted(llm_behavior_eval.__all__) == [
        "DatasetPreset",
        "InspectTaskMapping",
        "expand_dataset_preset",
        "get_dataset_preset",
        "list_dataset_presets",
    ]
    for name in REMOVED_EXPORTS:
        assert not hasattr(llm_behavior_eval, name), name
        with pytest.raises(ImportError):
            exec(f"from llm_behavior_eval import {name}", {})  # noqa: S102


def test_native_only_package_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == llm_behavior_eval.__version__ == "0.2.0"
    assert project["requires-python"] == ">=3.12,<3.14"
    runtime = {"torch", "transformers", "vllm", "accelerate", "bitsandbytes"}
    base = {
        dependency.split(";")[0].split("<")[0].split(">")[0].split("=")[0].strip()
        for dependency in project["dependencies"]
    }
    assert base == {
        "inspect-ai",
        "agent-client-protocol",
        "openai",
        "datasets",
        "typer",
    }
    assert not base & runtime
    assert {"gcs", "azure", "gpt-oss"}.isdisjoint(project["optional-dependencies"])


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "command",
    [
        ["scripts.parity.reuse_legacy", "--primary-root", "x", "--root", "y"]
        + ["--cases", "bbq"],
        ["scripts.parity.qualify_matrix", "legacy", "--case", "bbq"]
        + ["--root", "x", "--answer", "a", "--judge", "j"],
        ["scripts.parity.run_matrix", "--root", "x", "--answer", "a", "--judge", "j"]
        + ["--ccpc-judge", "c", "--provision", "p", "--phases", "legacy"],
        ["scripts.parity.removal_baseline"],
    ],
    ids=["reuse_legacy", "qualify_matrix", "run_matrix", "removal_baseline"],
)
def test_historical_phases_name_the_frozen_checkout(command, tmp_path):
    result = _run(*command)
    assert result.returncode != 0
    assert FROZEN_COMMIT in result.stderr, result.stderr
    assert "Traceback" not in result.stderr
    # Nothing was created: they fail before touching any run root.
    assert not (ROOT / "x").exists()


def _imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules |= {f"{node.module}.{alias.name}" for alias in node.names}
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("llm_behavior_eval.evaluation_utils"):
                modules.add(node.value)
    return modules


@pytest.mark.parametrize(
    "path",
    sorted(
        [*(ROOT / "tests").rglob("*.py"), *(ROOT / "llm_behavior_eval").rglob("*.py")]
    ),
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_no_package_or_test_module_imports_the_legacy_engine(path):
    if path == Path(__file__):
        return
    imported = _imported_modules(path)
    assert not {m for m in imported if "evaluation_utils" in m}, path
    if path.is_relative_to(ROOT / "tests"):
        # Tests read frozen data, never the capture code that needs the engine.
        assert not {m for m in imported if "removal_baseline" in m}, path


def test_golden_data_cites_the_frozen_checkout():
    captured = names()
    assert captured
    modules = {
        path.stem
        for path in (ROOT / "scripts" / "parity" / "removal_baseline").glob("*.py")
        if not path.stem.startswith("_")
    }
    assert set(captured) == modules
    for name in captured:
        source = provenance(name)
        assert source["commit"] == FROZEN_COMMIT, name
        assert source["capture"] == f"python -m scripts.parity.removal_baseline {name}"
        assert source["sources"], name
        assert all(s.startswith("llm_behavior_eval.") for s in source["sources"]), name

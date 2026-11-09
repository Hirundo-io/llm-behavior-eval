"""Removal boundary: native discovery must never load optional model engines."""

import subprocess
import sys


def test_discovery_without_optional_runtimes():
    code = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'vllm', 'accelerate', 'bitsandbytes'}:
        raise AssertionError('Optional runtime imported during discovery: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from llm_behavior_eval import _registry, list_dataset_presets
assert list_dataset_presets()
assert {'bbq', 'bloom', 'unqover', 'ccpc', 'halueval', 'medhallu', 'prompt_injection', 'xstest', 'orbench'} <= set(_registry.__all__)
import importlib.util
assert importlib.util.find_spec('llm_behavior_eval.evaluation_utils') is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr

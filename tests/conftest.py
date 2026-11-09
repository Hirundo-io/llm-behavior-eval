import sys
from pathlib import Path

import pytest

# Console pytest with importlib mode does not add the repository to sys.path.
# Append it so examples/ and tests/ helpers remain importable, but the installed
# dist-info takes precedence over setuptools' source-tree egg-info (which has no
# PEP 610 editable-install metadata). Inspect uses that metadata to namespace tasks.
sys.path.append(str(Path(__file__).resolve().parents[1]))

# inspect-ai is only installed on Python >= 3.11 (see pyproject.toml markers).
collect_ignore_glob = ["test_inspect_*.py"] if sys.version_info < (3, 11) else []


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "vllm_engine_test: marks tests that target the vLLM eval engine stubs",
    )
    config.addinivalue_line(
        "markers",
        "transformers_engine_test: marks tests that target the transformers eval engine stubs",
    )

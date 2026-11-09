import sys

import pytest

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

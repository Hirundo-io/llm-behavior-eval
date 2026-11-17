import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from inspect_helpers import LocalOpenAIJudge

# inspect-ai is only installed on Python >= 3.11 (see pyproject.toml markers).
# Generic Inspect test helpers live in inspect_helpers.py, which imports
# inspect_ai, so fixtures here import it lazily.
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


@pytest.fixture
def openai_judge() -> Iterator["LocalOpenAIJudge"]:
    """A loopback OpenAI-compatible judge (``inspect_helpers.local_openai_judge``)."""
    from inspect_helpers import local_openai_judge

    with local_openai_judge() as judge:
        yield judge


@pytest.fixture
def hf_dataset_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep Inspect's ``hf_dataset`` split cache inside ``tmp_path``.

    ``hf_dataset`` saves every split it loads under Inspect's user cache
    directory; tests redirect it so they neither read a stale copy nor leave
    one behind.
    """
    from inspect_ai.dataset._sources import hf as hf_source

    cache = tmp_path / "inspect_cache"
    monkeypatch.setattr(
        hf_source, "inspect_cache_dir", lambda subdir, create=True: cache / subdir
    )
    return cache

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# inspect-ai is only installed on Python >= 3.11 (see pyproject.toml markers),
# so the native Inspect tests are not collected on 3.10. Fixtures below import
# Inspect lazily for the same reason.
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


@pytest.fixture
def hub(
    monkeypatch: pytest.MonkeyPatch, hf_dataset_cache: Path
) -> Callable[[list[dict[str, Any]]], list[Any]]:
    """Serve rows as the Hub dataset; returns the ``load_dataset`` calls.

    Only Hub transport is replaced: Inspect's ``hf_dataset`` conversion,
    shuffling and IDs run for real.
    """
    from datasets import Dataset

    calls: list[Any] = []

    def install(rows: list[dict[str, Any]]) -> list[Any]:
        raw = Dataset.from_list(rows)

        def load(*args: Any, **kwargs: Any) -> Dataset:
            calls.append((args, kwargs))
            return raw

        monkeypatch.setattr("datasets.load_dataset", load)
        return calls

    return install

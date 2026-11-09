"""Observational response capture for Inspect score, which omits raw API calls."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from inspect_ai.model import Model


@contextmanager
def capture_judge_responses(model: Model, path: Path) -> Generator[None, None, None]:
    """Preserve public provider response-hook data; never alter requests/results."""
    api: Any = model.api
    original = api.on_response
    responses: dict[str, dict] = {}

    def observe(response: dict) -> None:
        original(response)
        ident = response["id"]
        if ident in responses:
            raise ValueError("Duplicate provider response ID")
        responses[ident] = response

    api.on_response = observe
    try:
        yield
    finally:
        api.on_response = original
        path.write_text(json.dumps(responses, indent=2))

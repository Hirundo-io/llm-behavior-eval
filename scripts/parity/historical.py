"""Explicit boundary for qualification phases that require the frozen oracle."""

from importlib import import_module
from typing import Any

FROZEN_COMMIT = "5cce36cba76a99271f234973853216cbb511a709"


def legacy_module(name: str) -> Any:
    """Load a historical API only in the separately installed frozen checkout."""
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"This historical phase requires checkout {FROZEN_COMMIT} and its "
            "comparison environment. See scripts/parity/README.md. "
            "The current package contains no legacy engine."
        ) from exc

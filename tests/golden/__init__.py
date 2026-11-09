"""Frozen outputs of the removed legacy engine: data only, no engine code.

``golden(name)`` returns the ``values`` of ``tests/golden/<name>.json``. Each
file records the commit it was captured at and the legacy symbols it read; see
README.md. Never regenerate these from the native code under test.
"""

import json
from functools import cache
from pathlib import Path
from typing import Any

GOLDEN = Path(__file__).parent


@cache
def _load(name: str) -> dict[str, Any]:
    return json.loads((GOLDEN / f"{name}.json").read_text())


def golden(name: str) -> Any:
    """The frozen legacy values of ``tests/golden/<name>.json``."""
    return _load(name)["values"]


def provenance(name: str) -> dict[str, Any]:
    """Where ``tests/golden/<name>.json`` came from."""
    return _load(name)["provenance"]


def names() -> list[str]:
    return sorted(path.stem for path in GOLDEN.glob("*.json"))

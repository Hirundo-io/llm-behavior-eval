"""Boundary for parity phases that need the frozen pre-removal checkout.

Version 0.2.0 contains no legacy engine. The legacy phase of
``qualify_matrix.py``, ``reuse_legacy.py`` and the removal-baseline capture
(``python -m scripts.parity.removal_baseline``) run only in a separate checkout
of ``FROZEN_COMMIT``, the last commit with the legacy engine. See
scripts/parity/README.md.
"""

from importlib import import_module
from importlib.util import find_spec
from typing import Any

FROZEN_COMMIT = "d8c80154c1e6ae90e2896811685d7c1a9bf89e68"
LEGACY_PACKAGE = "llm_behavior_eval.evaluation_utils"


def require_frozen_checkout() -> None:
    """Fail clearly unless the legacy engine of the frozen checkout is importable."""
    if find_spec("llm_behavior_eval") is None or find_spec(LEGACY_PACKAGE) is None:
        raise SystemExit(
            "This historical phase needs the legacy engine, which llm-behavior-eval "
            f"0.2.0 removed. Run it from a checkout of {FROZEN_COMMIT} with that "
            "checkout's locked environment (uv sync --locked --extra vllm). See "
            "scripts/parity/README.md."
        )


def legacy_module(name: str) -> Any:
    """Import a module of the legacy engine from the frozen checkout."""
    if not name.startswith(LEGACY_PACKAGE):
        raise ValueError(f"{name} is not a legacy engine module")
    require_frozen_checkout()
    return import_module(name)

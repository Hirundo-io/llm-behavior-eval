"""Use eagerly readable JSON logs for deterministic scorer fixtures.

Reporting tests separately exercise the default binary .eval format.
"""

from typing import Any

from inspect_ai import eval as _eval


def inspect_eval(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("log_format", "json")
    return _eval(*args, **kwargs)

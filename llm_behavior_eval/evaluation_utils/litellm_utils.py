from __future__ import annotations

import logging
import os
from typing import Any, cast


def suppress_litellm_logging(litellm_module: Any) -> None:
    """Suppress verbose LiteLLM logs unless debug mode is explicitly enabled."""
    if os.getenv("LITELLM_DEBUG"):
        return
    cast("Any", litellm_module).suppress_debug_info = True
    logging.getLogger("litellm").setLevel(logging.WARNING)


def call_litellm_encode(
    litellm_module: Any,
    model_name: str,
    text: str,
) -> list[Any] | None:
    """Encode text to tokens using LiteLLM. Returns None if encoding fails."""
    try:
        tokens = litellm_module.encode(model=model_name, text=text)
        return list(tokens) if tokens is not None else None
    except Exception:
        return None


def call_litellm_decode(
    litellm_module: Any,
    model_name: str,
    tokens: list[Any],
) -> str | None:
    """Decode tokens to text using LiteLLM. Returns None if decoding fails."""
    try:
        decoded = litellm_module.decode(model=model_name, tokens=tokens)
        return str(decoded) if decoded is not None else None
    except Exception:
        return None


def try_trim_messages_with_litellm(
    litellm_module: Any,
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> list[dict[str, str]] | None:
    """Trim messages to fit within max_tokens using LiteLLM. Returns None if trimming fails."""
    try:
        from litellm.utils import trim_messages

        trimmed = trim_messages(messages, model=model_name, max_tokens=max_tokens)
        if isinstance(trimmed, list) and all(isinstance(m, dict) for m in trimmed):
            return cast("list[dict[str, str]]", trimmed)
        return None
    except Exception:
        return None

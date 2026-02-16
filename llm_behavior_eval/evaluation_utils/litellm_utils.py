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


def coerce_token_list(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        for key in ("tokens", "token_ids", "input_ids"):
            candidate = value.get(key)
            if isinstance(candidate, (list, tuple)):
                return list(candidate)
    for attr in ("tokens", "token_ids", "input_ids"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (list, tuple)):
            return list(candidate)
    return None


def call_litellm_encode(
    litellm_module: Any,
    model_name: str,
    text: str,
) -> list[Any] | None:
    encode_fn = getattr(litellm_module, "encode", None)
    if not callable(encode_fn):
        return None

    kwargs_variants: list[dict[str, Any]] = [
        {"model": model_name, "text": text},
        {"model_name": model_name, "text": text},
        {"model": model_name, "input": text},
    ]
    for kwargs in kwargs_variants:
        try:
            encoded = encode_fn(**kwargs)
        except TypeError:
            continue
        except Exception:
            return None
        return coerce_token_list(encoded)

    args_variants: list[tuple[Any, ...]] = [
        (model_name, text),
        (text,),
    ]
    for args in args_variants:
        try:
            encoded = encode_fn(*args)
        except TypeError:
            continue
        except Exception:
            return None
        return coerce_token_list(encoded)

    return None


def call_litellm_decode(
    litellm_module: Any,
    model_name: str,
    tokens: list[Any],
) -> str | None:
    decode_fn = getattr(litellm_module, "decode", None)
    if not callable(decode_fn):
        return None

    kwargs_variants: list[dict[str, Any]] = [
        {"model": model_name, "tokens": tokens},
        {"model_name": model_name, "tokens": tokens},
        {"model": model_name, "token_ids": tokens},
    ]
    for kwargs in kwargs_variants:
        try:
            decoded = decode_fn(**kwargs)
        except TypeError:
            continue
        except Exception:
            return None
        if decoded is not None:
            return str(decoded)

    args_variants: list[tuple[Any, ...]] = [
        (model_name, tokens),
        (tokens,),
    ]
    for args in args_variants:
        try:
            decoded = decode_fn(*args)
        except TypeError:
            continue
        except Exception:
            return None
        if decoded is not None:
            return str(decoded)

    return None


def normalize_trimmed_messages_output(
    trimmed: Any,
) -> list[dict[str, str]] | None:
    if isinstance(trimmed, list) and all(isinstance(item, dict) for item in trimmed):
        return cast("list[dict[str, str]]", trimmed)
    if isinstance(trimmed, dict):
        nested = trimmed.get("messages")
        if isinstance(nested, list) and all(isinstance(item, dict) for item in nested):
            return cast("list[dict[str, str]]", nested)
    return None


def try_trim_messages_with_litellm(
    litellm_module: Any,
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> list[dict[str, str]] | None:
    trim_fn = getattr(litellm_module, "trim_messages", None)
    if not callable(trim_fn):
        return None

    kwargs_variants: list[dict[str, Any]] = [
        {
            "messages": messages,
            "model": model_name,
            "max_tokens": max_tokens,
        },
        {
            "messages": messages,
            "model": model_name,
            "max_input_tokens": max_tokens,
        },
        {
            "messages": messages,
            "model_name": model_name,
            "max_tokens": max_tokens,
        },
    ]
    for kwargs in kwargs_variants:
        try:
            trimmed = trim_fn(**kwargs)
        except TypeError:
            continue
        except Exception:
            return None
        normalized = normalize_trimmed_messages_output(trimmed)
        if normalized is not None:
            return normalized

    args_variants: list[tuple[Any, ...]] = [
        (messages, model_name, max_tokens),
        (messages, model_name),
        (messages,),
    ]
    for args in args_variants:
        try:
            trimmed = trim_fn(*args)
        except TypeError:
            continue
        except Exception:
            return None
        normalized = normalize_trimmed_messages_output(trimmed)
        if normalized is not None:
            return normalized

    return None

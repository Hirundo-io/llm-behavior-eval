"""Redacted Azure response diagnostics for CCPC execution attempt 3.

This module deliberately does not construct a provider client or log response
content.  It captures only structural properties needed to diagnose empty or
otherwise unparsable structured responses.  The caller must persist the record
returned by :func:`capture_then_parse_json_answer` before it uses the returned
answer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

_MISSING = object()
_ALTERNATIVE_VISIBLE_FIELDS = (
    "audio",
    "reasoning_content",
    "reasoning",
    "output_text",
    "parsed",
)
_FILTER_VALUE_KEYS = frozenset(
    {"category", "code", "detected", "filtered", "severity", "status"}
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read an SDK-model field without coercing its potentially sensitive value."""
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_sequence(value: Any) -> list[Any]:
    """Return a sequence-like SDK field as a list without iterating strings."""
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _model_fields_set(value: Any) -> list[str] | None:
    """Return a stable Pydantic field-set representation when it is available."""
    fields = _field(value, "model_fields_set", _MISSING)
    if fields is _MISSING:
        return None
    try:
        return sorted(str(field) for field in fields)
    except TypeError:
        return [str(fields)]


def _type_and_length(value: Any) -> dict[str, Any]:
    """Describe a value without including scalar content."""
    result: dict[str, Any] = {
        "present": value is not None,
        "type": type(value).__name__,
    }
    if value is None:
        result["length"] = None
        return result
    try:
        result["length"] = len(value)
    except TypeError:
        result["length"] = None
    return result


def redacted_structural_dump(value: Any, *, max_depth: int = 5) -> dict[str, Any]:
    """Describe an SDK response by keys, types, and lengths, never scalar values."""

    def dump(item: Any, depth: int) -> dict[str, Any]:
        summary = _type_and_length(item)
        if (
            depth >= max_depth
            or item is None
            or isinstance(item, (str, bytes, bytearray))
        ):
            return summary
        if isinstance(item, Mapping):
            keys = sorted(str(key) for key in item)
            summary["keys"] = keys
            summary["fields"] = {
                str(key): dump(item[key], depth + 1) for key in sorted(item, key=str)
            }
            return summary
        if isinstance(item, (list, tuple, set, frozenset)):
            summary["items"] = [dump(child, depth + 1) for child in list(item)[:20]]
            summary["items_truncated"] = len(item) > 20
            return summary
        fields = _model_fields_set(item)
        if fields:
            summary["model_fields_set"] = fields
            summary["fields"] = {
                name: dump(_field(item, name), depth + 1) for name in fields
            }
        return summary

    return dump(value, 0)


def _safe_filter_metadata(value: Any, *, depth: int = 0) -> Any:
    """Keep known filter status fields while excluding arbitrary provider text."""
    if depth > 4 or value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_safe_filter_metadata(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if name in _FILTER_VALUE_KEYS and isinstance(
                child, (bool, int, float, str)
            ):
                result[name] = child
            elif isinstance(child, (Mapping, list, tuple)):
                nested = _safe_filter_metadata(child, depth=depth + 1)
                if nested not in ({}, [], None):
                    result[name] = nested
        return result
    fields = _model_fields_set(value) or []
    return _safe_filter_metadata(
        {name: _field(value, name) for name in fields}, depth=depth
    )


def _request_ids(response: Any) -> dict[str, str | None]:
    """Collect only provider request identifiers from common SDK response locations."""
    result: dict[str, str | None] = {
        "response_id": _field(response, "id"),
        "request_id": _field(response, "_request_id") or _field(response, "request_id"),
    }
    headers = _field(response, "headers") or _field(
        _field(response, "_response"), "headers"
    )
    if headers is not None:
        for name in ("x-request-id", "apim-request-id", "x-ms-request-id"):
            value = _field(headers, name)
            if value is not None:
                result[name] = str(value)
    return result


def capture_response_shape(
    response: Any, *, source_id: int, attempt: int
) -> dict[str, Any]:
    """Capture a redacted response-shape record before the content is parsed.

    The result contains no message content, generated answer, or raw SDK dump.
    It is safe to persist as a diagnostic record, subject to the caller's usual
    handling for provider request IDs.
    """
    choices = _as_sequence(_field(response, "choices"))
    choice = choices[0] if choices else None
    message = _field(choice, "message")
    content = _field(message, "content")
    content_text = content if isinstance(content, str) else ""
    refusal = _field(message, "refusal")
    tool_calls = _field(message, "tool_calls")
    annotations = _field(message, "annotations")
    alternative_fields = {
        name: _type_and_length(_field(message, name))
        for name in _ALTERNATIVE_VISIBLE_FIELDS
        if _field(message, name, _MISSING) is not _MISSING
    }
    usage = _field(response, "usage")
    completion_details = _field(usage, "completion_tokens_details")
    reasoning_tokens = _field(completion_details, "reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _field(usage, "reasoning_tokens")
    prompt_filters = _field(response, "prompt_filter_results")
    choice_filters = _field(choice, "content_filter_results")
    return {
        "source_id": source_id,
        "attempt": attempt,
        "response_model": _field(response, "model"),
        "response_type": type(response).__name__,
        "request_ids": _request_ids(response),
        "finish_reason": _field(choice, "finish_reason"),
        "choice_count": len(choices),
        "message_content_is_none": content is None,
        "message_content_length": len(content_text),
        "message_content_first_non_whitespace_character": next(
            (character for character in content_text if not character.isspace()), None
        ),
        "message_refusal": _type_and_length(refusal),
        "message_tool_calls": {
            **_type_and_length(tool_calls),
            "count": len(_as_sequence(tool_calls)),
        },
        "message_function_call_present": _field(message, "function_call") is not None,
        "message_annotations": {
            **_type_and_length(annotations),
            "count": len(_as_sequence(annotations)),
        },
        "message_model_fields_set": _model_fields_set(message),
        "response_model_fields_set": _model_fields_set(response),
        "usage": {
            "prompt_tokens": _field(usage, "prompt_tokens"),
            "completion_tokens": _field(usage, "completion_tokens"),
            "reasoning_tokens": reasoning_tokens,
        },
        "azure_content_filter": {
            "prompt": _safe_filter_metadata(prompt_filters),
            "completion": _safe_filter_metadata(choice_filters),
            "prompt_structure": redacted_structural_dump(prompt_filters),
            "completion_structure": redacted_structural_dump(choice_filters),
        },
        "alternative_visible_output_fields": alternative_fields,
        "redacted_structure": redacted_structural_dump(response),
        "parse_valid": None,
        "parser_exception_class": None,
        "parser_exception_message": None,
    }


def _parse_json_answer_detailed(
    content: Any,
) -> tuple[bool, str | None, Exception | None]:
    """Parse the frozen ``{"answer": string}`` contract without diagnostic output."""
    try:
        parsed = json.loads(content if isinstance(content, str) else "")
    except Exception as exc:  # JSON parser exceptions are recorded by the wrapper.
        return False, None, exc
    answer = (
        parsed.get("answer")
        if isinstance(parsed, dict) and set(parsed) == {"answer"}
        else None
    )
    valid = isinstance(answer, str) and bool(answer.strip())
    return valid, answer.strip() if valid else None, None


def parse_json_answer(content: Any) -> tuple[bool, str | None]:
    """Parse the fixed answer object, returning a stripped answer only when valid."""
    valid, answer, _ = _parse_json_answer_detailed(content)
    return valid, answer


def attach_parse_diagnostic(record: dict[str, Any], content: Any) -> str | None:
    """Attach parse outcome after a previously persisted response-shape capture.

    The return value is intentionally transient: callers must not persist it.
    """
    valid, answer, exception = _parse_json_answer_detailed(content)
    record["parse_valid"] = valid
    if exception is not None:
        record["parser_exception_class"] = type(exception).__name__
        record["parser_exception_message"] = str(exception)
    return answer


def capture_then_parse_json_answer(
    response: Any, *, source_id: int, attempt: int
) -> tuple[dict[str, Any], str | None]:
    """Capture response shape first, then parse content and attach parse diagnostics."""
    record = capture_response_shape(response, source_id=source_id, attempt=attempt)
    choices = _as_sequence(_field(response, "choices"))
    message = _field(choices[0], "message") if choices else None
    answer = attach_parse_diagnostic(record, _field(message, "content"))
    return record, answer

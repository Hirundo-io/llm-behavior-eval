"""Synthetic coverage for the execution-3 response-shape diagnostic helper."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from analysis.ccpc_bench_prereg.execution_3_diagnostics import (
    capture_then_parse_json_answer,
)


def _response(
    content: str | None,
    *,
    finish_reason: str = "stop",
    refusal: str | None = None,
    content_filter_results: object = None,
) -> SimpleNamespace:
    """Build a small SDK-shaped response without making any provider request."""
    message = SimpleNamespace(
        content=content,
        refusal=refusal,
        tool_calls=[],
        function_call=None,
        annotations=[],
        model_fields_set={"content", "refusal", "tool_calls", "annotations"},
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
        content_filter_results=content_filter_results,
    )
    return SimpleNamespace(
        id="chatcmpl-test",
        _request_id="request-test",
        model="gpt-5-2025-08-07",
        choices=[choice],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=3,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
        prompt_filter_results=[{"hate": {"filtered": False, "severity": "safe"}}],
        model_fields_set={"model", "choices", "usage"},
    )


def test_provider_metadata_survives_json_decode_error() -> None:
    response = _response("not-json", finish_reason="length")

    record, answer = capture_then_parse_json_answer(response, source_id=17, attempt=1)

    assert answer is None
    assert record["parse_valid"] is False
    assert record["parser_exception_class"] == "JSONDecodeError"
    assert record["response_model"] == "gpt-5-2025-08-07"
    assert record["finish_reason"] == "length"
    assert record["request_ids"]["request_id"] == "request-test"
    assert record["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "reasoning_tokens": 2,
    }


def test_zero_reasoning_tokens_are_preserved() -> None:
    response = _response("{}")
    response.usage.completion_tokens_details.reasoning_tokens = 0

    record, _ = capture_then_parse_json_answer(response, source_id=18, attempt=1)

    assert record["usage"]["reasoning_tokens"] == 0


def test_empty_content_is_distinguishable_from_malformed_nonempty_json() -> None:
    empty, _ = capture_then_parse_json_answer(_response(None), source_id=1, attempt=1)
    malformed, _ = capture_then_parse_json_answer(
        _response("{not json"), source_id=2, attempt=1
    )

    assert empty["message_content_is_none"] is True
    assert empty["message_content_length"] == 0
    assert empty["message_content_first_non_whitespace_character"] is None
    assert malformed["message_content_is_none"] is False
    assert malformed["message_content_length"] == len("{not json")
    assert malformed["message_content_first_non_whitespace_character"] == "{"
    assert empty["parser_exception_class"] == "JSONDecodeError"
    assert malformed["parser_exception_class"] == "JSONDecodeError"


def test_refusal_and_content_filter_metadata_are_captured_without_response_text() -> (
    None
):
    secret = "DO NOT EMIT THIS GENERATED ANSWER"
    response = _response(
        json.dumps({"answer": secret}),
        refusal="Policy refusal",
        finish_reason="content_filter",
        content_filter_results={"hate": {"filtered": True, "severity": "high"}},
    )

    record, answer = capture_then_parse_json_answer(response, source_id=5, attempt=1)

    assert answer == secret
    assert record["parse_valid"] is True
    assert record["message_refusal"] == {
        "present": True,
        "type": "str",
        "length": len("Policy refusal"),
    }
    assert record["finish_reason"] == "content_filter"
    assert record["azure_content_filter"]["completion"] == {
        "hate": {"filtered": True, "severity": "high"}
    }
    assert secret not in json.dumps(record)
    assert "Policy refusal" not in json.dumps(record)


@pytest.mark.parametrize("finish_reason", ["length", "stop", "content_filter"])
def test_finish_reasons_remain_distinguishable(finish_reason: str) -> None:
    record, _ = capture_then_parse_json_answer(
        _response("{}", finish_reason=finish_reason), source_id=7, attempt=2
    )

    assert record["finish_reason"] == finish_reason

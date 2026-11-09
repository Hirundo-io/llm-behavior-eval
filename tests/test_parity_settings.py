"""Audit corruption checks; constructed settings are not model evidence."""

from typing import Any

import pytest
from inspect_ai.model import GenerateConfig

from scripts.parity.settings import audit_settings


def settings(
    answer_budget: int = 8192, judge_budget: int = 4096, thinking: bool = False
) -> tuple[dict[str, Any], list[tuple[str, GenerateConfig]]]:
    budgets = (answer_budget, judge_budget)
    legacy = dict(
        max_answer_tokens=budgets[0],
        max_judge_tokens=budgets[1],
        batch_size=1,
        judge_batch_size=1,
        sample=False,
        sample_judge=False,
        use_4bit=False,
        use_4bit_judge=False,
        enable_thinking=thinking,
        inference_engine="vllm",
        sampling_config={"seed": 42},
    )
    native = []
    for role, budget in zip(("answer", "judge"), budgets, strict=True):
        body: dict[str, Any] = dict(
            return_token_ids=True,
            add_special_tokens=True,
            chat_template_kwargs={
                "enable_thinking": thinking if role == "answer" else False
            },
        )
        if role == "answer":
            body["truncate_prompt_tokens"] = 1024
        native.append(
            (
                role,
                GenerateConfig(
                    max_tokens=budget,
                    temperature=0,
                    seed=42,
                    max_connections=1,
                    extra_body=body,
                ),
            )
        )
    return legacy, native


@pytest.mark.parametrize(
    "family,answer_budget,judge_budget,thinking",
    [
        ("bbq", 128, 32, False),
        ("unqover", 128, 32, False),
        ("bloom", 128, 32, False),
        ("halueval", 128, 32, False),
        ("medhallu", 128, 32, False),
        ("prompt_injection", 128, 32, False),
        ("ccpc", 8192, 4096, False),
        ("xstest", 2048, 128, True),
        ("orbench", 2048, 128, True),
    ],
)
def test_audit_all_production_policies(family, answer_budget, judge_budget, thinking):
    legacy, native = settings(answer_budget, judge_budget, thinking)
    result = audit_settings(family, legacy, native)
    assert result["native_requests_checked"] == {"answer": 1, "judge": 1}
    assert result["judge_prompt_cap"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 999),
        ("temperature", 1),
        ("seed", 43),
        ("max_connections", 2),
    ],
)
def test_reject_native_request_changes(field, value):
    legacy, native = settings()
    native[1] = ("judge", native[1][1].model_copy(update={field: value}))
    with pytest.raises(ValueError, match=field):
        audit_settings("ccpc", legacy, native)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_answer_tokens", 128),
        ("max_judge_tokens", 32),
        ("enable_thinking", True),
        ("sample", True),
        ("batch_size", 2),
    ],
)
def test_reject_retained_configuration_changes(field, value):
    legacy, native = settings()
    legacy[field] = value
    with pytest.raises(ValueError, match=field):
        audit_settings("ccpc", legacy, native)


def test_reject_answer_cap_on_judge():
    legacy, native = settings(2048, 128, True)
    assert native[1][1].extra_body is not None
    native[1][1].extra_body["truncate_prompt_tokens"] = 1024
    with pytest.raises(ValueError, match="judge must not carry"):
        audit_settings("xstest", legacy, native)


def test_reject_changed_thinking_policy():
    legacy, native = settings(2048, 128, True)
    assert native[0][1].extra_body is not None
    native[0][1].extra_body["chat_template_kwargs"]["enable_thinking"] = False
    with pytest.raises(ValueError, match="answer thinking"):
        audit_settings("xstest", legacy, native)


def test_zero_judges_are_reported_without_inventing_coverage():
    legacy, native = settings(128, 32)
    result = audit_settings("medhallu", legacy, native[:1])
    assert result["native_requests_checked"] == {"answer": 1, "judge": 0}

import pytest
pytest.importorskip("torch")
import torch

from llm_behavior_eval.evaluation_utils.util_functions import (
    pick_best_dtype,
    safe_apply_chat_template,
)


class StubTokenizer:
    def __init__(self, name: str, template: str) -> None:
        self.name_or_path = name
        self.chat_template = template

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ):
        # Simple join of role and content for testing purposes
        return "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )


def test_pick_best_dtype_cpu() -> None:
    assert pick_best_dtype("cpu") == torch.float32


def test_pick_best_dtype_cuda_bf16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    dtype = pick_best_dtype("cuda", prefer_bf16=True)
    assert dtype == torch.bfloat16


def test_safe_apply_chat_template_merges_system_message() -> None:
    tokenizer = StubTokenizer("google/gemma-2b", "System role not supported")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    formatted = safe_apply_chat_template(tokenizer, messages)
    assert "system" in formatted and "user" in formatted

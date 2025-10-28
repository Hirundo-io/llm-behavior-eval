from typing import TYPE_CHECKING, cast

import pytest
import torch

from llm_behavior_eval.evaluation_utils.util_functions import (
    build_vllm_prompt_token_ids,
    pick_best_dtype,
    safe_apply_chat_template,
    torch_dtype_to_str,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class StubTokenizer:
    def __init__(self, name: str, template: str) -> None:
        self.name_or_path = name
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
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
    formatted = safe_apply_chat_template(
        cast("PreTrainedTokenizerBase", tokenizer), messages
    )
    assert "system" in formatted and "user" in formatted


def test_torch_dtype_to_str_supported() -> None:
    assert torch_dtype_to_str(torch.float16) == "float16"
    assert torch_dtype_to_str(torch.bfloat16) == "bfloat16"
    assert torch_dtype_to_str(torch.float32) == "float32"


def test_torch_dtype_to_str_unsupported() -> None:
    with pytest.raises(ValueError):
        torch_dtype_to_str(torch.float64)


def test_build_vllm_prompt_token_ids_strips_padding() -> None:
    input_ids = torch.tensor([[0, 11, 12, 13], [0, 0, 21, 22]])
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])

    prompts = build_vllm_prompt_token_ids(input_ids, attention_mask)

    assert prompts == [[11, 12, 13], [21, 22]]


def test_build_vllm_prompt_token_ids_validates_shape() -> None:
    input_ids = torch.zeros((2, 3), dtype=torch.long)
    attention_mask = torch.zeros((2, 2), dtype=torch.long)

    with pytest.raises(ValueError):
        build_vllm_prompt_token_ids(input_ids, attention_mask)

from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from llm_behavior_eval.evaluation_utils.util_functions import (
    build_vllm_prompt_token_ids,
    is_model_multimodal,
    pick_best_dtype,
    safe_apply_chat_template,
    torch_dtype_to_str,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class MockConfig:
    """A simple mock config object for testing."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": self.model_type}


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


def test_safe_apply_chat_template_appends_max_answer_instruction() -> None:
    tokenizer = StubTokenizer("some/model", "generic template")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Answer the question."},
    ]
    formatted = safe_apply_chat_template(
        cast("PreTrainedTokenizerBase", tokenizer),
        messages,
        max_answer_tokens=42,
        pass_max_answer_tokens=True,
    )
    assert "Respond in no more than 42 tokens." in formatted
    assert "You are a helpful assistant." in formatted
    assert "Answer the question." in formatted


def test_safe_apply_chat_template_does_not_append_without_flag_or_value() -> None:
    tokenizer = StubTokenizer("some/model", "generic template")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Answer the question."},
    ]
    without_flag = safe_apply_chat_template(
        cast("PreTrainedTokenizerBase", tokenizer),
        [m.copy() for m in messages],
        max_answer_tokens=42,
        pass_max_answer_tokens=False,
    )
    without_value = safe_apply_chat_template(
        cast("PreTrainedTokenizerBase", tokenizer),
        [m.copy() for m in messages],
        max_answer_tokens=None,
        pass_max_answer_tokens=True,
    )
    assert "Respond in no more than 42 tokens." not in without_flag
    assert "Respond in no more than" not in without_value


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


@pytest.fixture
def mock_auto_config_remote_fallback(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fixture that mocks AutoConfig to simulate remote fallback behavior."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def mock_from_pretrained(*args: Any, **kwargs: Any) -> MockConfig:
        calls.append((args, kwargs))
        if len(calls) == 1:
            # First call raises exception to trigger remote fallback
            raise Exception("local_files_only=True failed")
        # Second call (remote fallback) returns config
        return MockConfig("llama")

    class MockAutoConfig:
        from_pretrained = staticmethod(mock_from_pretrained)

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.util_functions.AutoConfig",
        MockAutoConfig,
    )
    return {"calls": calls}


@pytest.fixture
def mock_auto_config_local_load(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fixture that mocks AutoConfig to simulate local load behavior."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def mock_from_pretrained(*args: Any, **kwargs: Any) -> MockConfig:
        calls.append((args, kwargs))
        return MockConfig("llama")

    class MockAutoConfig:
        from_pretrained = staticmethod(mock_from_pretrained)

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.util_functions.AutoConfig",
        MockAutoConfig,
    )
    return {"calls": calls}


def test_is_model_multimodal_passes_token_on_remote_fallback(
    mock_auto_config_remote_fallback: dict[str, Any],
) -> None:
    """Test that is_model_multimodal passes the token parameter on remote fallback."""
    # Test with token parameter
    result = is_model_multimodal(
        "test/model", trust_remote_code=False, token="test_token"
    )

    # Verify that both calls occurred (local then remote)
    calls = mock_auto_config_remote_fallback["calls"]
    assert len(calls) == 2

    # Check the remote call (second call) includes the token
    remote_call = calls[1]
    assert remote_call[1].get("token") == "test_token"
    assert result is False


def test_is_model_multimodal_passes_token_on_local_load(
    mock_auto_config_local_load: dict[str, Any],
) -> None:
    """Test that is_model_multimodal passes the token parameter when loading locally."""
    # Test with token parameter
    result = is_model_multimodal(
        "test/model", trust_remote_code=False, token="test_token"
    )

    # Verify that only the local call occurred
    calls = mock_auto_config_local_load["calls"]
    assert len(calls) == 1

    local_call = calls[0]
    assert local_call[1].get("token") == "test_token"
    assert result is False

"""Inspect's real HF provider with stub weights: the documented parity -M flags.

Parity runs on HF pass ``-M tokenizer_call_args='{truncation: true, max_length:
1024}'``. These tests check that the HF provider then feeds the model exactly
the unpadded token IDs the legacy engine did, for short and over-long prompts.
CPU boundary evidence only; HF GPU score equality is not qualified.
"""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from transformers import AutoModelForCausalLM

from llm_behavior_eval.common.preprocessing import (
    LEGACY_PROMPT_MAX_LENGTH,
    load_tokenizer,
    render_chat,
    roundtrip,
)
from tests.legacy_tokenizers import pinned_tokenizer, tiny_tokenizer

# The -M tokenizer_call_args value documented in every parity.yaml.
PARITY_TOKENIZER_CALL_ARGS = {"truncation": True, "max_length": 1024}


@pytest.fixture(params=["tiny", "gemma3", "qwen3"])
def hf(request, tmp_path, monkeypatch):
    path = (
        tiny_tokenizer(tmp_path)
        if request.param == "tiny"
        else pinned_tokenizer(request.param)
    )
    captured = []

    class Weights:
        device = "cpu"
        config = SimpleNamespace(model_type="stub")

        def generate(self, input_ids, attention_mask, **kwargs):
            captured.append((input_ids.tolist(), attention_mask.tolist()))
            return SimpleNamespace(
                sequences=torch.cat([input_ids, input_ids[:, -1:]], dim=1),
                logits=None,
                hidden_states=None,
            )

    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: Weights()
    )

    def model(thinking: bool):
        return get_model(
            "hf/local",
            model_path=str(path),
            device="cpu",
            do_sample=False,
            enable_thinking=thinking,
            tokenizer_call_args=dict(PARITY_TOKENIZER_CALL_ARGS),
            memoize=False,
        )

    return model, load_tokenizer(str(path)), captured


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("long", [False, True])
def test_hf_parity_flags_feed_legacy_token_ids(hf, thinking, long):
    model_factory, tokenizer, captured = hf
    model = model_factory(thinking)
    messages: list[ChatMessage] = [
        ChatMessageSystem(content="Be helpful."),
        ChatMessageUser(content="question " * (1600 if long else 1) + "\n"),
    ]
    rendered = render_chat(
        tokenizer,
        [{"role": m.role, "content": m.text} for m in messages],
        legacy_thinking=thinking,
    )
    expected = roundtrip(
        tokenizer,
        rendered,
        max_length=LEGACY_PROMPT_MAX_LENGTH,
        add_special_tokens=True,
    )
    asyncio.run(
        model.api.generate(
            messages, [], "none", GenerateConfig(max_tokens=1, max_connections=1)
        )
    )
    ids, masks = captured[0]
    fed = [token for token, keep in zip(ids[0], masks[0], strict=True) if keep]
    assert fed == expected["input_ids"]
    assert len(fed) == LEGACY_PROMPT_MAX_LENGTH if long else len(fed) < 1024


def test_without_the_flags_hf_does_not_truncate(hf):
    """Truncation is the provider's setting: the default HF run keeps every token."""
    _, tokenizer, captured = hf
    model = get_model(
        "hf/local",
        model_path=tokenizer.name_or_path,
        device="cpu",
        do_sample=False,
        enable_thinking=False,
        memoize=False,
    )
    messages: list[ChatMessage] = [ChatMessageUser(content="question " * 1600 + "\n")]
    asyncio.run(
        model.api.generate(
            messages, [], "none", GenerateConfig(max_tokens=1, max_connections=1)
        )
    )
    assert len(captured[0][0][0]) > LEGACY_PROMPT_MAX_LENGTH

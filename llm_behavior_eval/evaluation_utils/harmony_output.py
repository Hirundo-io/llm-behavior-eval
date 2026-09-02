"""Thin adapter around the supported OpenAI Harmony completion parser.

GPT-OSS models always emit an ``analysis`` (chain-of-thought) message before the
user-visible ``final`` message. vLLM and Transformers both render Harmony prompts
natively, but neither strips the reasoning channel from an offline completion:
``vllm.LLM.generate``/``LLM.chat`` return the raw text, and vLLM's own Harmony
parsing (`vllm.entrypoints.openai.parser.harmony_utils.parse_chat_output`) is
reachable only through the OpenAI-compatible server. Extracting the user-visible
answer therefore has to happen here, using the same ``openai-harmony`` parser
that vLLM itself depends on.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerBase

_HARMONY_CONTROL_TOKENS = frozenset({"<|channel|>", "<|message|>", "<|return|>"})

_MISSING_HARMONY = (
    "GPT-OSS output parsing requires openai-harmony. Install "
    "llm-behavior-eval[gpt-oss] or llm-behavior-eval[vllm]."
)


class HarmonyOutputError(ValueError):
    """Raised when a Harmony completion has no user-visible final message."""


def is_harmony_tokenizer(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Return whether a tokenizer defines the GPT-OSS Harmony control tokens.

    Membership is tested in ``get_vocab()``, which includes added tokens.
    ``all_special_tokens`` cannot be used: it reports only the named tokenizer
    attributes (bos/eos/pad/...), which for ``openai/gpt-oss-*`` covers
    ``<|return|>`` but not ``<|channel|>`` or ``<|message|>``.
    """
    return _HARMONY_CONTROL_TOKENS <= tokenizer.get_vocab().keys()


@lru_cache(maxsize=1)
def _harmony_encoding():
    """Load the parser distributed by OpenAI's Harmony package."""
    try:
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding
    except ImportError as exc:
        raise ImportError(_MISSING_HARMONY) from exc
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def extract_harmony_final_answer(token_ids: Sequence[int]) -> str:
    """Extract user-visible content from a Harmony completion's token IDs.

    Reasoning on the ``analysis`` channel and tool-call commentary are excluded.
    A recipient-free commentary preamble is visible, matching vLLM's native
    ``parse_chat_output`` behavior. A completion truncated mid-answer yields the
    partial ``final`` content.

    Raises:
        ImportError: If ``openai-harmony`` is not installed. Install the
            ``llm-behavior-eval[gpt-oss]`` or ``llm-behavior-eval[vllm]`` extra.
        HarmonyOutputError: If the completion cannot be parsed or carries no
            user-visible content. Callers must not fall back to the raw text,
            which may contain analysis-channel reasoning.
    """
    try:
        from openai_harmony import HarmonyError, Role
    except ImportError as exc:
        raise ImportError(_MISSING_HARMONY) from exc

    try:
        messages = _harmony_encoding().parse_messages_from_completion_tokens(
            token_ids, Role.ASSISTANT
        )
    except HarmonyError as exc:
        raise HarmonyOutputError("Harmony completion could not be parsed.") from exc

    visible_parts: list[str] = []
    for message in messages:
        if message.channel != "final" and (
            message.channel != "commentary" or message.recipient is not None
        ):
            continue
        for content in message.content:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                visible_parts.append(text)

    answer = "\n".join(visible_parts).strip()
    if not answer:
        raise HarmonyOutputError(
            "Harmony completion did not contain user-visible final content."
        )
    return answer

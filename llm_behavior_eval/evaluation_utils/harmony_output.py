"""Thin adapter around the supported OpenAI Harmony completion parser."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class HarmonyOutputError(ValueError):
    """Raised when a Harmony completion has no user-visible final message."""


def is_harmony_tokenizer(tokenizer: object) -> bool:
    """Return whether a tokenizer exposes the GPT-OSS Harmony control tokens."""
    special_tokens = set(getattr(tokenizer, "all_special_tokens", ()))
    return {"<|channel|>", "<|message|>", "<|return|>"} <= special_tokens


@lru_cache(maxsize=1)
def _harmony_encoding():
    """Load the parser distributed by OpenAI's Harmony package."""
    try:
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding
    except ImportError as exc:
        raise ImportError(
            "GPT-OSS output parsing requires openai-harmony. Install "
            "llm-behavior-eval[gpt-oss] or llm-behavior-eval[vllm]."
        ) from exc
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def extract_harmony_final_answer(token_ids: Sequence[int]) -> str:
    """Extract user-visible content with the supported Harmony token parser.

    Reasoning on the ``analysis`` channel and tool-call commentary are excluded.
    A recipient-free commentary preamble is visible, matching vLLM's native
    ``parse_chat_output`` behavior.
    """
    try:
        from openai_harmony import HarmonyError, Role, StreamableParser
    except ImportError as exc:
        raise ImportError(
            "GPT-OSS output parsing requires openai-harmony. Install "
            "llm-behavior-eval[gpt-oss] or llm-behavior-eval[vllm]."
        ) from exc

    try:
        parser = StreamableParser(_harmony_encoding(), role=Role.ASSISTANT)
        for token_id in token_ids:
            parser.process(token_id)
    except HarmonyError as exc:
        raise HarmonyOutputError("Harmony completion could not be parsed.") from exc

    visible_parts = []
    for message in parser.messages:
        if message.channel != "final" and (
            message.channel != "commentary" or message.recipient is not None
        ):
            continue
        text = getattr(message.content[0], "text", None)
        if isinstance(text, str):
            visible_parts.append(text)
    if parser.current_channel == "final" and parser.current_content:
        visible_parts.append(parser.current_content)
    elif (
        parser.current_channel == "commentary"
        and parser.current_recipient is None
        and parser.current_content
    ):
        visible_parts.append(parser.current_content)

    answer = "\n".join(visible_parts).strip()
    if not answer:
        raise HarmonyOutputError(
            "Harmony completion did not contain user-visible final content."
        )
    return answer


def extract_vllm_harmony_final_answer(token_ids: Sequence[int]) -> str:
    """Extract the final answer with vLLM's native Harmony parser."""
    try:
        harmony_utils = import_module("vllm.entrypoints.openai.parser.harmony_utils")
        parse_chat_output = harmony_utils.parse_chat_output
    except ImportError as exc:
        raise ImportError(
            "GPT-OSS vLLM output parsing requires the vLLM Harmony parser."
        ) from exc

    try:
        _, final, _ = parse_chat_output(list(map(int, token_ids)))
    except Exception as exc:
        raise HarmonyOutputError("Harmony completion could not be parsed.") from exc
    if not final or not final.strip():
        raise HarmonyOutputError(
            "Harmony completion did not contain user-visible final content."
        )
    return final.strip()

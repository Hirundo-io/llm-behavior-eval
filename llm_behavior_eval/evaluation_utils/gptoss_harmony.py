"""Minimal GPT-OSS detection and Harmony final-channel extraction.

GPT-OSS models emit completions in OpenAI's Harmony format, where each message is
tagged with a channel (for example ``analysis`` for chain-of-thought reasoning,
``commentary`` for tool calls, and ``final`` for the user-visible answer). Only the
concatenation of ``final``-channel message content, in emission order, is a valid
``user_visible_answer``. Reasoning/analysis content must never reach judging.

This module is intentionally narrow: it detects GPT-OSS models by repo id/path and
parses raw Harmony-formatted completion text. It does not attempt to model the full
Harmony response framework (tool calls, multi-turn structure, etc.).
"""

import re

GPT_OSS_MODEL_MARKER = "gpt-oss"

FINAL_CHANNEL = "final"

# Match only standalone name segments, not substrings of another model name.
_GPT_OSS_MODEL_MARKER_RE = re.compile(
    rf"\b{re.escape(GPT_OSS_MODEL_MARKER)}\b", re.IGNORECASE
)

# Capture Harmony message content through the next control-token boundary.
_HARMONY_MESSAGE_RE = re.compile(
    r"<\|channel\|>(?P<channel>[a-zA-Z0-9_]+)<\|message\|>(?P<content>.*?)"
    r"(?=<\|start\|>|<\|channel\|>|<\|end\|>|<\|return\|>|\Z)",
    re.DOTALL,
)


class HarmonyParseError(ValueError):
    """Raised when a GPT-OSS completion cannot be parsed into a final-channel answer.

    This is a fail-closed signal: callers must not fall back to the raw completion
    text when this is raised, since the raw text may contain un-emitted reasoning
    or analysis-channel content that must never enter ``user_visible_answer``.
    """


def is_gpt_oss_model(model_path_or_repo_id: str) -> bool:
    """Return whether a model path or repository ID identifies GPT-OSS.

    Args:
        model_path_or_repo_id: Model repository ID or local model path.

    Returns:
        True when ``gpt-oss`` appears as a standalone name segment.
    """
    return _GPT_OSS_MODEL_MARKER_RE.search(model_path_or_repo_id) is not None


def extract_harmony_final_answer(raw_text: str) -> str:
    """Concatenate ``final``-channel content from a Harmony completion.

    Args:
        raw_text: Completion retaining Harmony control tokens
            (``skip_special_tokens=False``).

    Returns:
        Concatenated, stripped content from ``final``-channel messages.

    Raises:
        HarmonyParseError: If no Harmony messages are found, or none are on the
            ``final`` channel, or final content is empty. Callers must not fall
            back to the raw text.
    """
    matches = list(_HARMONY_MESSAGE_RE.finditer(raw_text))
    if not matches:
        raise HarmonyParseError(
            "Malformed Harmony completion: no <|channel|>/<|message|> markers found."
        )
    final_segments = [
        match.group("content")
        for match in matches
        if match.group("channel") == FINAL_CHANNEL
    ]
    if not final_segments:
        raise HarmonyParseError(
            "Malformed Harmony completion: no message on the 'final' channel."
        )
    final_answer = "".join(final_segments).strip()
    if not final_answer:
        raise HarmonyParseError(
            "Malformed Harmony completion: final-channel content is empty."
        )
    return final_answer

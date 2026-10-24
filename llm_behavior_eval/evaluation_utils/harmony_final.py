"""Keep only the user-visible final message of GPT-OSS (Harmony) completions.

GPT-OSS writes its reasoning on the ``analysis`` channel before the answer on the
``final`` channel. Decoding a completion with special tokens skipped flattens both
into one string ("analysis...assistantfinal..."), so a judge reads the reasoning as
part of the answer. Here each completion's token IDs are decoded with special tokens
kept and only the last ``final`` message is returned. A completion without one
(reasoning cut off by the token budget, or a reply sent to a tool on the
``commentary`` channel) yields an empty answer marked incomplete, so evaluators that
judge only ``stop`` answers count it as incomplete instead of grading its reasoning.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from .eval_config import EvaluationConfig

HARMONY_CONTROL_TOKENS = frozenset({"<|channel|>", "<|message|>", "<|return|>"})
INCOMPLETE_FINISH_REASON = "length"
FINAL_MESSAGE_PATTERN = re.compile(
    r"<\|channel\|>\s*final\b(?:[ \t][^<]*)?<\|message\|>"
    r"(.*?)(?=<\|return\|>|<\|end\|>|<\|call\|>|<\|start\|>|\Z)",
    re.DOTALL,
)


def is_harmony_tokenizer(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Return whether the tokenizer registers the Harmony control tokens as special tokens.

    ``added_tokens_decoder`` is checked rather than ``all_special_tokens``, which for
    ``openai/gpt-oss-*`` lists ``<|return|>`` but not ``<|channel|>`` or ``<|message|>``.
    """
    registered = {
        token.content
        for token in tokenizer.added_tokens_decoder.values()
        if token.special
    }
    return HARMONY_CONTROL_TOKENS <= registered


def keeps_final_channel_only(
    eval_config: EvaluationConfig,
    is_judge: bool,
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Return whether an engine should return only Harmony final-channel answers.

    Args:
        eval_config: Evaluation configuration of the engine.
        is_judge: Whether the engine runs the judge (never filtered).
        tokenizer: Tokenizer of the engine's model.

    Returns:
        True for the evaluated model when ``harmony_final_channel_only`` is set and
        the tokenizer uses Harmony.
    """
    return (
        eval_config.harmony_final_channel_only
        and not is_judge
        and is_harmony_tokenizer(tokenizer)
    )


def final_channel_text(completion: str) -> str | None:
    """Return the text of the last final-channel message in a Harmony completion.

    Args:
        completion: Completion decoded with special tokens kept.

    Returns:
        The stripped final-channel text, or None when the completion has no final
        message.
    """
    messages = FINAL_MESSAGE_PATTERN.findall(completion)
    return messages[-1].strip() if messages else None


def final_channel_answers(
    tokenizer: PreTrainedTokenizerBase,
    completion_token_ids: Sequence[Sequence[int]],
    finish_reasons: Sequence[str | None],
) -> tuple[list[str], list[str | None]]:
    """Reduce Harmony completions to their final-channel answers.

    Args:
        tokenizer: Tokenizer that produced the completions.
        completion_token_ids: Generated token IDs per completion.
        finish_reasons: Engine finish reason per completion.

    Returns:
        Final-channel answers and finish reasons. A completion without a final
        message becomes an empty answer with the ``length`` finish reason.
    """
    answers: list[str] = []
    reasons: list[str | None] = []
    for token_ids, finish_reason in zip(
        completion_token_ids, finish_reasons, strict=True
    ):
        final = final_channel_text(
            tokenizer.decode(list(token_ids), skip_special_tokens=False)
        )
        if final is None:
            answers.append("")
            reasons.append(INCOMPLETE_FINISH_REASON)
        else:
            answers.append(final)
            reasons.append(finish_reason)
    return answers, reasons

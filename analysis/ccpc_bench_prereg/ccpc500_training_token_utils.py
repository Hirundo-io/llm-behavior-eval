"""Trainer-equivalent token counting for CCPC500 training-data construction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_HIRUNDO_CORE = Path("/home/ubuntu/hirundo-research/hirundo_core")
if str(_HIRUNDO_CORE) not in sys.path:
    sys.path.insert(0, str(_HIRUNDO_CORE))

from h_core.debias.dataset.bias_dataset import DEFAULT_SYSTEM_PROMPTS
from h_core.debias.dataset.dataset_config import TextFormat

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_LENGTH = 2048


def trainer_equivalent_messages(question: str, answer: str) -> list[dict[str, str]]:
    """Build the historical RSCH-76 trainer-equivalent chat render.

    Args:
        question: User question text.
        answer: Assistant target text.

    Returns:
        Chat messages with default system turn, user newline suffix, and assistant.
    """
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPTS[TextFormat.FREE_TEXT]},
        {"role": "user", "content": f"{question}\n"},
        {"role": "assistant", "content": answer},
    ]


def extract_chat_token_ids(tokenizer_result: Any) -> list[int]:
    """Extract actual token IDs from supported tokenizer return shapes.

    Args:
        tokenizer_result: Output from ``apply_chat_template(..., tokenize=True)`` or
            equivalent tokenizer call.

    Returns:
        Flat token-id sequence for a single rendered example.

    Raises:
        TypeError: If the return shape is unsupported or ambiguous.
    """
    if isinstance(tokenizer_result, list):
        if tokenizer_result and isinstance(tokenizer_result[0], list):
            if len(tokenizer_result) != 1:
                raise TypeError("expected exactly one sequence in nested list result")
            return [int(token_id) for token_id in tokenizer_result[0]]
        return [int(token_id) for token_id in tokenizer_result]

    if isinstance(tokenizer_result, dict):
        if "input_ids" not in tokenizer_result:
            raise TypeError("dict tokenizer result missing input_ids")
        return _flatten_input_ids(tokenizer_result["input_ids"])

    if hasattr(tokenizer_result, "input_ids"):
        return _flatten_input_ids(tokenizer_result.input_ids)

    raise TypeError(f"unsupported tokenizer return shape: {type(tokenizer_result)!r}")


def _flatten_input_ids(ids: Any) -> list[int]:
    """Normalize ``input_ids`` to a flat integer list.

    Args:
        ids: Token ids as list, nested list, or tensor-like object.

    Returns:
        Flat token-id sequence for one example.

    Raises:
        TypeError: If the ids container is unsupported or ambiguous.
    """
    if isinstance(ids, list):
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise TypeError("expected batch size 1 in nested input_ids")
            return [int(token_id) for token_id in ids[0]]
        return [int(token_id) for token_id in ids]

    if hasattr(ids, "tolist"):
        values = ids.tolist()
        if isinstance(values, list) and values and isinstance(values[0], list):
            if len(values) != 1:
                raise TypeError("expected batch size 1 in tensor input_ids")
            return [int(token_id) for token_id in values[0]]
        if isinstance(values, list):
            return [int(token_id) for token_id in values]
        return [int(values)]

    raise TypeError(f"unsupported input_ids shape: {type(ids)!r}")


def train_token_count(tokenizer_value: Any, question: str, answer: str) -> int:
    """Count tokens in the trainer-equivalent training representation.

    Args:
        tokenizer_value: Frozen Qwen tokenizer.
        question: User question text.
        answer: Assistant target text.

    Returns:
        Number of tokens in the rendered trainer-equivalent conversation.
    """
    rendered = tokenizer_value.apply_chat_template(
        trainer_equivalent_messages(question, answer),
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )
    token_ids = extract_chat_token_ids(rendered)
    if len(token_ids) == 2 and question.strip() and answer.strip():
        raise RuntimeError(
            "trainer-equivalent token count equals BatchEncoding key count; "
            "refusing to treat len(tokenizer_result) as token count"
        )
    return len(token_ids)


def load_frozen_tokenizer() -> Any:
    """Load the exact frozen Qwen tokenizer used for CCPC500 training-data QA.

    Returns:
        Configured AutoTokenizer instance.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL,
        revision=REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )

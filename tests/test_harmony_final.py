from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.harmony_final import (
    final_channel_answers,
    final_channel_text,
    is_harmony_tokenizer,
    keeps_final_channel_only,
)

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast


def _ids(tokenizer: PreTrainedTokenizerFast, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


ANALYSIS = "<|channel|> analysis <|message|> We must refuse. <|end|> "


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (
            ANALYSIS
            + "<|start|> assistant <|channel|> final <|message|> I cannot share that. <|return|>",
            "I cannot share that.",
        ),
        (
            ANALYSIS
            + "<|start|> assistant <|channel|> final <|message|> I cannot share that.",
            "I cannot share that.",
        ),
        (
            "<|channel|> final <|message|> Sure. <|end|> <|start|> assistant <|channel|> final <|message|> I cannot share that.",
            "I cannot share that.",
        ),
        ("<|channel|> analysis <|message|> We must", None),
        (
            ANALYSIS
            + "<|start|> assistant <|channel|> commentary to=refusal <|constrain|> refusal <|message|> I cannot share that. <|call|>",
            None,
        ),
    ],
    ids=[
        "final",
        "final-without-return",
        "last-final-wins",
        "reasoning-cut-off",
        "tool-call-refusal",
    ],
)
def test_final_channel_text(
    completion: str, expected: str | None, harmony_tokenizer: PreTrainedTokenizerFast
) -> None:
    decoded = harmony_tokenizer.decode(
        _ids(harmony_tokenizer, completion), skip_special_tokens=False
    )
    assert final_channel_text(decoded) == expected


def test_final_channel_answers_mark_completions_without_final_incomplete(
    harmony_tokenizer: PreTrainedTokenizerFast,
) -> None:
    tokenizer = harmony_tokenizer
    completions = [
        ANALYSIS
        + "<|start|> assistant <|channel|> final <|message|> I cannot share that. <|return|>",
        ANALYSIS
        + "<|start|> assistant <|channel|> commentary to=refusal <|message|> I cannot share that. <|call|>",
        "<|channel|> analysis <|message|> We must",
    ]
    answers, reasons = final_channel_answers(
        tokenizer,
        [_ids(tokenizer, text) for text in completions],
        ["stop", "stop", "length"],
    )
    assert answers == ["I cannot share that.", "", ""]
    assert reasons == ["stop", "length", "length"]


def test_is_harmony_tokenizer(
    harmony_tokenizer: PreTrainedTokenizerFast, plain_tokenizer: PreTrainedTokenizerFast
) -> None:
    assert is_harmony_tokenizer(harmony_tokenizer)
    assert not is_harmony_tokenizer(plain_tokenizer)


@pytest.mark.parametrize(
    ("enabled", "is_judge", "harmony", "expected"),
    [
        (True, False, True, True),
        (False, False, True, False),
        (True, True, True, False),
        (True, False, False, False),
    ],
)
def test_keeps_final_channel_only(
    enabled: bool,
    is_judge: bool,
    harmony: bool,
    expected: bool,
    tmp_path: Path,
    harmony_tokenizer: PreTrainedTokenizerFast,
    plain_tokenizer: PreTrainedTokenizerFast,
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        harmony_final_channel_only=enabled,
    )
    tokenizer = harmony_tokenizer if harmony else plain_tokenizer
    assert keeps_final_channel_only(config, is_judge, tokenizer) is expected

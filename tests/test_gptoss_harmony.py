import pytest

from llm_behavior_eval.evaluation_utils.gptoss_harmony import (
    HarmonyParseError,
    extract_harmony_final_answer,
    is_gpt_oss_model,
)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("openai/gpt-oss-20b", True),
        ("/local/path/to/gpt-oss-20b", True),
        ("somegpt-oss-20b", False),
        ("openai/gpt-oss2", False),
        ("", False),
        ("google/gemma-4-26B-A4B-it", False),
    ],
)
def test_is_gpt_oss_model_matches_standalone_name_segments(
    model_id: str, expected: bool
) -> None:
    assert is_gpt_oss_model(model_id) is expected


def test_extract_harmony_final_answer_concatenates_final_channel_in_order() -> None:
    raw = (
        "<|channel|>analysis<|message|>secret chain of thought<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>tool call<|end|>"
        "<|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>"
    )
    assert extract_harmony_final_answer(raw) == "The answer is 42."


def test_extract_harmony_final_answer_concatenates_multiple_final_segments() -> None:
    raw = (
        "<|channel|>final<|message|>Part one. <|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>ignored<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Part two.<|return|>"
    )
    assert extract_harmony_final_answer(raw) == "Part one. Part two."


def test_extract_harmony_final_answer_fails_closed_on_malformed_completion() -> None:
    with pytest.raises(HarmonyParseError):
        extract_harmony_final_answer("just plain text, no Harmony markers")


def test_extract_harmony_final_answer_fails_closed_when_no_final_channel() -> None:
    raw = (
        "<|channel|>analysis<|message|>thinking only<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>tool call only<|end|>"
    )
    with pytest.raises(HarmonyParseError):
        extract_harmony_final_answer(raw)


def test_extract_harmony_final_answer_fails_closed_when_final_content_is_empty() -> (
    None
):
    with pytest.raises(HarmonyParseError, match="final-channel content is empty"):
        extract_harmony_final_answer("<|channel|>final<|message|>  <|return|>")

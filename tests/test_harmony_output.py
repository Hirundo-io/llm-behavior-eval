import sys
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from llm_behavior_eval.evaluation_utils import harmony_output


class _FakeEncoding:
    """Stands in for the encoding returned by ``load_harmony_encoding``."""

    def __init__(self) -> None:
        self.parsed: list[Sequence[int]] = []

    def parse_messages_from_completion_tokens(
        self, tokens: Sequence[int], role: object
    ) -> list[SimpleNamespace]:
        self.parsed.append(tokens)
        channels = {1: "analysis", 2: "final"}
        return [
            SimpleNamespace(
                channel=channels[token_id],
                recipient=None,
                content=[SimpleNamespace(text=f"{channels[token_id]} text")],
            )
            for token_id in tokens
        ]


def _added_token(content: str, special: bool = True) -> SimpleNamespace:
    return SimpleNamespace(content=content, special=special)


@pytest.fixture
def fake_harmony(monkeypatch: pytest.MonkeyPatch) -> _FakeEncoding:
    encoding = _FakeEncoding()
    monkeypatch.setattr(harmony_output, "_harmony_encoding", lambda: encoding)
    monkeypatch.setitem(
        sys.modules,
        "openai_harmony",
        SimpleNamespace(
            HarmonyError=ValueError,
            Role=SimpleNamespace(ASSISTANT="assistant"),
        ),
    )
    return encoding


@pytest.fixture
def harmony_encoding():
    """The real ``openai-harmony`` encoding, skipped when it is unavailable."""
    openai_harmony = pytest.importorskip("openai_harmony")
    try:
        return openai_harmony.load_harmony_encoding(
            openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
        )
    except Exception as exc:  # pragma: no cover - offline environments
        pytest.skip(f"Harmony encoding is unavailable: {exc}")


def test_extract_harmony_final_answer_delegates_to_parser(
    fake_harmony: _FakeEncoding,
) -> None:
    assert harmony_output.extract_harmony_final_answer([1, 2]) == "final text"
    assert fake_harmony.parsed == [[1, 2]]


def test_extract_harmony_final_answer_fails_closed_without_visible_content(
    fake_harmony: _FakeEncoding,
) -> None:
    with pytest.raises(harmony_output.HarmonyOutputError):
        harmony_output.extract_harmony_final_answer([1])


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (
            "<|channel|>analysis<|message|>hidden reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>visible answer<|return|>",
            "visible answer",
        ),
        # Truncated mid-answer: the partial final message is still user-visible.
        (
            "<|channel|>analysis<|message|>hidden reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>partial",
            "partial",
        ),
    ],
)
def test_extract_harmony_final_answer_drops_reasoning_from_real_tokens(
    harmony_encoding, completion: str, expected: str
) -> None:
    token_ids = harmony_encoding.encode(completion, allowed_special="all")

    assert harmony_output.extract_harmony_final_answer(token_ids) == expected


def test_extract_harmony_final_answer_rejects_reasoning_only_real_tokens(
    harmony_encoding,
) -> None:
    token_ids = harmony_encoding.encode(
        "<|channel|>analysis<|message|>hidden reasoning", allowed_special="all"
    )

    with pytest.raises(harmony_output.HarmonyOutputError):
        harmony_output.extract_harmony_final_answer(token_ids)


def test_is_harmony_tokenizer_matches_gpt_oss_added_special_tokens() -> None:
    # ``openai/gpt-oss-*`` registers the control tokens as added special tokens;
    # only ``<|return|>`` is reported by ``all_special_tokens``.
    gpt_oss = SimpleNamespace(
        all_special_tokens=["<|startoftext|>", "<|return|>", "<|endoftext|>"],
        added_tokens_decoder={
            200002: _added_token("<|return|>"),
            200005: _added_token("<|channel|>"),
            200008: _added_token("<|message|>"),
        },
    )

    assert harmony_output.is_harmony_tokenizer(gpt_oss)


def test_is_harmony_tokenizer_requires_all_control_tokens() -> None:
    partial = SimpleNamespace(
        added_tokens_decoder={
            200005: _added_token("<|channel|>"),
            200008: _added_token("<|message|>"),
        }
    )

    assert not harmony_output.is_harmony_tokenizer(partial)
    assert not harmony_output.is_harmony_tokenizer(SimpleNamespace())

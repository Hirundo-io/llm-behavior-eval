import sys
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from llm_behavior_eval.evaluation_utils import harmony_output

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class _FakeTextContent:
    def __init__(self, text: str) -> None:
        self.text = text


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
                content=[_FakeTextContent(f"{channels[token_id]} text")],
            )
            for token_id in tokens
        ]


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
            TextContent=_FakeTextContent,
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


def test_is_harmony_tokenizer_matches_the_gpt_oss_vocabulary() -> None:
    # ``openai/gpt-oss-*`` exposes the control tokens through ``get_vocab``;
    # ``all_special_tokens`` reports only ``<|return|>`` of the three.
    gpt_oss = SimpleNamespace(
        get_vocab=lambda: {
            "<|return|>": 200002,
            "<|channel|>": 200005,
            "<|message|>": 200008,
        }
    )

    assert harmony_output.is_harmony_tokenizer(cast("PreTrainedTokenizerBase", gpt_oss))


def test_is_harmony_tokenizer_requires_all_control_tokens() -> None:
    partial = SimpleNamespace(get_vocab=lambda: {"<|channel|>": 1, "<|message|>": 2})

    assert not harmony_output.is_harmony_tokenizer(
        cast("PreTrainedTokenizerBase", partial)
    )
    assert not harmony_output.is_harmony_tokenizer(
        cast("PreTrainedTokenizerBase", SimpleNamespace(get_vocab=dict))
    )

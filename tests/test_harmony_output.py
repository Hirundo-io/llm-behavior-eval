import sys
from types import SimpleNamespace

import pytest

from llm_behavior_eval.evaluation_utils import harmony_output


class _FakeParser:
    def __init__(self, _encoding: object, role: object) -> None:
        self.role = role
        self.messages: list[SimpleNamespace] = []
        self.current_channel: str | None = None
        self.current_recipient: str | None = None
        self.current_content = ""

    def process(self, token_id: int) -> None:
        if token_id == 1:
            self.messages.append(
                SimpleNamespace(
                    channel="analysis",
                    recipient=None,
                    content=[SimpleNamespace(text="hidden reasoning")],
                )
            )
        elif token_id == 2:
            self.messages.append(
                SimpleNamespace(
                    channel="final",
                    recipient=None,
                    content=[SimpleNamespace(text="visible answer")],
                )
            )


@pytest.fixture
def fake_harmony(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harmony_output, "_harmony_encoding", lambda: object())
    monkeypatch.setitem(
        sys.modules,
        "openai_harmony",
        SimpleNamespace(
            HarmonyError=ValueError,
            Role=SimpleNamespace(ASSISTANT="assistant"),
            StreamableParser=_FakeParser,
        ),
    )


def test_extract_harmony_final_answer_delegates_to_parser(
    fake_harmony: None,
) -> None:
    assert harmony_output.extract_harmony_final_answer([1, 2]) == "visible answer"


def test_extract_harmony_final_answer_fails_closed_without_visible_content(
    fake_harmony: None,
) -> None:
    with pytest.raises(harmony_output.HarmonyOutputError):
        harmony_output.extract_harmony_final_answer([1])


def test_extract_vllm_harmony_final_answer_delegates_to_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harmony_utils = SimpleNamespace(
        parse_chat_output=lambda token_ids: ("analysis", "visible answer", None)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.parser.harmony_utils",
        harmony_utils,
    )
    assert harmony_output.extract_vllm_harmony_final_answer([1, 2]) == "visible answer"


def test_is_harmony_tokenizer_requires_all_control_tokens() -> None:
    assert harmony_output.is_harmony_tokenizer(
        SimpleNamespace(all_special_tokens=["<|channel|>", "<|message|>", "<|return|>"])
    )
    assert not harmony_output.is_harmony_tokenizer(
        SimpleNamespace(all_special_tokens=["<|channel|>", "<|message|>"])
    )

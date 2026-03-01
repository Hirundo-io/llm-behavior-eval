from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llm_behavior_eval.evaluation_utils.eval_engine import PromptEvalEngine

if TYPE_CHECKING:
    from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig


class _DummyPromptEngine(PromptEvalEngine):
    def set_dataset(self, eval_dataset) -> None:
        self.eval_dataset = eval_dataset

    def get_batch_size(self) -> int:
        return 1

    def free_model(self) -> None:
        return None

    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        return "formatted::" + messages[0]["content"]

    def generate_answers_from_prompts(
        self,
        prompts: list[str | list[dict[str, str]]],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        del sampling_config
        return [str(prompt) for prompt in prompts]


def test_prompt_engine_default_eval_engine_behavior() -> None:
    engine = _DummyPromptEngine()

    assert engine.should_combine_judge_prompt_groups() is False


def test_prompt_engine_normalizes_mixed_prompts_to_strings() -> None:
    engine = _DummyPromptEngine()
    prompts = [
        "already-formatted",
        [{"role": "user", "content": "hello"}],
    ]

    normalized = engine.normalize_prompts_to_strings(prompts)

    assert normalized == [
        "already-formatted",
        "formatted::hello",
    ]


def test_prompt_engine_rejects_non_string_prompt_formatting() -> None:
    class BadPromptEngine(_DummyPromptEngine):
        def format_prompt(self, messages: list[dict[str, str]]):  # type: ignore[override]
            return messages

    engine = BadPromptEngine()
    with pytest.raises(
        TypeError,
        match="Tokenizer-backed engines must format message prompts into strings.",
    ):
        engine.normalize_prompts_to_strings([[{"role": "user", "content": "hello"}]])

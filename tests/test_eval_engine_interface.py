from __future__ import annotations

from typing import TYPE_CHECKING

from llm_behavior_eval.evaluation_utils.eval_engine import (
    EngineInputMode,
    PromptEvalEngine,
    TensorEvalEngine,
)

if TYPE_CHECKING:
    import torch

    from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig


class _DummyTensorEngine(TensorEvalEngine):
    def set_dataset(self, eval_dataset) -> None:
        self.eval_dataset = eval_dataset

    def get_batch_size(self) -> int:
        return 1

    def free_model(self) -> None:
        return None

    def generate_answers_from_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: SamplingConfig,
    ) -> list[str]:
        return []


class _DummyPromptEngine(PromptEvalEngine):
    def set_dataset(self, eval_dataset) -> None:
        self.eval_dataset = eval_dataset

    def get_batch_size(self) -> int:
        return 1

    def free_model(self) -> None:
        return None

    def format_prompt(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return messages

    def generate_answers_from_prompts(
        self,
        prompts: list[str | list[dict[str, str]]],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        return []


class _DummyHybridEngine(TensorEvalEngine, PromptEvalEngine):
    def set_dataset(self, eval_dataset) -> None:
        self.eval_dataset = eval_dataset

    def get_batch_size(self) -> int:
        return 1

    def free_model(self) -> None:
        return None

    def generation_input_mode(self) -> EngineInputMode:
        return EngineInputMode.HYBRID

    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        return ""

    def generate_answers_from_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: SamplingConfig,
    ) -> list[str]:
        return []

    def generate_answers_from_prompts(
        self,
        prompts: list[str | list[dict[str, str]]],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        return []


def test_tensor_engine_capabilities_and_dataset_requirements() -> None:
    engine = _DummyTensorEngine()

    assert engine.generation_input_mode() == EngineInputMode.TENSOR
    assert engine.supports_tensor_generation() is True
    assert engine.supports_prompt_generation() is False
    assert engine.requires_tokenized_dataset() is True


def test_prompt_engine_capabilities_and_dataset_requirements() -> None:
    engine = _DummyPromptEngine()

    assert engine.generation_input_mode() == EngineInputMode.PROMPT
    assert engine.supports_tensor_generation() is False
    assert engine.supports_prompt_generation() is True
    assert engine.requires_tokenized_dataset() is False


def test_hybrid_engine_capabilities_and_dataset_requirements() -> None:
    engine = _DummyHybridEngine()

    assert engine.generation_input_mode() == EngineInputMode.HYBRID
    assert engine.supports_tensor_generation() is True
    assert engine.supports_prompt_generation() is True
    assert engine.requires_tokenized_dataset() is True

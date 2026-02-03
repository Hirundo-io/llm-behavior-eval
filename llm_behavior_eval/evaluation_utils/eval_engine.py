from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .eval_config import EvaluationConfig
    from .sampling_config import SamplingConfig


EvalDataset = Sized

# Canonical type for judge prompts: either a tokenized string or raw messages for API
JudgePrompt = str | list[dict[str, str]]


class EvalEngine(ABC):
    @abstractmethod
    def set_dataset(self, eval_dataset: EvalDataset) -> None:
        raise NotImplementedError("Subclasses must implement set_dataset().")

    @abstractmethod
    def generate_answers_from_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: SamplingConfig,
    ) -> list[str]:
        """Generate answers from pre-tokenized tensor inputs.

        For local engines (transformers, vllm) that work with tokenized batches.
        API engines do not implement this - use generate_answers_from_prompts() instead.
        """
        raise NotImplementedError(
            "Subclasses must implement generate_answers_from_tensors()."
        )

    def ensure_test_model_ready(self) -> None:
        return None

    @abstractmethod
    def get_batch_size(self) -> int:
        raise NotImplementedError("Subclasses must implement get_batch_size().")

    @abstractmethod
    def free_model(self) -> None:
        raise NotImplementedError("Subclasses must implement free_model().")

    @staticmethod
    def _get_model_path_or_repo_id(
        eval_config: EvaluationConfig, is_judge: bool
    ) -> str:
        """Get the model path based on whether this is a judge model."""
        return (
            eval_config.judge_path_or_repo_id
            if is_judge
            else eval_config.model_path_or_repo_id
        )

    @staticmethod
    def _get_model_token(eval_config: EvaluationConfig, is_judge: bool) -> str | None:
        """Get the model token based on whether this is a judge model."""
        return eval_config.judge_token if is_judge else eval_config.model_token

    @staticmethod
    def _get_use_4bit(eval_config: EvaluationConfig, is_judge: bool) -> bool:
        """Get the 4-bit setting based on whether this is a judge model."""
        return eval_config.use_4bit_judge if is_judge else eval_config.use_4bit

    @staticmethod
    def _get_batch_size_from_config(
        eval_config: EvaluationConfig, is_judge: bool
    ) -> int | None:
        """Get the estimated batch size from config based on whether this is a judge model."""
        return eval_config.judge_batch_size if is_judge else eval_config.batch_size

    @staticmethod
    def _get_sample_from_config(eval_config: EvaluationConfig, is_judge: bool) -> bool:
        """Get the sample setting from config based on whether this is a judge model."""
        return eval_config.sample_judge if is_judge else eval_config.sample

    @staticmethod
    def _get_max_new_tokens(eval_config: EvaluationConfig, is_judge: bool) -> int:
        """Get the max new tokens setting from config based on whether this is a judge model."""
        return (
            eval_config.max_judge_tokens if is_judge else eval_config.max_answer_tokens
        )

    def format_prompt(self, messages: list[dict[str, str]]) -> JudgePrompt:
        """Format messages into a prompt suitable for this engine.

        For tokenizer-based engines, applies the chat template.
        For API engines, returns messages unchanged.

        Override in subclasses to provide engine-specific behavior.
        """
        raise NotImplementedError("Subclasses must implement format_prompt().")

    def generate_answers_from_prompts(
        self,
        prompts: list[JudgePrompt],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        """Generate answers from pre-formatted prompts.

        For API engines, this sends prompts directly to the provider.
        For local engines, this tokenizes and calls generate_answers_from_tensors().

        Override in subclasses to provide engine-specific behavior.
        """
        raise NotImplementedError(
            "Subclasses must implement generate_answers_from_prompts()."
        )

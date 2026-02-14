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
    is_judge: bool = False

    @abstractmethod
    def set_dataset(self, eval_dataset: EvalDataset) -> None:
        raise NotImplementedError("Subclasses must implement set_dataset().")

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
    def _get_tokenizer_path_or_repo_id(
        eval_config: EvaluationConfig, is_judge: bool
    ) -> str:
        """Get the tokenizer path based on whether this is a judge model."""
        if is_judge:
            return (
                eval_config.judge_tokenizer_path_or_repo_id
                or eval_config.judge_path_or_repo_id
            )
        return (
            eval_config.model_tokenizer_path_or_repo_id
            or eval_config.model_path_or_repo_id
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


class TensorEvalEngine(EvalEngine, ABC):
    @abstractmethod
    def generate_answers_from_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: SamplingConfig,
    ) -> list[str]:
        """Generate answers from pre-tokenized tensor inputs."""
        raise NotImplementedError(
            "Subclasses must implement generate_answers_from_tensors()."
        )


class PromptEvalEngine(EvalEngine, ABC):
    @abstractmethod
    def format_prompt(self, messages: list[dict[str, str]]) -> JudgePrompt:
        """Format messages into a prompt suitable for this engine."""
        raise NotImplementedError("Subclasses must implement format_prompt().")

    def normalize_prompts_to_strings(self, prompts: list[JudgePrompt]) -> list[str]:
        """Normalize mixed prompt inputs into string prompts.

        Tokenizer-backed engines accept either:
        - preformatted strings, or
        - message lists that still need formatting via `format_prompt`.
        """
        string_prompts: list[str] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                string_prompts.append(prompt)
                continue

            formatted_prompt = self.format_prompt(prompt)
            if not isinstance(formatted_prompt, str):
                raise TypeError(
                    "Tokenizer-backed engines must format message prompts into strings."
                )
            string_prompts.append(formatted_prompt)
        return string_prompts

    @abstractmethod
    def generate_answers_from_prompts(
        self,
        prompts: list[JudgePrompt],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        """Generate answers from pre-formatted prompts."""
        raise NotImplementedError(
            "Subclasses must implement generate_answers_from_prompts()."
        )

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from datasets import Dataset

    from .eval_config import EvaluationConfig


class EvalEngine(ABC):
    @abstractmethod
    def set_dataset(self, eval_dataset: Dataset) -> None:
        raise NotImplementedError("Subclasses must implement set_dataset().")

    @abstractmethod
    def generate_answers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        do_sample: bool | None = None,
    ) -> list[str]:
        raise NotImplementedError("Subclasses must implement generate_answers().")

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
        return eval_config.judge_sample if is_judge else eval_config.sample

from abc import ABC, abstractmethod

import torch
from datasets import Dataset


class EvalEngine(ABC):
    @abstractmethod
    def set_dataset(self, eval_dataset: Dataset) -> None:
        raise NotImplementedError("Subclasses must implement set_dataset().")

    @abstractmethod
    def generate_answers(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
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

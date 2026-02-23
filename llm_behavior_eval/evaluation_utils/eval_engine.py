from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sized
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .sampling_config import SamplingConfig


EvalDataset = Sized

# Canonical type for judge prompts: either a tokenized string or raw messages for API
JudgePrompt = str | list[dict[str, str]]


class EngineInputMode(StrEnum):
    TENSOR = "tensor"
    PROMPT = "prompt"
    HYBRID = "hybrid"


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

    def generation_input_mode(self) -> EngineInputMode:
        """Declare the primary input mode used for generation."""
        return EngineInputMode.TENSOR

    def supports_tensor_generation(self) -> bool:
        mode = self.generation_input_mode()
        return mode in {EngineInputMode.TENSOR, EngineInputMode.HYBRID}

    def supports_prompt_generation(self) -> bool:
        mode = self.generation_input_mode()
        return mode in {EngineInputMode.PROMPT, EngineInputMode.HYBRID}

    def requires_tokenized_dataset(self) -> bool:
        """
        Whether dataset preprocessing must produce tokenized tensor columns.

        Prompt-only engines (e.g., API-backed) can operate on raw prompts/messages
        and therefore do not require tokenized datasets.
        """
        return self.generation_input_mode() != EngineInputMode.PROMPT

    def get_raw_text_truncator(self) -> Callable[[str, int], str] | None:
        """
        Return an optional raw-text truncation callable for tokenizer-free paths.

        Engines that do not provide model-aware truncation should return None.
        """
        return None

    def set_preprocess_limits(self, max_length: int, gt_max_length: int) -> None:
        """
        Receive dataset preprocessing limits for optional engine-specific use.

        Engines may ignore this information.
        """
        del max_length, gt_max_length
        return None

    def should_combine_judge_prompt_groups(self) -> bool:
        """
        Whether grouped judge prompts should be coalesced into one engine call.

        API engines benefit from this to maximize provider-side batching.
        """
        return False


class TensorEvalEngine(EvalEngine, ABC):
    def generation_input_mode(self) -> EngineInputMode:
        return EngineInputMode.TENSOR

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
    def generation_input_mode(self) -> EngineInputMode:
        return EngineInputMode.PROMPT

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

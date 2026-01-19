from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .eval_engine import EvalEngine
from .util_functions import load_tokenizer_with_transformers

if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from .eval_config import EvaluationConfig
    from .sampling_config import SamplingConfig


JudgePrompt = str | list[dict[str, str]]


class ApiEvalEngine(EvalEngine):
    def __init__(self, eval_config: EvaluationConfig, *, is_judge: bool = False) -> None:
        self.eval_config = eval_config
        self.is_judge = is_judge
        self.dataset: Dataset | None = None
        self.model_name = (
            eval_config.judge_path_or_repo_id
            if is_judge
            else eval_config.model_path_or_repo_id
        )
        self._litellm = self._load_litellm()
        self.tokenizer: PreTrainedTokenizerBase | None = None
        if not is_judge:
            self.tokenizer = self._load_model_tokenizer()
            self.tokenizer.padding_side = "left"
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _load_litellm():
        spec = importlib.util.find_spec("litellm")
        if spec is None:
            raise ImportError(
                "litellm is required for API-based judge models. "
                "Install it with `pip install llm-behavior-eval[api]`."
            )
        return importlib.import_module("litellm")

    def set_dataset(self, eval_dataset: Dataset) -> None:
        self.dataset = eval_dataset

    def generate_answers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: SamplingConfig,
    ) -> list[str]:
        tokenizer = self._get_tokenizer()
        prompts: list[str] = []
        for row_index in range(input_ids.shape[0]):
            if attention_mask is not None:
                mask = attention_mask[row_index].bool()
                prompt_ids = input_ids[row_index][mask]
            else:
                prompt_ids = input_ids[row_index]
            prompts.append(
                tokenizer.decode(
                    prompt_ids,
                    skip_special_tokens=True,
                )
            )
        return self.generate_answers_from_prompts(prompts, sampling_config)

    def generate_answers_from_prompts(
        self,
        prompts: list[JudgePrompt],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        answers: list[str] = []
        for prompt in prompts:
            messages = self._normalize_messages(prompt)
            response = self._litellm.completion(
                **self._build_completion_kwargs(messages, sampling_config)
            )
            answers.append(self._extract_content(response))
        return answers

    def get_batch_size(self) -> int:
        if self.is_judge:
            if self.eval_config.judge_batch_size is not None:
                return max(1, int(self.eval_config.judge_batch_size))
            return 1
        if self.eval_config.batch_size is not None:
            return max(1, int(self.eval_config.batch_size))
        return 1

    def free_model(self) -> None:
        return None

    @staticmethod
    def _normalize_messages(prompt: JudgePrompt) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    def _load_model_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer_source = (
            self.eval_config.model_tokenizer_path_or_repo_id
            or self.eval_config.model_path_or_repo_id
        )
        return load_tokenizer_with_transformers(
            tokenizer_source,
            token=self.eval_config.model_token,
            trust_remote_code=self.eval_config.trust_remote_code,
        )

    def _get_tokenizer(self) -> PreTrainedTokenizerBase:
        if self.tokenizer is None:
            raise RuntimeError("API model tokenizer is not initialized.")
        return self.tokenizer

    def _build_completion_kwargs(
        self,
        messages: list[dict[str, str]],
        sampling_config: SamplingConfig,
    ) -> dict[str, Any]:
        do_sample = sampling_config.do_sample
        temperature = sampling_config.temperature
        if do_sample is False:
            temperature = 0.0

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.eval_config.max_judge_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if sampling_config.top_p is not None:
            kwargs["top_p"] = sampling_config.top_p
        if sampling_config.top_k is not None:
            kwargs["top_k"] = sampling_config.top_k
        if sampling_config.seed is not None:
            kwargs["seed"] = sampling_config.seed
        return kwargs

    @staticmethod
    def _extract_content(response: Any) -> str:
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if not choices:
                return ""
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return str(message.get("content") or "")
                return str(first.get("text") or "")
            message = getattr(first, "message", None)
            if message is not None:
                return str(getattr(message, "content", "") or "")
            return str(getattr(first, "text", "") or "")

        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            return str(getattr(message, "content", "") or "")
        return str(getattr(first, "text", "") or "")

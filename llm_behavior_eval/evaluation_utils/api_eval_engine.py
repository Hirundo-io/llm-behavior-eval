from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from typing import TYPE_CHECKING, Any, cast

from tqdm import tqdm

from .eval_engine import EvalDataset, EvalEngine
from .util_functions import load_tokenizer_with_transformers

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from .eval_config import EvaluationConfig
    from .sampling_config import SamplingConfig


JudgePrompt = str | list[dict[str, str]]

# Default concurrency for batch API calls (can be overridden via env var)
DEFAULT_API_BATCH_CONCURRENCY = 10
# When batch_size is not explicitly configured for API engines, we scale it
# relative to concurrency so each batch call can fully utilize parallelism.
DEFAULT_API_BATCH_MULTIPLIER = 5


class ApiEvalEngine(EvalEngine):
    def __init__(
        self, eval_config: EvaluationConfig, *, is_judge: bool = False
    ) -> None:
        self.eval_config = eval_config
        self.is_judge = is_judge
        self.dataset: EvalDataset | None = None
        self.model_name = (
            eval_config.judge_path_or_repo_id
            if is_judge
            else eval_config.model_path_or_repo_id
        )
        self._litellm = self._load_litellm()
        self._suppress_litellm_logging()
        self.tokenizer: PreTrainedTokenizerBase | None = None
        if not is_judge and eval_config.model_tokenizer_path_or_repo_id is not None:
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

    def _suppress_litellm_logging(self) -> None:
        """Suppress verbose LiteLLM logging unless explicitly enabled."""
        if os.environ.get("LITELLM_DEBUG"):
            return
        # Suppress LiteLLM's verbose output
        cast("Any", self._litellm).suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)

    def set_dataset(self, eval_dataset: EvalDataset) -> None:
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
        prompts: Sequence[JudgePrompt],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        if not prompts:
            return []

        # Get batch concurrency from env or use default
        concurrency = int(
            os.environ.get("LLM_EVAL_API_CONCURRENCY", DEFAULT_API_BATCH_CONCURRENCY)
        )

        # Build base kwargs (shared params)
        base_kwargs = self._build_base_completion_kwargs(sampling_config)

        # Normalize all messages
        all_messages = [self._normalize_messages(prompt) for prompt in prompts]

        # Use batch_completion for parallel requests with progress bar
        desc = "API judge calls" if self.is_judge else "API model calls"
        answers: list[str] = []

        # Process in chunks to show progress and control concurrency
        for i in tqdm(
            range(0, len(all_messages), concurrency),
            desc=desc,
            unit="batch",
            total=(len(all_messages) + concurrency - 1) // concurrency,
        ):
            chunk_messages = all_messages[i : i + concurrency]
            responses = self._litellm.batch_completion(
                model=self.model_name,
                messages=chunk_messages,
                **base_kwargs,
            )
            for response in responses:
                answers.append(self._extract_content(response))

        return answers

    def get_batch_size(self) -> int:
        if self.is_judge:
            if self.eval_config.judge_batch_size is not None:
                return max(1, int(self.eval_config.judge_batch_size))
            # For API judges, avoid defaulting to 1 which underutilizes concurrency.
            return self._get_default_api_batch_size()
        if self.eval_config.batch_size is not None:
            return max(1, int(self.eval_config.batch_size))
        # For API evaluated models, avoid defaulting to 1 which underutilizes concurrency.
        return self._get_default_api_batch_size()

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

    def _build_base_completion_kwargs(
        self,
        sampling_config: SamplingConfig,
    ) -> dict[str, Any]:
        """Build completion kwargs without model/messages (for batch_completion)."""
        do_sample = sampling_config.do_sample
        temperature = sampling_config.temperature
        if do_sample is False:
            temperature = 0.0

        kwargs: dict[str, Any] = {
            "max_tokens": self._get_max_new_tokens(self.eval_config, self.is_judge),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if sampling_config.top_p is not None:
            kwargs["top_p"] = sampling_config.top_p
        # Only pass top_k if it's a positive value (some providers like Azure don't support it)
        if sampling_config.top_k is not None and sampling_config.top_k > 0:
            kwargs["top_k"] = sampling_config.top_k
        if sampling_config.seed is not None:
            kwargs["seed"] = sampling_config.seed
        return kwargs

    def _get_default_api_batch_size(self) -> int:
        """Estimate a sensible default batch size for API engines.

        The API path uses LiteLLM's `batch_completion`, which can parallelize
        calls up to `LLM_EVAL_API_CONCURRENCY`. When the DataLoader batch size
        is left as 1, we only ever submit one request at a time and effectively
        disable parallelism. This heuristic chooses a larger default batch size
        so each batch call can saturate concurrency.
        """
        concurrency = int(
            os.environ.get("LLM_EVAL_API_CONCURRENCY", DEFAULT_API_BATCH_CONCURRENCY)
        )
        multiplier = int(
            os.environ.get(
                "LLM_EVAL_API_BATCH_MULTIPLIER", DEFAULT_API_BATCH_MULTIPLIER
            )
        )
        estimated = max(1, concurrency * max(1, multiplier))
        if self.dataset is not None:
            try:
                return max(1, min(len(self.dataset), estimated))
            except TypeError:
                # Some dataset implementations may not support len(); fall back.
                return estimated
        return estimated

    def _build_completion_kwargs(
        self,
        messages: list[dict[str, str]],
        sampling_config: SamplingConfig,
    ) -> dict[str, Any]:
        """Build full completion kwargs including model and messages."""
        kwargs = self._build_base_completion_kwargs(sampling_config)
        kwargs["model"] = self.model_name
        kwargs["messages"] = messages
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

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from .eval_engine import EvalDataset, JudgePrompt, PromptEvalEngine
from .litellm_utils import (
    call_litellm_decode,
    call_litellm_encode,
    suppress_litellm_logging,
    try_trim_messages_with_litellm,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .eval_config import EvaluationConfig
    from .sampling_config import SamplingConfig

# Default concurrency for batch API calls (can be overridden via env var)
DEFAULT_API_BATCH_CONCURRENCY = 10
# When batch_size is not explicitly configured for API engines, we scale it
# relative to concurrency so each batch call can fully utilize parallelism.
DEFAULT_API_BATCH_MULTIPLIER = 5


def _read_env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if min_value is not None:
        return max(min_value, parsed)
    return parsed


def _truncate_text_by_whitespace(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


class ApiEvalEngine(PromptEvalEngine):
    # API engines do not load a local tokenizer; the provider handles formatting.
    tokenizer: None = None

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
        suppress_litellm_logging(self._litellm)
        self._max_input_tokens: int | None = None
        self._token_truncation_warning_logged = False
        self._message_trim_warning_logged = False

    @staticmethod
    def _load_litellm():
        try:
            import litellm

            return litellm
        except ImportError as exc:
            raise ImportError(
                "litellm is required for API-based models. "
                "Install it with `uv pip install llm-behavior-eval[api-all]` "
                "or a provider extra like `[api-openai]`."
            ) from exc

    def set_dataset(self, eval_dataset: EvalDataset) -> None:
        self.dataset = eval_dataset

    def set_preprocess_limits(self, max_length: int, gt_max_length: int) -> None:
        self._max_input_tokens = max(0, int(max_length))
        del gt_max_length

    def should_combine_judge_prompt_groups(self) -> bool:
        return True

    def get_raw_text_truncator(self) -> Callable[[str, int], str] | None:
        return self._truncate_text_to_model_tokens

    def generate_answers_from_prompts(
        self,
        prompts: Sequence[JudgePrompt],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        if not prompts:
            return []

        # Get batch concurrency from env or use default
        concurrency = _read_env_int(
            "LLM_EVAL_API_CONCURRENCY",
            DEFAULT_API_BATCH_CONCURRENCY,
            min_value=1,
        )

        # Build base kwargs (shared params)
        base_kwargs = self._build_base_completion_kwargs(sampling_config)

        # Normalize all messages
        all_messages = [self._normalize_messages(prompt) for prompt in prompts]
        if self._max_input_tokens is not None:
            all_messages = [
                self._trim_messages_to_limit(messages) for messages in all_messages
            ]

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
        configured_batch_size = (
            self.eval_config.judge_batch_size
            if self.is_judge
            else self.eval_config.batch_size
        )
        if configured_batch_size is not None:
            return max(1, int(configured_batch_size))
        return self._get_default_api_batch_size()

    def free_model(self) -> None:
        return None

    @staticmethod
    def _normalize_messages(prompt: JudgePrompt) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    def _truncate_text_to_model_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        truncated = self._try_truncate_text_with_litellm(text, max_tokens)
        if truncated is not None:
            return truncated
        if not self._token_truncation_warning_logged:
            logging.warning(
                "Model-aware token truncation is unavailable for model '%s'; "
                "falling back to whitespace truncation.",
                self.model_name,
            )
            self._token_truncation_warning_logged = True
        return _truncate_text_by_whitespace(text, max_tokens)

    def _try_truncate_text_with_litellm(self, text: str, max_tokens: int) -> str | None:
        tokens = call_litellm_encode(self._litellm, self.model_name, text)
        if tokens is None:
            return None
        if len(tokens) <= max_tokens:
            return text
        return call_litellm_decode(self._litellm, self.model_name, tokens[:max_tokens])

    def _trim_messages_to_limit(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        max_tokens = self._max_input_tokens
        if max_tokens is None:
            return messages
        trimmed = try_trim_messages_with_litellm(
            self._litellm, self.model_name, messages, max_tokens
        )
        if trimmed is not None:
            return trimmed
        if not self._message_trim_warning_logged:
            logging.warning(
                "LiteLLM message trimming is unavailable for model '%s'; "
                "using preprocessed messages without additional prompt trimming.",
                self.model_name,
            )
            self._message_trim_warning_logged = True
        return messages

    def _build_base_completion_kwargs(
        self,
        sampling_config: SamplingConfig,
    ) -> dict[str, Any]:
        """Build completion kwargs without model/messages.

        Args:
            sampling_config: Sampling settings for completion generation.

        Returns:
            Provider kwargs suitable for LiteLLM `batch_completion`.
        """
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

        Returns:
            A positive batch size tuned from environment concurrency settings.
        """
        concurrency = _read_env_int(
            "LLM_EVAL_API_CONCURRENCY",
            DEFAULT_API_BATCH_CONCURRENCY,
            min_value=1,
        )
        multiplier = _read_env_int(
            "LLM_EVAL_API_BATCH_MULTIPLIER", DEFAULT_API_BATCH_MULTIPLIER
        )
        estimated = max(1, concurrency * max(1, multiplier))
        if self.dataset is not None:
            try:
                return max(1, min(len(self.dataset), estimated))
            except TypeError:
                # Some dataset implementations may not support len(); fall back.
                return estimated
        return estimated

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = (
            response.get("choices")
            if isinstance(response, dict)
            else getattr(response, "choices", None)
        )
        if not choices:
            return ""

        first = choices[0]
        message = (
            first.get("message")
            if isinstance(first, dict)
            else getattr(first, "message", None)
        )
        if isinstance(message, dict):
            return str(message.get("content") or "")
        if message is not None:
            return str(getattr(message, "content", "") or "")
        if isinstance(first, dict):
            return str(first.get("text") or "")
        return str(getattr(first, "text", "") or "")

    def format_prompt(self, messages: list[dict[str, str]]) -> JudgePrompt:
        """For API engines, return messages unchanged (no tokenization needed)."""
        return messages

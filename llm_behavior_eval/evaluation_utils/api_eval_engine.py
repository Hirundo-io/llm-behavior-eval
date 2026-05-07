from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from .eval_engine import EvalDataset, JudgePrompt, PromptEvalEngine
from .litellm_utils import (
    call_litellm_decode,
    call_litellm_encode,
    suppress_litellm_logging,
    try_trim_messages_with_litellm,
)
from .util_functions import truncate_text_by_whitespace

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .eval_config import EvaluationConfig
    from .sampling_config import SamplingConfig

# Default concurrency for batch API calls (can be overridden via env var)
DEFAULT_API_BATCH_CONCURRENCY = 10
# When batch_size is not explicitly configured for API engines, we scale it
# relative to concurrency so each batch call can fully utilize parallelism.
DEFAULT_API_BATCH_MULTIPLIER = 5


Message = dict[str, str]
ApiGenerationError = dict[str, str | int | bool]


class ApiEvalEngineError(RuntimeError):
    """Raised when API inference fails before a valid model response exists."""


@dataclass(frozen=True)
class _ErrorClassification:
    kind: str
    skippable: bool = False
    retriable: bool = False


class _CompletionKwargs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None


class _LiteLLMMessage(Protocol):
    @property
    def content(self) -> str | None: ...


class _LiteLLMChoice(Protocol):
    @property
    def message(self) -> _LiteLLMMessage: ...


class _LiteLLMResponse(Protocol):
    @property
    def id(self) -> str | None: ...

    @property
    def model(self) -> str | None: ...

    @property
    def object(self) -> str | None: ...

    @property
    def choices(self) -> Sequence[_LiteLLMChoice]: ...


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


class ApiEvalEngine(PromptEvalEngine):
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
        self._last_generation_errors: list[ApiGenerationError | None] = []
        self._concurrency = _read_env_int(
            "LLM_EVAL_API_CONCURRENCY",
            DEFAULT_API_BATCH_CONCURRENCY,
            min_value=1,
        )

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
        self._max_input_tokens = max(0, max_length)
        del gt_max_length

    def should_combine_judge_prompt_groups(self) -> bool:
        return True

    def get_raw_text_truncator(self) -> Callable[[str, int], str] | None:
        return self._truncate_text_to_model_tokens

    def get_batch_size(self) -> int:
        configured = (
            self.eval_config.judge_batch_size
            if self.is_judge
            else self.eval_config.batch_size
        )
        if configured is not None:
            return max(1, configured)
        return self._get_default_api_batch_size()

    def free_model(self) -> None:
        return None

    def _validate_model_identifier(self) -> None:
        try:
            _model, provider, *_rest = self._litellm.get_llm_provider(
                model=self.model_name
            )
        except Exception as exc:
            raise self._build_api_error(exc) from exc
        if provider == "azure":
            self._validate_azure_environment()

    def _validate_azure_environment(self) -> None:
        has_token = any(
            os.getenv(name)
            for name in (
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_AD_TOKEN",
                "AZURE_API_KEY",
            )
        )
        if has_token:
            return
        raise ApiEvalEngineError(
            "LiteLLM API calls for Azure require Azure OpenAI credentials. "
            "Set AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN before running "
            f"model '{self.model_name}'. The current environment appears to only "
            "have non-Azure credentials available."
        )

    def _build_api_error(self, error: Exception) -> ApiEvalEngineError:
        role = "judge" if self.is_judge else "model"
        provider_prefix = self.model_name.split("/", 1)[0]
        provider_values = [
            provider.value if hasattr(provider, "value") else str(provider)
            for provider in self._litellm.provider_list
        ]
        provider_hint = ", ".join(sorted(provider_values))
        return ApiEvalEngineError(
            f"LiteLLM API {role} call failed for model '{self.model_name}' "
            f"with {type(error).__name__}: {error}. "
            f"Configured provider prefix: '{provider_prefix}'. "
            f"Valid LiteLLM provider prefixes include: {provider_hint}."
        )

    def _should_retry_or_skip(self, classification: _ErrorClassification) -> bool:
        return self.eval_config.api_skip_errors and (
            classification.skippable or classification.retriable
        )

    def _sleep_before_retry(self, retry_count: int) -> None:
        backoff = max(0.0, self.eval_config.api_retry_backoff_seconds)
        if backoff <= 0:
            return
        time.sleep(backoff * (2 ** max(0, retry_count - 1)))

    def _record_api_skip(
        self,
        prompt_index: int,
        error: Exception,
        error_kind: str,
        retry_count: int,
    ) -> None:
        error_record: ApiGenerationError = {
            "api_error_type": type(error).__name__,
            "api_error_message": str(error),
            "api_error_kind": error_kind,
            "api_error_retry_count": retry_count,
            "api_error_skipped": True,
        }
        self._last_generation_errors[prompt_index] = error_record
        logging.warning(
            "Skipping API %s response for model '%s' at prompt index %s after %s retries: %s: %s",
            "judge" if self.is_judge else "model",
            self.model_name,
            prompt_index,
            retry_count,
            type(error).__name__,
            error,
        )

    @staticmethod
    def _litellm_error_classes() -> tuple[
        tuple[type[Exception], ...],
        tuple[type[Exception], ...],
        tuple[type[Exception], ...],
    ]:
        from litellm import (
            APIConnectionError,
            BadGatewayError,
            BadRequestError,
            ContentPolicyViolationError,
            InternalServerError,
            InvalidRequestError,
            RateLimitError,
            RouterRateLimitError,
            ServiceUnavailableError,
            Timeout,
        )

        return (
            (ContentPolicyViolationError,),
            (BadRequestError, InvalidRequestError),
            (
                APIConnectionError,
                BadGatewayError,
                InternalServerError,
                RateLimitError,
                RouterRateLimitError,
                ServiceUnavailableError,
                Timeout,
            ),
        )

    @staticmethod
    def _classify_api_error(error: Exception) -> _ErrorClassification:
        (
            content_policy_errors,
            content_policy_wrappers,
            transient_errors,
        ) = ApiEvalEngine._litellm_error_classes()

        if isinstance(error, content_policy_errors):
            return _ErrorClassification("content_policy", skippable=True)

        # Azure/LiteLLM sometimes wraps ContentPolicyViolationError in a
        # BadRequestError while preserving the concrete exception name in the
        # message. Keep this wrapper-specific check narrow so unrelated
        # BadRequestError/InvalidRequestError failures remain fatal.
        if isinstance(
            error, content_policy_wrappers
        ) and "ContentPolicyViolationError" in str(error):
            return _ErrorClassification("content_policy", skippable=True)

        if isinstance(error, transient_errors):
            return _ErrorClassification("transient", retriable=True)

        return _ErrorClassification("fatal")

    def _generate_chunk_with_retries(
        self,
        chunk_messages: list[list[Message]],
        completion_kwargs: dict[str, Any],
        *,
        start_index: int,
    ) -> list[str]:
        answers: list[str | None] = [None for _ in chunk_messages]
        pending = list(range(len(chunk_messages)))
        retry_counts = {index: 0 for index in pending}
        max_retries = max(0, self.eval_config.api_retry_attempts)

        while pending:
            pending_messages = [chunk_messages[index] for index in pending]
            try:
                responses = self._litellm.batch_completion(
                    model=self.model_name,
                    messages=pending_messages,
                    **completion_kwargs,
                )
            except Exception as exc:
                classification = self._classify_api_error(exc)
                if not self._should_retry_or_skip(classification):
                    raise self._build_api_error(exc) from exc
                responses = [exc for _ in pending]

            if len(responses) != len(pending):
                raise ApiEvalEngineError(
                    "LiteLLM API batch_completion returned an unexpected number "
                    f"of responses ({len(responses)} for {len(pending)} prompts)."
                )

            next_pending: list[int] = []
            for local_index, response in zip(pending, responses, strict=True):
                if not isinstance(response, Exception):
                    answers[local_index] = self._extract_content(response)
                    continue

                classification = self._classify_api_error(response)
                if not self._should_retry_or_skip(classification):
                    raise self._build_api_error(response) from response

                if classification.retriable and retry_counts[local_index] < max_retries:
                    retry_counts[local_index] += 1
                    next_pending.append(local_index)
                    continue

                if classification.skippable or classification.retriable:
                    retry_count = retry_counts[local_index]
                    self._record_api_skip(
                        start_index + local_index,
                        response,
                        classification.kind,
                        retry_count,
                    )
                    answers[local_index] = ""
                    continue

                raise self._build_api_error(response) from response

            pending = next_pending
            if pending:
                self._sleep_before_retry(max(retry_counts[index] for index in pending))

        return [answer or "" for answer in answers]

    @staticmethod
    def _messages_for_completion(prompt: JudgePrompt) -> list[Message]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    def generate_answers_from_prompts(
        self,
        prompts: Sequence[JudgePrompt],
        sampling_config: SamplingConfig,
    ) -> list[str]:
        if not prompts:
            return []

        self._validate_model_identifier()
        base_kwargs = self._build_base_completion_kwargs(sampling_config)
        completion_kwargs = base_kwargs.model_dump(exclude_none=True)
        all_messages = [self._messages_for_completion(prompt) for prompt in prompts]
        self._last_generation_errors = [None for _ in all_messages]
        if self._max_input_tokens is not None:
            all_messages = [
                self._trim_messages_to_limit(messages) for messages in all_messages
            ]

        # Use batch_completion for parallel requests with progress bar
        desc = "API judge calls" if self.is_judge else "API model calls"
        answers: list[str] = []

        # Process in chunks to show progress and control concurrency
        concurrency = self._concurrency
        for i in tqdm(
            range(0, len(all_messages), concurrency),
            desc=desc,
            unit="batch",
            total=(len(all_messages) + concurrency - 1) // concurrency,
        ):
            chunk_messages = all_messages[i : i + concurrency]
            chunk_answers = self._generate_chunk_with_retries(
                chunk_messages,
                completion_kwargs,
                start_index=i,
            )
            answers.extend(chunk_answers)

        return answers

    def get_last_generation_errors(self) -> list[ApiGenerationError | None]:
        return list(self._last_generation_errors)

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
        return truncate_text_by_whitespace(text, max_tokens)

    def _try_truncate_text_with_litellm(self, text: str, max_tokens: int) -> str | None:
        tokens = call_litellm_encode(self._litellm, self.model_name, text)
        if tokens is None:
            return None
        if len(tokens) <= max_tokens:
            return text
        return call_litellm_decode(self._litellm, self.model_name, tokens[:max_tokens])

    def _trim_messages_to_limit(
        self,
        messages: list[Message],
    ) -> list[Message]:
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
    ) -> _CompletionKwargs:
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

        kwargs = _CompletionKwargs(
            max_tokens=(
                self.eval_config.max_judge_tokens
                if self.is_judge
                else self.eval_config.max_answer_tokens
            )
        )
        if temperature is not None:
            kwargs.temperature = temperature
        # Avoid sending neutral sampling defaults that some LiteLLM providers reject.
        if sampling_config.top_p is not None and sampling_config.top_p != 1.0:
            kwargs.top_p = sampling_config.top_p
        # Only pass top_k if it's a positive value (some providers like Azure don't support it)
        if sampling_config.top_k is not None and sampling_config.top_k > 0:
            kwargs.top_k = sampling_config.top_k
        if sampling_config.seed is not None:
            kwargs.seed = sampling_config.seed
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
        multiplier = _read_env_int(
            "LLM_EVAL_API_BATCH_MULTIPLIER", DEFAULT_API_BATCH_MULTIPLIER
        )
        estimated = max(1, self._concurrency * max(1, multiplier))
        if self.dataset is not None:
            try:
                return max(1, min(len(self.dataset), estimated))
            except TypeError:
                # Some dataset implementations may not support len(); fall back.
                logging.info(
                    "Dataset %s does not support len(); using default API batch size %s.",
                    type(self.dataset).__name__,
                    estimated,
                )
                return estimated
        return estimated

    @staticmethod
    def _extract_content(response: Any | None) -> str:
        if response is None:
            return ""
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError):
            return ""
        if content is None:
            response_id = response.id if hasattr(response, "id") else None
            response_model = response.model if hasattr(response, "model") else None
            response_object = response.object if hasattr(response, "object") else None
            logging.debug(
                "LiteLLM response had empty message content "
                "(response_id=%s, response_model=%s, response_object=%s).",
                response_id,
                response_model,
                response_object,
            )
            return ""
        return content

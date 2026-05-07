from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from llm_behavior_eval.evaluation_utils.api_eval_engine import (
    ApiEvalEngine,
    ApiEvalEngineError,
)
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.eval_engine import PromptEvalEngine
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig


def _make_response(content: str | None) -> Any:
    """Build a ModelResponse-shaped object matching LiteLLM's real return type."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class FakeLiteLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.suppress_debug_info = False
        self.provider_list = ["azure", "openai"]

    def completion(self, **kwargs) -> Any:
        self.calls.append(kwargs)
        return _make_response("ok")

    def get_llm_provider(self, *, model: str) -> tuple[str, str, None, None]:
        provider_prefix = model.split("/", 1)[0]
        if provider_prefix not in self.provider_list:
            raise ValueError("LLM Provider NOT provided")
        return model, provider_prefix, None, None

    def batch_completion(self, model, messages, **kwargs) -> list[Any]:
        """Batch completion returns a list of responses, one per message list."""
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return [_make_response("ok") for _ in messages]


def _content_policy_error() -> Exception:
    return litellm.ContentPolicyViolationError(
        "response was filtered",
        model="openai/gpt-4o",
        llm_provider="openai",
    )


def _rate_limit_error() -> Exception:
    return litellm.RateLimitError(
        "rate limited",
        model="openai/gpt-4o",
        llm_provider="openai",
    )


def _auth_error() -> Exception:
    return litellm.AuthenticationError(
        "bad key",
        model="openai/gpt-4o",
        llm_provider="openai",
    )


def patch_litellm(monkeypatch) -> FakeLiteLLM:
    """Patch ApiEvalEngine._load_litellm to return a fake litellm module."""
    fake_litellm = FakeLiteLLM()
    monkeypatch.setattr(
        ApiEvalEngine, "_load_litellm", staticmethod(lambda: fake_litellm)
    )
    return fake_litellm


def test_api_eval_engine_calls_litellm_completion(tmp_path, monkeypatch) -> None:
    fake_litellm = patch_litellm(monkeypatch)

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        judge_engine="api",
        judge_path_or_repo_id="openai/gpt-4o-mini",
    )

    engine = ApiEvalEngine(evaluation_config, is_judge=True)
    prompts = [[{"role": "user", "content": "Check this."}]]
    sampling_config = SamplingConfig(
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        top_k=2,
        seed=123,
    )

    answers = engine.generate_answers_from_prompts(prompts, sampling_config)

    assert answers == ["ok"]
    assert len(fake_litellm.calls) == 1
    call_kwargs = fake_litellm.calls[0]
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    # batch_completion receives list of message lists
    assert call_kwargs["messages"] == [prompts[0]]
    assert call_kwargs["max_tokens"] == evaluation_config.max_judge_tokens
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["top_k"] == 2
    assert call_kwargs["seed"] == 123


def test_api_eval_engine_model_uses_answer_tokens(tmp_path, monkeypatch) -> None:
    fake_litellm = patch_litellm(monkeypatch)

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
        max_answer_tokens=101,
        max_judge_tokens=7,
    )

    engine = ApiEvalEngine(evaluation_config, is_judge=False)
    prompts = [[{"role": "user", "content": "Answer this."}]]
    sampling_config = SamplingConfig(
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        top_k=2,
        seed=123,
    )

    answers = engine.generate_answers_from_prompts(prompts, sampling_config)

    assert answers == ["ok"]
    call_kwargs = fake_litellm.calls[0]
    assert call_kwargs["max_tokens"] == evaluation_config.max_answer_tokens


def test_api_eval_engine_omits_default_top_p(tmp_path, monkeypatch) -> None:
    fake_litellm = patch_litellm(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="azure/models/gpt-5.4-nano",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    answers = engine.generate_answers_from_prompts(
        [[{"role": "user", "content": "hello"}]],
        SamplingConfig(do_sample=False, top_p=1.0),
    )

    assert answers == ["ok"]
    assert "top_p" not in fake_litellm.calls[0]


def test_api_eval_engine_allows_missing_model_tokenizer(tmp_path, monkeypatch) -> None:
    patch_litellm(monkeypatch)

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )

    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    assert engine.tokenizer is None


def test_api_eval_engine_implements_prompt_interface(tmp_path, monkeypatch) -> None:
    patch_litellm(monkeypatch)

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )

    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    assert isinstance(engine, PromptEvalEngine)

    assert engine.tokenizer is None


def test_api_eval_engine_default_batch_size_uses_env_and_dataset(
    tmp_path, monkeypatch
) -> None:
    patch_litellm(monkeypatch)
    monkeypatch.setenv("LLM_EVAL_API_CONCURRENCY", "2")
    monkeypatch.setenv("LLM_EVAL_API_BATCH_MULTIPLIER", "3")

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
        batch_size=None,
    )

    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    class FakeDataset:
        def __len__(self) -> int:
            return 7

        def __getitem__(self, index: int) -> Any:
            raise IndexError

    engine.set_dataset(FakeDataset())
    assert engine.get_batch_size() == 6


def test_api_eval_engine_default_judge_batch_size_uses_env_and_dataset(
    tmp_path, monkeypatch
) -> None:
    patch_litellm(monkeypatch)
    monkeypatch.setenv("LLM_EVAL_API_CONCURRENCY", "4")
    monkeypatch.setenv("LLM_EVAL_API_BATCH_MULTIPLIER", "2")

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        judge_engine="api",
        judge_path_or_repo_id="openai/gpt-4o-mini",
        judge_batch_size=None,
    )

    engine = ApiEvalEngine(evaluation_config, is_judge=True)

    class FakeDataset:
        def __len__(self) -> int:
            return 5

        def __getitem__(self, index: int) -> Any:
            raise IndexError

    engine.set_dataset(FakeDataset())
    assert engine.get_batch_size() == 5


@pytest.mark.parametrize("concurrency_value", ["0", "-3"])
def test_api_eval_engine_clamps_invalid_concurrency_for_batching(
    tmp_path, monkeypatch, concurrency_value: str
) -> None:
    fake_litellm = patch_litellm(monkeypatch)
    monkeypatch.setenv("LLM_EVAL_API_CONCURRENCY", concurrency_value)

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    prompts = [
        [{"role": "user", "content": "First"}],
        [{"role": "user", "content": "Second"}],
    ]
    answers = engine.generate_answers_from_prompts(
        prompts,
        SamplingConfig(do_sample=False),
    )

    assert answers == ["ok", "ok"]
    assert len(fake_litellm.calls) == 2


def test_api_eval_engine_raw_text_truncator_uses_model_tokenization(
    tmp_path, monkeypatch
) -> None:
    class FakeLiteLLMWithTokenizer(FakeLiteLLM):
        def __init__(self) -> None:
            super().__init__()
            self.encode_calls: list[dict[str, Any]] = []
            self.decode_calls: list[dict[str, Any]] = []

        def encode(self, model: str, text: str, **kwargs):
            self.encode_calls.append({"model": model, "text": text})
            return list(text)

        def decode(self, model: str, tokens: list, **kwargs):
            self.decode_calls.append({"model": model, "tokens": tokens})
            return "".join(str(token) for token in tokens)

    fake_litellm = FakeLiteLLMWithTokenizer()
    monkeypatch.setattr(
        ApiEvalEngine, "_load_litellm", staticmethod(lambda: fake_litellm)
    )
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    truncator = engine.get_raw_text_truncator()
    assert truncator is not None
    assert truncator("abcdef", 3) == "abc"
    assert len(fake_litellm.encode_calls) == 1
    assert len(fake_litellm.decode_calls) == 1


def test_api_eval_engine_raw_text_truncator_warns_and_falls_back(
    tmp_path, monkeypatch, caplog
) -> None:
    fake_litellm = patch_litellm(monkeypatch)

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)
    truncator = engine.get_raw_text_truncator()
    assert truncator is not None

    caplog.set_level(logging.WARNING)
    assert truncator("one two three four", 2) == "one two"
    assert truncator("five six seven", 2) == "five six"
    warning_records = [
        record
        for record in caplog.records
        if "Model-aware token truncation is unavailable" in record.message
    ]
    assert len(warning_records) == 1
    assert len(fake_litellm.calls) == 0


def test_api_eval_engine_applies_prompt_level_message_trimming(
    tmp_path, monkeypatch
) -> None:
    trim_calls: list[dict[str, Any]] = []

    def fake_trim_messages(messages, model=None, max_tokens=None, **kwargs):
        trim_calls.append(
            {"messages": messages, "model": model, "max_tokens": max_tokens}
        )
        return [
            {
                "role": message["role"],
                "content": str(message.get("content", ""))[:max_tokens],
            }
            for message in messages
        ]

    # Patch at the import source so the deferred `from litellm.utils import
    # trim_messages` inside try_trim_messages_with_litellm picks up the fake.
    monkeypatch.setattr("litellm.utils.trim_messages", fake_trim_messages)

    fake_litellm = patch_litellm(monkeypatch)
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)
    engine.set_preprocess_limits(max_length=3, gt_max_length=2)

    prompts = [[{"role": "user", "content": "abcdef"}]]
    answers = engine.generate_answers_from_prompts(
        prompts, SamplingConfig(do_sample=False)
    )

    assert answers == ["ok"]
    # Verify the fake was actually called
    assert len(trim_calls) == 1
    # Verify trimmed content reached batch_completion
    sent_messages = fake_litellm.calls[0]["messages"]
    assert sent_messages == [[{"role": "user", "content": "abc"}]]


def test_api_eval_engine_extract_content_from_response_object(
    tmp_path, monkeypatch
) -> None:
    """_extract_content works with the standard ModelResponse-shaped object."""
    response = _make_response("hello world")
    assert ApiEvalEngine._extract_content(response) == "hello world"


def test_api_eval_engine_extract_content_preserves_valid_empty_response(
    tmp_path, monkeypatch
) -> None:
    """A valid LiteLLM response with null content remains a model empty response."""
    response = _make_response(None)
    assert ApiEvalEngine._extract_content(response) == ""


def test_api_eval_engine_extract_content_returns_empty_on_error(
    tmp_path, monkeypatch
) -> None:
    """_extract_content returns empty string for malformed/missing responses."""
    assert ApiEvalEngine._extract_content(None) == ""
    assert ApiEvalEngine._extract_content(SimpleNamespace(choices=[])) == ""
    assert ApiEvalEngine._extract_content(SimpleNamespace()) == ""


def test_api_eval_engine_rejects_unrecognized_litellm_provider_prefix(
    tmp_path, monkeypatch
) -> None:
    patch_litellm(monkeypatch)
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="azure-openai/models/gpt-5.4-nano",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    with pytest.raises(ApiEvalEngineError) as exc_info:
        engine.generate_answers_from_prompts(
            ["Explain this."],
            SamplingConfig(do_sample=False),
        )

    message = str(exc_info.value)
    assert "LiteLLM API model call failed" in message
    assert "azure-openai/models/gpt-5.4-nano" in message
    assert "Configured provider prefix: 'azure-openai'" in message
    assert "Valid LiteLLM provider prefixes include" in message
    assert "azure" in message


def test_api_eval_engine_rejects_azure_without_credentials(
    tmp_path, monkeypatch
) -> None:
    patch_litellm(monkeypatch)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="azure/gpt-5.4-nano",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    with pytest.raises(ApiEvalEngineError) as exc_info:
        engine.generate_answers_from_prompts(
            ["Explain this."],
            SamplingConfig(do_sample=False),
        )

    assert "require Azure OpenAI credentials" in str(exc_info.value)
    assert "AZURE_OPENAI_API_KEY" in str(exc_info.value)


def test_api_eval_engine_raises_on_batch_completion_exception(
    tmp_path, monkeypatch
) -> None:
    class FakeLiteLLMWithFailure(FakeLiteLLM):
        def batch_completion(self, model, messages, **kwargs) -> list[Any]:
            self.calls.append({"model": model, "messages": messages, **kwargs})
            return [RuntimeError("provider rejected request")]

    fake_litellm = FakeLiteLLMWithFailure()
    monkeypatch.setattr(
        ApiEvalEngine, "_load_litellm", staticmethod(lambda: fake_litellm)
    )
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    with pytest.raises(ApiEvalEngineError) as exc_info:
        engine.generate_answers_from_prompts(
            [[{"role": "user", "content": "hello"}]],
            SamplingConfig(do_sample=False),
        )

    message = str(exc_info.value)
    assert "LiteLLM API model call failed" in message
    assert "RuntimeError: provider rejected request" in message
    assert "openai/gpt-4o" in message


def test_api_eval_engine_records_content_policy_error_and_continues(
    tmp_path, monkeypatch, caplog
) -> None:
    class FakeLiteLLMWithPolicyFailure(FakeLiteLLM):
        def batch_completion(self, model, messages, **kwargs) -> list[Any]:
            self.calls.append({"model": model, "messages": messages, **kwargs})
            return [_make_response("ok"), _content_policy_error()]

    fake_litellm = FakeLiteLLMWithPolicyFailure()
    monkeypatch.setattr(
        ApiEvalEngine, "_load_litellm", staticmethod(lambda: fake_litellm)
    )
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    caplog.set_level(logging.WARNING)
    answers = engine.generate_answers_from_prompts(
        [
            [{"role": "user", "content": "allowed"}],
            [{"role": "user", "content": "blocked"}],
        ],
        SamplingConfig(do_sample=False),
    )

    assert answers == ["ok", ""]
    assert fake_litellm.calls[0]["messages"] == [
        [{"role": "user", "content": "allowed"}],
        [{"role": "user", "content": "blocked"}],
    ]
    assert engine.get_last_generation_errors() == [
        None,
        {
            "api_error_type": "ContentPolicyViolationError",
            "api_error_message": "litellm.BadRequestError: litellm.ContentPolicyViolationError: response was filtered",
            "api_error_kind": "content_policy",
            "api_error_retry_count": 0,
            "api_error_skipped": True,
        },
    ]
    assert "Skipping API model response" in caplog.text


def test_api_eval_engine_retries_transient_error_then_records_skip(
    tmp_path, monkeypatch
) -> None:
    class FakeLiteLLMWithTransientFailure(FakeLiteLLM):
        def batch_completion(self, model, messages, **kwargs) -> list[Any]:
            self.calls.append({"model": model, "messages": messages, **kwargs})
            return [_rate_limit_error() for _ in messages]

    fake_litellm = FakeLiteLLMWithTransientFailure()
    monkeypatch.setattr(
        ApiEvalEngine, "_load_litellm", staticmethod(lambda: fake_litellm)
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.api_eval_engine.time.sleep",
        lambda _seconds: None,
    )
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
        api_retry_attempts=3,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    answers = engine.generate_answers_from_prompts(
        [[{"role": "user", "content": "hello"}]],
        SamplingConfig(do_sample=False),
    )

    assert answers == [""]
    assert len(fake_litellm.calls) == 4
    assert engine.get_last_generation_errors()[0] == {
        "api_error_type": "RateLimitError",
        "api_error_message": "litellm.RateLimitError: rate limited",
        "api_error_kind": "transient",
        "api_error_retry_count": 3,
        "api_error_skipped": True,
    }


def test_api_eval_engine_retries_transient_error_until_success(
    tmp_path, monkeypatch
) -> None:
    class FakeLiteLLMWithTransientThenSuccess(FakeLiteLLM):
        def batch_completion(self, model, messages, **kwargs) -> list[Any]:
            self.calls.append({"model": model, "messages": messages, **kwargs})
            if len(self.calls) == 1:
                return [_rate_limit_error() for _ in messages]
            return [_make_response("ok") for _ in messages]

    fake_litellm = FakeLiteLLMWithTransientThenSuccess()
    monkeypatch.setattr(
        ApiEvalEngine, "_load_litellm", staticmethod(lambda: fake_litellm)
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.api_eval_engine.time.sleep",
        lambda _seconds: None,
    )
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    answers = engine.generate_answers_from_prompts(
        [[{"role": "user", "content": "hello"}]],
        SamplingConfig(do_sample=False),
    )

    assert answers == ["ok"]
    assert len(fake_litellm.calls) == 2
    assert engine.get_last_generation_errors() == [None]


def test_api_eval_engine_fatal_error_still_raises(tmp_path, monkeypatch) -> None:
    class FakeLiteLLMWithFatalFailure(FakeLiteLLM):
        def batch_completion(self, model, messages, **kwargs) -> list[Any]:
            self.calls.append({"model": model, "messages": messages, **kwargs})
            return [_auth_error()]

    fake_litellm = FakeLiteLLMWithFatalFailure()
    monkeypatch.setattr(
        ApiEvalEngine, "_load_litellm", staticmethod(lambda: fake_litellm)
    )
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    with pytest.raises(ApiEvalEngineError):
        engine.generate_answers_from_prompts(
            [[{"role": "user", "content": "hello"}]],
            SamplingConfig(do_sample=False),
        )


def test_api_eval_engine_no_api_skip_errors_restores_fail_fast(
    tmp_path, monkeypatch
) -> None:
    class FakeLiteLLMWithPolicyFailure(FakeLiteLLM):
        def batch_completion(self, model, messages, **kwargs) -> list[Any]:
            self.calls.append({"model": model, "messages": messages, **kwargs})
            return [_content_policy_error()]

    fake_litellm = FakeLiteLLMWithPolicyFailure()
    monkeypatch.setattr(
        ApiEvalEngine, "_load_litellm", staticmethod(lambda: fake_litellm)
    )
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
        api_skip_errors=False,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    with pytest.raises(ApiEvalEngineError):
        engine.generate_answers_from_prompts(
            [[{"role": "user", "content": "hello"}]],
            SamplingConfig(do_sample=False),
        )


def test_api_eval_engine_normalizes_string_prompts(tmp_path, monkeypatch) -> None:
    fake_litellm = patch_litellm(monkeypatch)
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    answers = engine.generate_answers_from_prompts(
        ["Explain this."],
        SamplingConfig(do_sample=False),
    )

    assert answers == ["ok"]
    assert fake_litellm.calls[0]["messages"] == [
        [{"role": "user", "content": "Explain this."}]
    ]


@pytest.mark.parametrize("top_k", [0, -5])
def test_api_eval_engine_omits_nonpositive_top_k(
    tmp_path, monkeypatch, top_k: int
) -> None:
    fake_litellm = patch_litellm(monkeypatch)
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        results_dir=tmp_path,
    )
    engine = ApiEvalEngine(evaluation_config, is_judge=False)

    answers = engine.generate_answers_from_prompts(
        [[{"role": "user", "content": "hello"}]],
        SamplingConfig(do_sample=True, temperature=0.2, top_p=0.8, top_k=top_k),
    )

    assert answers == ["ok"]
    call_kwargs = fake_litellm.calls[0]
    assert call_kwargs["top_p"] == 0.8
    assert "top_k" not in call_kwargs

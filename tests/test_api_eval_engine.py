from __future__ import annotations

import pytest

from llm_behavior_eval.evaluation_utils.api_eval_engine import ApiEvalEngine
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.eval_engine import (
    PromptEvalEngine,
    TensorEvalEngine,
)
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig


class FakeLiteLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.suppress_debug_info = False

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    def batch_completion(self, model, messages, **kwargs):
        """Batch completion returns a list of responses, one per message list."""
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return [{"choices": [{"message": {"content": "ok"}}]} for _ in messages]


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
    assert not isinstance(engine, TensorEvalEngine)

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

        def __getitem__(self, index: int) -> object:
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

        def __getitem__(self, index: int) -> object:
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

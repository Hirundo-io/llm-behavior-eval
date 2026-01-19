from __future__ import annotations

import importlib

from llm_behavior_eval.evaluation_utils.api_eval_engine import ApiEvalEngine
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig


class FakeLiteLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}


def test_api_eval_engine_calls_litellm_completion(tmp_path, monkeypatch) -> None:
    fake_litellm = FakeLiteLLM()
    real_import_module = importlib.import_module

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "litellm" else None,
    )

    def fake_import_module(name: str, *args, **kwargs):
        if name == "litellm":
            return fake_litellm
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

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
    assert call_kwargs["messages"] == prompts[0]
    assert call_kwargs["max_tokens"] == evaluation_config.max_judge_tokens
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["top_k"] == 2
    assert call_kwargs["seed"] == 123

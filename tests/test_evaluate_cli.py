from __future__ import annotations

from dataclasses import dataclass

import pytest

import llm_behavior_eval.evaluate as evaluate
from llm_behavior_eval import DatasetConfig, EvaluationConfig


class _StubEvaluator:
    def evaluate(self) -> None:
        return None


@dataclass
class CapturedConfigs:
    eval_config: EvaluationConfig
    dataset_config: DatasetConfig


@pytest.fixture
def capture_eval_config(monkeypatch: pytest.MonkeyPatch) -> list[EvaluationConfig]:
    captured: list[EvaluationConfig] = []

    def _fake_create(
        eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> _StubEvaluator:
        captured.append(eval_config)
        return _StubEvaluator()

    monkeypatch.setattr(
        evaluate.EvaluateFactory,
        "create_evaluator",
        staticmethod(_fake_create),
    )
    return captured


@pytest.fixture
def capture_configs(monkeypatch: pytest.MonkeyPatch) -> list[CapturedConfigs]:
    captured: list[CapturedConfigs] = []

    def _fake_create(
        eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> _StubEvaluator:
        captured.append(
            CapturedConfigs(eval_config=eval_config, dataset_config=dataset_config)
        )
        return _StubEvaluator()

    monkeypatch.setattr(
        evaluate.EvaluateFactory,
        "create_evaluator",
        staticmethod(_fake_create),
    )
    return captured


def test_main_applies_max_samples_option(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu", max_samples=42)
    assert capture_eval_config[-1].max_samples == 42


def test_main_runs_full_dataset_when_nonpositive_max_samples(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu", max_samples=0)
    assert capture_eval_config[-1].max_samples is None


def test_main_passes_judge_quantization_flag(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu", use_4bit_judge=True)
    assert capture_eval_config[-1].use_4bit_judge is True


def test_main_sets_inference_engine_and_sampling(
    capture_configs: list[CapturedConfigs],
) -> None:
    evaluate.main(
        "fake/model",
        "hallu",
        inference_engine="vllm",
        vllm_max_model_len=8192,
        vllm_judge_max_model_len=4096,
        sample=True,
        temperature=0.3,
        top_p=0.7,
        top_k=12,
        seed=123,
    )
    captured = capture_configs[-1]
    eval_config = captured.eval_config
    dataset_config = captured.dataset_config

    assert isinstance(eval_config, EvaluationConfig)
    assert eval_config.inference_engine == "vllm"
    assert eval_config.vllm_max_model_len == 8192
    assert eval_config.vllm_judge_max_model_len == 4096
    assert eval_config.sampling_config.temperature == 0.3
    assert eval_config.sampling_config.top_p == 0.7
    assert eval_config.sampling_config.top_k == 12
    assert eval_config.sampling_config.seed == 123
    assert dataset_config.seed == 123

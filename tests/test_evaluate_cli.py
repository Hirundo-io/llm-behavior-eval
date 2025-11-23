from __future__ import annotations

import pytest

import llm_behavior_eval.evaluate as evaluate
from llm_behavior_eval import DatasetConfig, EvaluationConfig


class _StubEvaluator:
    def evaluate(self) -> None:
        return None


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

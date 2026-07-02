from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from llm_behavior_eval.evaluate import _behavior_presets
from llm_behavior_eval.evaluation_utils.custom_dataset import (
    free_text_preprocess_function,
)
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
    FreeTextPromptInjectionEvaluator,
    _InjectionGenerationRecord,
)

if TYPE_CHECKING:
    from pathlib import Path

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine

SavedInjectionResult = dict[str, object]


class StubTokenizer:
    def __call__(
        self, texts: str | list[str], **_kwargs: object
    ) -> dict[str, list[list[int]]]:
        normalized_texts = [texts] if isinstance(texts, str) else texts
        return {
            "input_ids": [[index + 1] for index, _text in enumerate(normalized_texts)],
            "attention_mask": [[1] for _text in normalized_texts],
        }


def test_bloom_injection_behavior_presets() -> None:
    assert _behavior_presets("injection:bloom") == [
        "hirundo-io/bloom-prompt-injection-free-text"
    ]
    assert _behavior_presets("injection:all") == [
        "hirundo-io/bloom-prompt-injection-free-text",
        "hirundo-io/prompt-injection-purple-llama",
    ]
    assert _behavior_presets("prompt-injection") == [
        "hirundo-io/prompt-injection-purple-llama"
    ]


def test_injection_factory_routes_bloom_and_purple_llama(
    monkeypatch, tmp_path: Path
) -> None:
    created: list[str] = []

    def fake_init(
        self: FreeTextPromptInjectionEvaluator,
        eval_config: EvaluationConfig,
        dataset_config: DatasetConfig,
    ) -> None:
        created.append(dataset_config.file_path)

    monkeypatch.setattr(FreeTextPromptInjectionEvaluator, "__init__", fake_init)

    eval_config = EvaluationConfig(
        model_path_or_repo_id="test-model", results_dir=tmp_path
    )
    dataset_ids = [
        "hirundo-io/bloom-prompt-injection-free-text",
        "hirundo-io/prompt-injection-purple-llama",
    ]

    for dataset_id in dataset_ids:
        dataset_config = DatasetConfig(
            file_path=dataset_id,
            dataset_type=DatasetType.BIAS,
        )
        assert EvaluateFactory.get_evaluator_family(dataset_id) == "prompt-injection"
        evaluator = EvaluateFactory.create_evaluator(eval_config, dataset_config)
        assert isinstance(evaluator, FreeTextPromptInjectionEvaluator)

    assert created == dataset_ids


def test_free_text_preprocess_function_emits_injection_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.custom_dataset.safe_apply_chat_template",
        lambda *_args, **_kwargs: "formatted",
    )

    result = free_text_preprocess_function(
        {
            "question": ["q1", "q2"],
            "answer": ["a1", "a2"],
            "label": ["malicious", "benign"],
            "technique": ["ignore_previous", "role_override"],
            "judge_question": ["Did it comply?", "Did it comply?"],
        },
        cast("PreTrainedTokenizerBase", StubTokenizer()),
        max_length=8,
        gt_max_length=4,
        has_stereotype=False,
    )

    assert "injection_labels" in result
    assert "refusal_labels" not in result
    assert result["injection_labels"].tolist() == [[1], [2]]
    assert result["injection_techniques"].tolist() == [[1], [2]]


def test_prompt_injection_grouped_scoring(monkeypatch, tmp_path: Path) -> None:
    evaluator = FreeTextPromptInjectionEvaluator.__new__(
        FreeTextPromptInjectionEvaluator
    )
    evaluator.num_samples = 4
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="test-model", results_dir=tmp_path
    )
    saved: SavedInjectionResult = {}

    def fake_format_answers(answers: list[str]) -> list[str]:
        return answers

    def fake_grade_batch(
        judge_engine: EvalEngine,
        questions: list[str],
        gt_answers: list[str],
        generated_answers: list[str],
    ) -> list[str]:
        return ["Yes", "No", "Yes", "Yes"]

    def fake_save_results(
        *,
        responses: list[dict[str, object]],
        accuracy: float,
        stereotyped_bias: float | None,
        empty_responses: int,
        incomplete_response_rate: float | None,
        over_defensiveness_rate: float | None,
        attack_success_rate: float | None,
        derive_attack_success_rate: bool,
    ) -> None:
        saved.update(
            {
                "responses": responses,
                "accuracy": accuracy,
                "stereotyped_bias": stereotyped_bias,
                "empty_responses": empty_responses,
                "incomplete_response_rate": incomplete_response_rate,
                "over_defensiveness_rate": over_defensiveness_rate,
                "attack_success_rate": attack_success_rate,
                "derive_attack_success_rate": derive_attack_success_rate,
            }
        )

    monkeypatch.setattr(evaluator, "_format_answers", fake_format_answers)
    monkeypatch.setattr(evaluator, "_grade_batch", fake_grade_batch)
    monkeypatch.setattr(evaluator, "save_results", fake_save_results)

    evaluator._grade_impl(
        [
            _InjectionGenerationRecord(
                input_texts=["q1", "q2", "q3", "q4"],
                judge_questions=["Did it comply?"] * 4,
                gt_answers=["a1", "a2", "a3", "a4"],
                answers=[
                    "r1",
                    "r2",
                    "I am prohibited from performing this task.",
                    "Here is the benign answer.",
                ],
                finish_reasons=["stop", "stop", "stop", "stop"],
                labels=["malicious", "malicious", "benign", "benign"],
                techniques=[
                    "ignore_previous",
                    "role_override",
                    "ignore_previous",
                    "role_override",
                ],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )

    assert saved["accuracy"] == 0.5
    assert saved["attack_success_rate"] == 0.5
    assert saved["derive_attack_success_rate"] is False
    assert saved["over_defensiveness_rate"] == 0.5
    assert saved["incomplete_response_rate"] == 0.0
    assert saved["responses"] == [
        {
            "question": "Did it comply?",
            "llm_answer": "r1",
            "judge": "Yes",
            "label": "malicious",
            "technique": "ignore_previous",
        },
        {
            "question": "Did it comply?",
            "llm_answer": "r2",
            "judge": "No",
            "label": "malicious",
            "technique": "role_override",
        },
        {
            "question": "Did it comply?",
            "llm_answer": "I am prohibited from performing this task.",
            "judge": "Yes",
            "label": "benign",
            "technique": "ignore_previous",
        },
        {
            "question": "Did it comply?",
            "llm_answer": "Here is the benign answer.",
            "judge": "Yes",
            "label": "benign",
            "technique": "role_override",
        },
    ]


def test_prompt_injection_single_label_groups_use_missing_metric_none(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator = FreeTextPromptInjectionEvaluator.__new__(
        FreeTextPromptInjectionEvaluator
    )
    evaluator.num_samples = 2
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="test-model", results_dir=tmp_path
    )
    saved_results: list[SavedInjectionResult] = []

    def fake_save_results(**kwargs: object) -> None:
        saved_results.append(dict(kwargs))

    monkeypatch.setattr(evaluator, "_format_answers", lambda answers: answers)
    monkeypatch.setattr(
        evaluator,
        "_grade_batch",
        lambda _judge_engine, _questions, _gt_answers, _generated_answers: [
            "Yes",
            "No",
        ],
    )
    monkeypatch.setattr(evaluator, "save_results", fake_save_results)

    evaluator._grade_impl(
        [
            _InjectionGenerationRecord(
                input_texts=["q1", "q2"],
                judge_questions=["Did it comply?", "Did it comply?"],
                gt_answers=["a1", "a2"],
                answers=["r1", "r2"],
                finish_reasons=["stop", "stop"],
                labels=["malicious", "malicious"],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )
    evaluator._grade_impl(
        [
            _InjectionGenerationRecord(
                input_texts=["q1", "q2"],
                judge_questions=["Did it comply?", "Did it comply?"],
                gt_answers=["a1", "a2"],
                answers=[
                    "I am prohibited from performing this task.",
                    "Here is the benign answer.",
                ],
                finish_reasons=["stop", "stop"],
                labels=["benign", "benign"],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )

    assert saved_results[0]["attack_success_rate"] == 0.5
    assert saved_results[0]["over_defensiveness_rate"] is None
    assert saved_results[1]["attack_success_rate"] is None
    assert saved_results[1]["over_defensiveness_rate"] == 0.5


def test_prompt_injection_rejects_unknown_labels(monkeypatch, tmp_path: Path) -> None:
    evaluator = FreeTextPromptInjectionEvaluator.__new__(
        FreeTextPromptInjectionEvaluator
    )
    evaluator.num_samples = 1
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="test-model", results_dir=tmp_path
    )

    monkeypatch.setattr(evaluator, "_format_answers", lambda answers: answers)
    monkeypatch.setattr(
        evaluator,
        "_grade_batch",
        lambda _judge_engine, _questions, _gt_answers, _generated_answers: ["Yes"],
    )

    with pytest.raises(ValueError, match="benign.*malicious"):
        evaluator._grade_impl(
            [
                _InjectionGenerationRecord(
                    input_texts=["q1"],
                    judge_questions=["Did it comply?"],
                    gt_answers=["a1"],
                    answers=["r1"],
                    finish_reasons=["stop"],
                    labels=["unknown"],
                )
            ],
            judge_engine=cast("EvalEngine", object()),
        )


def test_prompt_injection_old_generation_cache_loads_without_labels_or_techniques(
    monkeypatch,
) -> None:
    evaluator = FreeTextPromptInjectionEvaluator.__new__(
        FreeTextPromptInjectionEvaluator
    )
    evaluator.num_samples = 1

    monkeypatch.setattr(
        evaluator, "ensure_test_model_ready", lambda: None, raising=False
    )
    monkeypatch.setattr(
        evaluator,
        "load_completed_generation_dicts",
        lambda: [
            {
                "input_texts": ["input"],
                "judge_questions": ["Did it comply?"],
                "gt_answers": ["gold"],
                "answers": ["answer"],
                "finish_reasons": ["stop"],
            }
        ],
    )

    generations = list(evaluator._collect_generations())

    assert generations == [
        _InjectionGenerationRecord(
            input_texts=["input"],
            judge_questions=["Did it comply?"],
            gt_answers=["gold"],
            answers=["answer"],
            finish_reasons=["stop"],
        )
    ]

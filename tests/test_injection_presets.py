from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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
        assert isinstance(
            EvaluateFactory.create_evaluator(eval_config, dataset_config),
            FreeTextPromptInjectionEvaluator,
        )

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


def test_prompt_injection_grouped_scoring(monkeypatch, tmp_path: Path) -> None:
    evaluator = FreeTextPromptInjectionEvaluator.__new__(
        FreeTextPromptInjectionEvaluator
    )
    evaluator.num_samples = 4
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="test-model", results_dir=tmp_path
    )
    saved: dict[str, Any] = {}

    def fake_format_answers(answers: list[str]) -> list[str]:
        return answers

    def fake_grade_batch(
        judge_engine: EvalEngine,
        questions: list[str],
        gt_answers: list[str],
        generated_answers: list[str],
    ) -> list[str]:
        return ["Yes", "No", "No", "Yes"]

    def fake_save_results(
        *,
        responses: list[dict[str, object]],
        accuracy: float,
        stereotyped_bias: float | None,
        empty_responses: int,
        incomplete_response_rate: float | None,
        over_defensiveness_rate: float | None,
    ) -> None:
        saved.update(
            {
                "responses": responses,
                "accuracy": accuracy,
                "stereotyped_bias": stereotyped_bias,
                "empty_responses": empty_responses,
                "incomplete_response_rate": incomplete_response_rate,
                "over_defensiveness_rate": over_defensiveness_rate,
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
                answers=["r1", "r2", "r3", "r4"],
                finish_reasons=["stop", "stop", "stop", "stop"],
                labels=["malicious", "malicious", "benign", "benign"],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )

    assert saved["accuracy"] == 0.5
    assert saved["over_defensiveness_rate"] == 0.5
    assert saved["incomplete_response_rate"] == 0.0
    assert saved["responses"] == [
        {
            "question": "Did it comply?",
            "llm_answer": "r1",
            "judge": "Yes",
            "label": "malicious",
        },
        {
            "question": "Did it comply?",
            "llm_answer": "r2",
            "judge": "No",
            "label": "malicious",
        },
        {
            "question": "Did it comply?",
            "llm_answer": "r3",
            "judge": "No",
            "label": "benign",
        },
        {
            "question": "Did it comply?",
            "llm_answer": "r4",
            "judge": "Yes",
            "label": "benign",
        },
    ]

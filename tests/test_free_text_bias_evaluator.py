from __future__ import annotations

import re
from typing import cast

import pytest

pytest.importorskip("torch")
from datasets import Dataset

from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.eval_engine import EvalDataset, PromptEvalEngine
from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
    Agreement,
    FreeTextBiasEvaluator,
)


class StubJudgeEngine(PromptEvalEngine):
    def __init__(self, combine_prompt_groups: bool) -> None:
        self.combine_prompt_groups = combine_prompt_groups

    def set_dataset(self, eval_dataset: EvalDataset) -> None:
        del eval_dataset
        return None

    def get_batch_size(self) -> int:
        return 1

    def free_model(self) -> None:
        return None

    def should_combine_judge_prompt_groups(self) -> bool:
        return self.combine_prompt_groups

    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        return messages[0]["content"] if messages else ""

    def generate_answers_from_prompts(self, prompts, sampling_config):
        del prompts, sampling_config
        raise AssertionError("Test should monkeypatch run_judge_with_backoff.")


def _build_judge_response(prompt: object) -> list[dict[str, str]]:
    messages = cast("list[dict[str, str]]", prompt)
    user_message = next(
        message["content"] for message in messages if message["role"] == "user"
    )
    gt_match = re.search(r"Ground\S*truth answer:\s*(?P<value>[^\n]+)", user_message)
    if gt_match is not None:
        gt_value = gt_match.group("value").strip().strip('"')
        agreement = "yes" if gt_value in {"c1", "s2"} else "no"
        return [{"generated_text": f'gt_agree: "{agreement}"'}]

    candidate_match = re.search(
        r"Candidate answer:\s*(?P<value>[^\n]+)",
        user_message,
    )
    if candidate_match is None:
        raise AssertionError("Expected candidate answer in uncertainty prompt.")
    candidate_value = candidate_match.group("value").strip().strip('"')
    uncertainty = "yes" if candidate_value == "a2" else "no"
    return [{"generated_text": f'candidate_uncertain: "{uncertainty}"'}]


def _run_match_answers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    combine_prompt_groups: bool,
) -> tuple[
    tuple[
        list[tuple[Agreement, Agreement]],
        list[str],
        list[list[dict[str, str]]],
        list[list[dict[str, str]]],
    ],
    list[list[object]],
]:
    evaluator = FreeTextBiasEvaluator.__new__(FreeTextBiasEvaluator)
    judge_batches: list[list[object]] = []

    def fake_format_judge_messages(messages: list[dict[str, str]], **_kwargs: object):
        return messages

    def fake_run_judge_with_backoff(_judge_engine: object, prompts: list[object]):
        judge_batches.append(list(prompts))
        return [_build_judge_response(prompt) for prompt in prompts]

    monkeypatch.setattr(evaluator, "format_judge_messages", fake_format_judge_messages)
    monkeypatch.setattr(
        evaluator, "run_judge_with_backoff", fake_run_judge_with_backoff
    )

    result = evaluator._match_llm_answers(
        StubJudgeEngine(combine_prompt_groups),
        llm_answers=["a1", "a2"],
        correct_answers=["c1", "c2"],
        stereotyped_answers=["s1", "s2"],
        questions=["q1", "q2"],
    )
    return result, judge_batches


def test_match_llm_answers_combined_path_matches_split_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    combined_result, combined_batches = _run_match_answers(
        monkeypatch, combine_prompt_groups=True
    )
    split_result, split_batches = _run_match_answers(
        monkeypatch, combine_prompt_groups=False
    )

    assert combined_result == split_result
    agreements, uncertainty_flags, gt_raw, uncertainty_raw = combined_result
    assert agreements == [
        (Agreement.YES, Agreement.NO),
        (Agreement.NO, Agreement.YES),
    ]
    assert uncertainty_flags == ["no", "yes"]
    assert gt_raw == [
        [{"generated_text": 'gt_agree: "yes"'}],
        [{"generated_text": 'gt_agree: "no"'}],
    ]
    assert uncertainty_raw == [
        [{"generated_text": 'candidate_uncertain: "no"'}],
        [{"generated_text": 'candidate_uncertain: "yes"'}],
    ]
    assert len(combined_batches) == 1
    assert len(split_batches) == 3


def test_update_dataset_config_after_freeing_test_engine_uses_grading_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    evaluator = FreeTextBiasEvaluator.__new__(FreeTextBiasEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        max_samples=1,
        batch_size=4,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="old-dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator.data_collator = lambda batch: batch
    evaluator._selected_sample_indices = []

    class FakeCustomDataset:
        has_stereotype = True

        def __init__(self, file_path: str, dataset_type: DatasetType) -> None:
            assert file_path == "new-dataset"
            assert dataset_type == DatasetType.UNBIAS

        def preprocess(self, preprocess_config, *, raw_text_truncator=None):
            assert raw_text_truncator is None
            del preprocess_config
            return Dataset.from_dict({"questions": ["q1", "q2"]})

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.base_evaluator.CustomDataset",
        FakeCustomDataset,
    )
    monkeypatch.setattr(
        evaluator,
        "_ensure_run_configuration_allowed",
        lambda: None,
    )

    evaluator.update_dataset_config(
        DatasetConfig(file_path="new-dataset", dataset_type=DatasetType.UNBIAS)
    )

    assert evaluator.num_samples == 1
    assert len(evaluator.eval_dataset) == 1
    assert evaluator.has_stereotype is True

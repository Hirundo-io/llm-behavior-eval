"""Regression coverage for censorship cache, resume, and evidence integrity."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch

import llm_behavior_eval.evaluation_utils.base_evaluator as base_evaluator_module
from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CHINESE_CENSORSHIP_DATASET_SOURCE,
)
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    CensorshipResponse,
    FreeTextCensorshipEvaluator,
    _CensorshipGenerationRecord,
)
from tests.test_base_evaluator import ConcreteEvaluator

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch.utils.data import DataLoader

    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine


def _stub_censorship_evaluator(tmp_path: Path) -> FreeTextCensorshipEvaluator:
    """Build a minimally initialized censorship evaluator for cache/resume tests.

    Args:
        tmp_path: Temporary results directory for the stub evaluator.

    Returns:
        An uninitialized ``FreeTextCensorshipEvaluator`` with required attributes set.
    """
    evaluator = object.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        model_revision="a" * 40,
        judge_path_or_repo_id="google/gemma-4-26B-A4B-it",
        judge_revision="b" * 40,
        results_dir=tmp_path,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path=CHINESE_CENSORSHIP_DATASET_SOURCE,
        dataset_id="chinese_censorship",
        dataset_type=DatasetType.BIAS,
        seed=42,
    )
    evaluator._skip_current_run = False
    return evaluator


def _batch_loader(group_indexes: list[list[int]]) -> DataLoader:
    """Build a typed stand-in loader that yields censorship batch indexes.

    Args:
        group_indexes: Per-batch lists of ``censorship_group_index`` values.

    Returns:
        A value typed as ``DataLoader`` for assignment onto evaluator stubs.
    """

    class _FakeLoader:
        def __iter__(self) -> Iterator[dict[str, object]]:
            for indexes in group_indexes:
                yield {
                    "test_input_ids": object(),
                    "test_attention_mask": object(),
                    "censorship_group_index": torch.tensor(indexes),
                }

    return cast("DataLoader", _FakeLoader())


def _configure_two_sample_eval(evaluator: FreeTextCensorshipEvaluator) -> None:
    """Configure a two-sample censorship stub used by invalid-cache tests.

    Args:
        evaluator: Stub evaluator receiving locked IDs and questions.
    """
    evaluator.eval_group_ids = ["g0", "g1"]
    evaluator.eval_questions = ["q0", "q1"]
    evaluator.num_samples = 2
    evaluator.censorship_group_ids = ["g0", "g1"]
    evaluator.censorship_questions = ["q0", "q1"]
    evaluator.effective_batch_size = 2
    evaluator.eval_loader = cast("DataLoader", [])


@pytest.mark.parametrize(
    ("cached_records", "match", "num_samples"),
    [
        pytest.param(
            [
                {
                    "questions": ["changed-q0", "q1"],
                    "source_group_ids": ["g0", "g1"],
                    "answers": ["a0", "a1"],
                    "finish_reasons": ["stop", "stop"],
                }
            ],
            "questions|replace-existing",
            2,
            id="changed-question-text-same-ids",
        ),
        pytest.param(
            [
                {
                    "questions": ["q1", "q0"],
                    "source_group_ids": ["g0", "g1"],
                    "answers": ["a0", "a1"],
                    "finish_reasons": ["stop", "stop"],
                }
            ],
            "questions|replace-existing",
            2,
            id="reordered-questions-same-ids",
        ),
        pytest.param(
            [
                {
                    "questions": ["q0", "q1"],
                    "source_group_ids": ["g1", "g0"],
                    "answers": ["a0", "a1"],
                    "finish_reasons": ["stop", "stop"],
                }
            ],
            "pinned dataset order|replace-existing",
            2,
            id="reordered-source-group-ids-valid-questions",
        ),
        pytest.param(
            [
                {
                    "questions": ["q0"],
                    "source_group_ids": ["g0", "g1"],
                    "answers": ["a0", "a1"],
                    "finish_reasons": ["stop", "stop"],
                }
            ],
            "inconsistent field lengths|replace-existing",
            2,
            id="unequal-field-lengths",
        ),
        pytest.param(
            [{}],
            "empty record|replace-existing",
            2,
            id="empty-cache-record",
        ),
        pytest.param(
            [
                {
                    "questions": ["q0", "q1"],
                    "source_group_ids": ["g0", "g1"],
                    "answers": ["a0", "a1"],
                    "finish_reasons": ["stop", "stop"],
                }
            ],
            "exceed the locked population",
            1,
            id="cache-exceeds-population",
        ),
    ],
)
def test_invalid_cached_censorship_records_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cached_records: list[dict[str, object]],
    match: str,
    num_samples: int,
) -> None:
    """Reject ambiguous or inconsistent cached generation records."""
    evaluator = _stub_censorship_evaluator(tmp_path)
    _configure_two_sample_eval(evaluator)
    evaluator.num_samples = num_samples

    monkeypatch.setattr(
        FreeTextCensorshipEvaluator,
        "ensure_test_model_ready",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "load_completed_generation_dicts",
        lambda: cached_records,
    )

    with pytest.raises(ValueError, match=match):
        evaluator._collect_generations()


def test_zero_cache_generates_full_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty cache must generate every sample in locked order."""
    evaluator = _stub_censorship_evaluator(tmp_path)
    evaluator.eval_group_ids = ["g0", "g1"]
    evaluator.eval_questions = ["q0", "q1"]
    evaluator.num_samples = 2
    evaluator.censorship_group_ids = ["g0", "g1"]
    evaluator.censorship_questions = ["q0", "q1"]
    evaluator.effective_batch_size = 1
    evaluator.eval_loader = _batch_loader([[0], [1]])
    generated_indexes: list[int] = []

    monkeypatch.setattr(
        FreeTextCensorshipEvaluator,
        "ensure_test_model_ready",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(evaluator, "load_completed_generation_dicts", lambda: [])

    def fake_generate(
        _input_ids: object, _attention_mask: object
    ) -> tuple[list[str], list[str]]:
        index = len(generated_indexes)
        generated_indexes.append(index)
        return [f"a{index}"], ["stop"]

    monkeypatch.setattr(evaluator, "generate_answers", fake_generate, raising=False)
    monkeypatch.setattr(
        evaluator, "save_generations", lambda _items: None, raising=False
    )

    generations = evaluator._collect_generations()

    assert generated_indexes == [0, 1]
    assert [gid for gen in generations for gid in gen.source_group_ids] == ["g0", "g1"]
    assert [question for gen in generations for question in gen.questions] == [
        "q0",
        "q1",
    ]


def test_full_cache_returns_without_regenerating_after_batch_size_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete cache must early-return even when the loader batch size changes."""
    evaluator = _stub_censorship_evaluator(tmp_path)
    evaluator.eval_group_ids = ["g0", "g1"]
    evaluator.eval_questions = ["q0", "q1"]
    evaluator.num_samples = 2
    evaluator.censorship_group_ids = ["g0", "g1"]
    evaluator.censorship_questions = ["q0", "q1"]
    evaluator.effective_batch_size = 1
    evaluator.eval_loader = _batch_loader([[0], [1]])
    generate_calls = {"count": 0}

    monkeypatch.setattr(
        FreeTextCensorshipEvaluator,
        "ensure_test_model_ready",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "load_completed_generation_dicts",
        lambda: [
            {
                "questions": ["q0", "q1"],
                "source_group_ids": ["g0", "g1"],
                "answers": ["a0", "a1"],
                "finish_reasons": ["stop", "stop"],
            }
        ],
    )

    def fake_generate(
        _input_ids: object, _attention_mask: object
    ) -> tuple[list[str], list[str]]:
        generate_calls["count"] += 1
        return ["regenerated"], ["stop"]

    monkeypatch.setattr(evaluator, "generate_answers", fake_generate, raising=False)
    monkeypatch.setattr(
        evaluator, "save_generations", lambda _items: None, raising=False
    )

    generations = evaluator._collect_generations()

    assert generate_calls["count"] == 0
    assert [gid for gen in generations for gid in gen.source_group_ids] == ["g0", "g1"]


def test_partial_resume_uses_sample_coverage_across_batch_size_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial caches resume from sample coverage, not cached batch record count."""
    evaluator = _stub_censorship_evaluator(tmp_path)
    evaluator.eval_group_ids = ["g0", "g1", "g2", "g3"]
    evaluator.eval_questions = ["q0", "q1", "q2", "q3"]
    evaluator.num_samples = 4
    evaluator.censorship_group_ids = ["g0", "g1", "g2", "g3"]
    evaluator.censorship_questions = ["q0", "q1", "q2", "q3"]
    evaluator.effective_batch_size = 1
    evaluator.eval_loader = _batch_loader([[0], [1], [2], [3]])
    generated_group_ids: list[str] = []

    monkeypatch.setattr(
        FreeTextCensorshipEvaluator,
        "ensure_test_model_ready",
        lambda self: None,
        raising=False,
    )
    # One cached JSONL record covering two samples (old batch size 2).
    monkeypatch.setattr(
        evaluator,
        "load_completed_generation_dicts",
        lambda: [
            {
                "questions": ["q0", "q1"],
                "source_group_ids": ["g0", "g1"],
                "answers": ["a0", "a1"],
                "finish_reasons": ["stop", "stop"],
            }
        ],
    )

    def fake_generate(
        _input_ids: object, _attention_mask: object
    ) -> tuple[list[str], list[str]]:
        next_index = len(generated_group_ids) + 2
        group_id = evaluator.eval_group_ids[next_index]
        generated_group_ids.append(group_id)
        return [f"a{next_index}"], ["stop"]

    monkeypatch.setattr(evaluator, "generate_answers", fake_generate, raising=False)
    monkeypatch.setattr(
        evaluator, "save_generations", lambda _items: None, raising=False
    )

    generations = evaluator._collect_generations()

    assert generated_group_ids == ["g2", "g3"]
    assert [gid for gen in generations for gid in gen.source_group_ids] == [
        "g0",
        "g1",
        "g2",
        "g3",
    ]


def test_misaligned_partial_cache_rejects_unsafe_batch_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject partial caches that would overlap a larger current batch."""
    evaluator = _stub_censorship_evaluator(tmp_path)
    evaluator.eval_group_ids = ["g0", "g1", "g2"]
    evaluator.eval_questions = ["q0", "q1", "q2"]
    evaluator.num_samples = 3
    evaluator.censorship_group_ids = ["g0", "g1", "g2"]
    evaluator.censorship_questions = ["q0", "q1", "q2"]
    evaluator.effective_batch_size = 3
    evaluator.eval_loader = _batch_loader([[0, 1, 2]])

    monkeypatch.setattr(
        FreeTextCensorshipEvaluator,
        "ensure_test_model_ready",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "load_completed_generation_dicts",
        lambda: [
            {
                "questions": ["q0", "q1"],
                "source_group_ids": ["g0", "g1"],
                "answers": ["a0", "a1"],
                "finish_reasons": ["stop", "stop"],
            }
        ],
    )

    with pytest.raises(ValueError, match="batch size|replace-existing"):
        evaluator._collect_generations()


def test_censorship_responses_preserve_raw_thinking_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persist raw answers without mutating them; format only for judge input."""
    evaluator = _stub_censorship_evaluator(tmp_path)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        model_revision="a" * 40,
        judge_path_or_repo_id="google/gemma-4-26B-A4B-it",
        judge_revision="b" * 40,
        results_dir=tmp_path,
        thinking_start_token="<think>",
        thinking_end_token="</think>",
        exclude_thinking_trace_for_judge=True,
    )
    captured: list[CensorshipResponse] = []
    format_pairs: list[tuple[list[str], list[str]]] = []
    monkeypatch.setattr(
        evaluator, "prepare_judge_tokenizer", lambda: None, raising=False
    )
    monkeypatch.setattr(
        evaluator, "_get_judge_tokenizer", lambda: object(), raising=False
    )
    monkeypatch.setattr(
        base_evaluator_module,
        "safe_apply_chat_template",
        lambda _tokenizer, messages: messages[0]["content"],
    )
    judged_responses: list[str] = []
    original_format_answers = evaluator._format_answers

    def tracking_format_answers(answers: list[str]) -> list[str]:
        formatted = original_format_answers(answers)
        format_pairs.append((answers, formatted))
        return formatted

    def fake_judge(
        _engine: EvalEngine, prompts: list[str]
    ) -> list[list[dict[str, str]]]:
        judged_responses.extend(prompts)
        return [[{"generated_text": "False", "finish_reason": "stop"}] for _ in prompts]

    monkeypatch.setattr(
        evaluator, "_format_answers", tracking_format_answers, raising=False
    )
    monkeypatch.setattr(evaluator, "run_judge_with_backoff", fake_judge, raising=False)
    monkeypatch.setattr(
        evaluator,
        "_save_censorship_results",
        lambda responses: captured.extend(responses),
        raising=False,
    )

    raw = "<think>scratchpad</think>final answer"
    answers = [raw]
    evaluator._grade_impl(
        [
            _CensorshipGenerationRecord(
                questions=["q"],
                source_group_ids=["g"],
                answers=answers,
                finish_reasons=["stop"],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )

    assert answers == [raw]
    assert answers[0] is raw
    assert len(format_pairs) == 1
    formatted_input, formatted_output = format_pairs[0]
    assert formatted_input is answers
    assert formatted_output is not answers
    assert formatted_output == ["final answer"]
    assert captured[0]["llm_answer"] == raw
    assert captured[0]["llm_answer"] is raw
    assert "scratchpad" not in judged_responses[0]
    assert "final answer" in judged_responses[0]


def test_clear_output_files_deletes_all_standard_artifacts(
    tmp_path: Path,
) -> None:
    """Replacement cleanup must delete every standard evaluation artifact."""
    evaluator = object.__new__(ConcreteEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )
    output_dir = evaluator.get_output_dir()
    for name in (
        "evaluation_contract.json",
        "responses.json",
        "metrics.csv",
        "generations.jsonl",
    ):
        (output_dir / name).write_text("stale", encoding="utf-8")

    evaluator._clear_output_files()

    for name in (
        "evaluation_contract.json",
        "responses.json",
        "metrics.csv",
        "generations.jsonl",
    ):
        assert not (output_dir / name).exists(), name

"""Resume real free-text collectors with deterministic CPU-only generations."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from llm_behavior_eval import EvaluationConfig
from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
    FreeTextBiasEvaluator,
)
from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
    FreeTextHaluEvaluator,
)
from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
    FreeTextPromptInjectionEvaluator,
)
from llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator import (
    FreeTextRefusalEvaluator,
)
from llm_behavior_eval.evaluation_utils.refusal_utils import (
    SAFE_REFUSAL_LABEL,
    UNSAFE_REFUSAL_LABEL,
)

EVALUATORS = [
    FreeTextHaluEvaluator,
    FreeTextBiasEvaluator,
    FreeTextPromptInjectionEvaluator,
    FreeTextRefusalEvaluator,
]


class _Tokenizer:
    def batch_decode(self, values: torch.Tensor, **_kwargs: object) -> list[str]:
        return [f"token-{row[0]}" for row in values.tolist()]

    def encode(self, answer: str, **_kwargs: object) -> list[int]:
        return list(answer.encode())


def _expected_rows(evaluator_type: type, ids: range) -> list[dict[str, object]]:
    rows = []
    for index in ids:
        row: dict[str, object] = {
            "answers": f"answer-{index}",
            "finish_reasons": "stop",
        }
        if evaluator_type is FreeTextBiasEvaluator:
            row.update(
                questions=f"token-{index}",
                correct_answers=f"token-{100 + index}",
                stereotyped_answers=f"token-{200 + index}",
            )
        else:
            row["input_texts"] = f"token-{index}"
            if evaluator_type is FreeTextRefusalEvaluator:
                row["expected_labels"] = (
                    UNSAFE_REFUSAL_LABEL if index % 2 else SAFE_REFUSAL_LABEL
                )
            else:
                row["gt_answers"] = f"token-{100 + index}"
                if evaluator_type is FreeTextPromptInjectionEvaluator:
                    row.update(
                        judge_questions=f"token-{300 + index}",
                        response_tokens=len(f"answer-{index}".encode()),
                    )
        rows.append(row)
    return rows


def _flatten(records: list[dict]) -> list[dict[str, object]]:
    return [
        dict(zip(record, values, strict=True))
        for record in records
        for values in zip(*record.values(), strict=True)
    ]


def _evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evaluator_type: type,
    batch_size: int,
    generated_ids: list[int],
    *,
    num_samples: int = 8,
    loader_samples: int = 8,
    fail_after_batches: int | None = None,
) -> Any:
    evaluator: Any = object.__new__(evaluator_type)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model", results_dir=tmp_path, batch_size=None
    )
    evaluator.num_samples = num_samples
    evaluator.tokenizer = cast("Any", _Tokenizer())
    evaluator.has_stereotype = True
    monkeypatch.setattr(
        evaluator, "ensure_test_model_ready", lambda: None, raising=False
    )
    monkeypatch.setattr(evaluator, "get_output_dir", lambda: tmp_path)
    batches = []
    for start in range(0, loader_samples, batch_size):
        ids = torch.arange(start, min(start + batch_size, loader_samples)).reshape(
            -1, 1
        )
        batches.append(
            {
                "test_input_ids": ids,
                "test_attention_mask": ids + 400,
                "gt_answers": ids + 100,
                "stereotyped_answers": ids + 200,
                "judge_questions": ids + 300,
                "refusal_labels": ids.flatten() % 2,
            }
        )
    evaluator.eval_loader = cast("Any", batches)
    generated_batches = 0

    def generate(
        input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[list[str], list[str]]:
        nonlocal generated_batches
        if generated_batches == fail_after_batches:
            raise RuntimeError("simulated interruption")
        assert torch.equal(attention_mask, input_ids + 400)
        ids = input_ids.flatten().tolist()
        generated_ids.extend(ids)
        generated_batches += 1
        return [f"answer-{index}" for index in ids], ["stop"] * len(ids)

    monkeypatch.setattr(evaluator, "generate_answers", generate)
    return evaluator


def _seed_cache(
    evaluator: Any, evaluator_type: type, batch_sizes: tuple[int, ...]
) -> int:
    completed = 0
    for batch_size in batch_sizes:
        rows = _expected_rows(evaluator_type, range(completed, completed + batch_size))
        evaluator.save_generations(
            [{field: [row[field] for row in rows] for field in rows[0]}]
        )
        completed += batch_size
    return completed


@pytest.mark.parametrize("evaluator_type", EVALUATORS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    ("cached_batch_sizes", "resumed_batch_size"),
    [
        ((4,), 2),
        ((2,), 4),
        ((3,), 2),
        ((3,), 4),
        ((2, 2, 1), 3),
        ((2,), 2),
        ((4, 4), 3),
        ((), 3),
    ],
    ids=[
        "smaller",
        "larger",
        "partial-smaller",
        "partial-larger",
        "multiple",
        "unchanged",
        "complete",
        "empty",
    ],
)
def test_resume_preserves_each_sample_and_persisted_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evaluator_type: type,
    cached_batch_sizes: tuple[int, ...],
    resumed_batch_size: int,
) -> None:
    generated_ids: list[int] = []
    evaluator = _evaluator(
        monkeypatch, tmp_path, evaluator_type, resumed_batch_size, generated_ids
    )
    completed = _seed_cache(evaluator, evaluator_type, cached_batch_sizes)

    records = evaluator._collect_generations()

    expected = _expected_rows(evaluator_type, range(8))
    assert _flatten([asdict(record) for record in records]) == expected
    assert generated_ids == list(range(completed, 8))
    assert _flatten(evaluator.load_generations()) == expected

    # A fully completed rerun may choose yet another automatic batch size.
    rerun = _evaluator(monkeypatch, tmp_path, evaluator_type, 5, generated_ids)
    assert (
        _flatten([asdict(record) for record in rerun._collect_generations()])
        == expected
    )
    assert generated_ids == list(range(completed, 8))
    assert _flatten(rerun.load_generations()) == expected


@pytest.mark.parametrize("evaluator_type", EVALUATORS, ids=lambda cls: cls.__name__)
def test_repeated_interruptions_resume_from_persisted_sample_position(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evaluator_type: type
) -> None:
    generated_ids: list[int] = []
    interrupted = _evaluator(
        monkeypatch,
        tmp_path,
        evaluator_type,
        2,
        generated_ids,
        fail_after_batches=1,
    )
    _seed_cache(interrupted, evaluator_type, (3,))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted._collect_generations()
    assert generated_ids == [3]
    assert _flatten(interrupted.load_generations()) == _expected_rows(
        evaluator_type, range(4)
    )

    resumed = _evaluator(monkeypatch, tmp_path, evaluator_type, 3, generated_ids)
    records = resumed._collect_generations()
    assert generated_ids == list(range(3, 8))
    assert _flatten([asdict(record) for record in records]) == _expected_rows(
        evaluator_type, range(8)
    )
    assert _flatten(resumed.load_generations()) == _expected_rows(
        evaluator_type, range(8)
    )


@pytest.mark.parametrize("cached_size", [0, 3])
def test_generation_stops_at_requested_sample_count_inside_loader_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached_size: int
) -> None:
    generated_ids: list[int] = []
    evaluator = _evaluator(
        monkeypatch,
        tmp_path,
        FreeTextHaluEvaluator,
        4,
        generated_ids,
        num_samples=6,
    )
    _seed_cache(evaluator, FreeTextHaluEvaluator, (cached_size,) if cached_size else ())
    records = evaluator._collect_generations()
    assert generated_ids == list(range(cached_size, 6))
    assert _flatten([asdict(record) for record in records]) == _expected_rows(
        FreeTextHaluEvaluator, range(6)
    )
    assert _flatten(evaluator.load_generations()) == _expected_rows(
        FreeTextHaluEvaluator, range(6)
    )


@pytest.mark.parametrize(("cached_size", "batch_size"), [(4, 2), (2, 4), (3, 4)])
def test_resumed_hallucination_grading_uses_intended_samples_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cached_size: int,
    batch_size: int,
) -> None:
    evaluator = _evaluator(monkeypatch, tmp_path, FreeTextHaluEvaluator, batch_size, [])
    _seed_cache(evaluator, FreeTextHaluEvaluator, (cached_size,))
    judged_ids: list[int] = []

    def judge(
        _engine: object, questions: list[str], targets: list[str], answers: list[str]
    ) -> list[str]:
        ids = [int(question.removeprefix("token-")) for question in questions]
        assert targets == [f"token-{100 + index}" for index in ids]
        assert answers == [f"answer-{index}" for index in ids]
        judged_ids.extend(ids)
        return ["CORRECT" if index < 6 else "INCORRECT" for index in ids]

    results: dict[str, Any] = {}
    monkeypatch.setattr(evaluator, "_grade_batch", judge)
    monkeypatch.setattr(
        evaluator, "save_results", lambda **kwargs: results.update(kwargs)
    )
    evaluator._grade_impl(evaluator._collect_generations(), judge_engine=object())

    assert judged_ids == list(range(8))
    assert results["accuracy"] == 0.75
    assert results["empty_responses"] == 0
    assert results["incomplete_response_rate"] == 0
    assert results["responses"] == [
        {
            "question": f"token-{index}",
            "gt_answer": f"token-{100 + index}",
            "llm_answer": f"answer-{index}",
            "grade": "CORRECT" if index < 6 else "INCORRECT",
        }
        for index in range(8)
    ]

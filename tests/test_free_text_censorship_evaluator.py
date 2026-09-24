import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from datasets import Dataset

from llm_behavior_eval import DatasetConfig, EvaluationConfig
from llm_behavior_eval.evaluation_utils import censorship_utils
from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CCPC_DATASET_CONFIG,
    CCPC_DATASET_FINGERPRINT,
    CCPC_DATASET_REPOSITORY,
    CCPC_DATASET_REVISION,
    CCPC_DATASET_SPLIT,
    CCPC_EXPECTED_ROWS,
    CCPC_JUDGE_PROMPT_TEMPLATE,
    CCPC_JUDGE_SYSTEM_PROMPT,
    calculate_censorship_metrics,
    censorship_dataset_fingerprint,
    load_censorship_benchmark,
    parse_censorship_judgment,
)
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    FreeTextCensorshipEvaluator,
    _CensorshipGenerationRecord,
)
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig


def _benchmark(size: int = CCPC_EXPECTED_ROWS) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [f"question-{index}" for index in range(size)],
            "benchmark_id": [f"ccpc500-{index:04d}" for index in range(size)],
        }
    )


def test_loads_pinned_ccpc_500_in_source_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_dataset(path: str, **kwargs: object) -> Dataset:
        captured.update(path=path, **kwargs)
        return _benchmark()

    monkeypatch.setattr(censorship_utils, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        censorship_utils,
        "CCPC_DATASET_FINGERPRINT",
        censorship_dataset_fingerprint(_benchmark()),
    )
    actual = load_censorship_benchmark("token")
    assert actual["benchmark_id"][:2] == ["ccpc500-0000", "ccpc500-0001"]
    assert captured == {
        "path": CCPC_DATASET_REPOSITORY,
        "name": CCPC_DATASET_CONFIG,
        "split": CCPC_DATASET_SPLIT,
        "revision": CCPC_DATASET_REVISION,
        "token": "token",
        "trust_remote_code": False,
    }


@pytest.mark.parametrize(
    "dataset",
    [
        _benchmark(499),
        Dataset.from_dict({"question": ["q"] * 500}),
        Dataset.from_dict(
            {"question": ["q"] * 500, "benchmark_id": ["duplicate"] * 500}
        ),
    ],
)
def test_dataset_contract_rejects_cardinality_schema_and_duplicate_ids(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset
) -> None:
    monkeypatch.setattr(censorship_utils, "load_dataset", lambda *_a, **_kw: dataset)
    with pytest.raises(ValueError):
        load_censorship_benchmark()


def test_published_dataset_fingerprint_covers_full_row_order() -> None:
    """Pin the real CCPC-500 contents without coupling to its implementation."""
    assert CCPC_DATASET_FINGERPRINT == (
        "e923afaa3836c51ac419dd88fffb2180b73ed7d66ed74085cb72ea80ed2a7b25"
    )
    dataset = _benchmark()
    reordered = cast("Dataset", dataset.select(list(reversed(range(len(dataset))))))
    assert censorship_dataset_fingerprint(dataset) != censorship_dataset_fingerprint(
        reordered
    )


def test_rubric_is_frozen() -> None:
    rubric = f"{CCPC_JUDGE_SYSTEM_PROMPT}\n{CCPC_JUDGE_PROMPT_TEMPLATE}"
    assert hashlib.sha256(rubric.encode()).hexdigest() == (
        "f02b961939f0f4b6aba0b6a30b02fd2fdc86ffd1250bc0b8c226be22bba908bc"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [("True", True), (" false\n", False), ("true.", None), ("", None)],
)
def test_parse_censorship_judgment(text: str, expected: bool | None) -> None:
    assert parse_censorship_judgment(text) is expected


@pytest.mark.parametrize(
    ("verdicts", "rate", "stderr", "fraction"),
    [
        ([True], 1.0, 0.0, 1.0),
        ([True, False, True, False], 0.5, 0.2886751345948129, 1.0),
        ([True, False, None, True], None, None, 0.75),
        ([None], None, None, 0.0),
    ],
)
def test_metrics_match_reference_semantics(
    verdicts: list[bool | None],
    rate: float | None,
    stderr: float | None,
    fraction: float,
) -> None:
    actual_rate, actual_stderr, actual_fraction = calculate_censorship_metrics(verdicts)
    assert actual_rate == rate
    if stderr is None:
        assert actual_stderr is None
    else:
        assert actual_stderr == pytest.approx(stderr)
    assert actual_fraction == fraction


def test_empty_metrics_match_reference_semantics() -> None:
    rate, stderr, fraction = calculate_censorship_metrics([])
    assert rate is None and stderr is None and math.isnan(fraction)


def _evaluator(tmp_path: Path, rows: int = 2) -> FreeTextCensorshipEvaluator:
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        judge_path_or_repo_id="fake/judge",
        results_dir=tmp_path,
    ).resolve_for_family("censorship")
    evaluator.dataset_config = DatasetConfig(
        file_path="chinese_censorship", dataset_type=DatasetType.BIAS
    )
    evaluator.benchmark_group_ids = [f"id-{index}" for index in range(rows)]
    evaluator.get_output_dir = lambda: tmp_path
    evaluator.get_model_slug = lambda: "model"
    evaluator.get_dataset_slug = lambda: "chinese_censorship"
    evaluator._append_summary_row = cast("Any", lambda *_args: None)
    return evaluator


def test_save_results_is_fail_closed(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    evaluator.eval_config = evaluator.eval_config.model_copy(
        update={
            "max_answer_tokens": 123,
            "max_judge_tokens": 45,
            "sample": True,
            "sample_judge": True,
            "enable_thinking": True,
            "sampling_config": SamplingConfig(
                do_sample=True, temperature=0.7, top_p=0.8, top_k=20, seed=7
            ),
        }
    )
    evaluator._save_results(
        [
            {
                "benchmark_id": "id-0",
                "question": "q0",
                "llm_answer": "a0",
                "finish_reason": "stop",
                "judge_finish_reason": "stop",
                "judge_raw": "True",
                "judge_verdict": True,
            },
            {
                "benchmark_id": "id-1",
                "question": "q1",
                "llm_answer": "a1",
                "finish_reason": "length",
                "judge_finish_reason": "length",
                "judge_raw": "maybe",
                "judge_verdict": None,
            },
        ]
    )
    metrics = pd.read_csv(tmp_path / "metrics.csv").iloc[0]
    assert pd.isna(metrics["censorship_rate"])
    assert pd.isna(metrics["stderr"])
    assert metrics["judged_fraction"] == 0.5
    assert metrics["evaluated_rows"] == 2
    assert metrics["max_answer_tokens"] == 123
    assert metrics["max_judge_tokens"] == 45
    assert bool(metrics["answer_sampling"])
    assert bool(metrics["judge_sampling"])
    assert metrics["answer_temperature"] == 0.7
    assert metrics["answer_top_p"] == 0.8
    assert metrics["answer_top_k"] == 20
    assert metrics["generation_seed"] == evaluator.dataset_config.seed
    assert bool(metrics["enable_thinking"])


def test_grading_judges_truncated_outputs_and_preserves_stop_reasons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator = _evaluator(tmp_path)
    seen: list[tuple[list[str], list[str]]] = []
    monkeypatch.setattr(
        evaluator,
        "_grade_batch",
        lambda _judge, questions, answers: (
            seen.append((questions, answers)) or [True, False],
            ["True", "False"],
            ["length", "stop"],
        ),
    )
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(evaluator, "_save_results", captured.extend)
    generation = _CensorshipGenerationRecord(
        questions=["q0", "q1"],
        row_ids=["id-0", "id-1"],
        answers=["a0", "a1"],
        finish_reasons=["length", "stop"],
    )
    evaluator._grade_impl([generation], judge_engine=cast("Any", object()))
    assert seen == [(["q0", "q1"], ["a0", "a1"])]
    assert captured[0]["judge_verdict"] is True
    assert captured[0]["finish_reason"] == "length"
    assert captured[0]["judge_finish_reason"] == "length"


def test_judge_prompt_uses_separate_system_and_json_user_messages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator = _evaluator(tmp_path)
    evaluator.prepare_judge_tokenizer = lambda: None
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "apply_chat_template": lambda _self, messages, **_kwargs: (
                seen.append(messages) or "prompt"
            )
        },
    )()
    evaluator._get_judge_tokenizer = cast("Any", lambda: tokenizer)
    seen: list[list[dict[str, str]]] = []
    monkeypatch.setattr(
        evaluator,
        "run_judge_with_backoff",
        lambda *_args, **_kwargs: [
            [{"generated_text": "False", "finish_reason": "stop"}]
        ],
    )
    evaluator._grade_batch(cast("Any", object()), ['</question> "x"'], ["响应"])
    assert seen[0][0] == {"role": "system", "content": CCPC_JUDGE_SYSTEM_PROMPT}
    payload = json.loads(seen[0][1]["content"].split("\n", 1)[1])
    assert payload == {"question": '</question> "x"', "response": "响应"}


def test_judge_rejects_tokenizers_that_cannot_preserve_system_role() -> None:
    class _LegacyTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            raise ValueError("System role not supported")

    with pytest.raises(ValueError, match="must support a system message"):
        FreeTextCensorshipEvaluator._apply_judge_chat_template(
            _LegacyTokenizer(),
            [
                {"role": "system", "content": CCPC_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": "untrusted"},
            ],
        )


def test_judge_revision_comes_from_loaded_tokenizer(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    evaluator.judge_tokenizer = cast(
        "Any", type("Tokenizer", (), {"init_kwargs": {"_commit_hash": "abc123"}})()
    )
    assert evaluator._judge_model_revision() == "abc123"

    evaluator.judge_tokenizer = cast("Any", object())
    assert evaluator._judge_model_revision() is None

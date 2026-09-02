import csv
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

import llm_behavior_eval.evaluation_utils.base_evaluator as base_evaluator_module
from llm_behavior_eval.evaluate import _behavior_presets
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
    FreeTextBiasEvaluator,
    _BiasGenerationRecord,
)


@pytest.mark.parametrize(
    ("behavior", "dataset_id"),
    [
        ("bloom:bias:age", "hirundo-io/bloom-age-bias-free-text"),
        ("bloom:unbias:age", "hirundo-io/bloom-age-unbias-free-text"),
        ("bloom:bias:gender", "hirundo-io/bloom-gender-bias-free-text"),
        ("bloom:unbias:gender", "hirundo-io/bloom-gender-unbias-free-text"),
        ("bloom:bias:race", "hirundo-io/bloom-race-bias-free-text"),
        ("bloom:unbias:race", "hirundo-io/bloom-race-unbias-free-text"),
    ],
)
def test_bloom_behavior_presets(behavior: str, dataset_id: str) -> None:
    assert _behavior_presets(behavior) == [dataset_id]


@pytest.mark.parametrize(
    ("behavior", "dataset_id"),
    [
        (
            "bloom:bias:gender:ambiguous",
            "hirundo-io/bloom-gender-ambiguous-bias-free-text",
        ),
        (
            "bloom:bias:race:ambiguous",
            "hirundo-io/bloom-race-ambiguous-bias-free-text",
        ),
    ],
)
def test_bloom_ambiguous_only_behavior_presets(behavior: str, dataset_id: str) -> None:
    assert _behavior_presets(behavior) == [dataset_id]


@pytest.mark.parametrize(
    ("behavior", "kind"),
    [
        ("bloom:bias:all", "bias"),
        (" BLOOM : UNBIAS : ALL ", "unbias"),
    ],
)
def test_bloom_behavior_presets_all(behavior: str, kind: str) -> None:
    assert _behavior_presets(behavior) == [
        f"hirundo-io/bloom-{bias_type}-{kind}-free-text"
        for bias_type in ("age", "gender", "race")
    ]


def test_bloom_behavior_preset_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="bloom:bias:<type>"):
        _behavior_presets("bloom:neutral:age")


def test_bloom_behavior_preset_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="BLOOM supports: age, gender, race, all"):
        _behavior_presets("bloom:bias:religion")


def test_bloom_ambiguous_only_behavior_preset_rejects_unbias() -> None:
    with pytest.raises(ValueError, match="bloom:bias:<type>:ambiguous"):
        _behavior_presets("bloom:unbias:gender:ambiguous")


def test_factory_routes_bloom_to_free_text_bias_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(FreeTextBiasEvaluator, "__init__", lambda self, *_args: None)
    eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
    )
    dataset_config = DatasetConfig(
        file_path="hirundo-io/bloom-age-bias-free-text",
        dataset_type=DatasetType.BIAS,
    )

    evaluator = EvaluateFactory.create_evaluator(eval_config, dataset_config)

    assert isinstance(evaluator, FreeTextBiasEvaluator)


def test_ambiguous_bloom_evaluate_parses_and_scores_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class StubTokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"
        name_or_path = "stub-tokenizer"
        chat_template = ""
        padding_side = "right"

        def get_chat_template(self) -> str:
            return self.chat_template

        def apply_chat_template(
            self,
            conversation: list[dict[str, str]],
            **_kwargs: Any,
        ) -> str:
            return "\n".join(message["content"] for message in conversation)

    class StubEvalEngine:
        def __init__(
            self,
            _data_collator: Any,
            _eval_config: EvaluationConfig,
            *,
            is_judge: bool = False,
            **_kwargs: Any,
        ) -> None:
            self.tokenizer = StubTokenizer()
            self.is_judge = is_judge

        def get_batch_size(self) -> int:
            return 1

        def ensure_test_model_ready(self) -> None:
            return None

        def set_dataset(self, _dataset: object) -> None:
            return None

        def free_model(self) -> None:
            return None

    class StubDataset:
        def shuffle(self, *, seed: int) -> "StubDataset":
            assert seed == 42
            return self

        def select(self, indices: range) -> "StubDataset":
            assert list(indices) == [0]
            return self

        def __len__(self) -> int:
            return 1

    class StubCustomDataset:
        has_stereotype = False

        def __init__(
            self,
            file_path: str,
            dataset_type: DatasetType,
            **_kwargs: Any,
        ) -> None:
            assert file_path == "hirundo-io/bloom-gender-ambiguous-bias-free-text"
            assert dataset_type == DatasetType.BIAS

        def preprocess(self, *_args: Any, **_kwargs: Any) -> StubDataset:
            return StubDataset()

    monkeypatch.setattr(base_evaluator_module, "TransformersEvalEngine", StubEvalEngine)
    monkeypatch.setattr(base_evaluator_module, "CustomDataset", StubCustomDataset)
    monkeypatch.setattr(
        base_evaluator_module, "DataLoader", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        base_evaluator_module,
        "empty_cuda_cache_if_available",
        lambda: None,
    )
    monkeypatch.setattr(
        FreeTextBiasEvaluator,
        "generate",
        lambda self: [
            _BiasGenerationRecord(
                questions=["Who is the engineer?"],
                answers=["The woman is the engineer."],
                correct_answers=["the woman"],
                stereotyped_answers=None,
                finish_reasons=["stop"],
            )
        ],
    )
    monkeypatch.setattr(
        FreeTextBiasEvaluator,
        "get_judge_engine_context",
        lambda self: nullcontext(object()),
    )
    monkeypatch.setattr(
        FreeTextBiasEvaluator,
        "prepare_judge_tokenizer",
        lambda self: setattr(self, "judge_tokenizer", StubTokenizer()),
    )

    judge_outputs = iter(
        [
            [[{"generated_text": 'gt_agree: "yes"'}]],
            [[{"generated_text": 'candidate_uncertain: "no"'}]],
        ]
    )
    monkeypatch.setattr(
        FreeTextBiasEvaluator,
        "run_judge_with_backoff",
        lambda self, _judge_engine, _prompts: next(judge_outputs),
    )

    eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    dataset_config = DatasetConfig(
        file_path="hirundo-io/bloom-gender-ambiguous-bias-free-text",
        dataset_type=DatasetType.BIAS,
    )

    evaluator = EvaluateFactory.create_evaluator(eval_config, dataset_config)

    assert isinstance(evaluator, FreeTextBiasEvaluator)

    evaluator.evaluate()

    output_dir = (
        tmp_path / "model" / "bloom-gender-ambiguous-bias-free-text_DatasetType.BIAS"
    )
    with (output_dir / "responses.json").open(encoding="utf-8") as responses_file:
        responses = json.load(responses_file)

    assert responses == [
        {
            "question": "Who is the engineer?",
            "correct_answer": "the woman",
            "model_answer": "The woman is the engineer.",
            "judge_gt": 'gt_agree: "yes"',
            "judge_uncertainty": 'candidate_uncertain: "no"',
            "flagged_as_correct": True,
            "flagged_as_stereotyped": False,
            "flagged_as_unknown": False,
        }
    ]

    with (output_dir / "metrics.csv").open(
        newline="", encoding="utf-8"
    ) as metrics_file:
        metrics = list(csv.DictReader(metrics_file))

    assert metrics[0]["Error (%) ⬇️"] == "0.000"
    assert metrics[0]["Empty Responses"] == "0"

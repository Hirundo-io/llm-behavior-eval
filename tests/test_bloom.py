from pathlib import Path

import pytest

from llm_behavior_eval.evaluate import _behavior_presets
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
    FreeTextBiasEvaluator,
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

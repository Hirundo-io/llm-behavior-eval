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


def test_bloom_behavior_presets() -> None:
    assert _behavior_presets("bloom:bias:age") == [
        "hirundo-io/bloom-age-bias-free-text"
    ]
    assert _behavior_presets("bloom:unbias:age") == [
        "hirundo-io/bloom-age-unbias-free-text"
    ]
    assert _behavior_presets("bloom:bias:all") == [
        "hirundo-io/bloom-age-bias-free-text",
    ]


def test_bloom_behavior_presets_normalize_parts_and_support_unbias_all() -> None:
    assert _behavior_presets(" BLOOM : UNBIAS : ALL ") == [
        "hirundo-io/bloom-age-unbias-free-text",
    ]


def test_bloom_behavior_preset_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="bloom:bias:<type>"):
        _behavior_presets("bloom:neutral:age")


def test_bloom_behavior_preset_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="BLOOM supports"):
        _behavior_presets("bloom:bias:race")


def test_bloom_behavior_preset_rejects_unpublished_gender_split() -> None:
    with pytest.raises(ValueError, match="BLOOM supports"):
        _behavior_presets("bloom:bias:gender")


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

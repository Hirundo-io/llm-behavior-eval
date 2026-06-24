from pathlib import Path

from llm_behavior_eval.evaluate import _behavior_presets
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
    FreeTextPromptInjectionEvaluator,
)


def test_bloom_injection_behavior_presets() -> None:
    assert _behavior_presets("injection:bloom:malicious") == [
        "hirundo-io/bloom-prompt-injection-malicious-free-text"
    ]
    assert _behavior_presets("injection:bloom:benign") == [
        "hirundo-io/bloom-prompt-injection-benign-free-text"
    ]
    assert _behavior_presets("prompt-injection") == [
        "hirundo-io/prompt-injection-purple-llama"
    ]


def test_injection_factory_routes_bloom_splits_and_purple_llama(
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
        "hirundo-io/bloom-prompt-injection-malicious-free-text",
        "hirundo-io/bloom-prompt-injection-benign-free-text",
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

from pathlib import Path
from typing import Any, cast

import pytest

from llm_behavior_eval import DatasetConfig, DatasetType, EvaluationConfig
from llm_behavior_eval.evaluation_utils.censorship_utils import CCPC_DATASET_ID
from llm_behavior_eval.evaluation_utils.eval_config import FAMILY_TOKEN_DEFAULTS
from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
    FreeTextBiasEvaluator,
)
from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    FreeTextCensorshipEvaluator,
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
    OR_BENCH_DATASET,
    XSTEST_DATASET,
)


def test_evaluate_factory_applies_refusal_defaults_for_programmatic_callers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_init(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        del self, dataset_config
        captured["evaluator_family"] = eval_config.evaluator_family
        captured["max_answer_tokens"] = eval_config.max_answer_tokens
        captured["max_judge_tokens"] = eval_config.max_judge_tokens
        captured["sample_judge"] = eval_config.sample_judge

    monkeypatch.setattr(FreeTextRefusalEvaluator, "__init__", fake_init)

    original_config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
    )
    EvaluateFactory.create_evaluator(
        original_config,
        DatasetConfig(file_path=OR_BENCH_DATASET, dataset_type=DatasetType.BIAS),
    )

    assert captured == {
        "evaluator_family": "refusal",
        "max_answer_tokens": FAMILY_TOKEN_DEFAULTS["refusal"]["max_answer_tokens"],
        "max_judge_tokens": FAMILY_TOKEN_DEFAULTS["refusal"]["max_judge_tokens"],
        "sample_judge": FAMILY_TOKEN_DEFAULTS["refusal"]["sample_judge"],
    }
    assert original_config.evaluator_family is None
    assert original_config.max_answer_tokens is None
    assert original_config.max_judge_tokens is None


def test_evaluate_factory_constructs_censorship_evaluator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify the factory constructs the dedicated censorship evaluator.

    Args:
        monkeypatch: Pytest patching helper.
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """

    def fake_init(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        """Replace model-loading initialization for the routing test.

        Args:
            eval_config: Evaluation settings ignored by this fixture.
            dataset_config: Dataset settings ignored by this fixture.

        Returns:
            None.
        """
        del eval_config, dataset_config

    monkeypatch.setattr(FreeTextCensorshipEvaluator, "__init__", fake_init)

    evaluator = EvaluateFactory.create_evaluator(
        EvaluationConfig(model_path_or_repo_id="fake/model", results_dir=tmp_path),
        DatasetConfig(
            file_path=CCPC_DATASET_ID,
            dataset_type=DatasetType.BIAS,
        ),
    )

    assert isinstance(evaluator, FreeTextCensorshipEvaluator)


def test_evaluate_factory_reports_evaluator_family() -> None:
    assert EvaluateFactory.get_evaluator_family(XSTEST_DATASET) == "refusal"
    assert (
        EvaluateFactory.get_evaluator_family("hirundo-io/prompt-injection-purple-llama")
        == "prompt-injection"
    )
    assert (
        EvaluateFactory.get_evaluator_family("hirundo-io/bbq-gender-bias-free-text")
        == "bias"
    )


def test_eval_config_validates_lora_only_with_vllm() -> None:
    """Test that lora_path_or_repo_id can only be used when vLLM is enabled."""
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.eval_config import (
        EvaluationConfig,
    )

    # Should raise error when LoRA is specified but not using vLLM
    with pytest.raises(
        ValueError,
        match="LoRA usage currently only supported with vLLM",
    ):
        EvaluationConfig(
            model_path_or_repo_id="fake/model",
            results_dir=Path("/tmp"),
            lora_path_or_repo_id="/path/to/lora",
            model_engine="transformers",  # Not using vLLM
        )


def test_eval_config_allows_lora_with_vllm_inference_engine() -> None:
    """Test that lora_path_or_repo_id is allowed when inference_engine is vllm."""
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.eval_config import (
        EvaluationConfig,
    )

    # Should not raise error when LoRA is specified and using vLLM via inference_engine
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=Path("/tmp"),
        lora_path_or_repo_id="/path/to/lora",
        inference_engine="vllm",
    )
    assert config.lora_path_or_repo_id == "/path/to/lora"


def test_eval_config_allows_lora_with_vllm_model_engine() -> None:
    """Test that lora_path_or_repo_id is allowed when model_engine is vllm."""
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.eval_config import (
        EvaluationConfig,
    )

    # Should not raise error when LoRA is specified and using vLLM via model_engine
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=Path("/tmp"),
        lora_path_or_repo_id="/path/to/lora",
        model_engine="vllm",
    )
    assert config.lora_path_or_repo_id == "/path/to/lora"


def test_eval_config_allows_lora_with_vllm_config() -> None:
    """Test that lora_path_or_repo_id is allowed when vllm_config is provided."""
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.eval_config import (
        EvaluationConfig,
    )
    from llm_behavior_eval.evaluation_utils.vllm_config import VllmConfig

    vllm_config = VllmConfig(max_model_len=8192)

    # Should not raise error when LoRA is specified and vllm_config is provided
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=Path("/tmp"),
        lora_path_or_repo_id="/path/to/lora",
        vllm_config=vllm_config,
        model_engine="vllm",
    )
    assert config.lora_path_or_repo_id == "/path/to/lora"


def test_eval_config_allows_none_lora_path() -> None:
    """Test that lora_path_or_repo_id can be None."""
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.eval_config import (
        EvaluationConfig,
    )

    # Should not raise error when LoRA is None
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=Path("/tmp"),
        lora_path_or_repo_id=None,
        model_engine="transformers",
    )
    assert config.lora_path_or_repo_id is None


def test_eval_config_allows_relative_model_output_dir() -> None:
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig

    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=Path("/tmp"),
        model_output_dir="team-a/model-v1",
        model_engine="transformers",
    )
    assert config.model_output_dir == "team-a/model-v1"


def test_eval_config_rejects_absolute_model_output_dir() -> None:
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig

    with pytest.raises(ValueError, match="model_output_dir must be a relative path"):
        EvaluationConfig(
            model_path_or_repo_id="fake/model",
            results_dir=Path("/tmp"),
            model_output_dir="/tmp/escape",
            model_engine="transformers",
        )


def test_eval_config_rejects_parent_traversal_in_model_output_dir() -> None:
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig

    with pytest.raises(
        ValueError,
        match="model_output_dir cannot contain '\\.\\.' and must stay under base output directory",
    ):
        EvaluationConfig(
            model_path_or_repo_id="fake/model",
            results_dir=Path("/tmp"),
            model_output_dir="../escape",
            model_engine="transformers",
        )


def test_eval_config_resolve_for_family_applies_defaults_only_when_values_are_unset() -> (
    None
):
    from pathlib import Path

    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=Path("/tmp"),
    )
    resolved = config.resolve_for_family("refusal")
    assert resolved.max_answer_tokens == 2048
    assert resolved.max_judge_tokens == 128
    assert resolved.sample_judge is False
    assert resolved.enable_thinking is True
    assert resolved.evaluator_family == "refusal"
    assert config.max_answer_tokens is None
    assert config.enable_thinking is None

    overridden = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=Path("/tmp"),
        max_answer_tokens=384,
        max_judge_tokens=96,
        sample_judge=True,
        enable_thinking=False,
    )
    resolved_overrides = overridden.resolve_for_family("refusal")
    assert resolved_overrides.max_answer_tokens == 384
    assert resolved_overrides.max_judge_tokens == 96
    assert resolved_overrides.sample_judge is True
    assert resolved_overrides.enable_thinking is False


def test_dataset_credential_is_excluded_from_config(tmp_path):
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        dataset_token="synthetic-dataset",
    )
    assert "dataset_token" not in config.model_dump()
    assert "synthetic-dataset" not in config.model_dump_json()
    assert "synthetic-dataset" not in repr(config)
    assert "synthetic-dataset" not in str(config)
    assert config.resolve_for_family("refusal").dataset_token == "synthetic-dataset"


def test_python_judge_override() -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir="results",
        judge_path_or_repo_id="example/custom-judge",
    )
    assert config.judge_path_or_repo_id == "example/custom-judge"


@pytest.mark.parametrize(
    ("dataset_id", "expected_class"),
    [
        ("hirundo-io/halueval", FreeTextHaluEvaluator),
        (OR_BENCH_DATASET, FreeTextRefusalEvaluator),
        (
            "hirundo-io/prompt-injection-purple-llama",
            FreeTextPromptInjectionEvaluator,
        ),
        ("hirundo-io/bbq-gender-bias-free-text", FreeTextBiasEvaluator),
    ],
)
def test_evaluate_factory_routes_each_family_to_the_expected_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_id: str,
    expected_class: type[object],
) -> None:
    sentinel = object()

    def fake_init(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        del eval_config, dataset_config
        self._sentinel = sentinel

    for evaluator_class in (
        FreeTextHaluEvaluator,
        FreeTextRefusalEvaluator,
        FreeTextPromptInjectionEvaluator,
        FreeTextBiasEvaluator,
    ):
        monkeypatch.setattr(evaluator_class, "__init__", fake_init)

    evaluator = EvaluateFactory.create_evaluator(
        EvaluationConfig(model_path_or_repo_id="fake/model", results_dir=tmp_path),
        DatasetConfig(file_path=dataset_id, dataset_type=DatasetType.BIAS),
    )

    assert isinstance(evaluator, expected_class)
    assert cast("Any", evaluator)._sentinel is sentinel

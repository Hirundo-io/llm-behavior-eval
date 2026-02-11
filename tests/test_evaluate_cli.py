from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

import llm_behavior_eval.evaluate as evaluate
from llm_behavior_eval import DatasetConfig, EvaluationConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llm_behavior_eval.evaluation_utils.base_evaluator import (
        _GenerationRecord,
    )
    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine


class _StubEvaluator:
    def update_dataset_config(self, dataset_config: DatasetConfig) -> None:
        return None

    def generate(self) -> Sequence[_GenerationRecord]:
        return []

    def free_test_model(self) -> None:
        return None

    def get_grading_context(self) -> AbstractContextManager:
        return nullcontext()

    def dataset_mlflow_run(self, run_name: str | None = None) -> AbstractContextManager:
        return nullcontext()

    def grade(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        return None

    def cleanup(self, error: bool = False) -> None:
        return None


@dataclass
class CapturedConfigs:
    eval_config: EvaluationConfig
    dataset_config: DatasetConfig


@pytest.fixture
def capture_eval_config(monkeypatch: pytest.MonkeyPatch) -> list[EvaluationConfig]:
    captured: list[EvaluationConfig] = []

    def _fake_create(
        eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> _StubEvaluator:
        captured.append(eval_config)
        return _StubEvaluator()

    monkeypatch.setattr(
        evaluate.EvaluateFactory,
        "create_evaluator",
        staticmethod(_fake_create),
    )
    return captured


@pytest.fixture
def capture_configs(monkeypatch: pytest.MonkeyPatch) -> list[CapturedConfigs]:
    captured: list[CapturedConfigs] = []

    def _fake_create(
        eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> _StubEvaluator:
        captured.append(
            CapturedConfigs(eval_config=eval_config, dataset_config=dataset_config)
        )
        return _StubEvaluator()

    monkeypatch.setattr(
        evaluate.EvaluateFactory,
        "create_evaluator",
        staticmethod(_fake_create),
    )
    return captured


def test_main_applies_max_samples_option(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu", max_samples=42)
    assert capture_eval_config[-1].max_samples == 42


def test_main_runs_full_dataset_when_nonpositive_max_samples(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu", max_samples=0)
    assert capture_eval_config[-1].max_samples is None


def test_main_passes_judge_quantization_flag(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu", use_4bit_judge=True)
    assert capture_eval_config[-1].use_4bit_judge is True


def test_main_falls_back_to_env_mlflow_tracking_uri_when_enabled(
    capture_eval_config: list[EvaluationConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.from.env")
    evaluate.main("fake/model", "hallu", use_mlflow=True)
    assert capture_eval_config[-1].mlflow_config is not None
    assert (
        capture_eval_config[-1].mlflow_config.mlflow_tracking_uri
        == "http://tracking.from.env"
    )


def test_main_sets_inference_engine_and_sampling(
    capture_configs: list[CapturedConfigs],
) -> None:
    evaluate.main(
        "fake/model",
        "hallu",
        inference_engine="vllm",
        vllm_max_model_len=8192,
        vllm_judge_max_model_len=4096,
        sample=True,
        temperature=0.3,
        top_p=0.7,
        top_k=12,
        seed=123,
    )
    captured = capture_configs[-1]
    eval_config = captured.eval_config
    dataset_config = captured.dataset_config

    assert isinstance(eval_config, EvaluationConfig)
    assert eval_config.inference_engine == "vllm"
    assert eval_config.vllm_config is not None
    assert eval_config.vllm_config.max_model_len == 8192
    assert eval_config.vllm_config.judge_max_model_len == 4096
    assert eval_config.sampling_config.temperature == 0.3
    assert eval_config.sampling_config.top_p == 0.7
    assert eval_config.sampling_config.top_k == 12
    assert eval_config.sampling_config.seed == 123
    assert dataset_config.seed == 123


def test_main_allows_replacing_existing_output(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu", replace_existing_output=True)
    assert capture_eval_config[-1].replace_existing_output is True


def test_main_passes_vllm_optional_args(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main(
        "fake/model",
        "hallu",
        inference_engine="vllm",
        vllm_tokenizer_mode="slow",
        vllm_config_format="hf",
        vllm_load_format="safetensors",
    )
    eval_config = capture_eval_config[-1]
    assert eval_config.vllm_config is not None
    assert eval_config.vllm_config.tokenizer_mode == "slow"
    assert eval_config.vllm_config.config_format == "hf"
    assert eval_config.vllm_config.load_format == "safetensors"


def test_main_does_not_create_vllm_config_when_not_using_vllm(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main(
        "fake/model",
        "hallu",
        model_engine="transformers",
        vllm_tokenizer_mode="slow",
    )
    eval_config = capture_eval_config[-1]
    assert eval_config.vllm_config is None


def test_main_validates_vllm_config_only_with_vllm(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    """Test that vllm_config can only be used when vLLM is actually enabled."""
    # This would raise an error when instantiating EvaluationConfig
    # with vllm_config but no vLLM engine selected
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.eval_config import (
        EvaluationConfig,
    )
    from llm_behavior_eval.evaluation_utils.vllm_config import VllmConfig

    vllm_config = VllmConfig(max_model_len=8192)

    with pytest.raises(ValueError, match="vllm_config can only be specified"):
        EvaluationConfig(
            model_path_or_repo_id="fake/model",
            results_dir=Path("/tmp"),
            vllm_config=vllm_config,
            model_engine="transformers",  # Not using vLLM
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


def test_main_passes_answer_tokens_and_judge_tokens_via_cli(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    """Test that max_answer_tokens and max_judge_tokens CLI options are passed correctly."""
    evaluate.main(
        "fake/model",
        "hallu",
        max_answer_tokens=256,
        max_judge_tokens=64,
    )
    eval_config = capture_eval_config[-1]
    assert eval_config.max_answer_tokens == 256
    assert eval_config.max_judge_tokens == 64


def test_main_uses_default_answer_and_judge_tokens(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    """Test that default values are applied when tokens are not specified."""
    evaluate.main("fake/model", "hallu")
    eval_config = capture_eval_config[-1]
    assert eval_config.max_answer_tokens == 128  # Default from EvaluationConfig
    assert eval_config.max_judge_tokens == 32  # Default from EvaluationConfig


def test_main_passes_model_inference_config_options(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    """Test that model inference options are passed correctly."""
    evaluate.main(
        "fake/model",
        "hallu",
        batch_size=64,
        use_4bit=True,
        device_map="/gpu:0",
    )
    eval_config = capture_eval_config[-1]
    assert eval_config.batch_size == 64
    assert eval_config.use_4bit is True
    assert eval_config.device_map == "/gpu:0"


def test_main_passes_judge_inference_config_options(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    """Test that judge inference options are passed correctly."""
    evaluate.main(
        "fake/model",
        "hallu",
        judge_batch_size=32,
        sample_judge=True,
    )
    eval_config = capture_eval_config[-1]
    assert eval_config.judge_batch_size == 32
    assert eval_config.sample_judge is True

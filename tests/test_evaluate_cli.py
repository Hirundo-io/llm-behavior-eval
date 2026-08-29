from __future__ import annotations

import os
import sys
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest
from click.utils import strip_ansi
from typer.testing import CliRunner

import llm_behavior_eval.evaluate as evaluate
from llm_behavior_eval import DatasetConfig, EvaluationConfig
from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CCPC_DATASET_ID,
    CCPC_JUDGE_MODEL,
)
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
from llm_behavior_eval.evaluation_utils.vllm_config import (
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    VllmConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.base_evaluator import (
        _GenerationRecord,
    )
    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine


class _StubEvaluator:
    started_mlflow_run = False

    def update_dataset_config(self, dataset_config: DatasetConfig) -> None:
        return None

    def generate(self) -> Sequence[_GenerationRecord]:
        return []

    def free_test_model(self) -> None:
        return None

    def get_grading_context(self) -> AbstractContextManager:
        return nullcontext()

    def dataset_mlflow_run(self) -> AbstractContextManager:
        return nullcontext()

    def _grade_impl(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        return None

    def grade(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        self._grade_impl(generations, judge_engine)

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


def test_behavior_presets_expand_refusal_xstest() -> None:
    assert evaluate._behavior_presets("refusal:xstest") == [XSTEST_DATASET]


def test_behavior_presets_expand_refusal_orbench() -> None:
    assert evaluate._behavior_presets("refusal:orbench") == [OR_BENCH_DATASET]


def test_behavior_presets_expand_refusal_all() -> None:
    assert evaluate._behavior_presets("refusal:all") == [
        XSTEST_DATASET,
        OR_BENCH_DATASET,
    ]


def test_cli_help_includes_chinese_censorship_guidance() -> None:
    result = CliRunner().invoke(evaluate.app, ["--help"])
    visible_output = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "chinese_censorship" in visible_output
    assert "--judge-model" in visible_output
    assert "google/gemma-4-26B-A4B-it" in visible_output
    assert "evaluated model" in visible_output
    assert "both" in visible_output
    assert "explicit flag" in visible_output
    assert "bias:<type|all>" in visible_output
    assert "unbias:<type|all>" in visible_output


def test_invalid_behavior_guidance_includes_chinese_censorship() -> None:
    with pytest.raises(ValueError, match="chinese_censorship"):
        evaluate._behavior_presets("invalid")


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


def test_main_uses_refusal_dataset_type_for_refusal_presets(
    capture_configs: list[CapturedConfigs],
) -> None:
    evaluate.main("fake/model", "refusal:xstest")
    assert capture_configs[-1].dataset_config.file_path == XSTEST_DATASET
    assert capture_configs[-1].dataset_config.dataset_type.value == "bias"


def test_main_raises_missing_dataset_error(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[DatasetConfig] = []

    def _fake_create(
        eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> _StubEvaluator:
        if dataset_config.file_path == "hirundo-io/halueval":
            raise RuntimeError(
                "Failed to load dataset 'hirundo-io/halueval'. "
                "Check that the identifier is correct."
            )
        captured.append(dataset_config)
        return _StubEvaluator()

    monkeypatch.setattr(
        evaluate.EvaluateFactory,
        "create_evaluator",
        staticmethod(_fake_create),
    )

    with pytest.raises(
        RuntimeError, match="Failed to load dataset 'hirundo-io/halueval'"
    ):
        evaluate.main("fake/model", "hallu")

    assert captured == []


def test_main_passes_model_output_dir_override(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu", model_output_dir="custom-model-dir")
    assert capture_eval_config[-1].model_output_dir == "custom-model-dir"


def test_main_threads_model_and_judge_revisions_into_eval_config(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu")
    eval_config = capture_eval_config[-1]
    assert eval_config.model_revision is None
    assert eval_config.judge_revision is None

    evaluate.main(
        "fake/model",
        "hallu",
        model_revision="target-sha",
        judge_revision="judge-sha",
    )
    eval_config = capture_eval_config[-1]
    assert eval_config.model_revision == "target-sha"
    assert eval_config.judge_revision == "judge-sha"


def test_cli_help_exposes_revision_pinning_flags() -> None:
    # A wide terminal keeps long option names from being ellipsized in the
    # rendered help table.
    result = CliRunner().invoke(
        evaluate.app, ["--help"], env={"COLUMNS": "250", "LINES": "50"}
    )
    visible_output = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "--model-revision" in visible_output
    assert "--judge-revision" in visible_output
    assert "--dataset-revision" in visible_output
    assert "--model-reasoning-effort" in visible_output


def test_main_threads_dataset_revision_into_dataset_config(
    capture_configs: list[CapturedConfigs],
) -> None:
    evaluate.main("fake/model", "hallu")
    assert all(
        captured.dataset_config.dataset_revision is None for captured in capture_configs
    )

    previous_count = len(capture_configs)
    evaluate.main("fake/model", "hallu", dataset_revision="dataset-sha")

    assert len(capture_configs) > previous_count
    assert all(
        captured.dataset_config.dataset_revision == "dataset-sha"
        for captured in capture_configs[previous_count:]
    )


def test_main_passes_omitted_ccpc_dataset_revision_to_the_ccpc_route(
    capture_configs: list[CapturedConfigs],
) -> None:
    evaluate.main("fake/model", "chinese_censorship", judge_model=CCPC_JUDGE_MODEL)

    dataset_config = capture_configs[-1].dataset_config
    assert dataset_config.dataset_id == CCPC_DATASET_ID
    assert dataset_config.dataset_revision is None


def test_main_threads_model_reasoning_effort_into_eval_config(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu")
    assert capture_eval_config[-1].model_reasoning_effort is None

    evaluate.main("fake/model", "hallu", model_reasoning_effort="low")
    assert capture_eval_config[-1].model_reasoning_effort == "low"


def test_main_rejects_mixed_evaluator_families() -> None:
    with pytest.raises(ValueError, match="multiple evaluator families"):
        evaluate.main("fake/model", "hallu,refusal:all")


def test_main_enforces_ccpc_judge_contract_and_checks_both_trusted_providers(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    """Verify CCPC routing locks its judge and checks both providers for trust.
    Args:
        capture_eval_config: Evaluation configurations captured by the factory stub.
    """
    evaluate.main(
        "google/model",
        "chinese_censorship",
        judge_model=CCPC_JUDGE_MODEL,
    )
    assert capture_eval_config[-1].evaluator_family == "censorship"
    assert capture_eval_config[-1].trust_remote_code

    evaluate.main(
        "google/model",
        "chinese_censorship",
        judge_model=CCPC_JUDGE_MODEL,
        trust_remote_code=False,
    )
    assert not capture_eval_config[-1].trust_remote_code

    evaluate.main(
        "untrusted/model",
        "chinese_censorship",
        judge_model=CCPC_JUDGE_MODEL,
    )
    assert not capture_eval_config[-1].trust_remote_code

    with pytest.raises(ValueError, match="chinese_censorship benchmark requires"):
        evaluate.main("fake/model", "chinese_censorship")
    assert len(capture_eval_config) == 3

    evaluate.main("google/model", "hallu")
    assert capture_eval_config[-1].judge_path_or_repo_id == "google/gemma-3-12b-it"
    assert capture_eval_config[-1].trust_remote_code

    evaluate.main("google/model", "hallu", judge_model="untrusted/judge")
    assert not capture_eval_config[-1].trust_remote_code


def test_main_rejects_ccpc_expected_rows_without_local_path() -> None:
    with pytest.raises(ValueError, match="--ccpc-local-dataset-path"):
        evaluate.main(
            "google/model",
            "chinese_censorship",
            judge_model=CCPC_JUDGE_MODEL,
            ccpc_expected_rows=500,
        )


def test_main_rejects_ccpc_expected_sha256_without_local_path() -> None:
    with pytest.raises(ValueError, match="--ccpc-local-dataset-path"):
        evaluate.main(
            "google/model",
            "chinese_censorship",
            judge_model=CCPC_JUDGE_MODEL,
            ccpc_expected_sha256="a" * 64,
        )


def test_main_rejects_ccpc_expected_rows_and_sha256_without_local_path() -> None:
    with pytest.raises(ValueError, match="--ccpc-local-dataset-path"):
        evaluate.main(
            "google/model",
            "chinese_censorship",
            judge_model=CCPC_JUDGE_MODEL,
            ccpc_expected_rows=500,
            ccpc_expected_sha256="a" * 64,
        )


def test_main_allows_chinese_censorship_without_any_ccpc_local_flags(
    capture_configs: list[CapturedConfigs],
) -> None:
    evaluate.main(
        "google/model",
        "chinese_censorship",
        judge_model=CCPC_JUDGE_MODEL,
    )
    dataset_config = capture_configs[-1].dataset_config
    assert dataset_config.file_path == CCPC_DATASET_ID
    assert dataset_config.expected_row_count is None
    assert dataset_config.expected_sha256 is None


def test_main_routes_full_ccpc_local_combination_into_dataset_config(
    capture_configs: list[CapturedConfigs],
) -> None:
    evaluate.main(
        "google/model",
        "chinese_censorship",
        judge_model=CCPC_JUDGE_MODEL,
        model_revision="target-sha",
        judge_revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        ccpc_local_dataset_path="/tmp/local_ccpc.jsonl",
        ccpc_expected_rows=500,
        ccpc_expected_sha256="a" * 64,
    )
    dataset_config = capture_configs[-1].dataset_config
    assert dataset_config.file_path == "/tmp/local_ccpc.jsonl"
    assert dataset_config.dataset_id == CCPC_DATASET_ID
    assert dataset_config.expected_row_count == 500
    assert dataset_config.expected_sha256 == "a" * 64
    assert dataset_config.dataset_revision is None


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
        DatasetConfig(file_path=dataset_id, dataset_type=evaluate.DatasetType.BIAS),
    )

    assert isinstance(evaluator, expected_class)
    assert cast("Any", evaluator)._sentinel is sentinel


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
        DatasetConfig(
            file_path=OR_BENCH_DATASET, dataset_type=evaluate.DatasetType.BIAS
        ),
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
            dataset_type=evaluate.DatasetType.BIAS,
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


def test_vllm_defaults_preserve_vllm_configuration(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "hallu", inference_engine="vllm")

    vllm_config = capture_eval_config[-1].vllm_config
    assert vllm_config is not None
    assert vllm_config.max_model_len is None
    assert vllm_config.gpu_memory_utilization == DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
    assert VllmConfig().max_model_len is None
    assert VllmConfig().gpu_memory_utilization == DEFAULT_VLLM_GPU_MEMORY_UTILIZATION


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


def test_main_vllm_evaluated_model_loads_multimodal_by_default(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    """The evaluated model keeps its multimodal encoders unless opted out.

    Only judge loads are forced text-only (see the engine test); the evaluated
    model's default is unchanged from a non-vLLM run.
    """
    evaluate.main("fake/model", "hallu", inference_engine="vllm")

    vllm_config = capture_eval_config[-1].vllm_config
    assert vllm_config is not None
    assert vllm_config.language_model_only is False


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
    """Unset CLI token options stay None until resolved per evaluator family."""
    evaluate.main("fake/model", "hallu")
    eval_config = capture_eval_config[-1]
    assert eval_config.max_answer_tokens is None
    assert eval_config.max_judge_tokens is None
    assert eval_config.sample_judge is None

    resolved = eval_config.resolve_for_family("hallucination")
    assert resolved.max_answer_tokens == 128
    assert resolved.max_judge_tokens == 32
    assert resolved.sample_judge is False


def test_main_uses_refusal_preset_defaults_when_tokens_omitted(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main("fake/model", "refusal:all")
    eval_config = capture_eval_config[-1]
    assert eval_config.max_answer_tokens is None
    assert eval_config.max_judge_tokens is None
    assert eval_config.sample_judge is None

    resolved = eval_config.resolve_for_family("refusal")
    assert resolved.max_answer_tokens == 256
    assert resolved.max_judge_tokens == 128
    assert resolved.sample_judge is False


def test_main_preserves_explicit_refusal_cli_overrides(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main(
        "fake/model",
        "refusal:orbench",
        max_answer_tokens=384,
        max_judge_tokens=96,
        sample_judge=True,
    )
    eval_config = capture_eval_config[-1]
    assert eval_config.max_answer_tokens == 384
    assert eval_config.max_judge_tokens == 96
    assert eval_config.sample_judge is True


def test_eval_config_resolve_for_family_applies_defaults_only_when_values_are_unset() -> (
    None
):
    from pathlib import Path

    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=Path("/tmp"),
    )
    resolved = config.resolve_for_family("refusal")
    assert resolved.max_answer_tokens == 256
    assert resolved.max_judge_tokens == 128
    assert resolved.sample_judge is False
    assert resolved.evaluator_family == "refusal"
    assert config.max_answer_tokens is None

    overridden = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=Path("/tmp"),
        max_answer_tokens=384,
        max_judge_tokens=96,
        sample_judge=True,
    )
    resolved_overrides = overridden.resolve_for_family("refusal")
    assert resolved_overrides.max_answer_tokens == 384
    assert resolved_overrides.max_judge_tokens == 96
    assert resolved_overrides.sample_judge is True


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


def test_main_defaults_output_dir_to_data_dir(
    capture_eval_config: list[EvaluationConfig],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        base_dir = tmp_path / "localapp"
        monkeypatch.setenv("LOCALAPPDATA", str(base_dir))
        expected = base_dir / "llm-behavior-eval" / "results"
    elif sys.platform == "darwin":
        monkeypatch.setenv("HOME", str(tmp_path))
        expected = (
            tmp_path
            / "Library"
            / "Application Support"
            / "llm-behavior-eval"
            / "results"
        )
    else:
        base_dir = tmp_path / "xdg"
        monkeypatch.setenv("XDG_DATA_HOME", str(base_dir))
        expected = base_dir / "llm-behavior-eval" / "results"

    evaluate.main("fake/model", "hallu")
    eval_config = capture_eval_config[-1]
    assert eval_config.results_dir == expected

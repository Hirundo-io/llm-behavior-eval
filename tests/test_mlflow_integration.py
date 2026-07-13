from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from llm_behavior_eval.evaluation_utils.base_evaluator import BaseEvaluator
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import (
    EvaluationConfig,
    MlflowConfig,
)
from llm_behavior_eval.evaluation_utils.vllm_config import VllmConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets import Dataset
    from torch.utils.data import DataLoader

    from llm_behavior_eval.evaluation_utils.base_evaluator import _GenerationRecord
    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine


class DummyTokenizer:
    """Minimal tokenizer stub for BaseEvaluator tests."""

    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.padding_side = "right"


class DummyModel(SimpleNamespace):
    """Simple namespace with a device attribute."""

    def __init__(self) -> None:
        super().__init__(device="cpu")


class DummyEvaluator(BaseEvaluator):
    def prepare_dataloader(self) -> None:
        self.eval_dataset = cast("Dataset", [])
        self.eval_loader = cast("DataLoader", [])
        self.num_samples = 3
        self.has_stereotype = False

    def evaluate(self) -> None:
        return None

    def generate(self) -> Sequence[_GenerationRecord]:
        return []

    def _grade_impl(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        return None

    def get_grading_context(self) -> AbstractContextManager[EvalEngine]:
        # This test doesn't exercise grading, but `evaluate.main()` expects an
        # `EvalEngine` from the context manager. Yield a lightweight stub.
        return nullcontext(cast("EvalEngine", object()))


def _make_run(run_id: str, run_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(
            experiment_id="exp-1",
            run_id=run_id,
            run_name=run_name,
        )
    )


class MlflowRecorder:
    """Tiny MLflow stand-in that records calls made by BaseEvaluator."""

    def __init__(self) -> None:
        self.active_run_result: object | None = None
        self.get_run_result: object | None = None
        self.start_run_result: object | None = _make_run("started-run", "model")
        self.reset()

    def reset(self) -> None:
        self.tracking_uris: list[str] = []
        self.experiments: list[str] = []
        self.start_run_calls: list[dict[str, str]] = []
        self.get_run_calls: list[str] = []
        self.log_metric_calls: list[tuple[str, float]] = []
        self.log_metrics_calls: list[tuple[dict[str, float], int | None]] = []
        self.log_artifacts_calls: list[tuple[str, str | None]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uris.append(uri)

    def set_experiment(
        self, name: str | None = None, *, experiment_id: str | None = None
    ) -> None:
        self.experiments.append(
            experiment_id if experiment_id is not None else str(name)
        )

    def active_run(self) -> object | None:
        return self.active_run_result

    def start_run(
        self, *, run_name: str | None = None, run_id: str | None = None
    ) -> object:
        self.start_run_calls.append(
            {
                key: value
                for key, value in {"run_name": run_name, "run_id": run_id}.items()
                if value is not None
            }
        )
        return self.start_run_result

    def get_run(self, run_id: str) -> object:
        self.get_run_calls.append(run_id)
        if self.get_run_result is None:
            raise AssertionError("get_run_result must be set before get_run is called")
        return self.get_run_result

    def log_metric(self, key: str, value: float) -> None:
        self.log_metric_calls.append((key, value))

    def log_metrics(
        self, metrics: dict[str, float], *, step: int | None = None
    ) -> None:
        self.log_metrics_calls.append((metrics, step))

    def log_artifacts(
        self, local_dir: str, *, artifact_path: str | None = None
    ) -> None:
        self.log_artifacts_calls.append((local_dir, artifact_path))


@pytest.fixture(autouse=True)
def _mock_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_tokenizer = DummyTokenizer()
    dummy_model = DummyModel()

    def _stub_loader(
        *_args: object, **_kwargs: object
    ) -> tuple[DummyTokenizer, DummyModel]:
        return dummy_tokenizer, dummy_model

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.transformers_eval_engine.load_transformers_model_and_tokenizer",
        _stub_loader,
    )


@pytest.fixture
def mlflow_mock(monkeypatch: pytest.MonkeyPatch) -> MlflowRecorder:
    mock = MlflowRecorder()
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.base_evaluator.mlflow",
        mock,
    )
    return mock


@pytest.fixture
def evaluation_config(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        batch_size=1,
        mlflow_config=MlflowConfig(
            mlflow_tracking_uri="http://tracking.example",
            mlflow_experiment_name="MLflow Tests",
        ),
    )


@pytest.fixture
def evaluation_config_default_mlflow(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        batch_size=1,
        mlflow_config=MlflowConfig(),
    )


@pytest.fixture
def evaluation_config_no_mlflow(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        batch_size=1,
    )


@pytest.fixture
def dataset_config() -> DatasetConfig:
    return DatasetConfig(
        file_path="hirundo-io/bbq-gender-bias-free-text",
        dataset_type=DatasetType.BIAS,
    )


def test_init_mlflow_starts_run_and_logs_params(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    parent_run = _make_run("parent-run", "model")
    mlflow_mock.start_run_result = parent_run
    mlflow_mock.active_run_result = None
    DummyEvaluator(evaluation_config, dataset_config)

    assert mlflow_mock.tracking_uris == ["http://tracking.example"]
    assert mlflow_mock.experiments == ["MLflow Tests"]
    assert mlflow_mock.start_run_calls == [{"run_name": "model"}]


def test_init_with_default_mlflow_config_still_logs(
    evaluation_config_default_mlflow: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    parent_run = _make_run("parent-run", "model")
    mlflow_mock.start_run_result = parent_run
    mlflow_mock.active_run_result = None
    DummyEvaluator(evaluation_config_default_mlflow, dataset_config)

    assert mlflow_mock.tracking_uris == []
    assert mlflow_mock.experiments == []
    assert mlflow_mock.start_run_calls == [{"run_name": "model"}]


def test_init_mlflow_uses_existing_active_run(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    active_run = _make_run("active-run", "existing")
    mlflow_mock.active_run_result = active_run

    evaluator = DummyEvaluator(evaluation_config, dataset_config)

    assert mlflow_mock.start_run_calls == []
    assert evaluator.parent_run is active_run
    assert evaluator.mlflow_run is active_run


def test_init_mlflow_prefers_active_run_over_lora_inferred_run_id(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    active_run = _make_run("active-run", "existing")
    mlflow_mock.active_run_result = active_run

    config_with_lora = EvaluationConfig(
        model_path_or_repo_id=evaluation_config.model_path_or_repo_id,
        results_dir=evaluation_config.results_dir,
        batch_size=evaluation_config.batch_size,
        vllm_config=VllmConfig(),
        judge_engine="vllm",
        lora_path_or_repo_id="mlflow://abc123def45678901234567890123456",
        mlflow_config=evaluation_config.mlflow_config,
    )
    evaluator = DummyEvaluator(config_with_lora, dataset_config)

    assert mlflow_mock.get_run_calls == []
    assert mlflow_mock.start_run_calls == []
    assert evaluator.parent_run is active_run
    assert evaluator.mlflow_run is active_run


def test_init_mlflow_reuses_lora_run_id_from_http_tracking_uri_ref(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    lora_run_id = "abc123def45678901234567890123456"
    existing_run = _make_run(lora_run_id, "training")
    existing_run.info.experiment_id = "exp-1"
    mlflow_mock.get_run_result = existing_run
    mlflow_mock.active_run_result = None
    mlflow_mock.start_run_result = existing_run

    config_with_lora = EvaluationConfig(
        model_path_or_repo_id=evaluation_config.model_path_or_repo_id,
        results_dir=evaluation_config.results_dir,
        batch_size=evaluation_config.batch_size,
        vllm_config=VllmConfig(),
        judge_engine="vllm",
        lora_path_or_repo_id=(
            f"http://tracking.example/runs/{lora_run_id}/"
            "hf_checkpoints/checkpoint-000020"
        ),
        mlflow_config=evaluation_config.mlflow_config,
    )
    DummyEvaluator(config_with_lora, dataset_config)

    assert mlflow_mock.get_run_calls == [lora_run_id]
    assert mlflow_mock.start_run_calls == [{"run_id": lora_run_id}]


def test_init_mlflow_reuses_lora_run_id_when_no_user_or_active_run(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    lora_run_id = "abc123def45678901234567890123456"
    existing_run = _make_run(lora_run_id, "training")
    existing_run.info.experiment_id = "exp-1"
    mlflow_mock.get_run_result = existing_run
    mlflow_mock.active_run_result = None
    mlflow_mock.start_run_result = existing_run

    config_with_lora = EvaluationConfig(
        model_path_or_repo_id=evaluation_config.model_path_or_repo_id,
        results_dir=evaluation_config.results_dir,
        batch_size=evaluation_config.batch_size,
        vllm_config=VllmConfig(),
        judge_engine="vllm",
        lora_path_or_repo_id=f"mlflow://{lora_run_id}",
        mlflow_config=evaluation_config.mlflow_config,
    )
    DummyEvaluator(config_with_lora, dataset_config)

    assert mlflow_mock.get_run_calls == [lora_run_id]
    assert mlflow_mock.start_run_calls == [{"run_id": lora_run_id}]


def test_init_without_mlflow_config_does_not_touch_mlflow(
    evaluation_config_no_mlflow: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    evaluator = DummyEvaluator(evaluation_config_no_mlflow, dataset_config)

    assert mlflow_mock.tracking_uris == []
    assert mlflow_mock.experiments == []
    assert mlflow_mock.start_run_calls == []
    assert evaluator.mlflow_config is None


def test_dataset_mlflow_run_requires_parent_run(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    parent_run = _make_run("parent-run", "model")
    mlflow_mock.start_run_result = parent_run
    mlflow_mock.active_run_result = None

    evaluator = DummyEvaluator(evaluation_config, dataset_config)
    evaluator.parent_run = None

    with pytest.raises(
        RuntimeError,
        match="Main MLFlow run not found, cannot launch dataset run before initializing MLFlow",
    ):
        with evaluator.dataset_mlflow_run():
            pass


def test_dataset_mlflow_run_logs_dataset_metrics_to_current_run(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    """dataset_mlflow_run logs dataset-related metrics (e.g. seed) and does not start a nested run."""
    parent_run = _make_run("parent-run", "model")
    mlflow_mock.start_run_result = parent_run
    mlflow_mock.active_run_result = None

    evaluator = DummyEvaluator(evaluation_config, dataset_config)
    mlflow_mock.reset()

    with evaluator.dataset_mlflow_run():
        pass

    assert mlflow_mock.start_run_calls == []
    assert mlflow_mock.log_metric_calls[-1] == ("datasets_attached", 1.0)
    assert mlflow_mock.log_metrics_calls[-1][0] == {
        "bbq_gender_bias_free_text_num_samples_evaluated": 3.0
    }


def test_dataset_mlflow_run_increments_datasets_attached_counter(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    parent_run = _make_run("parent-run", "model")
    mlflow_mock.start_run_result = parent_run
    mlflow_mock.active_run_result = None

    evaluator = DummyEvaluator(evaluation_config, dataset_config)
    mlflow_mock.reset()

    with evaluator.dataset_mlflow_run():
        pass
    with evaluator.dataset_mlflow_run():
        pass

    assert mlflow_mock.log_metric_calls == [
        ("datasets_attached", 1.0),
        ("datasets_attached", 2.0),
    ]


def test_dataset_mlflow_run_with_existing_run_id_logs_metrics(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    """When mlflow_run_id is set, dataset_mlflow_run still logs dataset metrics to the current run."""
    parent_run = _make_run("existing-123", "model")
    mlflow_mock.start_run_result = parent_run
    mlflow_mock.active_run_result = parent_run

    assert evaluation_config.mlflow_config is not None
    base_mlflow = evaluation_config.mlflow_config
    config_with_run_id = EvaluationConfig(
        model_path_or_repo_id=evaluation_config.model_path_or_repo_id,
        results_dir=evaluation_config.results_dir,
        batch_size=evaluation_config.batch_size,
        mlflow_config=MlflowConfig(
            mlflow_tracking_uri=base_mlflow.mlflow_tracking_uri,
            mlflow_experiment_name=base_mlflow.mlflow_experiment_name,
            mlflow_run_id="existing-123",
        ),
    )
    evaluator = DummyEvaluator(config_with_run_id, dataset_config)
    mlflow_mock.reset()

    with evaluator.dataset_mlflow_run():
        pass

    assert mlflow_mock.log_metric_calls[-1] == ("datasets_attached", 1.0)


def test_save_results_logs_mlflow_metrics_and_artifacts(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    evaluator = DummyEvaluator(evaluation_config, dataset_config)
    mlflow_mock.reset()

    responses = [{"prompt": "a", "response": "b"}]
    evaluator.save_results(
        responses=responses,
        accuracy=0.75,
        stereotyped_bias=0.1,
        empty_responses=2,
    )

    metrics, step = mlflow_mock.log_metrics_calls[-1]
    assert step is None
    assert metrics == {
        "bbq_gender_bias_free_text_accuracy": 0.75,
        "bbq_gender_bias_free_text_error": 0.25,
        "bbq_gender_bias_free_text_empty_responses": 2.0,
        "bbq_gender_bias_free_text_num_samples": 3.0,
        "bbq_gender_bias_free_text_stereotyped_bias": 0.1,
    }

    assert len(mlflow_mock.log_artifacts_calls) == 1
    artifact_dir, artifact_path = mlflow_mock.log_artifacts_calls[0]
    uploaded_dir = Path(artifact_dir)
    assert str(evaluation_config.results_dir / "model") in str(uploaded_dir)
    assert artifact_path in {None, ""} or artifact_path.startswith("llm-behavior-eval")

    output_dir = evaluation_config.results_dir / "model" / "bbq-gender-bias-free-text"
    assert (output_dir / "responses.json").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (uploaded_dir / "bbq-gender-bias-free-text" / "responses.json").exists()
    assert (uploaded_dir / "bbq-gender-bias-free-text" / "metrics.csv").exists()

    summary_metric_calls = dict(mlflow_mock.log_metric_calls)
    assert summary_metric_calls["Error"] == 25.0
    assert summary_metric_calls["Stereotype_Bias"] == 10.0
    assert summary_metric_calls["Empty_Responses"] == 2.0


def test_save_results_logs_per_label_prompt_injection_asr_as_ratios_to_mlflow(
    evaluation_config: EvaluationConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    injection_config = DatasetConfig(
        file_path="hirundo-io/bloom-prompt-injection-all-free-text",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = DummyEvaluator(evaluation_config, injection_config)
    mlflow_mock.reset()

    evaluator.save_results(
        responses=[{"prompt": "a", "response": "b"}],
        accuracy=0.6,
        stereotyped_bias=None,
        empty_responses=0,
        malicious_attack_success_rate=0.5,
        conflicting_signals_attack_success_rate=1 / 3,
        derive_attack_success_rate=False,
    )

    metrics = mlflow_mock.log_metrics_calls[-1][0]
    assert (
        metrics["bloom_prompt_injection_all_free_text_malicious_attack_success_rate"]
        == 0.5
    )
    assert metrics[
        "bloom_prompt_injection_all_free_text_conflicting_signals_attack_success_rate"
    ] == pytest.approx(1 / 3)


def test_save_results_logs_mlflow_metrics_without_inferred_checkpoint_step(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    config_with_lora_checkpoint = EvaluationConfig(
        model_path_or_repo_id=evaluation_config.model_path_or_repo_id,
        results_dir=evaluation_config.results_dir,
        batch_size=evaluation_config.batch_size,
        vllm_config=VllmConfig(),
        judge_engine="vllm",
        lora_path_or_repo_id="mlflow://abc123/hf_checkpoints/checkpoint-000020",
        mlflow_config=evaluation_config.mlflow_config,
    )
    mlflow_mock.get_run_result = _make_run("abc123", "existing")
    evaluator = DummyEvaluator(config_with_lora_checkpoint, dataset_config)
    mlflow_mock.reset()

    evaluator.save_results(
        responses=[{"prompt": "a", "response": "b"}],
        accuracy=0.5,
        stereotyped_bias=None,
        empty_responses=0,
    )

    assert mlflow_mock.log_metrics_calls[-1][1] is None


def test_save_results_logs_mlflow_metrics_with_inferred_checkpoint_step_same_run(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    lora_run_id = "abc123def45678901234567890123456"
    base_mlflow = evaluation_config.mlflow_config
    assert base_mlflow is not None
    config_with_lora_checkpoint = EvaluationConfig(
        model_path_or_repo_id=evaluation_config.model_path_or_repo_id,
        results_dir=evaluation_config.results_dir,
        batch_size=evaluation_config.batch_size,
        vllm_config=VllmConfig(),
        judge_engine="vllm",
        lora_path_or_repo_id=f"mlflow://{lora_run_id}/hf_checkpoints/checkpoint-000020",
        mlflow_config=MlflowConfig(
            mlflow_tracking_uri=base_mlflow.mlflow_tracking_uri,
            mlflow_experiment_name=base_mlflow.mlflow_experiment_name,
            mlflow_run_id=lora_run_id,
        ),
    )
    existing_run = _make_run(lora_run_id, "existing")
    mlflow_mock.get_run_result = existing_run
    mlflow_mock.active_run_result = existing_run
    evaluator = DummyEvaluator(config_with_lora_checkpoint, dataset_config)
    mlflow_mock.reset()

    evaluator.save_results(
        responses=[{"prompt": "a", "response": "b"}],
        accuracy=0.5,
        stereotyped_bias=None,
        empty_responses=0,
    )

    assert mlflow_mock.log_metrics_calls[-1][1] == 20


def test_log_mlflow_metrics_uses_distinct_dataset_prefixes(
    evaluation_config: EvaluationConfig,
    mlflow_mock: MlflowRecorder,
) -> None:
    parent_run = _make_run("parent-run", "model")
    mlflow_mock.start_run_result = parent_run
    mlflow_mock.active_run_result = None

    xstest_config = DatasetConfig(
        file_path="hirundo-io/XSTest",
        dataset_type=DatasetType.BIAS,
    )
    orbench_config = DatasetConfig(
        file_path="hirundo-io/or-bench",
        dataset_type=DatasetType.BIAS,
    )

    evaluator = DummyEvaluator(evaluation_config, xstest_config)
    mlflow_mock.reset()

    evaluator._log_mlflow_metrics({"safe_refusal_rate": 0.3})
    evaluator.update_dataset_config(orbench_config)
    evaluator._log_mlflow_metrics({"safe_refusal_rate": 0.7})

    assert mlflow_mock.log_metrics_calls == [
        ({"XSTest_safe_refusal_rate": 0.3}, None),
        ({"or_bench_safe_refusal_rate": 0.7}, None),
    ]

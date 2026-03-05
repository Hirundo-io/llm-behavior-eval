from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

from llm_behavior_eval.evaluation_utils.base_evaluator import BaseEvaluator
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import (
    EvaluationConfig,
    MlflowConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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
        self.eval_dataset = []
        self.eval_loader = []
        self.num_samples = 3
        self.has_stereotype = False

    def evaluate(self) -> None:
        return None

    def generate(self) -> Sequence[_GenerationRecord]:
        return []

    def _grade_impl(
        self, generations: Sequence[Any], judge_engine: EvalEngine | None = None
    ) -> None:
        return None

    def get_grading_context(self) -> AbstractContextManager[EvalEngine]:
        # This test doesn't exercise grading, but `evaluate.main()` expects an
        # `EvalEngine` from the context manager. Yield a lightweight stub.
        return nullcontext(cast("EvalEngine", MagicMock()))


def _make_run(run_id: str, run_name: str) -> MagicMock:
    run = MagicMock()
    run.info = SimpleNamespace(run_id=run_id, run_name=run_name)
    return run


@pytest.fixture(autouse=True)
def _mock_model_loading(monkeypatch):
    dummy_tokenizer = DummyTokenizer()
    dummy_model = DummyModel()

    def _stub_loader(*_args, **_kwargs):
        return dummy_tokenizer, dummy_model

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.transformers_eval_engine.load_transformers_model_and_tokenizer",
        _stub_loader,
    )


@pytest.fixture
def mlflow_mock(monkeypatch):
    mock = MagicMock()
    mock.start_run.return_value = MagicMock()
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
    mlflow_mock: MagicMock,
) -> None:
    parent_run = _make_run("parent-run", "model")
    mlflow_mock.start_run.return_value = parent_run
    mlflow_mock.active_run.return_value = None
    DummyEvaluator(evaluation_config, dataset_config)

    mlflow_mock.set_tracking_uri.assert_called_once_with("http://tracking.example")
    mlflow_mock.set_experiment.assert_called_once_with("MLflow Tests")
    mlflow_mock.start_run.assert_called_once_with(run_name="model")

    assert mlflow_mock.log_params.call_count == 1
    logged_params = mlflow_mock.log_params.call_args.args[0]
    assert logged_params["model_path_or_repo_id"] == "meta/model"
    assert "file_path" not in logged_params
    logged_param = mlflow_mock.log_param.call_args.args[0]
    logged_param_value = mlflow_mock.log_param.call_args.args[1]
    assert logged_param == "num_samples_evaluated"
    assert logged_param_value == 3


def test_init_with_default_mlflow_config_still_logs(
    evaluation_config_default_mlflow: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MagicMock,
) -> None:
    parent_run = _make_run("parent-run", "model")
    mlflow_mock.start_run.return_value = parent_run
    mlflow_mock.active_run.return_value = None
    DummyEvaluator(evaluation_config_default_mlflow, dataset_config)

    mlflow_mock.set_tracking_uri.assert_not_called()
    mlflow_mock.set_experiment.assert_not_called()
    mlflow_mock.start_run.assert_called_once_with(run_name="model")
    mlflow_mock.log_params.assert_called_once()


def test_init_mlflow_uses_existing_active_run(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MagicMock,
) -> None:
    active_run = _make_run("active-run", "existing")
    mlflow_mock.active_run.return_value = active_run

    evaluator = DummyEvaluator(evaluation_config, dataset_config)

    mlflow_mock.start_run.assert_not_called()
    assert evaluator.parent_run is active_run
    assert evaluator.mlflow_run is active_run


def test_init_without_mlflow_config_does_not_touch_mlflow(
    evaluation_config_no_mlflow: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MagicMock,
) -> None:
    evaluator = DummyEvaluator(evaluation_config_no_mlflow, dataset_config)

    mlflow_mock.set_tracking_uri.assert_not_called()
    mlflow_mock.set_experiment.assert_not_called()
    mlflow_mock.start_run.assert_not_called()
    mlflow_mock.log_params.assert_not_called()
    assert evaluator.mlflow_config is None


def test_dataset_mlflow_run_starts_nested_run_and_logs_dataset_params(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MagicMock,
) -> None:
    parent_run = _make_run("parent-run", "model")
    child_run = _make_run("child-run", "bbq-age-bias-free-text")
    mlflow_mock.start_run.side_effect = [parent_run, child_run]
    mlflow_mock.active_run.side_effect = [None, child_run, child_run]

    evaluator = DummyEvaluator(evaluation_config, dataset_config)
    evaluator.eval_engine.is_judge = True

    new_dataset_config = DatasetConfig(
        file_path="hirundo-io/bbq-age-bias-free-text",
        dataset_type=DatasetType.BIAS,
    )
    evaluator.update_dataset_config(new_dataset_config)

    with evaluator.dataset_mlflow_run(run_name="bbq-age-bias-free-text"):
        pass

    assert mlflow_mock.start_run.call_args_list[1].kwargs == {
        "run_name": "bbq-age-bias-free-text",
        "nested": True,
    }
    dataset_params = mlflow_mock.log_params.call_args_list[1].args[0]
    assert dataset_params["file_path"] == "hirundo-io/bbq-age-bias-free-text"
    assert dataset_params["dataset_type"] == DatasetType.BIAS
    assert dataset_params["seed"] == 42
    assert "model_path_or_repo_id" not in dataset_params
    mlflow_mock.end_run.assert_called_once()


def test_dataset_mlflow_run_requires_parent_run(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MagicMock,
) -> None:
    parent_run = _make_run("parent-run", "model")
    mlflow_mock.start_run.return_value = parent_run
    mlflow_mock.active_run.return_value = None

    evaluator = DummyEvaluator(evaluation_config, dataset_config)
    evaluator.parent_run = None

    with pytest.raises(
        RuntimeError,
        match="Main MLFlow run not found, cannot launch dataset run before initializing MLFlow",
    ):
        with evaluator.dataset_mlflow_run():
            pass


def test_save_results_logs_mlflow_metrics_and_artifacts(
    evaluation_config: EvaluationConfig,
    dataset_config: DatasetConfig,
    mlflow_mock: MagicMock,
) -> None:
    evaluator = DummyEvaluator(evaluation_config, dataset_config)
    mlflow_mock.reset_mock()

    responses = [{"prompt": "a", "response": "b"}]
    evaluator.save_results(
        responses=responses,
        accuracy=0.75,
        stereotyped_bias=0.1,
        empty_responses=2,
    )

    metrics_call = mlflow_mock.log_metrics.call_args
    assert metrics_call is not None
    metrics = metrics_call.args[0]
    assert metrics == {
        "accuracy": 0.75,
        "error": 0.25,
        "empty_responses": 2.0,
        "num_samples": 3.0,
        "stereotyped_bias": 0.1,
    }

    artifact_calls = {
        Path(call.args[0]).name for call in mlflow_mock.log_artifact.call_args_list
    }
    assert {"responses.json", "metrics.csv"}.issubset(artifact_calls)

    output_dir = evaluation_config.results_dir / "model" / "bbq-gender-bias-free-text"
    assert (output_dir / "responses.json").exists()
    assert (output_dir / "metrics.csv").exists()

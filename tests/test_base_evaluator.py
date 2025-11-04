from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

import llm_behavior_eval.evaluation_utils.base_evaluator as base_evaluator_module
from llm_behavior_eval.evaluation_utils.base_evaluator import BaseEvaluator
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sized
    from pathlib import Path


@dataclass
class CaptureState:
    data_collator: Callable[..., object] | None = None
    engine_dataset: Sized | None = None
    shuffle_seed: int | None = None
    select_indices: list[int] | None = None
    dataloader_args: tuple[Sized, int, bool, Callable[..., object] | None] | None = None
    tokenizer: object | None = None
    trust_remote_code: bool | None = None
    reasoning: bool | None = None
    init_args: tuple[str, DatasetType] | None = None


class StubTokenizer:
    def __init__(self) -> None:
        self.pad_token: str | None = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.padding_side = "right"


@pytest.fixture
def capture_state() -> CaptureState:
    return CaptureState()


@pytest.fixture
def stub_tokenizer() -> StubTokenizer:
    return StubTokenizer()


@pytest.fixture(autouse=True)
def patch_eval_engine(
    monkeypatch: pytest.MonkeyPatch,
    stub_tokenizer: StubTokenizer,
    capture_state: CaptureState,
) -> None:
    class StubEvalEngine:
        def __init__(
            self,
            data_collator: Callable[[Any], Any],
            eval_config: EvaluationConfig,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            self.tokenizer = stub_tokenizer
            self._explicit_batch_size = eval_config.batch_size
            self.dataset: Sized | None = None
            capture_state.data_collator = data_collator

        def get_batch_size(self) -> int:
            if self._explicit_batch_size is not None:
                return self._explicit_batch_size
            if self.dataset is None:
                raise RuntimeError("Dataset must be set before computing batch size")
            return len(self.dataset)

        def ensure_test_model_ready(self) -> None:
            return None

        def free_model(self) -> None:
            return None

        def set_dataset(self, dataset: Sized) -> None:
            capture_state.engine_dataset = dataset
            self.dataset = dataset

    monkeypatch.setattr(base_evaluator_module, "TransformersEvalEngine", StubEvalEngine)


@pytest.fixture(autouse=True)
def patch_custom_dataset(
    monkeypatch: pytest.MonkeyPatch,
    capture_state: CaptureState,
) -> None:
    class StubDataset:
        def __init__(self) -> None:
            self.has_stereotype = False

        def shuffle(self, *, seed: int) -> StubDataset:
            capture_state.shuffle_seed = seed
            return self

        def select(self, indices: range) -> StubDataset:
            capture_state.select_indices = list(indices)
            return self

        def __len__(self) -> int:
            return 3

    class StubCustomDataset:
        def __init__(self, file_path: str, dataset_type: DatasetType) -> None:
            capture_state.init_args = (file_path, dataset_type)
            self.has_stereotype = False

        def preprocess(
            self,
            tokenizer: StubTokenizer,
            _preprocess_config: object,
            *,
            trust_remote_code: bool,
            reasoning: bool,
        ) -> StubDataset:
            capture_state.tokenizer = tokenizer
            capture_state.trust_remote_code = trust_remote_code
            capture_state.reasoning = reasoning
            return StubDataset()

    monkeypatch.setattr(base_evaluator_module, "CustomDataset", StubCustomDataset)


@pytest.fixture(autouse=True)
def patch_dataloader(
    monkeypatch: pytest.MonkeyPatch,
    capture_state: CaptureState,
) -> None:
    def fake_dataloader(
        dataset: Sized,
        batch_size: int,
        shuffle: bool,
        collate_fn: Callable[..., object] | None,
    ) -> str:
        capture_state.dataloader_args = (dataset, batch_size, shuffle, collate_fn)
        return "loader"

    monkeypatch.setattr(base_evaluator_module, "DataLoader", fake_dataloader)


class ConcreteEvaluator(BaseEvaluator):
    def evaluate(self) -> None:
        return None


def test_prepare_dataloader_receives_eval_engine_tokenizer(
    tmp_path: Path,
    capture_state: CaptureState,
    stub_tokenizer: StubTokenizer,
) -> None:
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        batch_size=None,
        max_samples=10,
    )
    dataset_config_instance = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )

    evaluator = ConcreteEvaluator(evaluation_config, dataset_config_instance)

    assert capture_state.tokenizer is stub_tokenizer
    assert evaluator.tokenizer is stub_tokenizer
    assert capture_state.trust_remote_code == evaluation_config.trust_remote_code
    assert capture_state.dataloader_args is not None
    _, batch_size, _, _ = capture_state.dataloader_args
    assert batch_size == 3
    assert evaluator.eval_loader == "loader"
    assert evaluator.num_samples == 3
    assert capture_state.engine_dataset == evaluator.eval_dataset

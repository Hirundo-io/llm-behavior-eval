from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

import llm_behavior_eval.evaluation_utils.base_evaluator as base_evaluator_module
from llm_behavior_eval.evaluation_utils.base_evaluator import (
    BaseEvaluator,
    FreeTextSharedEvaluator,
)
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sized
    from pathlib import Path

    from datasets import Dataset
    from transformers.tokenization_utils_base import TruncationStrategy
    from transformers.utils.generic import PaddingStrategy, TensorType


TextInput = str
PreTokenizedInput = list[str]
EncodedInput = list[int]
TextInputPair = tuple[str, str]
PreTokenizedInputPair = tuple[list[str], list[str]]
EncodedInputPair = tuple[list[int], list[int]]


@dataclass
class CaptureState:
    data_collator: Callable[..., object] | None = None
    engine_dataset: Sized | None = None
    shuffle_seed: int | None = None
    select_indices: list[int] | None = None
    dataloader_args: tuple[Sized, int, bool, Callable[..., object] | None] | None = None
    tokenizer: object | None = None
    trust_remote_code: bool | None = None
    max_answer_tokens: int | None = None
    reasoning: bool | None = None
    pass_max_answer_tokens: bool | None = None
    token: str | None = None
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
            max_answer_tokens: int | None,
            reasoning: bool,
            pass_max_answer_tokens: bool,
            token: str | None = None,
        ) -> StubDataset:
            capture_state.tokenizer = tokenizer
            capture_state.trust_remote_code = trust_remote_code
            capture_state.max_answer_tokens = max_answer_tokens
            capture_state.reasoning = reasoning
            capture_state.pass_max_answer_tokens = pass_max_answer_tokens
            capture_state.token = token
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


def test_process_judge_prompts_batch_uses_sampling_config(tmp_path: Path) -> None:
    class StubJudgeTokenizer(PreTrainedTokenizerBase):
        def __call__(
            self,
            text: TextInput
            | PreTokenizedInput
            | list[TextInput]
            | list[PreTokenizedInput]
            | None = None,
            text_pair: TextInput
            | PreTokenizedInput
            | list[TextInput]
            | list[PreTokenizedInput]
            | None = None,
            text_target: TextInput
            | PreTokenizedInput
            | list[TextInput]
            | list[PreTokenizedInput]
            | None = None,
            text_pair_target: TextInput
            | PreTokenizedInput
            | list[TextInput]
            | list[PreTokenizedInput]
            | None = None,
            add_special_tokens: bool = True,
            padding: bool | str | PaddingStrategy = False,
            truncation: bool | str | TruncationStrategy | None = None,
            max_length: int | None = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: int | None = None,
            padding_side: str | None = None,
            return_tensors: str | TensorType | None = None,
            return_token_type_ids: bool | None = None,
            return_attention_mask: bool | None = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
        ) -> BatchEncoding:
            del text, return_tensors, padding
            input_ids = torch.tensor([[10], [11]])
            attention_mask = torch.ones_like(input_ids)
            return BatchEncoding(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )

    class RecordingJudgeEngine(EvalEngine):
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_answers(
            self,
            input_ids,
            attention_mask,
            sampling_config: SamplingConfig,
        ):
            self.calls.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "sampling_config": sampling_config,
                }
            )
            return ["yes"] * input_ids.shape[0]

        def free_model(self) -> None:
            return None

        def get_batch_size(self) -> int:
            return 1

        def set_dataset(self, eval_dataset: Dataset) -> None:
            return None

    class StubFreeTextEvaluator(FreeTextSharedEvaluator):
        def evaluate(self) -> None:
            return None

    evaluator = StubFreeTextEvaluator.__new__(StubFreeTextEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        sample_judge=True,
        sampling_config=SamplingConfig(
            temperature=0.5,
            top_p=0.9,
            top_k=4,
            seed=111,
        ),
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
        seed=777,
    )
    evaluator.judge_tokenizer = StubJudgeTokenizer()

    judge_engine = RecordingJudgeEngine()
    outputs = evaluator._process_judge_prompts_batch(
        judge_engine,
        ["prompt-a", "prompt-b"],
        do_sample=None,
    )

    assert outputs == [[{"generated_text": "yes"}], [{"generated_text": "yes"}]]
    assert len(judge_engine.calls) == 1
    sampling_config = judge_engine.calls[0]["sampling_config"]
    assert isinstance(sampling_config, SamplingConfig)
    assert sampling_config.do_sample is True
    assert sampling_config.temperature == 0.5
    assert sampling_config.top_p == 0.9
    assert sampling_config.top_k == 4
    assert sampling_config.seed == evaluator.dataset_config.seed

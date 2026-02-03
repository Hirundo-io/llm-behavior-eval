from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

import llm_behavior_eval.evaluation_utils.api_eval_engine as api_eval_engine_module
import llm_behavior_eval.evaluation_utils.base_evaluator as base_evaluator_module
from llm_behavior_eval.evaluation_utils.base_evaluator import (
    BaseEvaluator,
    FreeTextSharedEvaluator,
    _GenerationRecord,
)
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence, Sized
    from pathlib import Path

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
    engine_inits: list[bool] = field(default_factory=list)
    set_dataset_calls: list[tuple[bool, Sized]] = field(default_factory=list)
    free_model_calls: list[bool] = field(default_factory=list)
    grade_called_with_judge: bool | None = None
    grade_generations_count: int | None = None


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
            *,
            is_judge: bool = False,
            **_kwargs: object,
        ) -> None:
            self.tokenizer = stub_tokenizer
            self._explicit_batch_size = eval_config.batch_size
            self.dataset: Sized | None = None
            self.is_judge = is_judge
            capture_state.data_collator = data_collator
            capture_state.engine_inits.append(is_judge)

        def get_batch_size(self) -> int:
            if self._explicit_batch_size is not None:
                return self._explicit_batch_size
            if self.dataset is None:
                raise RuntimeError("Dataset must be set before computing batch size")
            return len(self.dataset)

        def ensure_test_model_ready(self) -> None:
            return None

        def free_model(self) -> None:
            capture_state.free_model_calls.append(self.is_judge)
            return None

        def set_dataset(self, dataset: Sized) -> None:
            capture_state.engine_dataset = dataset
            capture_state.set_dataset_calls.append((self.is_judge, dataset))
            self.dataset = dataset

    monkeypatch.setattr(base_evaluator_module, "TransformersEvalEngine", StubEvalEngine)

    class StubApiEvalEngine(StubEvalEngine):
        def __init__(
            self, eval_config: EvaluationConfig, *, is_judge: bool = False
        ) -> None:
            self.tokenizer = None
            self._explicit_batch_size = eval_config.batch_size
            self.dataset: Sized | None = None
            self.is_judge = is_judge
            capture_state.data_collator = None
            capture_state.engine_inits.append(is_judge)

    monkeypatch.setattr(api_eval_engine_module, "ApiEvalEngine", StubApiEvalEngine)


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
            tokenizer: StubTokenizer | None,
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

    def generate(self) -> Sequence[_GenerationRecord]:
        return []

    def grade(self, generations: object, judge_engine: object = None) -> None:
        del generations, judge_engine
        return None

    def get_grading_context(self) -> AbstractContextManager:
        # This test file doesn't exercise grading; we just need a valid context manager.
        return nullcontext()


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


def test_prepare_dataloader_api_raw_mode_uses_raw_collator(
    tmp_path: Path,
    capture_state: CaptureState,
) -> None:
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        model_tokenizer_path_or_repo_id=None,
        results_dir=tmp_path,
        batch_size=None,
        max_samples=10,
    )
    dataset_config_instance = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )

    evaluator = ConcreteEvaluator(evaluation_config, dataset_config_instance)

    assert capture_state.tokenizer is None
    assert evaluator.tokenizer is None
    assert evaluator.api_raw_mode is True
    assert capture_state.dataloader_args is not None
    _, batch_size, _, collate_fn = capture_state.dataloader_args
    assert batch_size == 3
    assert collate_fn is base_evaluator_module.raw_text_collator


def test_api_model_rebuilds_judge_dataset_with_judge_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_state: CaptureState,
    stub_tokenizer: StubTokenizer,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: stub_tokenizer,
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    class StubEvaluator(FreeTextSharedEvaluator):
        def evaluate(self) -> None:
            return None

        def generate(self) -> Sequence[_GenerationRecord]:
            return []

        def grade(
            self,
            generations: Sequence[_GenerationRecord],
            judge_engine: EvalEngine | None = None,
        ) -> None:
            del generations, judge_engine
            return None

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        model_engine="api",
        model_tokenizer_path_or_repo_id=None,
        model_token="model-token",
        judge_engine="transformers",
        judge_token="judge-token",
        results_dir=tmp_path,
        max_samples=1,
    )
    dataset_config_instance = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )

    evaluator = StubEvaluator(evaluation_config, dataset_config_instance)

    assert evaluator._judge_dataset_needs_rebuild is True

    with evaluator.get_grading_context() as _judge_engine:
        assert capture_state.set_dataset_calls
        is_judge, dataset = capture_state.set_dataset_calls[-1]
        assert is_judge is True
        assert dataset is not evaluator.eval_dataset

    assert capture_state.tokenizer is stub_tokenizer
    assert capture_state.token == "judge-token"


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

        def generate_answers_from_tensors(
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

        def generate_answers_from_prompts(
            self,
            prompts,
            sampling_config: SamplingConfig,
        ):
            self.calls.append(
                {
                    "prompts": prompts,
                    "sampling_config": sampling_config,
                }
            )
            return ["yes"] * len(prompts)

        def free_model(self) -> None:
            return None

        def get_batch_size(self) -> int:
            return 1

        def set_dataset(self, eval_dataset: Sized) -> None:
            return None

    class StubFreeTextEvaluator(FreeTextSharedEvaluator):
        def evaluate(self) -> None:
            return None

        def generate(self) -> Sequence[_GenerationRecord]:
            return []

        def grade(self, generations: object, judge_engine: object = None) -> None:
            del generations, judge_engine
            return None

        def get_grading_context(self) -> AbstractContextManager:
            return nullcontext()

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
    evaluator.judge_engine = "transformers"

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


def test_get_grading_context_creates_and_frees_judge_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_state: CaptureState,
) -> None:
    # Avoid loading any real tokenizer/model.
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    class StubEvaluator(FreeTextSharedEvaluator):
        def evaluate(self) -> None:
            return None

        def generate(self) -> Sequence[_GenerationRecord]:
            return []

        def grade(
            self,
            generations: Sequence[_GenerationRecord],
            judge_engine: EvalEngine | None = None,
        ) -> None:
            del generations, judge_engine
            return None

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    dataset_config_instance = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = StubEvaluator(evaluation_config, dataset_config_instance)

    # Entering the grading context should build a judge engine (is_judge=True) and set its dataset.
    with evaluator.get_grading_context() as judge_engine:
        assert getattr(judge_engine, "is_judge", False) is True
        assert capture_state.set_dataset_calls
        is_judge, dataset = capture_state.set_dataset_calls[-1]
        assert is_judge is True
        assert dataset is evaluator.eval_dataset

    # Exiting the context should free the judge engine.
    assert True in capture_state.free_model_calls


def test_get_grading_context_supports_api_judge_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubApiJudgeEngine:
        def __init__(self, _eval_config, *, is_judge: bool = False) -> None:
            self.is_judge = is_judge
            self.dataset = None

        def set_dataset(self, dataset: object) -> None:
            self.dataset = dataset

        def get_batch_size(self) -> int:
            return 1

        def free_model(self) -> None:
            return None

    # Need to use a FreeTextSharedEvaluator subclass to test get_judge_engine_context
    class StubApiEvaluator(FreeTextSharedEvaluator):
        def evaluate(self) -> None:
            return None

        def generate(self) -> Sequence[_GenerationRecord]:
            return []

        def grade(
            self,
            generations: Sequence[_GenerationRecord],
            judge_engine: EvalEngine | None = None,
        ) -> None:
            del generations, judge_engine
            return None

    monkeypatch.setattr(
        base_evaluator_module,
        "empty_cuda_cache_if_available",
        lambda: None,
    )
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Tokenizer should not be loaded for API judges.")
        ),
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.api_eval_engine.ApiEvalEngine",
        StubApiJudgeEngine,
    )

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
        judge_engine="api",
        judge_path_or_repo_id="openai/gpt-4o-mini",
    )
    dataset_config_instance = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )

    evaluator = StubApiEvaluator(evaluation_config, dataset_config_instance)

    with evaluator.get_grading_context() as judge_engine:
        assert isinstance(judge_engine, StubApiJudgeEngine)
        assert judge_engine.is_judge is True
        assert judge_engine.dataset is evaluator.eval_dataset


def test_api_model_engine_uses_api_eval_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubApiModelEngine:
        # API engines do not load a local tokenizer
        tokenizer = None

        def __init__(self, _eval_config, *, is_judge: bool = False) -> None:
            self.is_judge = is_judge
            self.dataset = None

        def set_dataset(self, dataset: object) -> None:
            self.dataset = dataset

        def get_batch_size(self) -> int:
            return 1

        def free_model(self) -> None:
            return None

        def ensure_test_model_ready(self) -> None:
            return None

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.api_eval_engine.ApiEvalEngine",
        StubApiModelEngine,
    )

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o-mini",
        results_dir=tmp_path,
        max_samples=1,
        model_engine="api",
    )
    dataset_config_instance = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )

    evaluator = ConcreteEvaluator(evaluation_config, dataset_config_instance)

    assert isinstance(evaluator.eval_engine, StubApiModelEngine)
    assert evaluator.eval_engine.is_judge is False
    assert evaluator.tokenizer is None


def test_evaluate_flow_can_use_generate_then_grade_in_grading_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_state: CaptureState,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    class FlowEvaluator(FreeTextSharedEvaluator):
        def evaluate(self) -> None:
            generations = self.generate()
            with self.get_grading_context() as judge_engine:
                self.grade(generations, judge_engine=judge_engine)

        def generate(self) -> Sequence[_GenerationRecord]:
            return [_GenerationRecord(answers=["a"])]

        def grade(
            self,
            generations: Sequence[_GenerationRecord],
            judge_engine: EvalEngine | None = None,
        ) -> None:
            capture_state.grade_generations_count = len(generations)
            capture_state.grade_called_with_judge = (
                judge_engine is not None
                and getattr(judge_engine, "is_judge", False) is True
            )

    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    dataset_config_instance = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )

    evaluator = FlowEvaluator(evaluation_config, dataset_config_instance)
    evaluator.evaluate()

    assert capture_state.grade_generations_count == 1
    assert capture_state.grade_called_with_judge is True

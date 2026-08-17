from __future__ import annotations

import csv
import json
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

import llm_behavior_eval.evaluation_utils.base_evaluator as base_evaluator_module
import llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator as censorship_module
import llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator as hallu_module
from llm_behavior_eval.evaluation_utils.base_evaluator import (
    BaseEvaluator,
    FreeTextSharedEvaluator,
    _GenerationRecord,
)
from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CENSORSHIP_JUDGE_PROMPT,
    CHINESE_CENSORSHIP_DATASET_SOURCE,
    CHINESE_CENSORSHIP_GROUP_COUNT,
    CensorshipJudgment,
    parse_censorship_judgment,
)
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.eval_config import (
    EvaluationConfig,
    MlflowConfig,
)
from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine
from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    GRADING_STATUS_JUDGE_INCOMPLETE,
    GRADING_STATUS_JUDGE_UNKNOWN_FINISH_REASON,
    GRADING_STATUS_MODEL_INCOMPLETE,
    GRADING_STATUS_UNKNOWN_FINISH_REASON,
    LEGAL_CENSORSHIP_STATUS_LABEL_PAIRS,
    CensorshipResponse,
    FreeTextCensorshipEvaluator,
    _CensorshipGenerationRecord,
    validate_censorship_status_label_pair,
)
from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    GRADING_STATUS_JUDGE_UNPARSEABLE as CENSORSHIP_STATUS_UNPARSEABLE,
)
from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    GRADING_STATUS_JUDGED as CENSORSHIP_STATUS_JUDGED,
)
from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
    FreeTextHaluEvaluator,
    _HalluGenerationRecord,
)
from llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator import (
    FreeTextRefusalEvaluator,
    _RefusalGenerationRecord,
)
from llm_behavior_eval.evaluation_utils.refusal_utils import (
    OR_BENCH_DATASET,
    XSTEST_DATASET,
)
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig
from llm_behavior_eval.evaluation_utils.vllm_config import VllmConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence, Sized
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
    enable_thinking: bool | None = None
    enable_thinking_arg_name: str | None = None
    thinking_start_token: str | None = None
    thinking_end_token: str | None = None
    pass_max_answer_tokens: bool | None = None
    token: str | None = None
    padding_side_at_preprocess: str | None = None
    init_args: tuple[str, DatasetType] | None = None
    custom_dataset_id: str | None = None
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
        def __init__(
            self,
            file_path: str,
            dataset_type: DatasetType,
            *,
            trust_remote_code: bool = False,
            token: str | None = None,
            dataset_id: str | None = None,
        ) -> None:
            capture_state.init_args = (file_path, dataset_type)
            capture_state.trust_remote_code = trust_remote_code
            capture_state.token = token
            capture_state.custom_dataset_id = dataset_id
            self.trust_remote_code = trust_remote_code
            self.dataset_id = dataset_id or file_path
            self.has_stereotype = False
            self.evidence_provenance: list[dict[str, str | int]] = []

        def preprocess(
            self,
            tokenizer: StubTokenizer,
            _preprocess_config: object,
            *,
            max_answer_tokens: int | None,
            enable_thinking: bool | None = None,
            enable_thinking_arg_name: str | None = None,
            thinking_start_token: str | None = None,
            thinking_end_token: str | None = None,
            pass_max_answer_tokens: bool,
            model_revision: str | None = None,
        ) -> StubDataset:
            capture_state.tokenizer = tokenizer
            # Capture tokenization-time padding before later tokenizer mutations.
            capture_state.padding_side_at_preprocess = tokenizer.padding_side
            capture_state.trust_remote_code = self.trust_remote_code
            capture_state.max_answer_tokens = max_answer_tokens
            capture_state.enable_thinking = enable_thinking
            capture_state.enable_thinking_arg_name = enable_thinking_arg_name
            capture_state.thinking_start_token = thinking_start_token
            capture_state.thinking_end_token = thinking_end_token
            capture_state.pass_max_answer_tokens = pass_max_answer_tokens
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

    def _grade_impl(self, generations: object, judge_engine: object = None) -> None:
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


def test_prepare_dataloader_propagates_explicit_and_default_dataset_id(
    tmp_path: Path, capture_state: CaptureState
) -> None:
    evaluation_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    evaluator = ConcreteEvaluator(
        evaluation_config,
        DatasetConfig(
            file_path="/opt/assets/halueval",
            dataset_id="hirundo-io/halueval",
            dataset_type=DatasetType.BIAS,
        ),
    )

    assert capture_state.custom_dataset_id == "hirundo-io/halueval"

    evaluator.dataset_config = DatasetConfig(
        file_path="repo/fallback-dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator.prepare_dataloader()

    assert capture_state.custom_dataset_id == "repo/fallback-dataset"


def test_mlflow_initializes_after_dataloader_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_state: CaptureState,
) -> None:
    class _MlflowRunInfo:
        run_id = "run-id"

    class _MlflowRun:
        info = _MlflowRunInfo()

    class _MlflowStub:
        def active_run(self) -> None:
            return None

        def set_tracking_uri(self, _tracking_uri: str) -> None:
            return None

        def set_experiment(self, _experiment_name: str) -> None:
            return None

        def start_run(self, *, run_name: str) -> _MlflowRun:
            del run_name
            assert capture_state.set_dataset_calls
            return _MlflowRun()

        def log_metric(self, _key: str, _value: float) -> None:
            return None

    monkeypatch.setattr(base_evaluator_module, "mlflow", _MlflowStub())

    ConcreteEvaluator(
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            max_samples=1,
            mlflow_config=MlflowConfig(),
        ),
        DatasetConfig(
            file_path="repo/dataset",
            dataset_type=DatasetType.BIAS,
        ),
    )


def test_dataset_is_tokenized_with_left_padding(
    tmp_path: Path,
    capture_state: CaptureState,
    stub_tokenizer: StubTokenizer,
) -> None:
    """Inputs must be left-padded before dataset tokenization."""
    assert stub_tokenizer.padding_side == "right"  # default before init

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

    ConcreteEvaluator(evaluation_config, dataset_config_instance)

    assert capture_state.padding_side_at_preprocess == "left"


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
            return ["yes"] * input_ids.shape[0], [None] * input_ids.shape[0]

        def free_model(self) -> None:
            return None

        def get_batch_size(self) -> int:
            return 1

        def set_dataset(self, eval_dataset: Dataset) -> None:
            return None

    class StubFreeTextEvaluator(FreeTextSharedEvaluator):
        def evaluate(self) -> None:
            return None

        def generate(self) -> Sequence[_GenerationRecord]:
            return []

        def _grade_impl(self, generations: object, judge_engine: object = None) -> None:
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
            repetition_penalty=1.2,
            seed=111,
        ),
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
        seed=0,
    )
    evaluator.judge_tokenizer = StubJudgeTokenizer()

    judge_engine = RecordingJudgeEngine()
    outputs = evaluator._process_judge_prompts_batch(
        judge_engine,
        ["prompt-a", "prompt-b"],
        do_sample=None,
    )

    assert outputs == [
        [{"generated_text": "yes", "finish_reason": None}],
        [{"generated_text": "yes", "finish_reason": None}],
    ]
    assert len(judge_engine.calls) == 1
    sampling_config = judge_engine.calls[0]["sampling_config"]
    assert isinstance(sampling_config, SamplingConfig)
    assert sampling_config.do_sample is True
    assert sampling_config.temperature == 0.5
    assert sampling_config.top_p == 0.9
    assert sampling_config.top_k == 4
    assert sampling_config.repetition_penalty == 1.2
    assert sampling_config.seed == evaluator.dataset_config.seed

    evaluator.eval_engine = judge_engine
    evaluator.generate_answers(torch.tensor([[1]]), torch.tensor([[1]]))

    model_sampling_config = judge_engine.calls[-1]["sampling_config"]
    assert isinstance(model_sampling_config, SamplingConfig)
    assert model_sampling_config.seed == 0
    assert model_sampling_config.repetition_penalty == 1.2


def test_fixed_judge_batch_size_replaces_stale_effective_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = FreeTextHaluEvaluator.__new__(FreeTextHaluEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        judge_batch_size=2,
    )
    evaluator.effective_judge_batch_size = 8
    seen_chunks: list[list[str]] = []

    def fake_process_batch(
        _judge_engine: EvalEngine, prompts: list[str]
    ) -> list[list[dict[str, str | None]]]:
        seen_chunks.append(prompts)
        return [
            [{"generated_text": prompt, "finish_reason": None}] for prompt in prompts
        ]

    monkeypatch.setattr(evaluator, "_process_judge_prompts_batch", fake_process_batch)

    outputs = evaluator.run_judge_with_backoff(
        cast("EvalEngine", object()), ["a", "b", "c"]
    )

    assert evaluator.effective_judge_batch_size == 2
    assert seen_chunks == [["a", "b"], ["c"]]
    assert [item[0]["generated_text"] for item in outputs] == ["a", "b", "c"]


def test_get_model_slug_includes_lora_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_args: dict[str, str | None] = {}

    def fake_get_lora_slug(
        adapter_ref: str, mlflow_tracking_uri: str | None = None
    ) -> str:
        captured_args["adapter_ref"] = adapter_ref
        captured_args["mlflow_tracking_uri"] = mlflow_tracking_uri
        return "adapter_test_slug"

    monkeypatch.setattr(base_evaluator_module, "get_lora_slug", fake_get_lora_slug)

    evaluator = ConcreteEvaluator.__new__(ConcreteEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        lora_path_or_repo_id="org/lora-adapter",
        inference_engine="vllm",
        results_dir=tmp_path,
        mlflow_config=MlflowConfig(mlflow_tracking_uri="http://tracking.example"),
    )

    assert evaluator.get_model_slug() == "model-lora-adapter_test_slug"
    assert captured_args == {
        "adapter_ref": "org/lora-adapter",
        "mlflow_tracking_uri": "http://tracking.example",
    }


def test_get_model_slug_prefers_model_output_dir_override(tmp_path: Path) -> None:
    evaluator = ConcreteEvaluator.__new__(ConcreteEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        model_output_dir="custom-model-output",
        results_dir=tmp_path,
    )

    assert evaluator.get_model_slug() == "custom-model-output"


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

        def _grade_impl(
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
            with self.dataset_mlflow_run(), self.get_grading_context() as judge_engine:
                self.grade(generations, judge_engine=judge_engine)

        def generate(self) -> Sequence[_GenerationRecord]:
            return [_GenerationRecord(answers=["a"])]

        def _grade_impl(
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


def test_format_answers_trims_thinking_trace_and_judge_prompt_uses_trimmed_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = FreeTextHaluEvaluator.__new__(FreeTextHaluEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        thinking_start_token="<think>",
        thinking_end_token="</think>",
        exclude_thinking_trace_for_judge=True,
    )
    evaluator.num_samples = 1

    answer_with_trace = "scratchpad thoughts </think> final answer"
    assert evaluator._format_answers([answer_with_trace]) == ["final answer"]

    monkeypatch.setattr(evaluator, "prepare_judge_tokenizer", lambda: None)
    monkeypatch.setattr(evaluator, "_get_judge_tokenizer", lambda: object())
    monkeypatch.setattr(
        hallu_module,
        "safe_apply_chat_template",
        lambda _tokenizer, messages: messages[-1]["content"],
    )
    monkeypatch.setattr(evaluator, "save_results", lambda **_kwargs: None)

    captured_prompts: list[str] = []

    def fake_run_judge_with_backoff(
        _judge_engine: EvalEngine, prompts: list[str]
    ) -> list[list[dict[str, str]]]:
        captured_prompts.extend(prompts)
        return [[{"generated_text": "A"}] for _ in prompts]

    monkeypatch.setattr(
        evaluator, "run_judge_with_backoff", fake_run_judge_with_backoff
    )

    evaluator._grade_impl(
        [
            _HalluGenerationRecord(
                input_texts=["question"],
                gt_answers=["gold"],
                answers=[answer_with_trace],
                finish_reasons=["stop"],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )

    assert len(captured_prompts) == 1
    assert "Predicted answer: final answer" in captured_prompts[0]
    assert "scratchpad thoughts" not in captured_prompts[0]


def test_format_answers_splits_on_last_thinking_end_token(
    tmp_path: Path,
) -> None:
    evaluator = FreeTextHaluEvaluator.__new__(FreeTextHaluEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        thinking_start_token="<think>",
        thinking_end_token="</think>",
        exclude_thinking_trace_for_judge=True,
    )

    token_in_answer_body = 'The answer includes "</think>" as a literal marker.'
    terminal_token_only = "scratchpad thoughts </think>"

    assert evaluator._format_answers([token_in_answer_body]) == [
        '" as a literal marker.'
    ]
    assert evaluator._format_answers([terminal_token_only]) == [""]


def test_save_results_drops_empty_metric_columns_and_uses_directional_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    evaluator = ConcreteEvaluator(
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            max_samples=1,
        ),
        DatasetConfig(
            file_path="hirundo-io/prompt-injection-purple-llama",
            dataset_type=DatasetType.BIAS,
        ),
    )

    evaluator.save_results(
        responses=[{"prompt": "test", "response": "value"}],
        accuracy=0.75,
        stereotyped_bias=None,
        empty_responses=0,
    )

    metrics_file_path = (
        tmp_path / "model" / "prompt-injection-purple-llama" / "metrics.csv"
    )
    with metrics_file_path.open(newline="", encoding="utf-8") as metrics_file:
        metrics_rows = list(csv.DictReader(metrics_file))

    assert metrics_rows == [
        {
            "Thinking": "off",
            "Attack success rate (%) ⬇️": "25.000",
            "Empty Responses": "0",
            "Incomplete response rate (%) ⬇️": "0.000",
        }
    ]

    summary_full_path = tmp_path / "model" / "summary_full.csv"
    with summary_full_path.open(newline="", encoding="utf-8") as summary_file:
        summary_rows = list(csv.DictReader(summary_file))

    assert summary_rows == [
        {
            "Model": "model",
            "Dataset": "prompt-injection-purple-llama",
            "Dataset Type": "DatasetType.BIAS",
            "Text Format": "free_text",
            "Thinking": "off",
            "Attack success rate (%) ⬇️": "25.000",
            "Empty Responses": "0",
            "Incomplete response rate (%) ⬇️": "0.000",
        }
    ]


def test_save_results_rewrites_summary_with_non_empty_columns_after_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    evaluator = ConcreteEvaluator(
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            max_samples=1,
        ),
        DatasetConfig(
            file_path="hirundo-io/bbq-gender-unbias-free-text",
            dataset_type=DatasetType.UNBIAS,
        ),
    )

    evaluator.save_results(
        responses=[{"prompt": "test", "response": "value"}],
        accuracy=0.80,
        stereotyped_bias=None,
        empty_responses=0,
    )

    evaluator.update_dataset_config(
        DatasetConfig(
            file_path="hirundo-io/bbq-gender-bias-free-text",
            dataset_type=DatasetType.BIAS,
        )
    )
    evaluator.save_results(
        responses=[{"prompt": "test2", "response": "value2"}],
        accuracy=0.60,
        stereotyped_bias=None,
        empty_responses=1,
    )

    summary_brief_path = tmp_path / "model" / "summary_brief.csv"
    with summary_brief_path.open(newline="", encoding="utf-8") as summary_file:
        summary_rows = list(csv.DictReader(summary_file))

    assert len(summary_rows) == 2
    assert summary_rows[0]["Accuracy (%) ⬆️"] == "80.000"
    assert summary_rows[0]["Error (%) ⬇️"] == ""
    assert summary_rows[0]["Thinking"] == "off"
    assert summary_rows[1]["Accuracy (%) ⬆️"] == ""
    assert summary_rows[1]["Error (%) ⬇️"] == "40.000"
    assert summary_rows[1]["Thinking"] == "off"
    assert "Attack success rate (%) ⬇️" not in summary_rows[0]
    assert "Attack success rate (%) ⬇️" not in summary_rows[1]


def test_save_results_uses_bloom_summary_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    evaluator = ConcreteEvaluator(
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            max_samples=1,
        ),
        DatasetConfig(
            file_path="hirundo-io/bloom-age-unbias-free-text",
            dataset_type=DatasetType.UNBIAS,
        ),
    )

    evaluator.save_results(
        responses=[{"prompt": "test", "response": "value"}],
        accuracy=0.80,
        stereotyped_bias=None,
        empty_responses=0,
    )

    summary_brief_path = tmp_path / "model" / "summary_brief.csv"
    with summary_brief_path.open(newline="", encoding="utf-8") as summary_file:
        summary_rows = list(csv.DictReader(summary_file))

    assert summary_rows[0]["Dataset"] == "Bloom: age unbias"
    assert summary_rows[0]["Accuracy (%) ⬆️"] == "80.000"
    assert "Error (%) ⬇️" not in summary_rows[0]


def test_save_results_uses_distinct_bloom_ambiguous_summary_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    evaluator = ConcreteEvaluator(
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            max_samples=1,
        ),
        DatasetConfig(
            file_path="hirundo-io/bloom-gender-ambiguous-bias-free-text",
            dataset_type=DatasetType.BIAS,
        ),
    )

    evaluator.save_results(
        responses=[{"prompt": "test", "response": "value"}],
        accuracy=0.80,
        stereotyped_bias=None,
        empty_responses=0,
    )

    summary_brief_path = tmp_path / "model" / "summary_brief.csv"
    with summary_brief_path.open(newline="", encoding="utf-8") as summary_file:
        summary_rows = list(csv.DictReader(summary_file))

    assert summary_rows[0]["Dataset"] == "Bloom: gender ambiguous bias"
    assert summary_rows[0]["Error (%) ⬇️"] == "20.000"


def test_save_results_marks_thinking_mode_on_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    evaluator = ConcreteEvaluator(
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            max_samples=1,
            enable_thinking=True,
        ),
        DatasetConfig(
            file_path="hirundo-io/bbq-gender-unbias-free-text",
            dataset_type=DatasetType.UNBIAS,
        ),
    )

    evaluator.save_results(
        responses=[{"prompt": "test", "response": "value"}],
        accuracy=0.80,
        stereotyped_bias=None,
        empty_responses=0,
    )

    metrics_file_path = (
        tmp_path / "model" / "bbq-gender-unbias-free-text" / "metrics.csv"
    )
    with metrics_file_path.open(newline="", encoding="utf-8") as metrics_file:
        metrics_rows = list(csv.DictReader(metrics_file))

    assert len(metrics_rows) == 1
    assert metrics_rows[0]["Thinking"] == "on"


def test_save_results_includes_incomplete_response_rate_when_finish_reasons_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    evaluator = ConcreteEvaluator(
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            max_samples=1,
        ),
        DatasetConfig(
            file_path="hirundo-io/prompt-injection-purple-llama",
            dataset_type=DatasetType.BIAS,
        ),
    )

    generations_path = (
        tmp_path / "model" / "prompt-injection-purple-llama" / "generations.jsonl"
    )
    generations_path.parent.mkdir(parents=True, exist_ok=True)
    generations_path.write_text(
        json.dumps(
            {
                "answers": ["a1", "a2", "a3", "a4"],
                "finish_reasons": ["length", "stop", "length", "stop"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluator.save_results(
        responses=[{"prompt": "test", "response": "value"}],
        accuracy=0.75,
        stereotyped_bias=None,
        empty_responses=0,
    )

    metrics_file_path = (
        tmp_path / "model" / "prompt-injection-purple-llama" / "metrics.csv"
    )
    with metrics_file_path.open(newline="", encoding="utf-8") as metrics_file:
        metrics_rows = list(csv.DictReader(metrics_file))

    assert metrics_rows == [
        {
            "Thinking": "off",
            "Attack success rate (%) ⬇️": "25.000",
            "Empty Responses": "0",
            "Incomplete response rate (%) ⬇️": "50.000",
        }
    ]

    summary_full_path = tmp_path / "model" / "summary_full.csv"
    with summary_full_path.open(newline="", encoding="utf-8") as summary_file:
        full_summary_rows = list(csv.DictReader(summary_file))
    assert full_summary_rows == [
        {
            "Model": "model",
            "Dataset": "prompt-injection-purple-llama",
            "Dataset Type": "DatasetType.BIAS",
            "Text Format": "free_text",
            "Thinking": "off",
            "Attack success rate (%) ⬇️": "25.000",
            "Empty Responses": "0",
            "Incomplete response rate (%) ⬇️": "50.000",
        }
    ]

    summary_brief_path = tmp_path / "model" / "summary_brief.csv"
    with summary_brief_path.open(newline="", encoding="utf-8") as summary_file:
        brief_summary_rows = list(csv.DictReader(summary_file))
    assert brief_summary_rows == [
        {
            "Dataset": "prompt-injection-purple-llama",
            "Thinking": "off",
            "Attack success rate (%) ⬇️": "25.000",
            "Incomplete response rate (%) ⬇️": "50.000",
        }
    ]


def test_refusal_evaluator_grade_impl_writes_metrics_and_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    evaluator = FreeTextRefusalEvaluator(
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            max_samples=4,
        ),
        DatasetConfig(
            file_path=XSTEST_DATASET,
            dataset_type=DatasetType.BIAS,
        ),
    )
    evaluator.num_samples = 4
    monkeypatch.setattr(evaluator, "prepare_judge_tokenizer", lambda: None)
    monkeypatch.setattr(evaluator, "_get_judge_tokenizer", lambda: object())
    monkeypatch.setattr(
        base_evaluator_module,
        "safe_apply_chat_template",
        lambda _tokenizer, messages: messages[-1]["content"],
    )

    judge_outputs = iter(
        [
            "1_full_compliance",
            "2_full_refusal",
            "3_partial_refusal",
        ]
    )
    monkeypatch.setattr(
        evaluator,
        "run_judge_with_backoff",
        lambda _judge, prompts: [
            [{"generated_text": next(judge_outputs), "finish_reason": "stop"}]
            for _ in prompts
        ],
    )

    evaluator._grade_impl(
        [
            _RefusalGenerationRecord(
                input_texts=[
                    "safe-comply",
                    "safe-refuse",
                    "unsafe-partial",
                    "unsafe-length",
                ],
                expected_labels=["safe", "safe", "unsafe", "unsafe"],
                answers=[
                    "Here is the answer.",
                    "Sorry, I can't help with that.",
                    "I can't help, but here is some related info.",
                    "truncated",
                ],
                finish_reasons=["stop", "stop", "stop", "length"],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )

    metrics_file_path = tmp_path / "model" / "XSTest" / "metrics.csv"
    with metrics_file_path.open(newline="", encoding="utf-8") as metrics_file:
        metrics_rows = list(csv.DictReader(metrics_file))

    assert metrics_rows == [
        {
            "Safe refusal rate (%) ⬇️": "50.000",
            "Unsafe refusal rate (%) ⬆️": "100.000",
            "Heuristic safe refusal rate (%) ⬇️": "50.000",
            "Heuristic unsafe refusal rate (%) ⬆️": "50.000",
            "Thinking": "off",
            "Safe Samples": "2",
            "Unsafe Samples": "2",
            "Incomplete Responses": "1",
            "Unknown Finish Reasons": "0",
            "Judge Unparseable": "0",
            "Judge Attempted Samples": "3",
            "Judged Samples": "3",
            "Safe Known Samples": "2",
            "Unsafe Known Samples": "1",
            "Incomplete response rate (%) ⬇️": "25.000",
            "Unknown finish reason rate (%) ⬇️": "0.000",
            "Judge unparseable rate (%) ⬇️": "0.000",
            "Judge parse success rate (%) ⬆️": "100.000",
        }
    ]

    summary_full_path = tmp_path / "model" / "summary_full.csv"
    with summary_full_path.open(newline="", encoding="utf-8") as summary_file:
        summary_rows = list(csv.DictReader(summary_file))
    assert summary_rows[0]["Dataset"] == "XSTest"
    assert summary_rows[0]["Dataset Type"] == "DatasetType.BIAS"
    assert summary_rows[0]["Safe refusal rate (%) ⬇️"] == "50.000"
    assert summary_rows[0]["Unsafe refusal rate (%) ⬆️"] == "100.000"

    responses_path = tmp_path / "model" / "XSTest" / "responses.json"
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    assert responses[-1]["finish_reason"] == "length"
    assert responses[-1]["judge_finish_reason"] is None
    assert responses[-1]["judge_label"] is None
    assert responses[-1]["grading_status"] == "model_incomplete"
    assert responses[2]["judge_success"] is True
    assert responses[2]["judge_finish_reason"] == "stop"
    assert responses[2]["grading_status"] == "judged"


def test_censorship_evaluator_writes_dedicated_standard_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = object.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        model_revision="model-revision",
        judge_path_or_repo_id="google/gemma-4-26B-A4B-it",
        judge_revision="judge-revision",
        results_dir=tmp_path,
        enable_thinking=False,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path=CHINESE_CENSORSHIP_DATASET_SOURCE,
        dataset_id="chinese_censorship",
        dataset_type=DatasetType.BIAS,
    )
    logged_metrics: dict[str, float] = {}
    logged_artifacts: list[bool] = []
    evaluator.mlflow_config = MlflowConfig()
    monkeypatch.setattr(evaluator, "_log_mlflow_metrics", logged_metrics.update)
    monkeypatch.setattr(
        evaluator, "_log_mlflow_artifacts", lambda: logged_artifacts.append(True)
    )
    evaluator.eval_group_ids = [f"group-{index:03d}" for index in range(288)]
    evaluator.effective_batch_size = 8
    evaluator.effective_judge_batch_size = None
    evaluator.model_engine = "transformers"
    evaluator.judge_engine = "transformers"
    attempted_batch_sizes: list[int] = []

    def fake_process_batch(
        _judge_engine: EvalEngine,
        prompts: list[str],
        batch_size: int | None = None,
        do_sample: bool | None = None,
    ) -> list[list[dict[str, str | None]]]:
        del do_sample
        assert batch_size is not None
        attempted_batch_sizes.append(batch_size)
        if batch_size > 4:
            raise RuntimeError("CUDA out of memory.")
        return [
            [{"generated_text": prompt, "finish_reason": "stop"}] for prompt in prompts
        ]

    monkeypatch.setattr(evaluator, "_process_judge_prompts_batch", fake_process_batch)
    evaluator.run_judge_with_backoff(cast("EvalEngine", object()), ["prompt"] * 8)
    responses: list[CensorshipResponse] = [
        {
            "dataset": "chinese_censorship",
            "source_group_id": group_id,
            "question": "question",
            "llm_answer": "answer",
            "finish_reason": "stop",
            "judge_finish_reason": "stop",
            "judge_raw": "False",
            "grading_status": CENSORSHIP_STATUS_JUDGED,
            "judge_label": "true" if index == 0 else "false",
        }
        for index, group_id in enumerate(evaluator.eval_group_ids)
    ]

    evaluator._save_censorship_results(responses)

    metrics_file_path = tmp_path / "model" / "chinese_censorship" / "metrics.csv"
    with metrics_file_path.open(newline="", encoding="utf-8") as metrics_file:
        metrics_rows = list(csv.DictReader(metrics_file))
    assert metrics_rows[0]["Censored Samples"] == "1"
    assert metrics_rows[0]["Samples"] == "288"
    assert metrics_rows[0]["Judge Unparseable"] == "0"
    assert metrics_rows[0]["Metric Valid"] == "True"
    assert "Safe refusal rate (%) ⬇️" not in metrics_rows[0]
    assert (tmp_path / "model" / "summary_full.csv").is_file()
    assert (tmp_path / "model" / "summary_brief.csv").is_file()
    assert (
        tmp_path / "model" / "chinese_censorship" / "evaluation_contract.json"
    ).is_file()
    contract = json.loads(
        (
            tmp_path / "model" / "chinese_censorship" / "evaluation_contract.json"
        ).read_text()
    )
    assert contract["model"]["revision"] == "model-revision"
    assert contract["judge"]["revision"] == "judge-revision"
    assert attempted_batch_sizes == [8, 4, 4]
    assert contract["effective"]["judge_batch_size"] == 4
    assert logged_metrics["censorship_rate"] == pytest.approx(1 / 288)
    assert logged_metrics["num_samples"] == 288
    assert logged_artifacts == [True]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            {
                "judge_label": "unsupported",
                "grading_status": CENSORSHIP_STATUS_JUDGED,
            },
            "unsupported judge label",
        ),
        (
            {
                "judge_label": None,
                "grading_status": "unsupported",
            },
            "unsupported grading status",
        ),
        (
            {
                "judge_label": None,
                "grading_status": CENSORSHIP_STATUS_JUDGED,
            },
            "illegal grading_status/judge_label pair",
        ),
        (
            {
                "judge_label": "unparseable",
                "grading_status": CENSORSHIP_STATUS_JUDGED,
            },
            "illegal grading_status/judge_label pair",
        ),
        (
            {
                "judge_label": "true",
                "grading_status": CENSORSHIP_STATUS_UNPARSEABLE,
            },
            "illegal grading_status/judge_label pair",
        ),
        (
            {
                "judge_label": "false",
                "grading_status": GRADING_STATUS_MODEL_INCOMPLETE,
            },
            "illegal grading_status/judge_label pair",
        ),
    ],
)
def test_censorship_results_reject_unsupported_categories(
    response: dict[str, str | None], message: str
) -> None:
    """Reject malformed evidence before any censorship artifacts are persisted."""
    evaluator = object.__new__(FreeTextCensorshipEvaluator)

    with pytest.raises(ValueError, match=message):
        evaluator._save_censorship_results(
            [cast("CensorshipResponse", response)] * CHINESE_CENSORSHIP_GROUP_COUNT
        )


@pytest.mark.parametrize(
    ("grading_status", "judge_label"),
    sorted(LEGAL_CENSORSHIP_STATUS_LABEL_PAIRS),
)
def test_censorship_status_label_legal_pairs_are_accepted(
    grading_status: str, judge_label: str | None
) -> None:
    """Every documented legal status/label pair validates successfully."""
    validate_censorship_status_label_pair(
        cast("Any", grading_status), cast("Any", judge_label)
    )


@pytest.mark.parametrize(
    ("grading_status", "judge_label"),
    [
        (CENSORSHIP_STATUS_JUDGED, None),
        (CENSORSHIP_STATUS_JUDGED, "unparseable"),
        (CENSORSHIP_STATUS_UNPARSEABLE, "true"),
        (CENSORSHIP_STATUS_UNPARSEABLE, "false"),
        (CENSORSHIP_STATUS_UNPARSEABLE, None),
        (GRADING_STATUS_MODEL_INCOMPLETE, "true"),
        (GRADING_STATUS_UNKNOWN_FINISH_REASON, "false"),
        (GRADING_STATUS_JUDGE_INCOMPLETE, "unparseable"),
        (GRADING_STATUS_JUDGE_UNKNOWN_FINISH_REASON, "true"),
    ],
)
def test_censorship_status_label_illegal_pairs_fail_fast(
    grading_status: str, judge_label: str | None
) -> None:
    """Impossible status/label combinations raise before metrics are derived."""
    with pytest.raises(ValueError, match="illegal grading_status/judge_label pair"):
        validate_censorship_status_label_pair(
            cast("Any", grading_status), cast("Any", judge_label)
        )


def test_censorship_adapter_manifest_ignores_symlinks(tmp_path: Path) -> None:
    """Exclude symlinked adapter paths from the persisted provenance digest."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_config.write_text("{}", encoding="utf-8")
    manifest_digest = censorship_module._adapter_manifest_sha256(str(adapter_dir))
    symlinked_adapter_dir = tmp_path / "adapter-link"
    symlinked_adapter_dir.symlink_to(adapter_dir, target_is_directory=True)
    outside_file = tmp_path / "outside.safetensors"
    outside_file.write_text("outside", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").symlink_to(outside_file)

    assert manifest_digest is not None
    assert (
        censorship_module._adapter_manifest_sha256(str(symlinked_adapter_dir)) is None
    )
    assert (
        censorship_module._adapter_manifest_sha256(str(adapter_dir)) == manifest_digest
    )


def test_censorship_grading_records_binary_labels_and_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = object.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        model_revision="model-revision",
        judge_path_or_repo_id="google/gemma-4-26B-A4B-it",
        judge_revision="judge-revision",
        results_dir=tmp_path,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path=CHINESE_CENSORSHIP_DATASET_SOURCE,
        dataset_id="chinese_censorship",
        dataset_type=DatasetType.BIAS,
    )
    evaluator.mlflow_config = None
    evaluator.eval_group_ids = [f"group-{index:03d}" for index in range(288)]
    evaluator.effective_batch_size = 8
    evaluator.effective_judge_batch_size = 8
    evaluator.model_engine = "transformers"
    evaluator.judge_engine = "transformers"
    monkeypatch.setattr(evaluator, "prepare_judge_tokenizer", lambda: None)
    monkeypatch.setattr(evaluator, "_get_judge_tokenizer", lambda: object())
    monkeypatch.setattr(
        base_evaluator_module,
        "safe_apply_chat_template",
        lambda _tokenizer, _messages: "judge prompt",
    )

    def judge_outputs(
        _judge_engine: EvalEngine, prompts: list[str]
    ) -> list[list[dict[str, str]]]:
        return [
            [
                {
                    "generated_text": "True" if index == 0 else "False",
                    "finish_reason": "stop",
                }
            ]
            for index, _prompt in enumerate(prompts)
        ]

    monkeypatch.setattr(evaluator, "run_judge_with_backoff", judge_outputs)
    generation = _CensorshipGenerationRecord(
        questions=["question"] * 288,
        source_group_ids=evaluator.eval_group_ids,
        answers=["answer"] * 288,
        finish_reasons=["stop"] * 288,
    )

    evaluator._grade_impl([generation], cast("EvalEngine", object()))

    responses_path = tmp_path / "model" / "chinese_censorship" / "responses.json"
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    assert responses[0]["grading_status"] == CENSORSHIP_STATUS_JUDGED
    assert responses[0]["judge_label"] == "true"
    assert responses[1]["grading_status"] == CENSORSHIP_STATUS_JUDGED
    assert responses[1]["judge_label"] == "false"
    metrics_path = tmp_path / "model" / "chinese_censorship" / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        metrics = next(csv.DictReader(metrics_file))
    assert metrics["Censored Samples"] == "1"
    assert float(metrics["Censorship rate (%) ⬇️"]) == pytest.approx(
        100 / 288, abs=0.001
    )


@pytest.mark.parametrize(
    ("model_revision", "judge_revision", "message"),
    [
        (None, "b" * 40, "model-revision"),
        ("a" * 40, None, "judge-revision"),
    ],
)
def test_censorship_evaluator_requires_immutable_model_and_judge_revisions(
    tmp_path: Path,
    model_revision: str | None,
    judge_revision: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FreeTextCensorshipEvaluator(
            EvaluationConfig(
                model_path_or_repo_id="meta/model",
                model_revision=model_revision,
                judge_path_or_repo_id="google/gemma-4-26B-A4B-it",
                judge_revision=judge_revision,
                results_dir=tmp_path,
            ),
            DatasetConfig(
                file_path=CHINESE_CENSORSHIP_DATASET_SOURCE,
                dataset_id="chinese_censorship",
                dataset_type=DatasetType.BIAS,
            ),
        )


def test_censorship_metric_is_unavailable_when_coverage_is_incomplete(
    tmp_path: Path,
) -> None:
    evaluator = object.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        model_revision="model-revision",
        judge_path_or_repo_id="google/gemma-4-26B-A4B-it",
        judge_revision="judge-revision",
        results_dir=tmp_path,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path=CHINESE_CENSORSHIP_DATASET_SOURCE,
        dataset_id="chinese_censorship",
        dataset_type=DatasetType.BIAS,
    )
    evaluator.mlflow_config = None
    evaluator.eval_group_ids = [f"group-{index:03d}" for index in range(288)]
    evaluator.effective_batch_size = 8
    evaluator.effective_judge_batch_size = 8
    evaluator.model_engine = "transformers"
    evaluator.judge_engine = "transformers"
    responses: list[CensorshipResponse] = [
        {
            "dataset": "chinese_censorship",
            "source_group_id": group_id,
            "question": "question",
            "llm_answer": "answer",
            "finish_reason": "stop",
            "judge_finish_reason": "stop",
            "judge_raw": "False",
            "grading_status": (
                CENSORSHIP_STATUS_UNPARSEABLE
                if index == 0
                else CENSORSHIP_STATUS_JUDGED
            ),
            "judge_label": "unparseable" if index == 0 else "false",
        }
        for index, group_id in enumerate(evaluator.eval_group_ids)
    ]

    evaluator._save_censorship_results(responses)

    metrics_path = tmp_path / "model" / "chinese_censorship" / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        row = next(csv.DictReader(metrics_file))
    assert row["Metric Valid"] == "False"
    assert row["Censorship rate (%) ⬇️"] == ""
    assert "Determinate censorship rate (%) ⬇️" not in row


def test_censorship_judge_uses_raw_question_and_rejects_truncated_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = object.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        model_revision="model-revision",
        judge_path_or_repo_id="google/gemma-4-26B-A4B-it",
        judge_revision="judge-revision",
        results_dir=tmp_path,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path=CHINESE_CENSORSHIP_DATASET_SOURCE,
        dataset_id="chinese_censorship",
        dataset_type=DatasetType.BIAS,
    )
    captured_prompts: list[str] = []
    captured_responses: list[CensorshipResponse] = []
    monkeypatch.setattr(evaluator, "prepare_judge_tokenizer", lambda: None)
    monkeypatch.setattr(evaluator, "_get_judge_tokenizer", lambda: object())
    monkeypatch.setattr(
        base_evaluator_module,
        "safe_apply_chat_template",
        lambda _tokenizer, messages: (
            captured_prompts.append(messages[0]["content"]) or messages[0]["content"]
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "run_judge_with_backoff",
        lambda _engine, _prompts: [
            [{"generated_text": "True", "finish_reason": "length"}]
        ],
    )
    monkeypatch.setattr(
        evaluator,
        "_save_censorship_results",
        lambda responses: captured_responses.extend(responses),
    )

    evaluator._grade_impl(
        [
            _CensorshipGenerationRecord(
                questions=["raw question"],
                source_group_ids=["group-1"],
                answers=["model answer"],
                finish_reasons=["stop"],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )

    assert captured_prompts == [
        CENSORSHIP_JUDGE_PROMPT.format(question="raw question", response="model answer")
    ]
    assert parse_censorship_judgment("True") == CensorshipJudgment.CENSORED
    assert captured_responses[0]["grading_status"] == "judge_incomplete"
    assert captured_responses[0]["judge_label"] is None


def test_refusal_evaluator_marks_unparseable_outputs_and_excludes_them_from_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    evaluator = FreeTextRefusalEvaluator(
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            max_samples=2,
        ),
        DatasetConfig(
            file_path=OR_BENCH_DATASET,
            dataset_type=DatasetType.BIAS,
        ),
    )
    evaluator.num_samples = 2
    monkeypatch.setattr(evaluator, "prepare_judge_tokenizer", lambda: None)
    monkeypatch.setattr(evaluator, "_get_judge_tokenizer", lambda: object())
    monkeypatch.setattr(
        base_evaluator_module,
        "safe_apply_chat_template",
        lambda _tokenizer, messages: messages[-1]["content"],
    )
    monkeypatch.setattr(
        evaluator,
        "run_judge_with_backoff",
        lambda _judge, prompts: [
            [{"generated_text": "other", "finish_reason": "length"}] for _ in prompts
        ],
    )

    evaluator._grade_impl(
        [
            _RefusalGenerationRecord(
                input_texts=["safe", "unsafe"],
                expected_labels=["safe", "unsafe"],
                answers=["answer", "answer"],
                finish_reasons=["stop", "stop"],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )

    metrics_file_path = tmp_path / "model" / "or-bench" / "metrics.csv"
    with metrics_file_path.open(newline="", encoding="utf-8") as metrics_file:
        metrics_rows = list(csv.DictReader(metrics_file))

    assert metrics_rows[0]["Safe refusal rate (%) ⬇️"] == "0.000"
    assert metrics_rows[0]["Unsafe refusal rate (%) ⬆️"] == "0.000"
    assert metrics_rows[0]["Judge Unparseable"] == "2"
    assert metrics_rows[0]["Judge Attempted Samples"] == "2"
    assert metrics_rows[0]["Judged Samples"] == "0"
    assert metrics_rows[0]["Safe Known Samples"] == "0"
    assert metrics_rows[0]["Unsafe Known Samples"] == "0"
    assert metrics_rows[0]["Judge unparseable rate (%) ⬇️"] == "100.000"
    assert metrics_rows[0]["Judge parse success rate (%) ⬆️"] == "0.000"

    responses_path = tmp_path / "model" / "or-bench" / "responses.json"
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    assert responses[0]["grading_status"] == "judge_unparseable"
    assert responses[0]["judge_finish_reason"] == "length"


def test_refusal_evaluator_only_counts_stop_rows_as_judge_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_evaluator_module,
        "load_tokenizer_with_transformers",
        lambda *_args, **_kwargs: StubTokenizer(),
    )
    monkeypatch.setattr(
        base_evaluator_module, "empty_cuda_cache_if_available", lambda: None
    )

    evaluator = FreeTextRefusalEvaluator(
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            max_samples=2,
        ),
        DatasetConfig(
            file_path=OR_BENCH_DATASET,
            dataset_type=DatasetType.BIAS,
        ),
    )
    evaluator.num_samples = 2
    monkeypatch.setattr(evaluator, "prepare_judge_tokenizer", lambda: None)
    monkeypatch.setattr(evaluator, "_get_judge_tokenizer", lambda: object())
    monkeypatch.setattr(
        base_evaluator_module,
        "safe_apply_chat_template",
        lambda _tokenizer, messages: messages[-1]["content"],
    )
    monkeypatch.setattr(
        evaluator,
        "run_judge_with_backoff",
        lambda _judge, prompts: [
            [{"generated_text": "[[direct_refusal]]", "finish_reason": "stop"}]
            for _ in prompts
        ],
    )

    evaluator._grade_impl(
        [
            _RefusalGenerationRecord(
                input_texts=["stop-row", "unknown-finish-row"],
                expected_labels=["unsafe", "unsafe"],
                answers=["I won't help with that.", "answer"],
                finish_reasons=["stop", None],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )

    metrics_file_path = tmp_path / "model" / "or-bench" / "metrics.csv"
    with metrics_file_path.open(newline="", encoding="utf-8") as metrics_file:
        metrics_rows = list(csv.DictReader(metrics_file))

    assert metrics_rows[0]["Judge Attempted Samples"] == "1"
    assert metrics_rows[0]["Judged Samples"] == "1"
    assert metrics_rows[0]["Unknown Finish Reasons"] == "1"
    assert metrics_rows[0]["Judge Unparseable"] == "0"
    assert metrics_rows[0]["Unknown finish reason rate (%) ⬇️"] == "50.000"
    assert metrics_rows[0]["Judge unparseable rate (%) ⬇️"] == "0.000"
    assert metrics_rows[0]["Judge parse success rate (%) ⬆️"] == "100.000"

    responses_path = tmp_path / "model" / "or-bench" / "responses.json"
    responses = json.loads(responses_path.read_text(encoding="utf-8"))
    assert responses[1]["finish_reason"] is None
    assert responses[1]["judge_label"] is None
    assert responses[1]["grading_status"] == "unknown_finish_reason"


def test_run_config_matching_generation_contract_reuses_cached_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matching generation contracts continue without prompting or clearing cache."""
    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = ConcreteEvaluator(eval_config, dataset_config)
    generations_path = evaluator.generations_path()
    generations_path.write_text(
        json.dumps({"answers": ["cached"]}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        base_evaluator_module.typer,
        "prompt",
        lambda *_args, **_kwargs: pytest.fail("matching config prompted"),
    )

    ConcreteEvaluator(eval_config, dataset_config)

    assert generations_path.read_text(encoding="utf-8") == (
        json.dumps({"answers": ["cached"]}) + "\n"
    )


def test_run_config_judge_only_mismatch_allows_skip_reusing_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Judge/postprocess-only mismatches may reuse cached generations via skip."""
    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        judge_path_or_repo_id="google/gemma-3-12b-it",
        results_dir=tmp_path,
        max_samples=1,
    )
    dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = ConcreteEvaluator(eval_config, dataset_config)
    run_config_path = evaluator.run_config_path()
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["evaluation_config"]["judge_path_or_repo_id"] = "other/judge"
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    generations_path = evaluator.generations_path()
    generations_path.write_text(
        json.dumps({"answers": ["cached"]}) + "\n", encoding="utf-8"
    )
    responses_path = evaluator.get_output_dir() / "responses.json"
    responses_path.write_text('{"preserved": true}', encoding="utf-8")

    class _StubStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(base_evaluator_module.sys, "stdin", _StubStdin())
    monkeypatch.setattr(base_evaluator_module.typer, "prompt", lambda *_a, **_k: "s")
    monkeypatch.setattr(base_evaluator_module.typer, "confirm", lambda *_a, **_k: False)

    ConcreteEvaluator(eval_config, dataset_config)

    assert responses_path.read_text(encoding="utf-8") == '{"preserved": true}'
    assert generations_path.read_text(encoding="utf-8") == (
        json.dumps({"answers": ["cached"]}) + "\n"
    )
    persisted = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert persisted["evaluation_config"]["judge_path_or_repo_id"] == "other/judge"


def test_run_config_judge_only_vllm_fields_do_not_invalidate_transformer_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Judge-only vLLM settings must not block reuse when the model is transformers."""
    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
        model_engine="transformers",
        judge_engine="vllm",
        vllm_config=VllmConfig(
            gpu_memory_utilization=0.5,
            judge_max_model_len=1024,
            enforce_eager=True,
        ),
    )
    dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = ConcreteEvaluator(eval_config, dataset_config)
    run_config_path = evaluator.run_config_path()
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["evaluation_config"]["vllm_config"]["gpu_memory_utilization"] = 0.9
    run_config["evaluation_config"]["vllm_config"]["judge_max_model_len"] = 2048
    run_config["evaluation_config"]["vllm_config"]["enforce_eager"] = False
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    generations_path = evaluator.generations_path()
    generations_payload = json.dumps({"answers": ["cached"]}) + "\n"
    generations_path.write_text(generations_payload, encoding="utf-8")

    class _StubStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(base_evaluator_module.sys, "stdin", _StubStdin())
    monkeypatch.setattr(base_evaluator_module.typer, "prompt", lambda *_a, **_k: "s")
    monkeypatch.setattr(base_evaluator_module.typer, "confirm", lambda *_a, **_k: False)

    ConcreteEvaluator(eval_config, dataset_config)

    assert generations_path.read_text(encoding="utf-8") == generations_payload


def test_generation_contract_vllm_fields_depend_on_model_engine() -> None:
    """vLLM settings enter the generation contract only when the model uses vLLM."""
    base_run_config = {
        "decoding_contract_version": 2,
        "evaluation_config": {
            "model_path_or_repo_id": "meta/model",
            "model_engine": "vllm",
            "vllm_config": {
                "max_model_len": 8192,
                "judge_max_model_len": 1024,
                "gpu_memory_utilization": 0.5,
            },
        },
        "dataset_config": {
            "file_path": "repo/dataset",
            "dataset_id": "repo/dataset",
        },
    }
    changed_generation = json.loads(json.dumps(base_run_config))
    changed_generation["evaluation_config"]["vllm_config"]["max_model_len"] = 4096
    assert BaseEvaluator._generation_contract(
        base_run_config
    ) != BaseEvaluator._generation_contract(changed_generation)

    changed_judge_only = json.loads(json.dumps(base_run_config))
    changed_judge_only["evaluation_config"]["vllm_config"]["judge_max_model_len"] = 2048
    assert BaseEvaluator._generation_contract(
        base_run_config
    ) == BaseEvaluator._generation_contract(changed_judge_only)

    transformers_run_config = json.loads(json.dumps(base_run_config))
    transformers_run_config["evaluation_config"]["model_engine"] = "transformers"
    transformers_changed = json.loads(json.dumps(transformers_run_config))
    transformers_changed["evaluation_config"]["vllm_config"][
        "gpu_memory_utilization"
    ] = 0.9
    assert BaseEvaluator._generation_contract(
        transformers_run_config
    ) == BaseEvaluator._generation_contract(transformers_changed)


def test_run_config_malformed_dataset_config_raises_replace_error(
    tmp_path: Path,
) -> None:
    """Malformed nested run-config shapes must raise RuntimeError, not TypeError."""
    output_dir = tmp_path / "model" / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "evaluation_config": {"model_path_or_repo_id": "meta/model"},
                "dataset_config": ["not-a-mapping"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="malformed"):
        ConcreteEvaluator(
            EvaluationConfig(
                model_path_or_repo_id="meta/model",
                results_dir=tmp_path,
                max_samples=1,
            ),
            DatasetConfig(
                file_path="repo/dataset",
                dataset_type=DatasetType.BIAS,
            ),
        )


def test_judge_only_regrade_persists_current_run_config(
    tmp_path: Path,
) -> None:
    """Judge-only regrade without metrics must rewrite run_config to the current contract."""
    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        judge_path_or_repo_id="google/gemma-3-12b-it",
        results_dir=tmp_path,
        max_samples=1,
    )
    dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = ConcreteEvaluator(eval_config, dataset_config)
    run_config_path = evaluator.run_config_path()
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["evaluation_config"]["judge_path_or_repo_id"] = "other/judge"
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    evaluator.eval_engine.is_judge = True

    evaluator._ensure_run_configuration_allowed()

    persisted = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert (
        persisted["evaluation_config"]["judge_path_or_repo_id"]
        == "google/gemma-3-12b-it"
    )

    # A later non-interactive invocation should treat configs as matching.
    ConcreteEvaluator(eval_config, dataset_config)


def test_censorship_grade_rejects_generic_generation_records(
    tmp_path: Path,
) -> None:
    """Generic _GenerationRecord inputs must fail fast instead of AttributeError."""
    evaluator = object.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        model_revision="a" * 40,
        judge_path_or_repo_id="google/gemma-4-26B-A4B-it",
        judge_revision="b" * 40,
        results_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="_CensorshipGenerationRecord"):
        evaluator._grade_impl(
            [_GenerationRecord(answers=["answer"])],
            judge_engine=cast("EvalEngine", object()),
        )


@pytest.mark.parametrize(
    ("mutate_existing", "match"),
    [
        (
            lambda run_config: run_config.__setitem__("decoding_contract_version", 1),
            "generation contract",
        ),
        (
            lambda run_config: run_config["evaluation_config"].__setitem__(
                "sample", True
            ),
            "generation contract",
        ),
        (
            lambda run_config: run_config["evaluation_config"].__setitem__(
                "model_path_or_repo_id", "other/model"
            ),
            "generation contract",
        ),
    ],
)
def test_run_config_generation_mismatch_interactive_skip_cannot_reuse_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate_existing: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    """Interactive skip must fail closed on generation-affecting contract changes."""
    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
        sample=False,
    )
    dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = ConcreteEvaluator(eval_config, dataset_config)
    run_config_path = evaluator.run_config_path()
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    mutate_existing(run_config)
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    generations_path = evaluator.generations_path()
    generations_payload = json.dumps({"answers": ["cached"]}) + "\n"
    generations_path.write_text(generations_payload, encoding="utf-8")

    class _StubStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(base_evaluator_module.sys, "stdin", _StubStdin())
    monkeypatch.setattr(base_evaluator_module.typer, "prompt", lambda *_a, **_k: "s")
    monkeypatch.setattr(base_evaluator_module.typer, "confirm", lambda *_a, **_k: False)

    with pytest.raises(RuntimeError, match=match):
        ConcreteEvaluator(eval_config, dataset_config)

    assert generations_path.read_text(encoding="utf-8") == generations_payload


def test_run_config_decoding_contract_mismatch_non_tty_cannot_reuse_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-interactive decoding-contract mismatches must not reuse generations."""
    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = ConcreteEvaluator(eval_config, dataset_config)
    run_config_path = evaluator.run_config_path()
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["decoding_contract_version"] = 1
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    generations_path = evaluator.generations_path()
    generations_payload = json.dumps({"answers": ["cached"]}) + "\n"
    generations_path.write_text(generations_payload, encoding="utf-8")

    class _StubStdin:
        @staticmethod
        def isatty() -> bool:
            return False

    monkeypatch.setattr(base_evaluator_module.sys, "stdin", _StubStdin())

    with pytest.raises(RuntimeError, match="generation contract"):
        ConcreteEvaluator(eval_config, dataset_config)

    assert generations_path.read_text(encoding="utf-8") == generations_payload


def test_legacy_run_config_without_dataset_id_reuses_cached_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = ConcreteEvaluator(eval_config, dataset_config)
    run_config_path = evaluator.run_config_path()
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["dataset_config"].pop("dataset_id")
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    monkeypatch.setattr(
        base_evaluator_module.typer,
        "prompt",
        lambda *_args, **_kwargs: pytest.fail("matching legacy config prompted"),
    )

    ConcreteEvaluator(eval_config, dataset_config)

    persisted = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert "dataset_id" not in persisted["dataset_config"]


@pytest.mark.parametrize("legacy_version", [None, 1])
def test_legacy_decoding_contract_cannot_reuse_cached_outputs(
    tmp_path: Path,
    legacy_version: int | None,
) -> None:
    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
        sampling_config=SamplingConfig(repetition_penalty=1.2),
    )
    dataset_config = DatasetConfig(
        file_path="repo/dataset",
        dataset_type=DatasetType.BIAS,
    )
    evaluator = ConcreteEvaluator(eval_config, dataset_config)
    run_config_path = evaluator.run_config_path()
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if legacy_version is None:
        run_config.pop("decoding_contract_version")
    else:
        run_config["decoding_contract_version"] = legacy_version
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    with pytest.raises(
        RuntimeError, match="generation contract|different configuration"
    ):
        ConcreteEvaluator(eval_config, dataset_config)


def test_run_config_mismatch_cancel_still_raises_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "model" / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(
        json.dumps(
            {
                "evaluation_config": {"model_path_or_repo_id": "different/model"},
                "dataset_config": {"file_path": "repo/other-dataset"},
            }
        ),
        encoding="utf-8",
    )

    class _StubStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(base_evaluator_module.sys, "stdin", _StubStdin())
    monkeypatch.setattr(base_evaluator_module.typer, "prompt", lambda *_a, **_k: "c")
    monkeypatch.setattr(base_evaluator_module.typer, "confirm", lambda *_a, **_k: False)

    with pytest.raises(RuntimeError, match="--replace-existing-output"):
        ConcreteEvaluator(
            EvaluationConfig(
                model_path_or_repo_id="meta/model",
                results_dir=tmp_path,
                max_samples=1,
            ),
            DatasetConfig(
                file_path="repo/dataset",
                dataset_type=DatasetType.BIAS,
            ),
        )


def _write_judge_only_mismatched_run_config(evaluator: ConcreteEvaluator) -> None:
    """Persist a run config that differs only in judge identity.

    Args:
        evaluator: Evaluator whose current matching run config should be mutated.
    """
    run_config_path = evaluator.run_config_path()
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["evaluation_config"]["judge_path_or_repo_id"] = "other/judge"
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")


def test_run_config_choice_remembered_for_rest_of_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    first_dataset = DatasetConfig(
        file_path="repo/dataset-first",
        dataset_type=DatasetType.BIAS,
    )
    second_dataset = DatasetConfig(
        file_path="repo/dataset-second",
        dataset_type=DatasetType.BIAS,
    )
    first_evaluator = ConcreteEvaluator(eval_config, first_dataset)
    _write_judge_only_mismatched_run_config(first_evaluator)
    second_evaluator = ConcreteEvaluator(eval_config, second_dataset)
    _write_judge_only_mismatched_run_config(second_evaluator)

    class _StubStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    prompt_calls = {"count": 0}
    confirm_calls = {"count": 0}

    def _prompt(*_args: object, **_kwargs: object) -> str:
        prompt_calls["count"] += 1
        return "s"

    def _confirm(*_args: object, **_kwargs: object) -> bool:
        confirm_calls["count"] += 1
        return True

    monkeypatch.setattr(base_evaluator_module.sys, "stdin", _StubStdin())
    monkeypatch.setattr(base_evaluator_module.typer, "prompt", _prompt)
    monkeypatch.setattr(base_evaluator_module.typer, "confirm", _confirm)

    evaluator = ConcreteEvaluator(eval_config, first_dataset)
    evaluator.update_dataset_config(second_dataset)

    assert prompt_calls["count"] == 1
    assert confirm_calls["count"] == 1


def test_run_config_choice_not_remembered_prompts_again_without_second_remember_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    first_dataset = DatasetConfig(
        file_path="repo/dataset-first",
        dataset_type=DatasetType.BIAS,
    )
    second_dataset = DatasetConfig(
        file_path="repo/dataset-second",
        dataset_type=DatasetType.BIAS,
    )
    first_evaluator = ConcreteEvaluator(eval_config, first_dataset)
    _write_judge_only_mismatched_run_config(first_evaluator)
    second_evaluator = ConcreteEvaluator(eval_config, second_dataset)
    _write_judge_only_mismatched_run_config(second_evaluator)

    class _StubStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    prompt_calls = {"count": 0}
    confirm_calls = {"count": 0}

    def _prompt(*_args: object, **_kwargs: object) -> str:
        prompt_calls["count"] += 1
        return "s"

    def _confirm(*_args: object, **_kwargs: object) -> bool:
        confirm_calls["count"] += 1
        return False

    monkeypatch.setattr(base_evaluator_module.sys, "stdin", _StubStdin())
    monkeypatch.setattr(base_evaluator_module.typer, "prompt", _prompt)
    monkeypatch.setattr(base_evaluator_module.typer, "confirm", _confirm)

    evaluator = ConcreteEvaluator(eval_config, first_dataset)
    evaluator.update_dataset_config(second_dataset)

    assert prompt_calls["count"] == 2
    assert confirm_calls["count"] == 1

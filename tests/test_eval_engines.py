from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import torch
from datasets import Dataset

from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig
from llm_behavior_eval.evaluation_utils.transformers_eval_engine import (
    TransformersEvalEngine,
)
from llm_behavior_eval.evaluation_utils.vllm_eval_engine import VllmEvalEngine


class RecordingTokenizer:
    def __init__(self, pad_token_id: int = 0, eos_token_id: int = 2) -> None:
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.batch_decode_calls: list[dict[str, object]] = []
        self.pad_token: str | None = None
        self.eos_token = "<eos>"

    def batch_decode(self, tokens: torch.Tensor, skip_special_tokens: bool = True):
        self.batch_decode_calls.append(
            {"tokens": tokens.clone(), "skip_special_tokens": skip_special_tokens}
        )
        return ["decoded"] * tokens.size(0)


class DummyTransformersModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []
        self.eval_called = False
        self.cpu_called = False
        self.device = torch.device("cpu")

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        input_ids: torch.Tensor = kwargs["input_ids"]
        extra = torch.full(
            (input_ids.size(0), 1),
            fill_value=9,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, extra], dim=1)

    def eval(self):
        self.eval_called = True

    def cpu(self):
        self.cpu_called = True
        return self


class BuildPromptRecorder:
    def __init__(self) -> None:
        self.last_input_ids: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None
        self.return_value = [[101], [102]]

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.last_input_ids = input_ids
        self.last_attention_mask = attention_mask
        return self.return_value


class SamplingParamsRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(**kwargs)


class ReturnValueStub:
    def __init__(self, value) -> None:
        self.value = value
        self.calls: list[dict[str, object]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.value


def always_false() -> bool:
    return False


class DummyVllmModel:
    def __init__(self, outputs_log: list[object]) -> None:
        self.outputs_log = outputs_log
        self.llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(
                shutdown=lambda: self.outputs_log.append("shutdown")
            )
        )

    def generate(self, **kwargs):
        self.outputs_log.append(kwargs)
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text="first")]),
            SimpleNamespace(outputs=[]),
        ]


class TransformModelLoaderStub:
    def __init__(self, tokenizer: RecordingTokenizer, model: DummyTransformersModel):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.tokenizer, self.model


class TokenizerLoaderStub:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, _model_id, _token: str | None = None):
        return self.tokenizer


class ConstantCollator:
    def __call__(self, _batch):
        return {
            "test_input_ids": torch.tensor([[1, 2]]),
            "test_attention_mask": torch.tensor([[1, 1]]),
        }


class ExecutableBatchWrapper:
    def __init__(self, fn, starting_batch_size: int) -> None:
        self.fn = fn
        self.starting_batch_size = starting_batch_size

    def __call__(self):
        return self.fn(self.starting_batch_size)


class FindExecutableBatchSizeRecorder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, fn, starting_batch_size, reduce_batch_size_fn):
        self.calls.append(starting_batch_size)
        return ExecutableBatchWrapper(fn, starting_batch_size)


class CandidateRecorder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def record(self, candidate_bs: int) -> int:
        self.calls.append(candidate_bs)
        return candidate_bs


@dataclass
class VllmPatchBundle:
    tokenizer: SimpleNamespace
    build_recorder: BuildPromptRecorder
    sampling_recorder: SamplingParamsRecorder
    outputs_log: list[object]
    tokenizer_loader: TokenizerLoaderStub
    model_loader: ReturnValueStub


@dataclass
class TransformersPatchBundle:
    tokenizer: RecordingTokenizer
    model: DummyTransformersModel
    loader_stub: TransformModelLoaderStub
    data_collator: ConstantCollator
    find_recorder: FindExecutableBatchSizeRecorder
    candidate_recorder: CandidateRecorder


@pytest.fixture
def vllm_bundle() -> VllmPatchBundle:
    tokenizer = SimpleNamespace(
        pad_token=None,
        eos_token="<eos>",
        eos_token_id=7,
    )
    build_recorder = BuildPromptRecorder()
    sampling_recorder = SamplingParamsRecorder()
    outputs_log: list[object] = []
    model = DummyVllmModel(outputs_log)
    return VllmPatchBundle(
        tokenizer=tokenizer,
        build_recorder=build_recorder,
        sampling_recorder=sampling_recorder,
        outputs_log=outputs_log,
        tokenizer_loader=TokenizerLoaderStub(tokenizer),
        model_loader=ReturnValueStub(model),
    )


@pytest.fixture
def transformers_bundle() -> TransformersPatchBundle:
    tokenizer = RecordingTokenizer()
    model = DummyTransformersModel()
    loader_stub = TransformModelLoaderStub(tokenizer, model)
    data_collator = ConstantCollator()
    find_recorder = FindExecutableBatchSizeRecorder()
    candidate_recorder = CandidateRecorder()
    return TransformersPatchBundle(
        tokenizer=tokenizer,
        model=model,
        loader_stub=loader_stub,
        data_collator=data_collator,
        find_recorder=find_recorder,
        candidate_recorder=candidate_recorder,
    )


@pytest.fixture(autouse=True)
def _apply_vllm_patching(request, monkeypatch):
    if "vllm_engine_test" not in request.keywords:
        return
    bundle: VllmPatchBundle = request.getfixturevalue("vllm_bundle")
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.vllm_eval_engine.load_tokenizer_with_transformers",
        bundle.tokenizer_loader,
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.vllm_eval_engine.build_vllm_prompt_token_ids",
        bundle.build_recorder,
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.vllm_eval_engine.torch.cuda.is_available",
        always_false,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(SamplingParams=bundle.sampling_recorder),
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.vllm_eval_engine.load_vllm_model",
        bundle.model_loader,
    )


@pytest.fixture(autouse=True)
def _apply_transformers_patching(request, monkeypatch):
    if "transformers_engine_test" not in request.keywords:
        return
    bundle: TransformersPatchBundle = request.getfixturevalue("transformers_bundle")
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.transformers_eval_engine.load_transformers_model_and_tokenizer",
        bundle.loader_stub,
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.transformers_eval_engine.find_executable_batch_size",
        bundle.find_recorder,
    )
    monkeypatch.setattr(
        TransformersEvalEngine,
        "_get_first_non_oom_batch_size",
        bundle.candidate_recorder.record,
    )


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_generate_answers(vllm_bundle, tmp_path) -> None:
    dataset = Dataset.from_dict({"question": ["q1", "q2"]})
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        answer_tokens=16,
        sample=False,
        batch_size=None,
    )

    engine = VllmEvalEngine(config)
    engine.set_dataset(dataset)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    responses = engine.generate_answers(
        input_ids,
        attention_mask,
        sampling_config=SamplingConfig(
            do_sample=config.sample,
            temperature=config.sampling_config.temperature,
            top_p=config.sampling_config.top_p,
            top_k=config.sampling_config.top_k,
            seed=None,
        ),
    )
    assert responses == ["first", ""]
    assert vllm_bundle.build_recorder.last_input_ids is input_ids
    assert vllm_bundle.build_recorder.last_attention_mask is attention_mask
    call_kwargs = vllm_bundle.sampling_recorder.calls[0]
    assert call_kwargs["max_tokens"] == config.answer_tokens
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["stop_token_ids"] == [vllm_bundle.tokenizer.eos_token_id]
    assert vllm_bundle.tokenizer.pad_token == vllm_bundle.tokenizer.eos_token
    assert engine.get_batch_size() == len(dataset)


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_sampling_overrides_config(vllm_bundle, tmp_path) -> None:
    dataset = Dataset.from_dict({"question": ["q1", "q2"]})
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        answer_tokens=8,
        sample=False,
        batch_size=None,
    )

    engine = VllmEvalEngine(config)
    engine.set_dataset(dataset)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    responses = engine.generate_answers(
        input_ids,
        attention_mask,
        sampling_config=SamplingConfig(
            do_sample=True,
            temperature=None,
            top_p=0.9,
            top_k=5,
            seed=99,
        ),
    )
    assert responses == ["first", ""]
    call_kwargs = vllm_bundle.sampling_recorder.calls[-1]
    assert call_kwargs["temperature"] == 1.0
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["top_k"] == 5
    assert call_kwargs["seed"] == 99


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_passes_optional_kwargs(vllm_bundle, tmp_path) -> None:
    from llm_behavior_eval.evaluation_utils.vllm_config import VllmConfig

    vllm_config = VllmConfig(
        tokenizer_mode="slow",
        config_format="hf-torch",
        load_format="dummy",
        tool_call_parser="json",
        enable_auto_tool_choice=True,
    )
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        model_engine="vllm",
        vllm_config=vllm_config,
    )

    VllmEvalEngine(config)

    last_call = vllm_bundle.model_loader.calls[-1]["kwargs"]
    assert last_call["tokenizer_mode"] == "slow"
    assert last_call["config_format"] == "hf-torch"
    assert last_call["load_format"] == "dummy"
    assert last_call["tool_call_parser"] == "json"
    assert last_call["enable_auto_tool_choice"] is True


@pytest.mark.transformers_engine_test
def test_transformers_eval_engine_generate_answers(
    transformers_bundle, tmp_path
) -> None:
    dataset = Dataset.from_dict({"prompt": ["hi"]})
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        answer_tokens=3,
        sample=True,
        batch_size=2,
    )

    engine = TransformersEvalEngine(
        transformers_bundle.data_collator,
        config,
    )
    engine.set_dataset(dataset)

    input_ids = torch.tensor([[5, 6]])
    attention_mask = torch.tensor([[1, 1]])
    sampling_config = SamplingConfig(
        do_sample=config.sample,
        temperature=0.7,
        top_p=0.8,
        top_k=5,
        seed=123,
    )
    answers = engine.generate_answers(
        input_ids,
        attention_mask,
        sampling_config=sampling_config,
    )

    assert answers == ["decoded"]
    generate_call = transformers_bundle.model.generate_calls[0]
    assert generate_call["do_sample"] == config.sample
    assert generate_call["max_new_tokens"] == config.answer_tokens
    assert generate_call["pad_token_id"] == transformers_bundle.tokenizer.pad_token_id
    assert generate_call["eos_token_id"] == transformers_bundle.tokenizer.eos_token_id
    assert generate_call["temperature"] == sampling_config.temperature
    assert generate_call["top_p"] == sampling_config.top_p
    assert generate_call["top_k"] == sampling_config.top_k
    assert generate_call["seed"] == sampling_config.seed
    decode_call = transformers_bundle.tokenizer.batch_decode_calls[0]
    assert decode_call["skip_special_tokens"] is True
    assert torch.equal(
        decode_call["tokens"],
        torch.tensor([[9]], dtype=torch.long),
    )

    engine.ensure_test_model_ready()
    assert transformers_bundle.model.eval_called

    engine.free_model()
    assert transformers_bundle.model.cpu_called
    assert not hasattr(engine, "model")


@pytest.mark.transformers_engine_test
def test_transformers_eval_engine_sampling_config_overrides_defaults(
    transformers_bundle, tmp_path
) -> None:
    dataset = Dataset.from_dict({"prompt": ["hi"]})
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        answer_tokens=3,
        sample=False,
        batch_size=1,
    )

    engine = TransformersEvalEngine(
        transformers_bundle.data_collator,
        config,
    )
    engine.set_dataset(dataset)

    input_ids = torch.tensor([[7, 8]])
    attention_mask = torch.tensor([[1, 1]])
    sampling_config = SamplingConfig(
        do_sample=True,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=321,
    )
    engine.generate_answers(
        input_ids,
        attention_mask,
        sampling_config=sampling_config,
    )
    generate_call = transformers_bundle.model.generate_calls[-1]
    assert generate_call["do_sample"] is True
    assert generate_call["temperature"] == 1.0
    assert generate_call["top_p"] == 1.0
    assert generate_call["top_k"] == 0
    assert generate_call["seed"] == sampling_config.seed


@pytest.mark.transformers_engine_test
def test_transformers_eval_engine_get_batch_size_autotune(
    transformers_bundle, tmp_path
) -> None:
    dataset = Dataset.from_dict({"prompt": list(range(5))})
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        answer_tokens=2,
        sample=False,
        batch_size=None,
    )

    engine = TransformersEvalEngine(
        transformers_bundle.data_collator,
        config,
    )
    engine.set_dataset(dataset)
    batch_size = engine.get_batch_size()

    assert batch_size == len(dataset)
    assert transformers_bundle.find_recorder.calls == [len(dataset)]
    assert transformers_bundle.candidate_recorder.calls == [len(dataset)]

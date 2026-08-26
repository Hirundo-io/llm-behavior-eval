from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("torch")
import torch
from datasets import Dataset

from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig
from llm_behavior_eval.evaluation_utils.transformers_eval_engine import (
    TransformersEvalEngine,
)
from llm_behavior_eval.evaluation_utils.vllm_config import VllmConfig
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
        sequences = torch.cat([input_ids, extra], dim=1)
        if kwargs.get("return_dict_in_generate", False):
            return SimpleNamespace(sequences=sequences)
        return sequences

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


@dataclass(frozen=True)
class RecordedCall:
    args: tuple[object, ...]
    kwargs: dict[str, object]


class ReturnValueStub:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[RecordedCall] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append(RecordedCall(args=args, kwargs=kwargs))
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
        self.calls: list[RecordedCall] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(RecordedCall(args=args, kwargs=kwargs))
        return self.tokenizer, self.model


class TokenizerLoaderStub:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.calls: list[RecordedCall] = []

    def __call__(
        self,
        _model_id,
        _token: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
    ):
        self.calls.append(
            RecordedCall(
                args=(_model_id,),
                kwargs={
                    "token": _token,
                    "trust_remote_code": trust_remote_code,
                    "revision": revision,
                },
            )
        )
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


@dataclass(frozen=True)
class CompilationConfigStub:
    cudagraph_specialize_lora: bool


@dataclass(frozen=True)
class VllmConstructorCall:
    model: str
    runner: str
    trust_remote_code: bool
    dtype: str
    enforce_eager: bool
    quantization: str | None
    tensor_parallel_size: int
    max_num_seqs: int
    hf_token: str | None
    max_model_len: int | None
    tokenizer_mode: str
    config_format: str | None
    load_format: str | None
    gpu_memory_utilization: float
    enable_lora: bool
    max_lora_rank: int
    language_model_only: bool
    compilation_config: CompilationConfigStub
    revision: str | None = None
    tokenizer_revision: str | None = None


class RecordingLlm:
    calls: list[VllmConstructorCall] = []

    def __init__(
        self,
        model: str,
        *,
        runner: str,
        trust_remote_code: bool,
        dtype: str,
        enforce_eager: bool,
        quantization: str | None,
        tensor_parallel_size: int,
        max_num_seqs: int,
        hf_token: str | None,
        max_model_len: int | None,
        tokenizer_mode: str,
        config_format: str | None,
        load_format: str | None,
        gpu_memory_utilization: float,
        enable_lora: bool,
        max_lora_rank: int,
        language_model_only: bool,
        compilation_config: CompilationConfigStub,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
    ) -> None:
        self.calls.append(
            VllmConstructorCall(
                model=model,
                runner=runner,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                enforce_eager=enforce_eager,
                quantization=quantization,
                tensor_parallel_size=tensor_parallel_size,
                max_num_seqs=max_num_seqs,
                hf_token=hf_token,
                max_model_len=max_model_len,
                tokenizer_mode=tokenizer_mode,
                config_format=config_format,
                load_format=load_format,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_lora=enable_lora,
                max_lora_rank=max_lora_rank,
                language_model_only=language_model_only,
                compilation_config=compilation_config,
                revision=revision,
                tokenizer_revision=tokenizer_revision,
            )
        )


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
def test_vllm_eval_engine_generate_answers(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    """Verify vLLM receives the default repetition penalty.

    Args:
        vllm_bundle: Patched vLLM dependencies and call recorders.
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    dataset = Dataset.from_dict({"question": ["q1", "q2"]})
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        max_answer_tokens=16,
        sample=False,
        batch_size=None,
    )

    engine = VllmEvalEngine(config)
    engine.set_dataset(dataset)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    responses, finish_reasons = engine.generate_answers(
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
    assert finish_reasons == [None, None]
    assert vllm_bundle.build_recorder.last_input_ids is input_ids
    assert vllm_bundle.build_recorder.last_attention_mask is attention_mask
    call_kwargs = vllm_bundle.sampling_recorder.calls[0]
    assert call_kwargs["max_tokens"] == config.max_answer_tokens
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["repetition_penalty"] == 1.0
    assert call_kwargs["stop_token_ids"] == [vllm_bundle.tokenizer.eos_token_id]
    assert vllm_bundle.tokenizer.pad_token == vllm_bundle.tokenizer.eos_token
    assert engine.get_batch_size() == len(dataset)


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_sampling_overrides_config(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    """Verify vLLM receives explicit sampling and repetition settings.

    Args:
        vllm_bundle: Patched vLLM dependencies and call recorders.
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    dataset = Dataset.from_dict({"question": ["q1", "q2"]})
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        max_answer_tokens=8,
        sample=False,
        batch_size=None,
    )

    engine = VllmEvalEngine(config)
    engine.set_dataset(dataset)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    responses, finish_reasons = engine.generate_answers(
        input_ids,
        attention_mask,
        sampling_config=SamplingConfig(
            do_sample=True,
            temperature=None,
            top_p=0.9,
            top_k=5,
            seed=99,
        ),
        repetition_penalty=1.1,
    )
    assert responses == ["first", ""]
    assert finish_reasons == [None, None]
    call_kwargs = vllm_bundle.sampling_recorder.calls[-1]
    assert call_kwargs["temperature"] == 1.0
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["top_k"] == 5
    assert call_kwargs["repetition_penalty"] == 1.1
    assert call_kwargs["seed"] == 99


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_passes_optional_kwargs(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    vllm_config = VllmConfig(
        max_model_len=8192,
        tokenizer_mode="slow",
        config_format="hf-torch",
        load_format="dummy",
        enforce_eager=True,
    )
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        model_engine="vllm",
        vllm_config=vllm_config,
    )

    VllmEvalEngine(config)

    last_call = vllm_bundle.model_loader.calls[-1].kwargs
    assert last_call["max_model_len"] == 8192
    assert last_call["tokenizer_mode"] == "slow"
    assert last_call["config_format"] == "hf-torch"
    assert last_call["load_format"] == "dummy"
    assert last_call["language_model_only"] is False
    assert last_call["enforce_eager"] is True


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_allows_multimodal_loading(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        model_engine="vllm",
        vllm_config=VllmConfig(language_model_only=False),
    )

    VllmEvalEngine(config)

    last_call = vllm_bundle.model_loader.calls[-1].kwargs
    assert last_call["language_model_only"] is False


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_forces_text_only_generation_for_judge(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        judge_path_or_repo_id="fake/judge",
        results_dir=tmp_path,
        judge_engine="vllm",
        vllm_config=VllmConfig(language_model_only=False),
    )

    VllmEvalEngine(config, is_judge=True, max_model_len=2048)

    last_call = vllm_bundle.model_loader.calls[-1]
    assert last_call.args[0] == "fake/judge"
    assert last_call.kwargs["max_model_len"] == 2048
    assert last_call.kwargs["language_model_only"] is True


def test_load_vllm_model_uses_text_generation_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_behavior_eval.evaluation_utils.util_functions import load_vllm_model

    RecordingLlm.calls.clear()
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=RecordingLlm))
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        types.SimpleNamespace(CompilationConfig=CompilationConfigStub),
    )

    load_vllm_model(
        "fake/model",
        torch.bfloat16,
        trust_remote_code=False,
        batch_size=16,
        tensor_parallel_size=2,
        max_model_len=4096,
        tokenizer_mode="slow",
        config_format="hf",
        load_format="safetensors",
        gpu_memory_utilization=0.8,
    )

    assert RecordingLlm.calls == [
        VllmConstructorCall(
            model="fake/model",
            runner="generate",
            trust_remote_code=False,
            dtype="bfloat16",
            enforce_eager=False,
            quantization=None,
            tensor_parallel_size=2,
            max_num_seqs=16,
            hf_token=None,
            max_model_len=4096,
            tokenizer_mode="slow",
            config_format="hf",
            load_format="safetensors",
            gpu_memory_utilization=0.8,
            enable_lora=False,
            max_lora_rank=128,
            language_model_only=False,
            compilation_config=CompilationConfigStub(cudagraph_specialize_lora=False),
        )
    ]


def test_load_vllm_model_forwards_multimodal_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_behavior_eval.evaluation_utils.util_functions import load_vllm_model

    RecordingLlm.calls.clear()
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=RecordingLlm))
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        types.SimpleNamespace(CompilationConfig=CompilationConfigStub),
    )

    load_vllm_model(
        "fake/multimodal-model",
        torch.bfloat16,
        trust_remote_code=False,
        batch_size=8,
        tensor_parallel_size=1,
        language_model_only=False,
    )

    assert len(RecordingLlm.calls) == 1
    call = RecordingLlm.calls[0]
    assert call.model == "fake/multimodal-model"
    assert call.language_model_only is False


@pytest.mark.parametrize(
    ("config_format", "load_format", "expected_config", "expected_load"),
    [
        (None, None, "auto", "auto"),
        ("mistral", "tensorizer", "mistral", "tensorizer"),
    ],
)
def test_load_vllm_model_config_and_load_format_defaults_and_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    config_format: str | None,
    load_format: str | None,
    expected_config: str,
    expected_load: str,
) -> None:
    from llm_behavior_eval.evaluation_utils.util_functions import load_vllm_model

    RecordingLlm.calls.clear()
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=RecordingLlm))
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        types.SimpleNamespace(CompilationConfig=CompilationConfigStub),
    )

    load_vllm_model(
        "fake/model",
        torch.bfloat16,
        trust_remote_code=False,
        batch_size=16,
        tensor_parallel_size=2,
        config_format=config_format,
        load_format=load_format,
    )

    assert len(RecordingLlm.calls) == 1
    call = RecordingLlm.calls[0]
    assert call.config_format == expected_config
    assert call.load_format == expected_load


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_explicit_length_overrides_config(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        model_engine="vllm",
        vllm_config=VllmConfig(max_model_len=8192),
    )

    VllmEvalEngine(config, max_model_len=4096)

    last_call = vllm_bundle.model_loader.calls[-1].kwargs
    assert last_call["max_model_len"] == 4096


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_uses_float16_on_t4(
    vllm_bundle: VllmPatchBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.vllm_eval_engine.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        model_engine="vllm",
    )

    VllmEvalEngine(config)

    assert vllm_bundle.model_loader.calls[-1].args[1] == torch.float16


@pytest.mark.transformers_engine_test
def test_transformers_eval_engine_generate_answers(
    transformers_bundle: TransformersPatchBundle, tmp_path: Path
) -> None:
    """Verify Transformers receives the default repetition penalty.

    Args:
        transformers_bundle: Patched Transformers dependencies and call recorders.
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    dataset = Dataset.from_dict({"prompt": ["hi"]})
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        max_answer_tokens=3,
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
    answers, finish_reasons = engine.generate_answers(
        input_ids,
        attention_mask,
        sampling_config=sampling_config,
    )

    assert answers == ["decoded"]
    assert finish_reasons == ["length"]
    generate_call = transformers_bundle.model.generate_calls[0]
    assert generate_call["do_sample"] == config.sample
    assert generate_call["max_new_tokens"] == config.max_answer_tokens
    assert generate_call["pad_token_id"] == transformers_bundle.tokenizer.pad_token_id
    assert generate_call["eos_token_id"] == transformers_bundle.tokenizer.eos_token_id
    assert generate_call["temperature"] == sampling_config.temperature
    assert generate_call["top_p"] == sampling_config.top_p
    assert generate_call["top_k"] == sampling_config.top_k
    assert generate_call["repetition_penalty"] == 1.0
    decode_call = transformers_bundle.tokenizer.batch_decode_calls[0]
    assert decode_call["skip_special_tokens"] is True
    assert torch.equal(
        cast("torch.Tensor", decode_call["tokens"]),
        torch.tensor([[9]], dtype=torch.long),
    )

    engine.ensure_test_model_ready()
    assert transformers_bundle.model.eval_called

    engine.free_model()
    assert transformers_bundle.model.cpu_called
    assert not hasattr(engine, "model")


@pytest.mark.transformers_engine_test
def test_transformers_eval_engine_sampling_config_overrides_defaults(
    transformers_bundle: TransformersPatchBundle, tmp_path: Path
) -> None:
    """Verify Transformers receives explicit sampling and repetition settings.

    Args:
        transformers_bundle: Patched Transformers dependencies and call recorders.
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    dataset = Dataset.from_dict({"prompt": ["hi"]})
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        max_answer_tokens=3,
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
        repetition_penalty=1.1,
    )
    generate_call = transformers_bundle.model.generate_calls[-1]
    assert generate_call["do_sample"] is True
    assert generate_call["temperature"] == 1.0
    assert generate_call["top_p"] == 1.0
    assert generate_call["top_k"] == 0
    assert generate_call["repetition_penalty"] == 1.1


@pytest.mark.transformers_engine_test
def test_transformers_eval_engine_get_batch_size_autotune(
    transformers_bundle, tmp_path
) -> None:
    dataset = Dataset.from_dict({"prompt": list(range(5))})
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        max_answer_tokens=2,
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


class _HarmonyTextVllmModel:
    """A DummyVllmModel-alike that returns a fixed, configurable completion text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda: None)
        )

    def generate(self, **kwargs):
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text=self.text, finish_reason="stop")]
            )
        ]


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_disables_skip_special_tokens_for_gpt_oss(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-oss-20b",
        results_dir=tmp_path,
        max_answer_tokens=16,
    )
    engine = VllmEvalEngine(config)
    engine.set_dataset(Dataset.from_dict({"question": ["q1"]}))

    engine.generate_answers(
        torch.tensor([[1, 2]]),
        torch.tensor([[1, 1]]),
        sampling_config=SamplingConfig(do_sample=False),
    )

    assert vllm_bundle.sampling_recorder.calls[-1]["skip_special_tokens"] is False


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_keeps_skip_special_tokens_for_non_gpt_oss(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        max_answer_tokens=16,
    )
    engine = VllmEvalEngine(config)
    engine.set_dataset(Dataset.from_dict({"question": ["q1"]}))

    engine.generate_answers(
        torch.tensor([[1, 2]]),
        torch.tensor([[1, 1]]),
        sampling_config=SamplingConfig(do_sample=False),
    )

    assert vllm_bundle.sampling_recorder.calls[-1]["skip_special_tokens"] is True


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_extracts_harmony_final_channel_for_gpt_oss(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    harmony_text = (
        "<|channel|>analysis<|message|>secret reasoning<|end|>"
        "<|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>"
    )
    vllm_bundle.model_loader.value = _HarmonyTextVllmModel(harmony_text)
    config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-oss-20b",
        results_dir=tmp_path,
        max_answer_tokens=16,
    )
    engine = VllmEvalEngine(config)
    engine.set_dataset(Dataset.from_dict({"question": ["q1"]}))

    responses, finish_reasons = engine.generate_answers(
        torch.tensor([[1, 2]]),
        torch.tensor([[1, 1]]),
        sampling_config=SamplingConfig(do_sample=False),
    )

    assert responses == ["The answer is 42."]
    assert "secret reasoning" not in responses[0]
    assert finish_reasons == ["stop"]


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_gpt_oss_fails_closed_on_malformed_harmony(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    vllm_bundle.model_loader.value = _HarmonyTextVllmModel("plain text, no markers")
    config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-oss-20b",
        results_dir=tmp_path,
        max_answer_tokens=16,
    )
    engine = VllmEvalEngine(config)
    engine.set_dataset(Dataset.from_dict({"question": ["q1"]}))

    responses, finish_reasons = engine.generate_answers(
        torch.tensor([[1, 2]]),
        torch.tensor([[1, 1]]),
        sampling_config=SamplingConfig(do_sample=False),
    )

    assert responses == [""]
    assert finish_reasons == ["harmony_parse_error"]


@pytest.mark.vllm_engine_test
def test_vllm_eval_engine_threads_each_role_only_its_own_revision(
    vllm_bundle: VllmPatchBundle, tmp_path: Path
) -> None:
    unpinned = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
    )
    VllmEvalEngine(unpinned)
    assert vllm_bundle.tokenizer_loader.calls[-1].kwargs["revision"] is None
    assert vllm_bundle.model_loader.calls[-1].kwargs["revision"] is None
    assert vllm_bundle.model_loader.calls[-1].kwargs["tokenizer_revision"] is None

    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        model_revision="cafebabe",
        judge_path_or_repo_id="fake/judge",
        judge_revision="deadbeef",
        results_dir=tmp_path,
        judge_engine="vllm",
    )

    VllmEvalEngine(config)
    assert vllm_bundle.tokenizer_loader.calls[-1].kwargs["revision"] == "cafebabe"
    assert vllm_bundle.model_loader.calls[-1].kwargs["revision"] == "cafebabe"
    assert vllm_bundle.model_loader.calls[-1].kwargs["tokenizer_revision"] == "cafebabe"

    VllmEvalEngine(config, is_judge=True)
    assert vllm_bundle.tokenizer_loader.calls[-1].kwargs["revision"] == "deadbeef"
    assert vllm_bundle.model_loader.calls[-1].kwargs["revision"] == "deadbeef"
    assert vllm_bundle.model_loader.calls[-1].kwargs["tokenizer_revision"] == "deadbeef"


@pytest.mark.transformers_engine_test
def test_transformers_eval_engine_threads_each_role_only_its_own_revision(
    transformers_bundle: TransformersPatchBundle, tmp_path: Path
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        model_revision="cafebabe",
        judge_path_or_repo_id="fake/judge",
        judge_revision="deadbeef",
        results_dir=tmp_path,
    )

    TransformersEvalEngine(transformers_bundle.data_collator, config)
    target_call = transformers_bundle.loader_stub.calls[-1]
    assert target_call.args[0] == "fake/model"
    assert target_call.kwargs["revision"] == "cafebabe"

    TransformersEvalEngine(transformers_bundle.data_collator, config, is_judge=True)
    judge_call = transformers_bundle.loader_stub.calls[-1]
    assert judge_call.args[0] == "fake/judge"
    assert judge_call.kwargs["revision"] == "deadbeef"


@pytest.mark.transformers_engine_test
def test_transformers_eval_engine_rejects_gpt_oss_models(
    transformers_bundle: TransformersPatchBundle, tmp_path: Path
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-oss-20b",
        results_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="GPT-OSS"):
        TransformersEvalEngine(transformers_bundle.data_collator, config)

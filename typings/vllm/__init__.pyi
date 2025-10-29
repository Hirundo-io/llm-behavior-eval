from typing import Any, Sequence


class SamplingOutput:
    outputs: Sequence[Any]


class SamplingParams:
    def __init__(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int] | None = ...,
    ) -> None: ...


class LLM:
    llm_engine: Any

    def __init__(
        self,
        *,
        model: str,
        trust_remote_code: bool,
        dtype: str,
        enforce_eager: bool = ...,
        quantization: str | None = ...,
        tensor_parallel_size: int | None = ...,
        max_num_seqs: int | None = ...,
    ) -> None: ...

    def generate(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        sampling_params: Any,
        use_tqdm: bool = ...,
    ) -> Sequence[SamplingOutput]: ...

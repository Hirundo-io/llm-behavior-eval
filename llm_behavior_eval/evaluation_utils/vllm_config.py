from pydantic import BaseModel
from torch import cuda

from .vllm_types import TokenizerModeOption


class VllmConfig(BaseModel):
    """
    Configuration for vLLM inference.

    Keep this separate from the main EvaluationConfig to avoid
    coupling vLLM-specific settings with core evaluation logic.
    Only used when inference_engine or model_engine/judge_engine is set to "vllm".

    Args:
        max_model_len: Maximum model length for vLLM model inference (optional).
        judge_max_model_len: Maximum model length for vLLM judge inference (optional).
            Defaults to the same value as max_model_len if not specified.
        tokenizer_mode: Tokenizer mode forwarded to vLLM (e.g. 'auto', 'slow', 'mistral', 'custom').
        config_format: Model config format hint forwarded to vLLM (optional).
        load_format: Checkpoint load format hint forwarded to vLLM (optional).
        enable_lora: Whether to enable LoRA.
        max_lora_rank: The maximum LoRA rank (do not set too high to avoid wasting memory).
        enforce_eager: Whether to enforce eager execution (useful for CPU-only setups or for saving memory on CUDA graphs).
    """

    max_model_len: int | None = None
    judge_max_model_len: int | None = None
    tokenizer_mode: TokenizerModeOption | None = None
    # vLLM expects string values for these fields. Default to "auto".
    config_format: str = "auto"
    load_format: str = "auto"
    gpu_memory_utilization: float = 0.9
    enable_lora: bool = False
    max_lora_rank: int = 128
    enforce_eager: bool = not cuda.is_available()

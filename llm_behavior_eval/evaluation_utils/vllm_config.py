from pydantic import BaseModel

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
    """

    max_model_len: int | None = None
    judge_max_model_len: int | None = None

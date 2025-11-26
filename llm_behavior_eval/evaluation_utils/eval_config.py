from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic.functional_validators import model_validator

from .sampling_config import SamplingConfig
from .vllm_config import VllmConfig


class EvaluationConfig(BaseModel):
    """
    Configuration for bias evaluation.

    Args:
        max_samples: Optional limit on the number of examples to process. Use None to evaluate the full set.
        batch_size: Batch size for model inference. Depends on GPU memory (commonly 16-64). If None, will be adjusted for GPU limits.
        sample: Whether to sample outputs (True) or generate deterministically (False).
        use_4bit: Whether to load the model in 4-bit mode (using bitsandbytes).
            This is only relevant for the model under test.
        device_map: Device map for model inference. If None, will be set to "auto".
        answer_tokens: Number of tokens to generate per answer. Typical range is 32-256.
        model_path_or_repo_id: HF repo ID or path of the model under test (e.g. "meta-llama/Llama-3.1-8B-Instruct").
        model_token: HuggingFace token for the model under test.
        judge_batch_size: Batch size for the judge model (free-text tasks only). If None, will be adjusted for GPU limits.
        judge_output_tokens: Number of tokens to generate with the judge model. Typical range is 16-64.
        judge_path_or_repo_id: HF repo ID or path of the judge model (e.g. "meta-llama/Llama-3.3-70B-Instruct").
        judge_token: HuggingFace token for the judge model. Defaults to the value of `model_token` if not provided.
        sample_judge: Whether to sample outputs from the judge model (True) or generate deterministically (False). Defaults to False.
        use_4bit_judge: Whether to load the judge model in 4-bit mode (using bitsandbytes).
            This is only relevant for the judge model.
        inference_engine: Whether to run inference with vLLM instead of transformers. Overrides model_engine and judge_engine arguments.
        model_engine: Whether to run model under test inference with vLLM instead of transformers. DO NOT combine with the inference_engine argument.
        judge_engine: Whether to run judge model inference with vLLM instead of transformers. DO NOT combine with the inference_engine argument.
        vllm_config: vLLM-specific configuration (optional). Only used when inference_engine or model_engine/judge_engine is set to "vllm".
        results_dir: Directory where evaluation output files (CSV/JSON) will be saved.
        reasoning: Whether to enable chat-template reasoning (if supported by tokenizer/model).
        trust_remote_code: Whether to trust remote code when loading models.
        sampling_config: Sampling configuration for model inference.
        mlflow_config: MLflow configuration for tracking (optional).
    """

    max_samples: None | int = 500
    batch_size: None | int = None
    sample: bool = False
    use_4bit: bool = False
    device_map: str | dict[str, int] | None = "auto"
    answer_tokens: int = 128
    model_path_or_repo_id: str
    model_token: str | None = None
    judge_batch_size: None | int = None
    judge_output_tokens: int = 32
    judge_path_or_repo_id: str = "google/gemma-3-12b-it"
    judge_token: str | None = None
    sample_judge: bool = False
    use_4bit_judge: bool = False
    inference_engine: Literal["vllm", "transformers"] | None = None
    model_engine: Literal["vllm", "transformers"] = "transformers"
    judge_engine: Literal["vllm", "transformers"] = "transformers"
    vllm_config: VllmConfig | None = None
    results_dir: Path
    reasoning: bool = False
    trust_remote_code: bool = False
    sampling_config: SamplingConfig = SamplingConfig()
    mlflow_config: "MlflowConfig | None" = None

    @model_validator(mode="after")
    def set_judge_token(self):
        if self.judge_token is None:
            self.judge_token = self.model_token
        return self

    @model_validator(mode="after")
    def validate_vllm_config_usage(self):
        """Ensure vllm_config is only provided when using vLLM."""
        if self.vllm_config is not None:
            using_vllm = (
                self.inference_engine == "vllm"
                or self.model_engine == "vllm"
                or self.judge_engine == "vllm"
            )
            if not using_vllm:
                raise ValueError(
                    "vllm_config can only be specified when using vLLM "
                    "(set inference_engine='vllm' or model_engine='vllm' or judge_engine='vllm')"
                )
        return self


class MlflowConfig(BaseModel):
    """
    Configuration for MLflow tracking (optional).

    Keep this separate from the main EvaluationConfig to avoid
    coupling MLflow-specific settings with core evaluation logic.
    """

    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None

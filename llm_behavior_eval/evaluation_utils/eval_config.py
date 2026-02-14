from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic.functional_validators import model_validator

from .sampling_config import SamplingConfig
from .vllm_config import VllmConfig

DEFAULT_JUDGE_MODEL_PATH_OR_REPO_ID = "google/gemma-3-12b-it"


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
        max_answer_tokens: Number of tokens to generate per answer. Typical range is 32-256.
        pass_max_answer_tokens: Whether to pass max_answer_tokens to the model.
        model_path_or_repo_id: HF repo ID or path of the model under test (e.g. "meta-llama/Llama-3.1-8B-Instruct").
        model_tokenizer_path_or_repo_id: Optional tokenizer repo ID or path for the model under test.
            Only used with transformers/vllm engines. Not supported with model_engine='api'.
        lora_path_or_repo_id: Optional LoRA adapter path/repo for evaluated model (vLLM only).
        model_token: HuggingFace token for the model under test.
        judge_batch_size: Batch size for the judge model (free-text tasks only). If None, will be adjusted for GPU limits.
        max_judge_tokens: Number of tokens to generate with the judge model. Typical range is 16-64.
        judge_path_or_repo_id: HF repo ID or path of the judge model (e.g. "meta-llama/Llama-3.3-70B-Instruct").
            When judge_engine="api", this should be an API model identifier (e.g. "openai/gpt-4o-mini").
        judge_tokenizer_path_or_repo_id: Optional tokenizer repo ID or path for the judge model.
            Only used with transformers/vllm engines. Not supported with judge_engine='api'.
        judge_token: HuggingFace token for the judge model. Defaults to the value of `model_token` if not provided.
        sample_judge: Whether to sample outputs from the judge model (True) or generate deterministically (False). Defaults to False.
        use_4bit_judge: Whether to load the judge model in 4-bit mode (using bitsandbytes).
            This is only relevant for the judge model.
        inference_engine: Optional shared backend for both evaluated model and judge.
            Overrides model_engine and judge_engine when provided.
        model_engine: Model inference backend when inference_engine is not provided.
        judge_engine: Judge model inference backend when inference_engine is not provided.
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
    max_answer_tokens: int = 128
    pass_max_answer_tokens: bool = False
    model_path_or_repo_id: str
    model_tokenizer_path_or_repo_id: str | None = None
    lora_path_or_repo_id: str | None = None
    model_token: str | None = None
    judge_batch_size: None | int = None
    max_judge_tokens: int = 32
    judge_path_or_repo_id: str = DEFAULT_JUDGE_MODEL_PATH_OR_REPO_ID
    judge_tokenizer_path_or_repo_id: str | None = None
    judge_token: str | None = None
    sample_judge: bool = False
    use_4bit_judge: bool = False
    inference_engine: Literal["vllm", "transformers", "api"] | None = None
    model_engine: Literal["vllm", "transformers", "api"] = "transformers"
    judge_engine: Literal["vllm", "transformers", "api"] = "transformers"
    vllm_config: VllmConfig | None = None
    results_dir: Path
    reasoning: bool = False
    trust_remote_code: bool = False
    sampling_config: SamplingConfig = SamplingConfig()
    mlflow_config: "MlflowConfig | None" = None
    replace_existing_output: bool = False

    @model_validator(mode="after")
    def set_judge_token(self):
        if self.judge_token is None:
            self.judge_token = self.model_token
        return self

    @model_validator(mode="after")
    def validate_vllm_config_usage(self):
        """Ensure vllm_config is only provided when using vLLM."""
        if self.vllm_config is not None:
            vllm_related_args = [
                self.inference_engine,
                self.model_engine,
                self.judge_engine,
            ]
            using_vllm = any([arg == "vllm" for arg in vllm_related_args])
            if not using_vllm:
                raise ValueError(
                    "vllm_config can only be specified when using vLLM "
                    "(set inference_engine='vllm' or model_engine='vllm' or judge_engine='vllm')"
                )
        return self

    @model_validator(mode="after")
    def validate_api_no_tokenizer(self):
        """Reject local tokenizer overrides when using API engines.

        API providers handle chat template formatting internally, so providing
        local tokenizer overrides is not supported.
        """
        effective_model_engine = self.inference_engine or self.model_engine
        effective_judge_engine = self.inference_engine or self.judge_engine
        if effective_model_engine == "api" and self.model_tokenizer_path_or_repo_id:
            raise ValueError(
                "model_tokenizer_path_or_repo_id cannot be used with model_engine='api'. "
                "API providers handle chat formatting internally."
            )
        if effective_judge_engine == "api" and self.judge_tokenizer_path_or_repo_id:
            raise ValueError(
                "judge_tokenizer_path_or_repo_id cannot be used with judge_engine='api'. "
                "API providers handle chat formatting internally."
            )
        return self

    @model_validator(mode="after")
    def validate_api_judge_model_identifier(self):
        """Require an explicit API-compatible judge model when judge uses API."""
        effective_judge_engine = self.inference_engine or self.judge_engine
        if (
            effective_judge_engine == "api"
            and self.judge_path_or_repo_id == DEFAULT_JUDGE_MODEL_PATH_OR_REPO_ID
        ):
            raise ValueError(
                "judge_path_or_repo_id must be set to an API model identifier when "
                "judge_engine='api' (or inference_engine='api'). "
                "For example: openai/gpt-4o-mini."
            )
        return self

    @model_validator(mode="after")
    def validate_lora_path_or_repo_id(self):
        # LoRA usage currently only supported with vLLM
        using_vllm = self.vllm_config is not None or any(
            [arg == "vllm" for arg in [self.inference_engine, self.model_engine]]
        )
        if self.lora_path_or_repo_id is not None and not using_vllm:
            raise ValueError(
                "LoRA usage currently only supported with vLLM (Either inference_engine or model_engine must be set to 'vllm')"
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

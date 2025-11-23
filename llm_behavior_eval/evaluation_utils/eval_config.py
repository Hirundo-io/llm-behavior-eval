from pathlib import Path

from pydantic import BaseModel
from pydantic.functional_validators import model_validator


class EvaluationConfig(BaseModel):
    """
    Configuration for bias evaluation.

    Args:
        max_samples: Optional limit on the number of examples to process. Use None to evaluate the full set.
        batch_size: Batch size for model inference. Depends on GPU memory (commonly 16-64). If None, will be adjusted for GPU limits.
        sample: Whether to sample outputs (True) or generate deterministically (False).
        use_4bit: Whether to load the model in 4-bit mode (using bitsandbytes).
                 This is only relevant for the model under test.
        judge_type: Metric type to compute. Only JudgeType.BIAS is currently supported.
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
        use_vllm_for_judge: Whether to run judge model inference with vLLM instead of transformers.
        results_dir: Directory where evaluation output files (CSV/JSON) will be saved.
        reasoning: Whether to enable chat-template reasoning (if supported by tokenizer/model).
        use_vllm: Whether to run model inference with vLLM instead of transformers.
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
    use_vllm_for_judge: bool = False
    vllm_judge_max_model_len: int | None = None
    results_dir: Path
    reasoning: bool = False
    use_vllm: bool = False
    vllm_max_model_len: int | None = None

    mlflow_config: "MlflowConfig | None" = None

    @model_validator(mode="after")
    def set_judge_token(self):
        if self.judge_token is None:
            self.judge_token = self.model_token
        return self

    @property
    def trust_remote_code(self) -> bool:
        return self.model_path_or_repo_id.startswith("nvidia/")


class MlflowConfig(BaseModel):
    """
    Configuration for MLflow tracking (optional).

    Keep this separate from the main EvaluationConfig to avoid
    coupling MLflow-specific settings with core evaluation logic.
    """

    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None

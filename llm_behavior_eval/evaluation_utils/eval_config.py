from pathlib import Path

from pydantic import BaseModel


class EvaluationConfig(BaseModel):
    """
    Configuration for bias evaluation.

    Args:
        max_samples: Optional limit on the number of examples to process. Use None to evaluate the full set.
        batch_size: Batch size for model inference. Depends on GPU memory (commonly 16–64). If None, will be adjusted for GPU limits.
        sample: Whether to sample outputs (True) or generate deterministically (False).
        use_4bit: Whether to load the model in 4-bit mode (using bitsandbytes).
                 This is only relevant for the model under test.
        judge_type: Metric type to compute. Only JudgeType.BIAS is currently supported.
        answer_tokens: Number of tokens to generate per answer. Typical range is 32–256.
        model_path_or_repo_id: HF repo ID or path of the model under test (e.g. "meta-llama/Llama-3.1-8B-Instruct").
        judge_batch_size: Batch size for the judge model (free-text tasks only). If None, will be adjusted for GPU limits.
        judge_output_tokens: Number of tokens to generate with the judge model. Typical range is 16–64.
        judge_path_or_repo_id: HF repo ID or path of the judge model (e.g. "meta-llama/Llama-3.3-70B-Instruct").
        use_4bit_judge: Whether to load the judge model in 4-bit mode (using bitsandbytes).
                        This is only relevant for the judge model.
        results_dir: Directory where evaluation output files (CSV/JSON) will be saved.
    """

    max_samples: None | int = 500
    batch_size: None | int = None
    sample: bool = False
    use_4bit: bool = False
    device_map: str | dict[str, int] | None = "auto"
    answer_tokens: int = 128
    model_path_or_repo_id: str
    judge_batch_size: None | int = None
    judge_output_tokens: int = 32
    judge_path_or_repo_id: str = "google/gemma-3-12b-it"
    use_4bit_judge: bool = False
    results_dir: Path

    mlflow_config: "MlflowConfig | None" = None


class MlflowConfig(BaseModel):
    """
    Configuration for MLflow tracking (optional).

    Keep this separate from the main EvaluationConfig to avoid
    coupling MLflow-specific settings with core evaluation logic.
    """

    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None

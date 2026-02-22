import gc
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer

os.environ["TORCHDYNAMO_DISABLE"] = "1"

from llm_behavior_eval import (
    DatasetConfig,
    DatasetType,
    EvaluateFactory,
    EvaluationConfig,
    PreprocessConfig,
    SamplingConfig,
)
from llm_behavior_eval.evaluation_utils.enums import AnswerFormat
from llm_behavior_eval.evaluation_utils.util_functions import (
    empty_cuda_cache_if_available,
)
from llm_behavior_eval.evaluation_utils.vllm_config import VllmConfig
from llm_behavior_eval.evaluation_utils.vllm_types import TokenizerModeOption

torch.set_float32_matmul_precision("high")

BIAS_KINDS = {"bias", "unbias"}
HALUEVAL_ALIAS = {"hallu", "hallucination"}
MEDHALLU_ALIAS = {"hallu-med", "hallucination-med"}
INJECTION_ALIAS = {"prompt-injection"}
TRUSTED_MODEL_PROVIDERS = {
    "hirundo-io",
    "nvidia",
    "meta-llama",
    "google",
    "aisingapore",
    "LGAI-EXAONE",
}
DEFAULT_MAX_SAMPLES = EvaluationConfig.model_fields["max_samples"].default
DEFAULT_BATCH_SIZE = EvaluationConfig.model_fields["batch_size"].default
DEFAULT_USE_4BIT = EvaluationConfig.model_fields["use_4bit"].default
DEFAULT_DEVICE_MAP = EvaluationConfig.model_fields["device_map"].default
DEFAULT_MAX_ANSWER_TOKENS = EvaluationConfig.model_fields["max_answer_tokens"].default
DEFAULT_JUDGE_BATCH_SIZE = EvaluationConfig.model_fields["judge_batch_size"].default
DEFAULT_MAX_JUDGE_TOKENS = EvaluationConfig.model_fields["max_judge_tokens"].default
DEFAULT_SAMPLE_JUDGE = EvaluationConfig.model_fields["sample_judge"].default
DEFAULT_SEED = SamplingConfig.model_fields["seed"].default
DEFAULT_TOP_P = SamplingConfig.model_fields["top_p"].default
DEFAULT_TOP_K = SamplingConfig.model_fields["top_k"].default
DEFAULT_MAX_LORA_RANK = VllmConfig.model_fields["max_lora_rank"].default


def _default_results_dir() -> Path:
    if os.name == "nt":
        base_dir = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base_dir:
            return Path(base_dir) / "llm-behavior-eval" / "results"
        return Path.home() / "AppData" / "Local" / "llm-behavior-eval" / "results"
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "llm-behavior-eval"
            / "results"
        )
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "llm-behavior-eval" / "results"
    return Path.home() / ".local" / "share" / "llm-behavior-eval" / "results"


def _behavior_presets(behavior: str) -> list[str]:
    """
    Map behavior presets to dataset identifiers.

    New formats:
    - BBQ: "bias:<bias_type>" or "unbias:<bias_type>"
    - CBBQ short presets: "cbbq:bias_basic" | "cbbq:bias_all" | "cbbq:unbias_basic" | "cbbq:unbias_all"
    - CBBQ explicit: "cbbq:bias:<bias_type>" or "cbbq:unbias:<bias_type>"
    - UNQOVER: "unqover:bias:<bias_type>" (UNQOVER does not support 'unbias')
    - Hallucinations: "hallu" or "hallu-med"
    - Prompt injection: "prompt-injection"
    """
    behavior_parts = [part.strip().lower() for part in behavior.split(":")]

    # Hallucination shortcuts
    if behavior in HALUEVAL_ALIAS:
        return ["hirundo-io/halueval"]
    if behavior in MEDHALLU_ALIAS:
        return ["hirundo-io/medhallu"]
    if behavior in INJECTION_ALIAS:
        return ["hirundo-io/prompt-injection-purple-llama"]

    # Expected structures:
    # ["cbbq", <bias_basic|bias_all|unbias_basic|unbias_all>] for CBBQ short presets
    # [kind, bias_type] for BBQ, where kind in {bias, unbias}
    #   - bias_type can be a concrete type or 'all'
    # ["cbbq", kind, bias_type] for CBBQ, where kind in {bias, unbias}
    #   - bias_type can be a concrete type or 'all'
    # ["unqover", kind, bias_type] for UNQOVER (kind must be 'bias')
    #   - bias_type can be a concrete type or 'all'
    if len(behavior_parts) == 2 and behavior_parts[0] == "cbbq":
        from llm_behavior_eval.evaluation_utils.enums import (
            CBBQ_ALL_TYPES,
            CBBQ_BASIC_TYPES,
        )

        _, preset = behavior_parts
        if preset not in {"bias_basic", "bias_all", "unbias_basic", "unbias_all"}:
            raise ValueError(
                "For CBBQ short presets use one of: 'cbbq:bias_basic', 'cbbq:bias_all', 'cbbq:unbias_basic', 'cbbq:unbias_all'"
            )
        kind = "unbias" if preset.startswith("unbias") else "bias"
        bias_types = CBBQ_BASIC_TYPES if preset.endswith("basic") else CBBQ_ALL_TYPES
        return [
            f"hirundo-io/cbbq-{bias_type}-{kind}-free-text" for bias_type in bias_types
        ]

    if len(behavior_parts) == 2:
        kind, bias_type = behavior_parts
        if kind not in BIAS_KINDS:
            raise ValueError("For BBQ use 'bias:<bias_type>' or 'unbias:<bias_type>'")
        from llm_behavior_eval.evaluation_utils.enums import BBQ_BIAS_TYPES

        if bias_type == "all":
            return [
                f"hirundo-io/bbq-{bias_type}-{kind}-free-text"
                for bias_type in sorted(BBQ_BIAS_TYPES)
            ]
        if bias_type not in BBQ_BIAS_TYPES:
            allowed = ", ".join(sorted(list(BBQ_BIAS_TYPES)) + ["all"])
            raise ValueError(f"BBQ supports: {allowed}")
        return [f"hirundo-io/bbq-{bias_type}-{kind}-free-text"]

    if len(behavior_parts) == 3 and behavior_parts[0] == "cbbq":
        _, kind, bias_type = behavior_parts
        if kind not in BIAS_KINDS:
            raise ValueError(
                "For CBBQ use 'cbbq:bias:<bias_type>' or 'cbbq:unbias:<bias_type>'"
            )
        from llm_behavior_eval.evaluation_utils.enums import CBBQ_BIAS_TYPES

        normalized_bias_types = {
            bias_type_value.lower(): bias_type_value
            for bias_type_value in CBBQ_BIAS_TYPES
        }
        if bias_type == "all":
            return [
                f"hirundo-io/cbbq-{bias_type_value}-{kind}-free-text"
                for bias_type_value in sorted(CBBQ_BIAS_TYPES, key=str.lower)
            ]
        if bias_type not in normalized_bias_types:
            allowed = ", ".join(sorted(list(CBBQ_BIAS_TYPES)) + ["all"])
            raise ValueError(f"CBBQ supports: {allowed}")
        canonical_bias_type = normalized_bias_types[bias_type]
        return [f"hirundo-io/cbbq-{canonical_bias_type}-{kind}-free-text"]

    if len(behavior_parts) == 3 and behavior_parts[0] == "unqover":
        _, kind, bias_type = behavior_parts
        if kind != "bias":
            raise ValueError(
                "UNQOVER supports only 'bias:<bias_type>' (no 'unbias' for UNQOVER)"
            )
        from llm_behavior_eval.evaluation_utils.enums import UNQOVER_BIAS_TYPES

        if bias_type == "all":
            return [
                f"hirundo-io/unqover-{bt}-{kind}-free-text"
                for bt in sorted(UNQOVER_BIAS_TYPES)
            ]
        if bias_type not in UNQOVER_BIAS_TYPES:
            allowed = ", ".join(sorted(list(UNQOVER_BIAS_TYPES)) + ["all"])
            raise ValueError(f"UNQOVER supports: {allowed}")
        return [f"hirundo-io/unqover-{bias_type}-{kind}-free-text"]

    raise ValueError(
        "--behavior must be 'bias:<type|all>' | 'unbias:<type|all>' | 'cbbq:bias_basic|bias_all|unbias_basic|unbias_all' | 'cbbq:bias:<type|all>' | 'cbbq:unbias:<type|all>' | 'unqover:bias:<type|all>' | 'hallu' | 'hallu-med' | 'prompt-injection'"
    )


def _infer_dataset_type(file_path: str) -> DatasetType:
    if "-unbias-" in file_path:
        return DatasetType.UNBIAS
    return DatasetType.BIAS


def main(
    model: Annotated[
        str,
        typer.Argument(
            help="Model repo id or path, e.g. meta-llama/Llama-3.1-8B-Instruct"
        ),
    ],
    behavior: Annotated[
        str,
        typer.Argument(
            help="Behavior preset(s). Can be comma-separated for multiple behaviors. BBQ: 'bias:<type>' or 'unbias:<type>'; CBBQ short: 'cbbq:bias_basic'|'cbbq:bias_all'|'cbbq:unbias_basic'|'cbbq:unbias_all'; CBBQ explicit: 'cbbq:bias:<type>' or 'cbbq:unbias:<type>'; UNQOVER: 'unqover:bias:<type>'; Hallucination: 'hallu' | 'hallu-med'; Prompt injection: 'prompt-injection'"
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help=(
                "Output directory for evaluation results (optional). "
                "Defaults to: macOS ~/Library/Application Support/llm-behavior-eval/results; "
                "Linux $XDG_DATA_HOME/llm-behavior-eval/results (or ~/.local/share/llm-behavior-eval/results); "
                "Windows %LOCALAPPDATA%\\llm-behavior-eval\\results."
            ),
        ),
    ] = None,
    model_token: Annotated[
        str | None,
        typer.Option(
            "--model-token", help="HuggingFace token for the model (optional)"
        ),
    ] = None,
    judge_token: Annotated[
        str | None,
        typer.Option(
            "--judge-token", help="HuggingFace token for the judge model (optional)"
        ),
    ] = None,
    judge_model: Annotated[
        str,
        typer.Option("--judge-model", help="Judge repo id or path (optional)"),
    ] = "google/gemma-3-12b-it",
    use_mlflow: Annotated[
        bool,
        typer.Option(
            "--use-mlflow", help="Enable MLflow tracking for this evaluation run"
        ),
    ] = False,
    mlflow_tracking_uri: Annotated[
        str | None,
        typer.Option("--mlflow-tracking-uri", help="MLflow tracking URI (optional)"),
    ] = None,
    mlflow_experiment_name: Annotated[
        str | None,
        typer.Option(
            "--mlflow-experiment-name", help="MLflow experiment name (optional)"
        ),
    ] = None,
    mlflow_run_name: Annotated[
        str | None,
        typer.Option(
            "--mlflow-run-name",
            help="MLflow run name (optional, auto-generates if not specified)",
        ),
    ] = None,
    lora_path_or_repo_id: Annotated[
        str | None,
        typer.Option(
            "--lora-path-or-repo-id",
            help="LoRA path or repo ID (optional), can be local path, HF repo or remote location with a valid scheme: mlflow://<run_id>/[<artifact_path>], git://<repo_url>[#<rev>[:<subdir>]], s3://<bucket>/<path>, gs://<bucket>/<path>",
        ),
    ] = None,
    inference_engine: Annotated[
        Literal["vllm", "transformers"] | None,
        typer.Option(
            "--inference-engine",
            help="""Inference engine to use for model and judge inference. "vllm" or "transformers". Overrides model_engine and judge_engine arguments.""",
        ),
    ] = None,
    trust_remote_code: Annotated[
        bool | None,
        typer.Option(
            "--trust-remote-code/--no-trust-remote-code",
            help=(
                "Trust remote code when loading models. "
                "Automatically set to True for providers defined in TRUSTED_MODEL_PROVIDERS."
            ),
        ),
    ] = None,
    model_engine: Annotated[
        Literal["vllm", "transformers"],
        typer.Option(
            "--model-engine",
            help="""Model engine to use for model inference. "vllm" or "transformers". DO NOT combine with the inference_engine argument.""",
        ),
    ] = "transformers",
    vllm_max_model_len: Annotated[
        int | None,
        typer.Option(
            "--vllm-max-model-len",
            help="Maximum model length for vLLM (optional)",
        ),
    ] = None,
    judge_engine: Annotated[
        Literal["vllm", "transformers"],
        typer.Option(
            "--judge-engine",
            help="""Judge engine to use for judge model inference. "vllm" or "transformers". DO NOT combine with the inference_engine argument.""",
        ),
    ] = "transformers",
    vllm_judge_max_model_len: Annotated[
        int | None,
        typer.Option(
            "--vllm-judge-max-model-len",
            help="Maximum model length for vLLM judge (optional). Defaults to the same value as model inference",
        ),
    ] = None,
    vllm_tokenizer_mode: Annotated[
        TokenizerModeOption | None,
        typer.Option(
            "--vllm-tokenizer-mode",
            help="Tokenizer mode forwarded to vLLM (e.g. 'auto', 'slow').",
        ),
    ] = None,
    vllm_config_format: Annotated[
        str | None,
        typer.Option(
            "--vllm-config-format",
            help="Model config format hint forwarded to vLLM.",
        ),
    ] = "auto",
    vllm_load_format: Annotated[
        str | None,
        typer.Option(
            "--vllm-load-format",
            help="Checkpoint load format hint forwarded to vLLM.",
        ),
    ] = "auto",
    vllm_gpu_memory_utilization: Annotated[
        float,
        typer.Option(
            "--vllm-gpu-memory-utilization",
            help="GPU memory utilization for vLLM (must be between 0 and 1).",
            min=0.001,
            max=1.0,
        ),
    ] = 0.9,
    vllm_enforce_eager: Annotated[
        bool,
        typer.Option(
            "--vllm-enforce-eager",
            help="Enforce eager execution for vLLM.",
        ),
    ] = False,
    vllm_max_lora_rank: Annotated[
        int,
        typer.Option(
            "--vllm-max-lora-rank",
            help="Maximum LoRA rank for vLLM.",
        ),
    ] = DEFAULT_MAX_LORA_RANK,
    replace_existing_output: Annotated[
        bool,
        typer.Option(
            "--replace-existing-output/--no-replace-existing-output",
            help=(
                "Replace any existing evaluation outputs when the configuration "
                "differs from previous runs. Defaults to keeping existing "
                "results."
            ),
        ),
    ] = False,
    reasoning: Annotated[
        bool,
        typer.Option(
            "--reasoning/--no-reasoning",
            help="Enable chat-template reasoning if supported",
        ),
    ] = False,
    max_samples: Annotated[
        int,
        typer.Option(
            "--max-samples",
            help=(
                "Maximum number of samples to evaluate per dataset. "
                "Use a value <= 0 to run the full dataset."
            ),
            show_default=str(DEFAULT_MAX_SAMPLES),
        ),
    ] = DEFAULT_MAX_SAMPLES,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            help="Batch size for model inference. If None, will be adjusted for GPU limits.",
        ),
    ] = DEFAULT_BATCH_SIZE,
    use_4bit: Annotated[
        bool,
        typer.Option(
            "--use-4bit/--no-use-4bit",
            help="Load the model in 4-bit mode (using bitsandbytes).",
        ),
    ] = DEFAULT_USE_4BIT,
    device_map: Annotated[
        str | None,
        typer.Option(
            "--device-map",
            help="Device map for model inference. If None, will be set to 'auto'.",
        ),
    ] = DEFAULT_DEVICE_MAP,
    judge_batch_size: Annotated[
        int | None,
        typer.Option(
            "--judge-batch-size",
            help="Batch size for the judge model. If None, will be adjusted for GPU limits.",
        ),
    ] = DEFAULT_JUDGE_BATCH_SIZE,
    sample_judge: Annotated[
        bool,
        typer.Option(
            "--sample-judge/--no-sample-judge",
            help="Whether to sample outputs from the judge model.",
        ),
    ] = DEFAULT_SAMPLE_JUDGE,
    use_4bit_judge: Annotated[
        bool,
        typer.Option(
            "--use-4bit-judge/--no-use-4bit-judge",
            help="Load the judge model using 4-bit quantization (bitsandbytes).",
        ),
    ] = False,
    sample: Annotated[
        bool | None,
        typer.Option(
            "--sample/--no-sample",
            help="Whether to sample from the model. DO NOT combine with the temperature parameter.",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--temperature",
            help="The temperature for sampling. DO NOT combine with the do_sample parameter.",
        ),
    ] = None,
    top_p: Annotated[
        float,
        typer.Option(
            "--top-p",
            help="The top-p value for sampling.",
        ),
    ] = DEFAULT_TOP_P,
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            help="The top-k value for sampling.",
        ),
    ] = DEFAULT_TOP_K,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed for the evaluation.",
        ),
    ] = DEFAULT_SEED,
    max_answer_tokens: Annotated[
        int,
        typer.Option(
            "--max-answer-tokens",
            help="Maximum number of tokens to generate per answer.",
            show_default=str(DEFAULT_MAX_ANSWER_TOKENS),
        ),
    ] = DEFAULT_MAX_ANSWER_TOKENS,
    pass_max_answer_tokens: Annotated[
        bool,
        typer.Option(
            "--pass-max-answer-tokens/--no-pass-max-answer-tokens",
            help="Pass max_answer_tokens to the model.",
        ),
    ] = False,
    max_judge_tokens: Annotated[
        int,
        typer.Option(
            "--max-judge-tokens",
            help="Maximum number of tokens to generate with the judge model.",
            show_default=str(DEFAULT_MAX_JUDGE_TOKENS),
        ),
    ] = DEFAULT_MAX_JUDGE_TOKENS,
) -> None:
    model_path_or_repo_id = model
    judge_path_or_repo_id = judge_model
    result_dir = output_dir if output_dir is not None else _default_results_dir()
    # Split behavior by commas and collect all file paths
    behaviors = [behavior.strip() for behavior in behavior.split(",")]
    file_paths = []
    for behavior in behaviors:
        file_paths.extend(_behavior_presets(behavior))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Compose MLflow config separately
    if use_mlflow or mlflow_tracking_uri or mlflow_experiment_name or mlflow_run_name:
        from llm_behavior_eval.evaluation_utils.eval_config import MlflowConfig

        mlflow_config = MlflowConfig(
            mlflow_tracking_uri=mlflow_tracking_uri
            or os.environ.get("MLFLOW_TRACKING_URI"),
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_run_name=mlflow_run_name,
        )
    else:
        mlflow_config = None

    # Compose vLLM config separately, only if using vLLM
    vllm_related_args = [inference_engine, model_engine, judge_engine]
    using_vllm = any([arg == "vllm" for arg in vllm_related_args])
    if using_vllm:
        vllm_config = VllmConfig(
            max_model_len=vllm_max_model_len,
            judge_max_model_len=vllm_judge_max_model_len
            if vllm_judge_max_model_len is not None
            else vllm_max_model_len,
            tokenizer_mode=vllm_tokenizer_mode,
            config_format=vllm_config_format,
            load_format=vllm_load_format,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            enable_lora=lora_path_or_repo_id is not None,
            max_lora_rank=vllm_max_lora_rank,
            enforce_eager=vllm_enforce_eager,
        )
    else:
        vllm_config = None

    eval_config = EvaluationConfig(
        model_path_or_repo_id=model_path_or_repo_id,
        model_token=model_token,
        lora_path_or_repo_id=lora_path_or_repo_id,
        judge_path_or_repo_id=judge_path_or_repo_id,
        judge_token=judge_token,
        results_dir=result_dir,
        mlflow_config=mlflow_config,
        vllm_config=vllm_config,
        reasoning=reasoning,
        trust_remote_code=trust_remote_code
        if trust_remote_code is not None
        else model_path_or_repo_id.split("/")[0] in TRUSTED_MODEL_PROVIDERS,
        inference_engine=inference_engine,
        model_engine=model_engine,
        judge_engine=judge_engine,
        replace_existing_output=replace_existing_output,
        max_samples=None if max_samples <= 0 else max_samples,
        batch_size=batch_size,
        use_4bit=use_4bit,
        device_map=device_map,
        max_answer_tokens=max_answer_tokens,
        pass_max_answer_tokens=pass_max_answer_tokens,
        judge_batch_size=judge_batch_size,
        max_judge_tokens=max_judge_tokens,
        sample_judge=sample_judge,
        use_4bit_judge=use_4bit_judge,
        sampling_config=SamplingConfig(
            do_sample=sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        ),
    )

    evaluator = None
    generation_lists = []
    dataset_configs = []
    evaluation_error = True
    try:
        # generation loop
        try:
            for file_path in file_paths:
                logging.info(
                    f"Evaluating {file_path} with {model_path_or_repo_id}"
                    + (
                        f" and LoRA from {lora_path_or_repo_id}"
                        if lora_path_or_repo_id is not None
                        else ""
                    )
                )
                dataset_config = DatasetConfig(
                    file_path=file_path,
                    dataset_type=_infer_dataset_type(file_path),
                    preprocess_config=PreprocessConfig(),
                    answer_format=(
                        AnswerFormat.MULTIPLE_CHOICE
                        if "/cbbq-" in file_path
                        else AnswerFormat.FREE_TEXT
                    ),
                    seed=seed,
                )
                if evaluator is None:
                    evaluator = EvaluateFactory.create_evaluator(
                        eval_config, dataset_config
                    )
                else:
                    evaluator.update_dataset_config(dataset_config)

                dataset_configs.append(dataset_config)
                generation_lists.append(evaluator.generate())
        finally:
            if evaluator is not None:
                evaluator.free_test_model()

        if evaluator is None:
            # Type-checking hint
            raise ValueError("Evaluator does not exist.")

        # Grading loop
        with evaluator.get_grading_context() as judge:
            for generations, dataset_config, file_path in zip(
                generation_lists, dataset_configs, file_paths, strict=True
            ):
                logging.info("Grading %s with %s", file_path, judge_path_or_repo_id)
                evaluator.update_dataset_config(dataset_config)
                with evaluator.dataset_mlflow_run():
                    evaluator.grade(generations, judge)
        evaluation_error = False
    finally:
        if evaluator is not None:
            evaluator.cleanup(error=evaluation_error)
        del evaluator
        gc.collect()
        empty_cuda_cache_if_available()


app = typer.Typer()
app.command()(main)

if __name__ == "__main__":
    app()

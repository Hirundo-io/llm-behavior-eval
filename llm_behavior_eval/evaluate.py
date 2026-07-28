import gc
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer

os.environ["TORCHDYNAMO_DISABLE"] = "1"

from llm_behavior_eval.evaluation_utils.dataset_config import (
    DatasetConfig,
    PreprocessConfig,
)
from llm_behavior_eval.evaluation_utils.enums import (
    BBQ_BIAS_BEHAVIOR,
    BEHAVIOR_PRESET_ERROR,
    BLOOM_BIAS_TYPES,
    HALUEVAL_ALIAS,
    INJECTION_ALIAS,
    MEDHALLU_ALIAS,
    REFUSAL_ALIAS,
    THREE_PART_BIAS_BEHAVIORS,
    TRUSTED_MODEL_PROVIDERS,
    DatasetType,
)
from llm_behavior_eval.evaluation_utils.eval_config import (
    FAMILY_TOKEN_DEFAULTS,
    EvaluationConfig,
    EvaluatorFamily,
)
from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig
from llm_behavior_eval.evaluation_utils.util_functions import (
    empty_cuda_cache_if_available,
)
from llm_behavior_eval.evaluation_utils.vllm_config import (
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_VLLM_MAX_MODEL_LEN,
    VllmConfig,
)
from llm_behavior_eval.evaluation_utils.vllm_types import TokenizerModeOption
from llm_behavior_eval.presets import build_bias_dataset_id, expand_dataset_preset

torch.set_float32_matmul_precision("high")

DEFAULT_MAX_SAMPLES = EvaluationConfig.model_fields["max_samples"].default
DEFAULT_BATCH_SIZE = EvaluationConfig.model_fields["batch_size"].default
DEFAULT_USE_4BIT = EvaluationConfig.model_fields["use_4bit"].default
DEFAULT_DEVICE_MAP = EvaluationConfig.model_fields["device_map"].default
DEFAULT_MAX_ANSWER_TOKENS = FAMILY_TOKEN_DEFAULTS["bias"]["max_answer_tokens"]
DEFAULT_JUDGE_BATCH_SIZE = EvaluationConfig.model_fields["judge_batch_size"].default
DEFAULT_MAX_JUDGE_TOKENS = FAMILY_TOKEN_DEFAULTS["bias"]["max_judge_tokens"]
DEFAULT_SAMPLE_JUDGE = FAMILY_TOKEN_DEFAULTS["bias"]["sample_judge"]
DEFAULT_SEED = SamplingConfig.model_fields["seed"].default
DEFAULT_TOP_P = SamplingConfig.model_fields["top_p"].default
DEFAULT_TOP_K = SamplingConfig.model_fields["top_k"].default
DEFAULT_MAX_LORA_RANK = VllmConfig.model_fields["max_lora_rank"].default


def _resolve_bias_behavior(
    prefix: str,
    kind: str,
    bias_type: str,
    allowed_types: set[str],
    allowed_kinds: set[str],
    kind_error: str,
    support_label: str,
) -> list[str]:
    if kind not in allowed_kinds:
        raise ValueError(kind_error)

    allowed_with_all = ", ".join(sorted(list(allowed_types)) + ["all"])
    if bias_type == "all":
        return [
            build_bias_dataset_id(prefix, allowed_bias_type, kind)
            for allowed_bias_type in sorted(allowed_types)
        ]

    if bias_type not in allowed_types:
        raise ValueError(f"{support_label} supports: {allowed_with_all}")
    return [build_bias_dataset_id(prefix, bias_type, kind)]


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
    Map behavior presets to dataset identifiers (free‑text only).

    New formats:
    - BBQ: "bias:<bias_type>" or "unbias:<bias_type>"
    - UNQOVER: "unqover:bias:<bias_type>" (UNQOVER does not support 'unbias')
    - Bloom: "bloom:bias:<bias_type>" or "bloom:unbias:<bias_type>"
    - Hallucinations: "hallu" or "hallu-med"
    - Prompt injection: "prompt-injection"
    - Refusal: "refusal:xstest" | "refusal:orbench" | "refusal:all"
    """
    behavior_parts = [part.strip().lower() for part in behavior.split(":")]

    # Hallucination shortcuts
    if behavior in HALUEVAL_ALIAS:
        return expand_dataset_preset("hallu")
    if behavior in MEDHALLU_ALIAS:
        return expand_dataset_preset("hallu-med")
    if behavior in INJECTION_ALIAS:
        return expand_dataset_preset("prompt-injection")
    if len(behavior_parts) == 2 and behavior_parts[0] in REFUSAL_ALIAS:
        _, refusal_dataset = behavior_parts
        if refusal_dataset not in {"xstest", "orbench", "all"}:
            raise ValueError("Refusal supports: xstest, orbench, all")
        return expand_dataset_preset(f"refusal:{refusal_dataset}")

    # Expected structures:
    # [kind, bias_type] for BBQ, where kind in {bias, unbias}
    #   - bias_type can be a concrete type or 'all'
    # ["unqover", kind, bias_type] for UNQOVER (kind must be 'bias')
    #   - bias_type can be a concrete type or 'all'
    # ["bloom", kind, bias_type] for Bloom, where kind in {bias, unbias}
    #   - bias_type can be a concrete type or 'all'
    # ["bloom", "bias", bias_type, "ambiguous"] for Bloom ambiguous-only bias
    if len(behavior_parts) == 2:
        kind, bias_type = behavior_parts
        allowed_types, allowed_kinds, kind_error, support_label = BBQ_BIAS_BEHAVIOR
        _resolve_bias_behavior(
            "bbq",
            kind,
            bias_type,
            allowed_types,
            allowed_kinds,
            kind_error,
            support_label,
        )
        return expand_dataset_preset(f"{kind}:{bias_type}")

    if len(behavior_parts) == 3:
        prefix, kind, bias_type = behavior_parts
        if prefix not in THREE_PART_BIAS_BEHAVIORS:
            raise ValueError(BEHAVIOR_PRESET_ERROR)

        allowed_types, allowed_kinds, kind_error, support_label = (
            THREE_PART_BIAS_BEHAVIORS[prefix]
        )
        _resolve_bias_behavior(
            prefix,
            kind,
            bias_type,
            allowed_types,
            allowed_kinds,
            kind_error,
            support_label,
        )
        return expand_dataset_preset(f"{prefix}:{kind}:{bias_type}")

    if len(behavior_parts) == 4:
        prefix, kind, bias_type, context = behavior_parts
        if (
            prefix == "bloom"
            and kind == "bias"
            and context == "ambiguous"
            and bias_type in BLOOM_BIAS_TYPES
        ):
            return expand_dataset_preset(f"bloom:bias:{bias_type}:ambiguous")
        raise ValueError("Use 'bloom:bias:<type>:ambiguous'")

    raise ValueError(BEHAVIOR_PRESET_ERROR)


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
            help="Behavior preset(s). Can be comma-separated for multiple behaviors. BBQ: 'bias:<type>' or 'unbias:<type>'; UNQOVER: 'unqover:bias:<type>'; Bloom: 'bloom:bias:<type|all>' or 'bloom:unbias:<type|all>'; Hallucination: 'hallu' | 'hallu-med'; Prompt injection: 'prompt-injection'; Refusal: 'refusal:xstest' | 'refusal:orbench' | 'refusal:all'"
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--base-output-dir",
            help=(
                "Output directory for evaluation results (optional). "
                "Defaults to: macOS ~/Library/Application Support/llm-behavior-eval/results; "
                "Linux $XDG_DATA_HOME/llm-behavior-eval/results (or ~/.local/share/llm-behavior-eval/results); "
                "Windows %LOCALAPPDATA%\\llm-behavior-eval\\results."
            ),
        ),
    ] = None,
    model_output_dir: Annotated[
        str | None,
        typer.Option(
            "--model-output-dir",
            help=(
                "Override the model output directory name used under --base-output-dir. "
                "By default this is derived from the model (and LoRA, if provided)."
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
        typer.Option(
            "--mlflow-tracking-uri",
            help="MLflow tracking URI (optional). For auth, set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD environment variables or use --mlflow-username / --mlflow-password.",
        ),
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
    mlflow_run_id: Annotated[
        str | None,
        typer.Option(
            "--mlflow-run-id",
            help="Existing MLflow run ID to log metrics and upload artifacts to (instead of creating a new run).",
        ),
    ] = None,
    mlflow_username: Annotated[
        str | None,
        typer.Option(
            "--mlflow-username",
            help="MLflow tracking server username (optional). Overrides MLFLOW_TRACKING_USERNAME env var.",
        ),
    ] = None,
    mlflow_password: Annotated[
        str | None,
        typer.Option(
            "--mlflow-password",
            help="MLflow tracking server password (optional). Overrides MLFLOW_TRACKING_PASSWORD env var.",
        ),
    ] = None,
    mlflow_artifact_path_subfolder: Annotated[
        str | None,
        typer.Option(
            "--mlflow-artifact-path-subfolder",
            help='Optional subfolder for MLflow artifact path. Use "timestamp" to auto pre-append current time. If set to any other value, the artifact folder will be "llm-behavior-eval/{mlflow_artifact_path_subfolder}" and if not set (i.e. `None`) then no folder will be set',
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
    ] = DEFAULT_VLLM_MAX_MODEL_LEN,
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
    ] = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
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
    vllm_language_model_only: Annotated[
        bool,
        typer.Option(
            "--vllm-language-model-only",
            help="Load only the language model for vLLM.",
        ),
    ] = False,
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
    enable_thinking: Annotated[
        bool,
        typer.Option(
            "--thinking-on/--thinking-off",
            help="Enable thinking (if supported by the tokenizer and model)",
        ),
    ] = False,
    enable_thinking_arg_name: Annotated[
        str | None,
        typer.Option(
            "--enable-thinking-arg-name",
            help="Enable thinking argument name in tokenizer's `apply_chat_template` (e.g. 'enable_thinking').",
        ),
    ] = None,
    thinking_start_token: Annotated[
        str | None,
        typer.Option(
            "--thinking-start-token",
            help=(
                "Thinking start token to use for the model (e.g. '<think>')."
                "Must be specified when running with `--exclude-thinking-trace-for-judge`."
            ),
        ),
    ] = None,
    thinking_end_token: Annotated[
        str | None,
        typer.Option(
            "--thinking-end-token",
            help=(
                "Thinking end token to use for the model (e.g. '</think>'). "
                "Must be specified when running with `--exclude-thinking-trace-for-judge`."
            ),
        ),
    ] = None,
    exclude_thinking_trace_for_judge: Annotated[
        bool,
        typer.Option(
            "--exclude-thinking-trace-for-judge/--include-thinking-trace-for-judge",
            help="Exclude thinking trace from judgement.",
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
        bool | None,
        typer.Option(
            "--sample-judge/--no-sample-judge",
            help="Whether to sample outputs from the judge model.",
            show_default=str(DEFAULT_SAMPLE_JUDGE),
        ),
    ] = None,
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
        int | None,
        typer.Option(
            "--max-answer-tokens",
            help="Maximum number of tokens to generate per answer.",
            show_default=str(DEFAULT_MAX_ANSWER_TOKENS),
        ),
    ] = None,
    pass_max_answer_tokens: Annotated[
        bool,
        typer.Option(
            "--pass-max-answer-tokens/--no-pass-max-answer-tokens",
            help="Pass max_answer_tokens to the model.",
        ),
    ] = False,
    max_judge_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-judge-tokens",
            help="Maximum number of tokens to generate with the judge model.",
            show_default=str(DEFAULT_MAX_JUDGE_TOKENS),
        ),
    ] = None,
) -> None:
    model_path_or_repo_id = model
    judge_path_or_repo_id = judge_model
    result_dir = output_dir if output_dir is not None else _default_results_dir()
    # Split behavior by commas and collect all file paths
    behaviors = [behavior.strip() for behavior in behavior.split(",")]
    file_paths = []
    for behavior in behaviors:
        file_paths.extend(_behavior_presets(behavior))
    evaluator_families: set[EvaluatorFamily] = {
        EvaluateFactory.get_evaluator_family(file_path) for file_path in file_paths
    }
    if len(evaluator_families) > 1:
        # TODO: Support mixed evaluator families by instantiating a separate
        # evaluator per dataset instead of reusing one evaluator across the full run.
        raise ValueError(
            "Cannot evaluate behaviors from multiple evaluator families in one invocation."
        )
    evaluator_family: EvaluatorFamily | None = next(iter(evaluator_families), None)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Compose MLflow config separately
    if (
        use_mlflow
        or mlflow_tracking_uri
        or mlflow_experiment_name
        or mlflow_run_name
        or mlflow_run_id
    ):
        # Set MLflow auth env vars so the client can authenticate (MLflow reads MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD)
        if mlflow_username is not None:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        if mlflow_password is not None:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
        from llm_behavior_eval.evaluation_utils.eval_config import MlflowConfig

        mlflow_config = MlflowConfig(
            mlflow_tracking_uri=mlflow_tracking_uri
            or os.environ.get("MLFLOW_TRACKING_URI"),
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_run_name=mlflow_run_name,
            mlflow_run_id=mlflow_run_id,
            mlflow_artifact_path_subfolder=mlflow_artifact_path_subfolder,
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
            language_model_only=vllm_language_model_only,
            enforce_eager=vllm_enforce_eager,
        )
    else:
        vllm_config = None

    eval_config = EvaluationConfig(
        model_path_or_repo_id=model_path_or_repo_id,
        model_output_dir=model_output_dir,
        model_token=model_token,
        lora_path_or_repo_id=lora_path_or_repo_id,
        judge_path_or_repo_id=judge_path_or_repo_id,
        judge_token=judge_token,
        results_dir=result_dir,
        mlflow_config=mlflow_config,
        vllm_config=vllm_config,
        enable_thinking=enable_thinking,
        enable_thinking_arg_name=enable_thinking_arg_name,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
        exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
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
        evaluator_family=evaluator_family,
    )

    evaluator = None
    generation_lists = []
    dataset_configs = []
    dataset_file_paths = []
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
                    dataset_type=DatasetType.UNBIAS
                    if "-unbias-" in file_path
                    else DatasetType.BIAS,
                    preprocess_config=PreprocessConfig(),
                    seed=seed,
                )
                if evaluator is None:
                    evaluator = EvaluateFactory.create_evaluator(
                        eval_config, dataset_config
                    )
                else:
                    evaluator.update_dataset_config(dataset_config)

                generation_records = evaluator.generate()

                dataset_configs.append(dataset_config)
                dataset_file_paths.append(file_path)
                generation_lists.append(generation_records)
        finally:
            if evaluator is not None:
                evaluator.free_test_model()

        if evaluator is None:
            logging.warning(
                "Evaluator could not be created; no datasets were evaluated. "
                "See logs above for details."
            )
            evaluation_error = False
            return

        # Grading loop
        with evaluator.get_grading_context() as judge:
            for generations, dataset_config, file_path in zip(
                generation_lists, dataset_configs, dataset_file_paths, strict=True
            ):
                logging.info("Grading %s with %s", file_path, judge_path_or_repo_id)
                evaluator.update_dataset_config(dataset_config)
                with evaluator.dataset_mlflow_run():
                    evaluator.grade(generations, judge)
        evaluation_error = False
    finally:
        if evaluator is not None and evaluator.started_mlflow_run:
            evaluator.cleanup(error=evaluation_error)
        del evaluator
        gc.collect()
        empty_cuda_cache_if_available()


app = typer.Typer()
app.command()(main)

if __name__ == "__main__":
    app()

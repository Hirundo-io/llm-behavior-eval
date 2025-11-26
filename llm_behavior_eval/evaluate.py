import gc
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer
from transformers.trainer_utils import set_seed

os.environ["TORCHDYNAMO_DISABLE"] = "1"

from llm_behavior_eval import (
    DatasetConfig,
    DatasetType,
    EvaluateFactory,
    EvaluationConfig,
    PreprocessConfig,
    SamplingConfig,
)
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
DEFAULT_MAX_SAMPLES = EvaluationConfig.model_fields["max_samples"].default
DEFAULT_SEED = SamplingConfig.model_fields["seed"].default
DEFAULT_TOP_P = SamplingConfig.model_fields["top_p"].default
DEFAULT_TOP_K = SamplingConfig.model_fields["top_k"].default


def _behavior_presets(behavior: str) -> list[str]:
    """
    Map behavior presets to dataset identifiers (free‑text only).

    New formats:
    - BBQ: "bias:<bias_type>" or "unbias:<bias_type>"
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
    # [kind, bias_type] for BBQ, where kind in {bias, unbias}
    #   - bias_type can be a concrete type or 'all'
    # ["unqover", kind, bias_type] for UNQOVER (kind must be 'bias')
    #   - bias_type can be a concrete type or 'all'
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

    if len(behavior_parts) == 3 and behavior_parts[0] == "unqover":
        _, kind, bias_type = behavior_parts
        if kind != "bias":
            raise ValueError(
                "UNQOVER supports only 'bias:<bias_type>' (no 'unbias' for UNQOVER)"
            )
        from llm_behavior_eval.evaluation_utils.enums import UNQOVER_BIAS_TYPES

        if bias_type == "all":
            return [
                f"unqover/unqover-{bt}-{kind}-free-text"
                for bt in sorted(UNQOVER_BIAS_TYPES)
            ]
        if bias_type not in UNQOVER_BIAS_TYPES:
            allowed = ", ".join(sorted(list(UNQOVER_BIAS_TYPES)) + ["all"])
            raise ValueError(f"UNQOVER supports: {allowed}")
        return [f"unqover/unqover-{bias_type}-{kind}-free-text"]

    raise ValueError(
        "--behavior must be 'bias:<type|all>' | 'unbias:<type|all>' | 'unqover:bias:<type|all>' | 'hallu' | 'hallu-med' | 'prompt-injection'"
    )


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
            help="Behavior preset. BBQ: 'bias:<type>' or 'unbias:<type>'; UNQOVER: 'unqover:bias:<type>'; Hallucination: 'hallu' | 'hallu-med'"
        ),
    ],
    output_dir: Annotated[
        str | None,
        typer.Option(
            "--output-dir", help="Output directory for evaluation results (optional)"
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
                "Automatically set to True for NVIDIA models on huggingface."
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
    ] = None,
    vllm_load_format: Annotated[
        str | None,
        typer.Option(
            "--vllm-load-format",
            help="Checkpoint load format hint forwarded to vLLM.",
        ),
    ] = None,
    vllm_tool_call_parser: Annotated[
        str | None,
        typer.Option(
            "--vllm-tool-call-parser",
            help="Tool-call parser identifier forwarded to vLLM.",
        ),
    ] = None,
    vllm_enable_auto_tool_choice: Annotated[
        bool | None,
        typer.Option(
            "--vllm-enable-auto-tool-choice/--no-vllm-enable-auto-tool-choice",
            help=(
                "Enable vLLM automatic tool selection (leave unset to keep vLLM default)."
            ),
        ),
    ] = None,
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
) -> None:
    model_path_or_repo_id = model
    judge_path_or_repo_id = judge_model
    result_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).parent / "results"
    )
    file_paths = _behavior_presets(behavior)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for file_path in file_paths:
        logging.info("Evaluating %s with %s", file_path, model_path_or_repo_id)
        dataset_config = DatasetConfig(
            file_path=file_path,
            dataset_type=DatasetType.UNBIAS
            if "-unbias-" in file_path
            else DatasetType.BIAS,
            preprocess_config=PreprocessConfig(),
            seed=seed,
        )

        # Compose MLflow config separately
        if (
            use_mlflow
            or mlflow_tracking_uri
            or mlflow_experiment_name
            or mlflow_run_name
        ):
            from llm_behavior_eval.evaluation_utils.eval_config import MlflowConfig

            mlflow_config = MlflowConfig(
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_experiment_name=mlflow_experiment_name,
                mlflow_run_name=mlflow_run_name,
            )
        else:
            mlflow_config = None

        # Compose vLLM config separately, only if using vLLM
        using_vllm = (
            inference_engine == "vllm"
            or model_engine == "vllm"
            or judge_engine == "vllm"
        )
        if using_vllm:
            vllm_config = VllmConfig(
                max_model_len=vllm_max_model_len,
                judge_max_model_len=vllm_judge_max_model_len
                if vllm_judge_max_model_len is not None
                else vllm_max_model_len,
                tokenizer_mode=vllm_tokenizer_mode,
                config_format=vllm_config_format,
                load_format=vllm_load_format,
                tool_call_parser=vllm_tool_call_parser,
                enable_auto_tool_choice=vllm_enable_auto_tool_choice,
            )
        else:
            vllm_config = None

        eval_config = EvaluationConfig(
            model_path_or_repo_id=model_path_or_repo_id,
            model_token=model_token,
            judge_path_or_repo_id=judge_path_or_repo_id,
            judge_token=judge_token,
            results_dir=result_dir,
            mlflow_config=mlflow_config,
            vllm_config=vllm_config,
            reasoning=reasoning,
            trust_remote_code=trust_remote_code
            if trust_remote_code is not None
            else model_path_or_repo_id.startswith("nvidia/"),
            # ⬆️ Default logic: trust remote code for NVIDIA models on huggingface
            inference_engine=inference_engine,
            model_engine=model_engine,
            judge_engine=judge_engine,
            max_samples=None if max_samples <= 0 else max_samples,
            use_4bit_judge=use_4bit_judge,
            sampling_config=SamplingConfig(
                do_sample=sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            ),
        )
        if dataset_config.seed is not None:
            set_seed(dataset_config.seed)
        elif eval_config.sampling_config.seed is not None:
            set_seed(eval_config.sampling_config.seed)
        evaluator = EvaluateFactory.create_evaluator(eval_config, dataset_config)
        try:
            with torch.inference_mode():
                evaluator.evaluate()
        finally:
            del evaluator
            gc.collect()
            empty_cuda_cache_if_available()


app = typer.Typer()
app.command()(main)

if __name__ == "__main__":
    app()

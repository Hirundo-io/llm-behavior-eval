import gc
import logging
import os
from pathlib import Path
from typing import Annotated

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
)
from llm_behavior_eval.evaluation_utils.util_functions import (
    empty_cuda_cache_if_available,
)

torch.set_float32_matmul_precision("high")

BIAS_KINDS = {"bias", "unbias"}
HALUEVAL_ALIAS = {"hallu", "hallucination"}
MEDHALLU_ALIAS = {"hallu-med", "hallucination-med"}
INJECTION_ALIAS = {"prompt-injection"}


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
    use_vllm: Annotated[
        bool,
        typer.Option(
            "--use-vllm/--no-use-vllm",
            help="Use vLLM for model inference instead of transformers",
        ),
    ] = False,
    use_vllm_for_judge: Annotated[
        bool | None,
        typer.Option(
            "--use-vllm-for-judge/--no-use-vllm-for-judge",
            help="Use vLLM for judge model inference instead of transformers. Defaults to same choice for model inference.",
        ),
    ] = None,
    reasoning: Annotated[
        bool,
        typer.Option(
            "--reasoning/--no-reasoning",
            help="Enable chat-template reasoning if supported",
        ),
    ] = False,
) -> None:
    model_path_or_repo_id = model
    judge_path_or_repo_id = judge_model
    result_dir = Path(__file__).parent / "results"
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

        eval_config = EvaluationConfig(
            model_path_or_repo_id=model_path_or_repo_id,
            model_token=model_token,
            judge_path_or_repo_id=judge_path_or_repo_id,
            judge_token=judge_token,
            results_dir=result_dir,
            mlflow_config=mlflow_config,
            reasoning=reasoning,
            use_vllm=use_vllm,
            use_vllm_for_judge=use_vllm_for_judge
            if use_vllm_for_judge is not None
            else use_vllm,
        )
        set_seed(dataset_config.seed)
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

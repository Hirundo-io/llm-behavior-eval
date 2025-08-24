import gc
import logging
from pathlib import Path

import torch
from transformers.trainer_utils import set_seed

from llm_behavior_eval import (
    DatasetConfig,
    DatasetType,
    EvaluateFactory,
    EvaluationConfig,
    PreprocessConfig,
)
import typer
from typing_extensions import Annotated

torch.set_float32_matmul_precision("high")


def _behavior_presets(behavior: str) -> list[str]:
    """
    Map behavior presets to dataset identifiers (free‑text only).

    New formats:
    - BBQ: "bias:<bias_type>" or "unbias:<bias_type>"
    - UNQOVER: "unqover:bias:<bias_type>" (UNQOVER does not support 'unbias')
    - Hallucination retained for backward compatibility: "hallu" or "hallu-med"
    """
    behavior_parts = [p.strip().lower() for p in behavior.split(":")]

    # Hallucination shortcuts
    if behavior in {"hallu", "hallucination"}:
        return ["hirundo-io/halueval"]
    if behavior in {"hallu-med", "hallucination-med"}:
        return ["hirundo-io/medhallu"]

    # Expected structures:
    # [kind, bias_type] for BBQ, where kind in {bias, unbias}
    # ["unqover", kind, bias_type] for UNQOVER (kind must be 'bias')
    if len(behavior_parts) == 2:
        kind, bias_type = behavior_parts
        if kind not in {"bias", "unbias"}:
            raise ValueError("For BBQ use 'bias:<bias_type>' or 'unbias:<bias_type>'")
        from llm_behavior_eval.evaluation_utils.enums import BBQ_BIAS_TYPES

        if bias_type not in BBQ_BIAS_TYPES:
            allowed = ", ".join(sorted(BBQ_BIAS_TYPES))
            raise ValueError(f"BBQ supports: {allowed}")
        return [f"hirundo-io/bbq-{bias_type}-{kind}-free-text"]

    if len(behavior_parts) == 3 and behavior_parts[0] == "unqover":
        _, kind, bias_type = behavior_parts
        if kind != "bias":
            raise ValueError(
                "UNQOVER supports only 'bias:<bias_type>' (no 'unbias' for UNQOVER)"
            )
        from llm_behavior_eval.evaluation_utils.enums import UNQOVER_BIAS_TYPES

        if bias_type not in UNQOVER_BIAS_TYPES:
            allowed = ", ".join(sorted(UNQOVER_BIAS_TYPES))
            raise ValueError(f"UNQOVER supports: {allowed}")
        return [f"unqover/unqover-{bias_type}-{kind}-free-text"]

    raise ValueError(
        "--behavior must be 'bias:<bias_type>' | 'unbias:<bias_type>' | 'unqover:bias:<bias_type>' | 'hallu' | 'hallu-med'"
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
        typer.Argument(help="Behavior preset. BBQ: 'bias:<type>' or 'unbias:<type>'; "),
    ],
) -> None:
    model_path_or_repo_id = model
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
        eval_config = EvaluationConfig(
            model_path_or_repo_id=model_path_or_repo_id,
            results_dir=result_dir,
        )
        set_seed(dataset_config.seed)
        evaluator = EvaluateFactory.create_evaluator(eval_config, dataset_config)
        evaluator.evaluate()
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()


app = typer.Typer()
app.command()(main)

if __name__ == "__main__":
    app()

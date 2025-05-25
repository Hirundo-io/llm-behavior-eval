import gc
import logging
from pathlib import Path

import torch
from transformers.trainer_utils import set_seed

from llm_behavior_eval import (
    BiasEvaluatorFactory,
    DatasetConfig,
    PreprocessConfig,
    DatasetType,
    TextFormat,
    EvaluationConfig,
    JudgeType,
)

if __name__ == "__main__":
    model_path_or_repo_ids = [
        "google/gemma-3-12b-it",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-7b-it",
        "google/gemma-2b-it",
        "google/gemma-3-4b-it",
    ]
    judge_path_or_repo_id = "google/gemma-3-12b-it"
    result_dir = Path(__file__).parent / "results"
    file_paths = [
        "hirundo-io/bbq-physical-bias-free-text",
        "hirundo-io/bbq-physical-bias-multi-choice",
        "hirundo-io/bbq-physical-unbias-free-text",
        "hirundo-io/bbq-physical-unbias-multi-choice",
        "hirundo-io/bbq-race-bias-free-text",
        "hirundo-io/bbq-race-bias-multi-choice",
        "hirundo-io/bbq-race-unbias-free-text",
        "hirundo-io/bbq-race-unbias-multi-choice",
        "hirundo-io/bbq-nationality-bias-free-text",
        "hirundo-io/bbq-nationality-bias-multi-choice",
        "hirundo-io/bbq-nationality-unbias-free-text",
        "hirundo-io/bbq-nationality-unbias-multi-choice",
        "hirundo-io/bbq-gender-bias-free-text",
        "hirundo-io/bbq-gender-bias-multi-choice",
        "hirundo-io/bbq-gender-unbias-free-text",
        "hirundo-io/bbq-gender-unbias-multi-choice",
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for model_path_or_repo_id in model_path_or_repo_ids:
        for file_path in file_paths:
            logging.info("Evaluating %s with %s", file_path, model_path_or_repo_id)
            dataset_config = DatasetConfig(
                file_path=file_path,
                dataset_type=DatasetType.UNBIAS
                if "unbias" in file_path
                else DatasetType.BIAS,
                text_format=TextFormat.FREE_TEXT
                if "free-text" in file_path
                else TextFormat.MULTIPLE_CHOICE,
                preprocess_config=PreprocessConfig(),
            )
            eval_config = EvaluationConfig(
                max_samples=None,  # Set to 100 or lower for testing the pipeline works
                batch_size=64,
                sample=False,
                judge_type=JudgeType.BIAS,
                answer_tokens=128,
                model_path_or_repo_id=model_path_or_repo_id,
                judge_batch_size=64,  # relevant only if the text format is free text
                judge_output_tokens=32,  # relevant only if the text format is free text
                judge_path_or_repo_id=judge_path_or_repo_id,  # relevant only if the text format is free text
                use_4bit_judge=False,  # relevant only if the text format is free text
                results_dir=result_dir,
            )
            set_seed(dataset_config.seed)
            evaluator = BiasEvaluatorFactory.create_evaluator(
                eval_config, dataset_config
            )
            evaluator.evaluate()
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()

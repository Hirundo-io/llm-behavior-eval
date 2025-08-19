import gc
import logging
from pathlib import Path

import torch
from transformers.trainer_utils import set_seed

from llm_behavior_eval import (
    EvaluateFactory,
    DatasetConfig,
    DatasetType,
    EvaluationConfig,
    PreprocessConfig,
)


def main() -> None:
    # Customize here
    model = "meta-llama/Llama-3.1-8B-Instruct"
    judge_model = "google/gemma-3-12b-it"
    result_dir = Path(__file__).resolve().parents[1] / "results"
    # Choose a specific dataset from hirundo-io datasets collections
    dataset_id = "hirundo-io/bbq-gender-bias-free-text"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_config = DatasetConfig(
        file_path=dataset_id,
        dataset_type=DatasetType.BIAS if "-bias-" in dataset_id else DatasetType.UNBIAS,
        preprocess_config=PreprocessConfig(
            max_length=256, gt_max_length=32, preprocess_batch_size=512
        ),
    )

    eval_config = EvaluationConfig(
        max_samples=200,  # set to None to run full dataset
        batch_size=32,
        sample=False,
        answer_tokens=128,
        model_path_or_repo_id=model,
        judge_batch_size=64,
        judge_output_tokens=32,
        judge_path_or_repo_id=judge_model,
        use_4bit_judge=False,
        results_dir=result_dir,
    )

    set_seed(dataset_config.seed)
    evaluator = EvaluateFactory.create_evaluator(eval_config, dataset_config)
    evaluator.evaluate()
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

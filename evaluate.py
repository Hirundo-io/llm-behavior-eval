import gc
import logging

import torch
from transformers.trainer_utils import set_seed

from evaluation_utils.bias_evaluate_factory import BiasEvaluatorFactory
from evaluation_utils.dataset_config import DatasetConfig, PreprocessConfig
from evaluation_utils.enums import DatasetType, TextFormat
from evaluation_utils.eval_config import EvaluationConfig, JudgeType

set_seed(42)

if __name__ == "__main__":
    model_path_or_repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    result_dir = "/home/ubuntu/bias-evaluation/results"
    file_paths = [
        # "hirundo-io/bbq-physical-bias-free-text",
        # "hirundo-io/bbq-physical-bias-multi-choice",
        "hirundo-io/bbq-physical-unbias-free-text",
        # "hirundo-io/bbq-physical-unbias-multi-choice",
        # "hirundo-io/bbq-race-bias-free-text",
        # "hirundo-io/bbq-race-bias-multi-choice",
        "hirundo-io/bbq-race-unbias-free-text",
        # "hirundo-io/bbq-race-unbias-multi-choice",
        # "hirundo-io/bbq-nationality-bias-free-text",
        # "hirundo-io/bbq-nationality-bias-multi-choice",
        "hirundo-io/bbq-nationality-unbias-free-text",
        # "hirundo-io/bbq-nationality-unbias-multi-choice",
        # "hirundo-io/bbq-gender-bias-free-text",
        # "hirundo-io/bbq-gender-bias-multi-choice",
        "hirundo-io/bbq-gender-unbias-free-text",
        # "hirundo-io/bbq-gender-unbias-multi-choice",
    ]
    for file_path in file_paths:
        logging.info("Evaluating %s...", file_path)
        dataset_config = DatasetConfig(
            file_path=file_path,
            dataset_type=DatasetType.UNBIAS
            if "unbias" in file_path
            else DatasetType.BIAS,
            text_format=TextFormat.FREE_TEXT
            if "free-text" in file_path
            else TextFormat.MULTIPLE_CHOICE,
            preprocess_config=PreprocessConfig(
                max_length=512,
                gt_max_length=64,
            ),
        )
        eval_config = EvaluationConfig(
            max_samples=200,
            batch_size=64,
            sample=False,
            judge_type=JudgeType.BIAS,
            answer_tokens=128,
            model_path_or_repo_id=model_path_or_repo_id,
            judge_batch_size=64,  # relevant only if the text format is free text
            judge_output_tokens=32,  # relevant only if the text format is free text
            judge_path_or_repo_id="meta-llama/Llama-3.3-70B-Instruct",  # relevant only if the text format is free text
            use_4bit_judge=True,  # relevant only if the text format is free text
            results_dir=result_dir,
        )
        set_seed(dataset_config.seed)
        evaluator = BiasEvaluatorFactory.create_evaluator(eval_config, dataset_config)
        evaluator.evaluate()
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()

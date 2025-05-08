import sys
from pathlib import Path

from evaluation_utils.bias_evaluate_factory import BiasEvaluatorFactory
from evaluation_utils.dataset_config import DatasetConfig, PreprocessConfig
from evaluation_utils.enums import DatasetType, TextFormat
from evaluation_utils.eval_config import EvaluationConfig, JudgeType

from transformers.trainer_utils import set_seed

set_seed(42)

if __name__ == "__main__":
    file_paths = [
        "hirundo-io/bbq-physical-biased-free-text",
        "hirundo-io/bbq-physical-biased-multiple-choice",
        "hirundo-io/bbq-physical-unbiased-free-text",
        "hirundo-io/bbq-physical-unbiased-multiple-choice",
        "hirundo-io/bbq-race-biased-free-text",
        "hirundo-io/bbq-race-biased-multiple-choice",
        "hirundo-io/bbq-race-unbiased-free-text",
        "hirundo-io/bbq-race-unbiased-multiple-choice",
        "hirundo-io/bbq-nationality-biased-free-text",
        "hirundo-io/bbq-nationality-biased-multiple-choice",
        "hirundo-io/bbq-nationality-unbiased-free-text",
        "hirundo-io/bbq-nationality-unbiased-multiple-choice",
        "hirundo-io/bbq-gender-biased-free-text",
        "hirundo-io/bbq-gender-biased-multiple-choice",
        "hirundo-io/bbq-gender-unbiased-free-text",
        "hirundo-io/bbq-gender-unbiased-multiple-choice",
    ]
    for file_path in file_paths:
        print(f"Evaluating {file_path}...")
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
            max_samples=100,
            batch_size=32,
            sample=False,
            judge_type=JudgeType.BIAS,
            answer_tokens=128,
            temperature=0.6,
            model_path_or_repo_id="meta-llama/Llama-3.1-8B-Instruct",
            judge_batch_size=32,  # relevant only if the text format is free text
            judge_output_tokens=32,  # relevant only if the text format is free text
            judge_path_or_repo_id="meta-llama/Llama-3.1-8B-Instruct",  # relevant only if the text format is free text
            results_dir="/home/ubuntu/bias-evaluation/results",
        )
        evaluator = BiasEvaluatorFactory.create_evaluator(eval_config, dataset_config)
        evaluator.evaluate()

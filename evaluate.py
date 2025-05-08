import sys
from pathlib import Path

from evaluation_utils.bias_evaluate_factory import BiasEvaluatorFactory
from evaluation_utils.dataset_config import DatasetConfig, PreprocessConfig
from evaluation_utils.enums import DatasetType, TextFormat
from evaluation_utils.eval_config import EvaluationConfig, JudgeType

from transformers.trainer_utils import set_seed

set_seed(42)

if __name__ == "__main__":
    result_dir = "/home/ubuntu/bias-evaluation/results"
    file_paths = [
        # "hirundo-io/bbq-physical-bias-free-text",
        # "hirundo-io/bbq-physical-bias-multi-choice",
        "hirundo-io/bbq-physical-unbias-free-text",
        # "hirundo-io/bbq-physical-unbias-multi-choice",
        # "hirundo-io/bbq-race-bias-free-text",
        # "hirundo-io/bbq-race-bias-multi-choice",
        # "hirundo-io/bbq-race-unbias-free-text",
        # "hirundo-io/bbq-race-unbias-multi-choice",
        # "hirundo-io/bbq-nationality-bias-free-text",
        # "hirundo-io/bbq-nationality-bias-multi-choice",
        # "hirundo-io/bbq-nationality-unbias-free-text",
        # "hirundo-io/bbq-nationality-unbias-multi-choice",
        # "hirundo-io/bbq-gender-bias-free-text",
        # "hirundo-io/bbq-gender-bias-multi-choice",
        # "hirundo-io/bbq-gender-unbias-free-text",
        # "hirundo-io/bbq-gender-unbias-multi-choice",
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
            model_path_or_repo_id="meta-llama/Llama-3.1-8B-Instruct",
            judge_batch_size=32,  # relevant only if the text format is free text
            judge_output_tokens=32,  # relevant only if the text format is free text
            judge_path_or_repo_id="meta-llama/Llama-3.1-8B-Instruct",  # relevant only if the text format is free text
            results_dir=result_dir,
        )
        evaluator = BiasEvaluatorFactory.create_evaluator(eval_config, dataset_config)
        evaluator.evaluate()

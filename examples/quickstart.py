from pathlib import Path
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


def main() -> None:
    dataset_cfg = DatasetConfig(
        file_path="hirundo-io/bbq-race-bias-free-text",
        dataset_type=DatasetType.BIAS,
        text_format=TextFormat.FREE_TEXT,
        preprocess_config=PreprocessConfig(max_length=512, gt_max_length=32),
    )
    eval_cfg = EvaluationConfig(
        max_samples=10,
        batch_size=2,
        sample=False,
        judge_type=JudgeType.BIAS,
        answer_tokens=128,
        model_path_or_repo_id="google/gemma-3-4b-it",
        judge_batch_size=2,
        judge_output_tokens=32,
        judge_path_or_repo_id="google/gemma-3-12b-it",
        results_dir=Path(__file__).parent / "results",
    )
    set_seed(dataset_cfg.seed)
    evaluator = BiasEvaluatorFactory.create_evaluator(eval_cfg, dataset_cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    main()

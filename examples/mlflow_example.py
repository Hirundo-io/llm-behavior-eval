"""
Minimal example showing how to run an evaluation with local MLflow tracking enabled.

This example assumes you have MLflow installed (`pip install mlflow`).
"""

from pathlib import Path

from llm_behavior_eval import (
    DatasetConfig,
    DatasetType,
    EvaluateFactory,
    EvaluationConfig,
    PreprocessConfig,
    MlflowConfig,
)


def main():
    dataset_config = DatasetConfig(
        file_path="hirundo-io/bbq-gender-bias-free-text",
        dataset_type=DatasetType.BIAS,
        preprocess_config=PreprocessConfig(),
    )

    mlflow_config = MlflowConfig()

    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta-llama/Llama-3.1-8B-Instruct",
        results_dir=Path("./results"),
        max_samples=50,
        mlflow_config=mlflow_config,
    )

    evaluator = EvaluateFactory.create_evaluator(eval_config, dataset_config)
    evaluator.evaluate()


if __name__ == "__main__":
    print("Run `mlflow ui` in another terminal to view results after this finishes.")
    main()

 
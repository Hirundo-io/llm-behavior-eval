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
    MlflowConfig,
    PreprocessConfig,
)


def main():
    dataset_config = DatasetConfig(
        file_path="hirundo-io/bbq-gender-bias-free-text",
        dataset_type=DatasetType.BIAS,
        preprocess_config=PreprocessConfig(),
    )

    mlflow_config = MlflowConfig()
    # You can also set `mlflow_tracking_uri`, `mlflow_experiment_name` or `mlflow_run_name` here
    # e.g.
    # ```python
    # mlflow_config = MlflowConfig(
    #     mlflow_tracking_uri="http://tracking.example",
    #     mlflow_experiment_name="MLflow Tests",
    #     mlflow_run_name="MLflow Run Name",
    # )
    # ```
    # Note: All of these parameters are optional, as per MLflow's behavior.

    eval_config = EvaluationConfig(
        model_path_or_repo_id="meta-llama/Llama-3.1-8B-Instruct",
        model_token=None,  # Don't forget to set this to your HuggingFace token after accepting the terms of service for the gated model
        results_dir=Path("./results"),
        max_samples=50,
        mlflow_config=mlflow_config,
    )

    evaluator = EvaluateFactory.create_evaluator(eval_config, dataset_config)
    evaluator.evaluate()


if __name__ == "__main__":
    print("Run `mlflow ui` in another terminal to view results after this finishes.")
    main()

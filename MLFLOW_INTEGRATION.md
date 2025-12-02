# MLflow Integration Summary

## Overview

Optional MLflow support is available. It is implemented so that core evaluation
logic is unaffected when MLflow is not used.

## What is Logged

- Parameters: model, dataset, dataset type, seed, and evaluation settings
- Metrics: accuracy, error, stereotyped bias (when available), empty responses
- Artifacts: `responses.json`, `metrics.csv`, `generations.jsonl` (when present)

## How to Enable

1. Install MLflow: `pip install llm-behavior-eval[mlflow]` or `pip install mlflow`.
2. CLI: add `--use-mlflow` (and optional `--mlflow-tracking-uri`,
   `--mlflow-experiment-name`, `--mlflow-run-name`) to the command.
3. Programmatically: construct a `MlflowConfig` and pass it to
   `EvaluationConfig.mlflow_config`.

Example (programmatic):

```python
from pathlib import Path
from llm_behavior_eval import EvaluationConfig, MlflowConfig

mlflow_cfg = MlflowConfig()
eval_config = EvaluationConfig(
    model_path_or_repo_id="meta-llama/Llama-3.1-8B-Instruct",
    results_dir=Path("./results"),
    mlflow_config=mlflow_cfg,
)
```

The repository includes a minimal example at `examples/mlflow_example.py`.

## Notes

- MLflow import is optional and guarded; if MLflow is not installed the
  evaluator logs a warning and proceeds without tracking.
- The evaluator handles MLflow run lifecycle; you do not need to call
  internal cleanup helpers after `evaluate()` returns.

"""Report a native Inspect run to a local MLflow file store.

    uv pip install -e '.[mlflow]'
    python examples/mlflow_example.py

This runs the offline ``inspect_template`` wiring task on a mock model. For a
real benchmark, keep the same environment and run, for example:

    LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED=1 MLFLOW_TRACKING_URI=... \\
      inspect eval llm_behavior_eval/halueval --model hf/org/model \\
        --model-role 'grader={model: hf/google/gemma-4-12B-it, temperature: 0.0}'

Reporting is configured only through the environment (see
MLFLOW_INTEGRATION.md). Inference settings for the model under test and the
grader are passed to Inspect for each role; the hook neither reads nor sets them.
"""

import os
from pathlib import Path

from inspect_ai import eval as inspect_eval


def main() -> None:
    os.environ["LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED"] = "1"
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI", Path("results/mlruns").resolve().as_uri()
    )
    # MLflow 3.x requires explicit opt-in to its deprecated local file store.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "llm-behavior-eval-example")
    inspect_eval(
        "llm_behavior_eval/inspect_template",
        model="mockllm/model",
        log_dir="results/inspect",
    )


if __name__ == "__main__":
    main()

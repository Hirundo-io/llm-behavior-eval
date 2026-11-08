"""Run a tiny native Inspect evaluation with a local MLflow file store.

uv pip install -e '.[mlflow]'
python examples/mlflow_example.py
"""

import os
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

os.environ["LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED"] = "1"
os.environ["MLFLOW_TRACKING_URI"] = Path("results/mlruns").resolve().as_uri()
# MLflow 3.x requires explicit opt-in to its deprecated local file store.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

if __name__ == "__main__":
    inspect_eval(
        Task(
            dataset=[Sample(input="Say hello", target="Default output")],
            solver=generate(),
            scorer=exact(),
            name="mlflow_example",
        ),
        model="mockllm/model",
        log_dir="results/inspect",
    )

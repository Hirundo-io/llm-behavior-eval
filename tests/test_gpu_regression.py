from __future__ import annotations

import csv
import importlib.util
import os
from typing import TYPE_CHECKING, Literal

import pytest

pytest.importorskip("torch")
import torch

from llm_behavior_eval.evaluate import main as run_evaluation

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_BEHAVIORS = "bias:gender,unqover:bias:gender,prompt-injection"
DEFAULT_DATASET_COUNT = 3
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@pytest.mark.gpu_regression
@pytest.mark.parametrize("engine", ["transformers", "vllm"])
def test_gpu_local_local_regression(
    engine: Literal["transformers", "vllm"],
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU regression tests.")

    if engine == "vllm" and importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM is not installed; install the 'vllm' extra to run this case.")

    model = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MODEL", DEFAULT_MODEL)
    judge_model = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_JUDGE", DEFAULT_JUDGE)

    behavior = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_BEHAVIOR", DEFAULT_BEHAVIORS)
    max_samples = int(os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_SAMPLES", "4"))
    max_answer_tokens = int(
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_ANSWER_TOKENS", "32")
    )
    max_judge_tokens = int(
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_JUDGE_TOKENS", "16")
    )
    trust_remote_code = _read_bool_env(
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_TRUST_REMOTE_CODE", False
    )
    model_token = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MODEL_TOKEN")
    judge_token = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_JUDGE_TOKEN")
    output_dir = tmp_path / f"{engine}-local-local-regression"

    run_evaluation(
        model=model,
        behavior=behavior,
        output_dir=str(output_dir),
        model_token=model_token,
        judge_token=judge_token,
        judge_model=judge_model,
        model_engine=engine,
        judge_engine=engine,
        replace_existing_output=True,
        max_samples=max_samples,
        batch_size=1,
        judge_batch_size=1,
        sample=False,
        sample_judge=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=7,
        max_answer_tokens=max_answer_tokens,
        max_judge_tokens=max_judge_tokens,
        use_4bit=False,
        use_4bit_judge=False,
        trust_remote_code=trust_remote_code,
        vllm_enforce_eager=True,
    )

    model_slug = model.split("/")[-1]
    metrics_files = sorted((output_dir / model_slug).rglob("metrics.csv"))
    assert len(metrics_files) == DEFAULT_DATASET_COUNT

    for metrics_file in metrics_files:
        with metrics_file.open(newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))
        assert rows, f"Expected non-empty metrics file: {metrics_file}"
        first_row = rows[0]
        metric_key = "Error (%)" if "Error (%)" in first_row else "Accuracy (%)"
        score = float(first_row[metric_key])
        assert 0.0 <= score <= 100.0

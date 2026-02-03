from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_eval_config_rejects_api_model_engine_with_inference_engine(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match="model_engine='api' cannot be combined with inference_engine"
    ):
        EvaluationConfig(
            model_path_or_repo_id="openai/gpt-4o",
            results_dir=tmp_path,
            inference_engine="transformers",
            model_engine="api",
        )


def test_eval_config_rejects_api_judge_engine_with_inference_engine(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match="judge_engine='api' cannot be combined with inference_engine"
    ):
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            results_dir=tmp_path,
            inference_engine="vllm",
            judge_engine="api",
        )


def test_eval_config_rejects_model_tokenizer_with_api_model_engine(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="model_tokenizer_path_or_repo_id cannot be used with model_engine='api'",
    ):
        EvaluationConfig(
            model_path_or_repo_id="openai/gpt-4o",
            model_tokenizer_path_or_repo_id="meta/tokenizer",
            results_dir=tmp_path,
            model_engine="api",
        )

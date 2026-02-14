from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_eval_config_allows_inference_engine_to_override_model_engine(
    tmp_path: Path,
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="openai/gpt-4o",
        results_dir=tmp_path,
        inference_engine="transformers",
        model_engine="api",
    )
    assert config.inference_engine == "transformers"
    assert config.model_engine == "api"


def test_eval_config_allows_inference_engine_to_override_judge_engine(
    tmp_path: Path,
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        inference_engine="vllm",
        judge_engine="api",
    )
    assert config.inference_engine == "vllm"
    assert config.judge_engine == "api"


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


def test_eval_config_rejects_model_tokenizer_with_api_inference_engine(
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
            inference_engine="api",
        )


def test_eval_config_rejects_judge_tokenizer_with_api_judge_engine(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="judge_tokenizer_path_or_repo_id cannot be used with judge_engine='api'",
    ):
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            judge_path_or_repo_id="openai/gpt-4o-mini",
            judge_tokenizer_path_or_repo_id="meta/judge-tokenizer",
            results_dir=tmp_path,
            judge_engine="api",
        )


def test_eval_config_rejects_judge_tokenizer_with_api_inference_engine(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="judge_tokenizer_path_or_repo_id cannot be used with judge_engine='api'",
    ):
        EvaluationConfig(
            model_path_or_repo_id="meta/model",
            judge_tokenizer_path_or_repo_id="meta/judge-tokenizer",
            results_dir=tmp_path,
            inference_engine="api",
        )


def test_eval_config_allows_judge_tokenizer_with_local_judge_engine(
    tmp_path: Path,
) -> None:
    config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        judge_tokenizer_path_or_repo_id="meta/judge-tokenizer",
        results_dir=tmp_path,
        judge_engine="transformers",
    )
    assert config.judge_tokenizer_path_or_repo_id == "meta/judge-tokenizer"

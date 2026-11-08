"""Deterministic legacy parity, not live provider/GPU parity."""

import asyncio
import importlib
from collections.abc import Callable
from typing import Any, cast

import pytest
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageUser, ModelName, ModelOutput, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Target
from inspect_ai.solver import TaskState

from llm_behavior_eval.common.metrics import incomplete_response_rate
from llm_behavior_eval.hallucination.prompts import GRADER_TEMPLATE
from llm_behavior_eval.hallucination.scorer import (
    accuracy,
    empty_responses,
    hallucination_scorer,
    parse_grade,
)
from llm_behavior_eval.halueval.dataset import DATASET_ID, PARITY_GAPS, dataset
from llm_behavior_eval.halueval.task import halueval


@pytest.mark.parametrize(
    "raw",
    ["A", "B", "C", "garbage", "", "AA", "a", "B then A", "(A)", "<think>B</think>A"],
)
def test_parse_legacy(raw):
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        GRADER_TEMPLATE as LEGACY_TEMPLATE,
    )
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        FreeTextHaluEvaluator,
    )

    assert (
        parse_grade(raw)
        == FreeTextHaluEvaluator._map_judge_outputs([[{"generated_text": raw}]])[0]
    )
    assert GRADER_TEMPLATE == LEGACY_TEMPLATE


def test_dataset(tmp_path):
    path = tmp_path / "snapshot"
    Dataset.from_list(
        [{"question": f"q{i}", "answer": "a"} for i in range(502)]
    ).save_to_disk(path)
    limited = dataset(dataset_path=str(path))
    full = dataset(0, dataset_path=str(path))
    assert len(limited) == 500
    assert len(full) == len(dataset(-1, dataset_path=str(path))) == 502
    assert limited[0].id == full[0].id == f"{DATASET_ID}:0"
    assert limited.name == DATASET_ID
    assert limited[0].input == "q0"
    assert limited[0].metadata is not None
    assert limited[0].metadata["parity_gaps"] == PARITY_GAPS
    Dataset.from_list(
        [{"question": "q", "answer": "a"}, {"question": "", "answer": "a"}]
    ).save_to_disk(tmp_path / "invalid")
    with pytest.raises(ValueError):
        dataset(1, dataset_path=str(tmp_path / "invalid"))


def test_native_scorer_and_legacy_aggregate(monkeypatch):
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        GRADER_TEMPLATE as LEGACY_TEMPLATE,
    )
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        FreeTextHaluEvaluator,
        _HalluGenerationRecord,
    )

    module = importlib.import_module("llm_behavior_eval.hallucination.scorer")
    raw = iter(["A", "B", "garbage", "C"])
    calls = []
    grader = get_model("mockllm/model")

    async def generate(prompt, *, config):
        calls.append(prompt)
        assert config.max_tokens == 32 and config.temperature == 0
        return ModelOutput.from_content("mockllm/model", next(raw))

    monkeypatch.setattr(grader, "generate", generate)
    monkeypatch.setattr(module, "get_model", lambda **kwargs: grader)
    stops = ["stop", "stop", "stop", "stop", "max_tokens", "unknown", None]
    scores = []
    for stop in stops:
        output = ModelOutput.from_content(
            "mockllm/model", "<think>trace</think> answer"
        )
        if stop is None:
            output.choices = []
        else:
            output.choices[0].stop_reason = cast("Any", stop)
        state = TaskState(
            model=ModelName("mockllm/model"),
            sample_id=1,
            epoch=1,
            input="question",
            messages=[ChatMessageUser(content="question")],
            output=output,
        )
        scores.append(
            SampleScore(
                score=asyncio.run(hallucination_scorer()(state, Target("gold")))
            )
        )
    assert len(calls) == 4
    assert calls[0] == LEGACY_TEMPLATE.format(
        question="question", target="gold", predicted_answer="answer"
    )
    assert scores[0].score.metadata is not None
    assert scores[0].score.metadata["raw_judge_text"] == "A"
    assert scores[2].score.metadata is not None
    assert scores[2].score.metadata["parse_error"]
    assert all(
        (s.score.metadata or {}).get("grade") == "NOT_ATTEMPTED" for s in scores[4:]
    )

    legacy = FreeTextHaluEvaluator.__new__(FreeTextHaluEvaluator)
    legacy.num_samples = len(stops)
    monkeypatch.setattr(legacy, "_format_answers", lambda answers: answers)
    monkeypatch.setattr(
        legacy,
        "_grade_batch",
        lambda *args: ["CORRECT", "INCORRECT", "NOT_ATTEMPTED", "NOT_ATTEMPTED"],
    )
    saved = {}
    monkeypatch.setattr(legacy, "save_results", lambda **kwargs: saved.update(kwargs))
    legacy._grade_impl(
        [
            _HalluGenerationRecord(
                answers=["answer"] * len(stops),
                input_texts=["question"] * len(stops),
                gt_answers=["gold"] * len(stops),
                finish_reasons=["length" if s == "max_tokens" else s for s in stops],
            )
        ],
        cast("Any", object()),
    )
    assert rate(accuracy())(scores) == pytest.approx(100 * saved["accuracy"])
    assert rate(empty_responses())(scores) == saved["empty_responses"] == 5
    assert rate(incomplete_response_rate())(scores) == saved["incomplete_response_rate"]
    assert (
        rate(accuracy())(
            [SampleScore(score=Score(value=1, metadata={"grade": "NOT_ATTEMPTED"}))]
        )
        == 100
    )


def test_native_task(tmp_path):
    path = tmp_path / "snapshot"
    Dataset.from_list([{"question": "question", "answer": "gold"}]).save_to_disk(path)
    calls = []

    def answer(messages, tools, choice, config):
        calls.append(messages)
        assert config.max_tokens == 128 and config.temperature == 0
        return ModelOutput.from_content("mockllm/model", "answer")

    logs = inspect_eval(
        halueval(dataset_path=str(path)),
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={
            "grader": get_model(
                "mockllm/grader",
                custom_outputs=lambda *args: ModelOutput.from_content(
                    "mockllm/grader", "B"
                ),
            )
        },
        log_dir=str(tmp_path / "logs"),
        display="none",
        log_format="json",
    )
    assert logs[0].status == "success", logs[0].error
    assert calls[0][-1].text == "question\n"
    assert logs[0].results is not None
    metrics = logs[0].results.scores[0].metrics
    assert metrics["accuracy"].value == 0
    assert metrics["empty_responses"].value == 0
    import json
    from pathlib import Path

    saved_log = json.loads(Path(logs[0].location).read_text())
    assert saved_log["samples"][0]["scores"]["hallucination_scorer"]["value"] == 0
    missing = inspect_eval(
        halueval(dataset_path=str(path)),
        model="mockllm/model",
        log_dir=str(tmp_path / "missing"),
        display="none",
        log_format="json",
    )
    assert missing[0].status == "error"


def test_documented_tokenizer_parity_gaps(monkeypatch, tmp_path):
    from llm_behavior_eval.evaluation_utils import custom_dataset

    class Tokenizer:
        def __call__(self, texts, *, max_length, **kwargs):
            ids = [[ord(c) for c in text[:max_length]] for text in texts]
            return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

    monkeypatch.setattr(
        custom_dataset,
        "safe_apply_chat_template",
        lambda tokenizer, messages, **kwargs: "TEMPLATE:" + str(messages),
    )
    question, target = "q" * 1200, "a" * 300
    legacy = custom_dataset.free_text_preprocess_function(
        {"question": [question], "answer": [target]},
        cast("Any", Tokenizer()),
        1024,
        256,
        False,
    )
    assert legacy["test_input_ids"].shape[1] == 1024
    assert legacy["gt_answers"].shape[1] == 256
    decoded = "".join(chr(c) for c in legacy["test_input_ids"][0])
    assert decoded.startswith("TEMPLATE:") and decoded != question
    path = tmp_path / "long"
    Dataset.from_list([{"question": question, "answer": target}]).save_to_disk(path)
    sample = dataset(dataset_path=str(path))[0]
    assert sample.input == question and sample.target == target
    assert sample.metadata is not None
    assert len(sample.metadata["parity_gaps"]) == 3


def test_thinking_contracts(monkeypatch):
    from types import SimpleNamespace

    from llm_behavior_eval.hallucination import solver

    model = get_model("mockllm/model")
    monkeypatch.setattr(
        solver, "ModelName", lambda model: SimpleNamespace(api="openai-api")
    )
    with pytest.raises(ValueError, match="unsupported"):
        solver.configured_generation(model)
    assert solver.configured_generation(model, thinking_disabled=True).max_tokens == 128
    monkeypatch.setattr(solver, "ModelName", lambda model: SimpleNamespace(api="hf"))
    with pytest.raises(ValueError, match="model args"):
        solver.configured_generation(model)
    monkeypatch.setattr(model.api, "enable_thinking", False, raising=False)
    monkeypatch.setattr(model.api, "do_sample", False, raising=False)
    assert solver.configured_generation(model, judge=True).max_tokens == 32
    monkeypatch.setattr(solver, "ModelName", lambda model: SimpleNamespace(api="vllm"))
    assert solver.configured_generation(model).extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_independent_api_grader(tmp_path, monkeypatch):
    """Check independent API role routing with deterministic generation, not HTTP parity."""
    path = tmp_path / "snapshot"
    Dataset.from_list([{"question": "q", "answer": "a"}]).save_to_disk(path)
    grader = get_model(
        "openai-api/airgap/judge",
        base_url="http://judge.invalid/v1",
        api_key="fixture-not-secret",
    )
    requests = []

    async def generate(prompt, *, config):
        requests.append((prompt, config))
        return ModelOutput.from_content("judge", "B")

    monkeypatch.setattr(grader, "generate", generate)
    logs = inspect_eval(
        halueval(dataset_path=str(path), grader_thinking_disabled=True),
        model="mockllm/model",
        model_roles={"grader": grader},
        display="none",
        log_format="json",
        log_dir=str(tmp_path / "logs"),
    )
    assert logs[0].status == "success", logs[0].error
    assert str(cast("Any", grader.api).client.base_url) == "http://judge.invalid/v1/"
    assert len(requests) == 1
    assert requests[0][1].max_tokens == 32
    assert logs[0].results is not None
    assert logs[0].results.scores[0].metrics["accuracy"].value == 0


def test_cli_registration(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "snapshot"
    Dataset.from_list([{"question": "q", "answer": "a"}]).save_to_disk(path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "llm_behavior_eval/halueval",
            "--model",
            "mockllm/model",
            "--model-role",
            "grader=mockllm/grader",
            "-T",
            f"dataset_path={path}",
            "--log-dir",
            str(tmp_path / "logs"),
            "--log-format",
            "json",
            "--display",
            "none",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def rate(metric: Metric) -> Callable[[list[SampleScore]], float]:
    return cast("Callable[[list[SampleScore]], float]", metric)

"""Deterministic fixtures only; no live/GPU or publication parity claim."""

import asyncio
import json
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
from datasets import Dataset, DatasetDict
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelName, ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, SampleScore, Score, Target
from inspect_ai.solver import TaskState

from llm_behavior_eval.ccpc import contract, dataset
from llm_behavior_eval.ccpc.generation import PROMPT_PARITY, ccpc_config
from llm_behavior_eval.ccpc.metrics import (
    censorship_rate,
    censorship_rate_stderr,
    judged_fraction,
    parse_verdict,
)
from llm_behavior_eval.ccpc.scorer import _judge_prompt, censorship_judge
from llm_behavior_eval.ccpc.task import ccpc
from llm_behavior_eval.evaluation_utils import censorship_utils as legacy


@pytest.fixture(autouse=True)
def writable_inspect_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture
def rows(monkeypatch):
    data = Dataset.from_dict(
        {
            "question": [f"问题 {i}" for i in range(500)],
            "benchmark_id": [f"id-{i}" for i in range(500)],
        }
    )
    fingerprint = legacy.censorship_dataset_fingerprint(data)
    monkeypatch.setattr(dataset, "CCPC_DATASET_FINGERPRINT", fingerprint)
    monkeypatch.setattr(dataset, "load_dataset", lambda *a, **kw: data)
    return data


def test_contract():
    for key in vars(contract):
        if key.startswith("CCPC_"):
            assert getattr(contract, key) == getattr(legacy, key)


@pytest.mark.parametrize(
    "text",
    ["True", " FALSE\n", "true.", "yes", "", "<think>x</think>True", "True\nFalse"],
)
def test_parser(text):
    expected = legacy.parse_censorship_judgment(text)
    assert parse_verdict(text) == (
        NOANSWER if expected is None else CORRECT if expected else INCORRECT
    )


@pytest.mark.parametrize("raw", [[], ["True"], ["True", "False"], ["True", "oops"]])
def test_metrics(raw):
    expected = legacy.calculate_censorship_metrics(
        [legacy.parse_censorship_judgment(x) for x in raw]
    )
    scores = [
        SampleScore(score=Score(value=0, metadata={"verdict": parse_verdict(x)}))
        for x in raw
    ]
    for metric, old in zip(
        [censorship_rate(), censorship_rate_stderr(), judged_fraction()],
        expected,
        strict=True,
    ):
        actual = cast("Any", metric)(scores)
        if old is None or math.isnan(old):
            assert math.isnan(actual)
        else:
            assert actual == old


def test_pinned_load_and_order(rows, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        dataset, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or rows
    )
    assert dataset.load_ccpc_dataset(max_samples=1)[0].id == "id-0"
    assert calls[0] == (
        (contract.CCPC_DATASET_REPOSITORY,),
        dict(
            name="default",
            split="test",
            revision=contract.CCPC_DATASET_REVISION,
            token=None,
        ),
    )
    for source in (rows, DatasetDict({"test": rows, "train": rows.select([1])})):
        path = tmp_path / str(len(list(tmp_path.iterdir())))
        source.save_to_disk(str(path))
        for cap, n in [(None, 500), (-1, 500), (0, 500), (3, 3), (500, 500)]:
            result = dataset.load_ccpc_dataset(file_path=str(path), max_samples=cap)
            assert len(result) == n
            assert [r.id for r in result] == [f"id-{i}" for i in range(n)]
            assert result.name == "chinese_censorship"


@pytest.mark.parametrize(
    "mutation", ["reorder", "last", "duplicate", "columns", "count", "blank"]
)
def test_validation_before_cap(rows, monkeypatch, mutation):
    data = rows.to_dict()
    if mutation == "reorder":
        changed = rows.select(list(reversed(range(500))))
    elif mutation == "columns":
        changed = rows.add_column("extra", [0] * 500)
    elif mutation == "count":
        changed = rows.select(range(499))
    else:
        if mutation == "last":
            data["question"][-1] = "changed"
        elif mutation == "blank":
            data["question"][-1] = " "
        else:
            data["benchmark_id"][-1] = data["benchmark_id"][0]
        changed = Dataset.from_dict(data)
    monkeypatch.setattr(dataset, "load_dataset", lambda *a, **kw: changed)
    with pytest.raises(ValueError):
        dataset.load_ccpc_dataset(max_samples=1)


@pytest.mark.parametrize("stop", ["stop", "max_tokens", "unknown", None])
def test_native_judge_all_finishes(monkeypatch, stop):
    calls = []

    class Judge:
        def __str__(self):
            return "mockllm/judge"

        async def generate(self, messages, config):
            calls.append((messages, config))
            result = ModelOutput.from_content("judge", " TRUE ")
            result.choices[0].stop_reason = "max_tokens"
            return result

    monkeypatch.setattr("llm_behavior_eval.ccpc.scorer.get_model", lambda **kw: Judge())
    output = ModelOutput.from_content("model", "<think>hidden</think>答案")
    if stop is None:
        output.choices = []
    else:
        output.choices[0].stop_reason = stop
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="a",
        epoch=1,
        input="问题",
        messages=[],
        output=output,
    )
    score = asyncio.run(censorship_judge()(state, Target("")))
    assert score is not None and score.metadata is not None
    assert score.metadata["verdict"] == CORRECT
    assert score.metadata["raw_judge_text"] == " TRUE "
    assert score.metadata["stop_reason"] == stop
    messages, config = calls[0]
    assert messages[0].role == "system"
    assert messages[0].text == legacy.CCPC_JUDGE_SYSTEM_PROMPT
    assert messages[1].text == legacy.CCPC_JUDGE_PROMPT_TEMPLATE.format(
        serialized_judge_input=json.dumps(
            {"question": "问题", "response": "答案"},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    assert config.max_tokens == 4096
    assert config.temperature == 0


def test_thinking_controls():
    class FakeModel:
        config = SimpleNamespace(extra_body=None)

        def __init__(self, name, **kw):
            self.name, self.api = name, SimpleNamespace(**kw)

        def __str__(self):
            return self.name

    Model = cast("Any", FakeModel)
    with pytest.raises(ValueError, match="Unsupported"):
        ccpc_config(Model("openai-api/judge"))
    with pytest.raises(ValueError, match="HF requires"):
        ccpc_config(Model("hf/judge"))
    assert (
        ccpc_config(
            Model("hf/judge", enable_thinking=False, do_sample=False)
        ).max_tokens
        == 8192
    )
    assert ccpc_config(Model("vllm/judge"), judge=True).extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert (
        ccpc_config(Model("openai-api/judge"), externally_configured=True).temperature
        == 0
    )


@pytest.mark.parametrize("raw", ["False", "unparseable"])
def test_eval_serialization(rows, tmp_path, raw):
    log = inspect_eval(
        ccpc(max_samples=2),
        model=get_model(
            "mockllm/answer",
            custom_outputs=[
                ModelOutput.from_content("answer", "text") for _ in range(2)
            ],
        ),
        model_roles={
            "grader": get_model(
                "mockllm/judge",
                custom_outputs=[
                    ModelOutput.from_content("judge", "True"),
                    ModelOutput.from_content("judge", raw),
                ],
            )
        },
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "success", log.error
    with open(log.location, "rb") as stream:
        restored = read_eval_log(stream)
    assert restored.results is not None and restored.samples is not None
    metrics = next(iter(restored.results.scores)).metrics
    rate = next(m.value for m in metrics.values() if m.name.endswith("censorship_rate"))
    assert math.isnan(rate) if raw == "unparseable" else rate == 0.5
    from llm_behavior_eval.ccpc.metrics import reported_rate

    assert reported_rate(rate, percentage=True) == (
        None if raw == "unparseable" else 50.0
    )
    assert json.loads(
        json.dumps({"censorship_rate": reported_rate(rate)}, allow_nan=False)
    )["censorship_rate"] == (None if raw == "unparseable" else 0.5)
    assert restored.samples[0].messages[0].role == "user"
    assert restored.samples[0].messages[0].text == "问题 0\n"
    assert restored.samples[0].metadata["prompt_parity"] == PROMPT_PARITY
    assert "1024" in PROMPT_PARITY
    assert len(restored.samples) == 2


def test_required_grader(rows, tmp_path):
    log = inspect_eval(
        ccpc(max_samples=1),
        model="mockllm/model",
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "error"
    assert log.error is not None
    assert "grader" in log.error.message


def test_no_prompt_truncation_hidden():
    question = "long " * 2000
    assert (
        json.loads(_judge_prompt(question, "answer").split("\n", 1)[1])["question"]
        == question
    )
    assert "unresolved" in PROMPT_PARITY


def test_independent_endpoint(rows, monkeypatch, tmp_path):
    from inspect_ai.model import Model

    grader = get_model(
        "openai-api/airgap/judge",
        base_url="http://judge.invalid/v1",
        api_key="fixture-only",
    )
    original = Model.generate
    calls = []

    async def generate(self, input, *args, **kwargs):  # noqa: A002
        if self.name == grader.name:
            calls.append(input)
            assert kwargs["config"].max_tokens == 4096
            return ModelOutput.from_content("judge", "True")
        return await original(self, input, *args, **kwargs)

    monkeypatch.setattr(Model, "generate", generate)
    log = inspect_eval(
        ccpc(max_samples=1, judge_thinking_configured=True),
        model="mockllm/answer",
        model_roles={"grader": grader},
        display="none",
        log_dir=str(tmp_path),
    )[0]
    assert log.status == "success", log.error
    assert len(calls) == 1
    assert calls[0][0].role == "system"

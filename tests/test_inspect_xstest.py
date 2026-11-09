"""Deterministic legacy parity only: no live model/provider parity claim."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from datasets import Dataset
from inspect_ai.model import ModelName, ModelOutput, StopReason, get_model
from inspect_ai.scorer import SampleScore, Target
from inspect_ai.solver import TaskState

from llm_behavior_eval.common.judge import strip_thinking
from llm_behavior_eval.refusal.contract import normalize_refusal_label
from llm_behavior_eval.refusal.dataset import refusal_dataset
from llm_behavior_eval.refusal.generation import refusal_config
from llm_behavior_eval.refusal.metrics import aggregate_refusal
from llm_behavior_eval.refusal.scorer import refusal_heuristic
from llm_behavior_eval.xstest.scorer import parse_xstest_judgment, scorer
from llm_behavior_eval.xstest.task import xstest
from tests.golden import BASELINE as GOLDEN
from tests.inspect_helpers import inspect_eval


@pytest.mark.parametrize(
    "text",
    [
        "1_full_compliance",
        "2_FULL_REFUSAL",
        "3_partial_refusal",
        "nothing",
        "x1_full_compliance",
        "3_partial_refusal then 1_full_compliance",
    ],
)
def test_parser(text):
    expected = {
        "1_full_compliance": "comply",
        "2_FULL_REFUSAL": "refuse",
        "3_partial_refusal": "partial",
        "nothing": "unparseable",
        "x1_full_compliance": "unparseable",
        "3_partial_refusal then 1_full_compliance": "partial",
    }
    assert parse_xstest_judgment(text) == expected[text]


@pytest.mark.parametrize(
    "answer",
    [
        "  Sorry!",
        "I cannot do that",
        "That is illegal",
        "Sure, here",
        "",
        "<think>x</think> I will not",
        "No.",
    ],
)
def test_heuristic(answer):
    expected = {
        "  Sorry!": True,
        "I cannot do that": True,
        "That is illegal": True,
        "Sure, here": False,
        "": False,
        "<think>x</think> I will not": True,
        "No.": True,
    }
    assert refusal_heuristic(answer) == expected[answer]


@pytest.mark.parametrize("label", ["safe", " SAFE ", "Unsafe", 1, None, "other"])
def test_labels(label, monkeypatch):
    monkeypatch.setattr(
        "llm_behavior_eval.refusal.dataset.load_hf_dataset",
        lambda *a, **kw: [{"prompt": "q", "label": label}],
    )
    try:
        expected = normalize_refusal_label(label)
    except ValueError:
        with pytest.raises(ValueError):
            refusal_dataset("hirundo-io/XSTest")
    else:
        metadata = refusal_dataset("hirundo-io/XSTest")[0].metadata
        assert metadata is not None
        assert metadata["label"] == expected


def test_offline_caps_and_validation(tmp_path, monkeypatch):
    path = tmp_path / "data"
    Dataset.from_dict({"question": ["q"] * 503, "label": ["safe"] * 503}).save_to_disk(
        str(path)
    )
    for cap, count in [(500, 500), (0, 503), (-1, 503), (None, 503), (1, 1)]:
        data = refusal_dataset(
            "hirundo-io/XSTest", dataset_path=str(path), max_samples=cap
        )
        assert len(data) == count
        assert data.name == "hirundo-io/XSTest"
        assert data[0].id == "hirundo-io/XSTest:0"
        assert data[0].target == ""
    monkeypatch.setattr(
        "llm_behavior_eval.refusal.dataset.load_hf_dataset",
        lambda *a, **kw: [
            {"question": "q", "label": "safe"},
            {"question": "q", "label": "invalid"},
        ],
    )
    with pytest.raises(ValueError):
        refusal_dataset("hirundo-io/XSTest", max_samples=1)


def test_xstest_shuffle_before_cap(monkeypatch):
    from llm_behavior_eval.xstest.dataset import dataset

    source = Dataset.from_dict(
        {"question": [f"q{i}" for i in range(20)], "label": ["safe"] * 20}
    )
    monkeypatch.setattr(
        "llm_behavior_eval.refusal.dataset.load_hf_dataset", lambda *a, **kw: source
    )
    expected = list(source.shuffle(seed=42)["question"])
    for cap in (3, 0):
        data = dataset(max_samples=cap)
        questions = expected[:cap] if cap else expected
        assert [s.input for s in data] == questions
        assert [s.id for s in data] == [f"hirundo-io/XSTest:{q[1:]}" for q in questions]


def test_legacy_metric_parity(monkeypatch):
    labels = ["safe", "safe", "safe", "unsafe", "unsafe", "unsafe", "safe"]
    answers = [
        "sure",
        "Sorry",
        "illegal",
        "I cannot",
        "sure",
        "I will not",
        "<think>x</think> sure",
    ]
    stops = ["stop", "length", "weird", "stop", "length", None, "stop"]
    raw = ["1_full_compliance", "3_partial_refusal", "2_full_refusal", "bad", "bad"]
    captured = GOLDEN["xstest_aggregate"]
    outputs = iter(raw)
    calls = []

    class Grader:
        def __str__(self):
            return "mockllm/grader"

        async def generate(self, prompt, config):
            calls.append((prompt, config))
            return ModelOutput.from_content("judge", next(outputs))

    monkeypatch.setattr(
        "llm_behavior_eval.refusal.scorer.get_model", lambda **kw: Grader()
    )
    scores = []
    for label, answer, stop in zip(labels, answers, stops, strict=True):
        output = ModelOutput.from_content("model", answer)
        if stop is None:
            output.choices = []
        else:
            output.choices[0].stop_reason = cast(
                "StopReason",
                {
                    "length": "max_tokens",
                    "weird": "unknown",
                }.get(stop, stop),
            )
        state = TaskState(
            model=ModelName("mockllm/model"),
            sample_id=1,
            epoch=1,
            input="q",
            messages=[],
            metadata={"label": label},
            output=output,
        )
        scores.append(SampleScore(score=asyncio.run(scorer()(state, Target("")))))
    assert aggregate_refusal(scores) == captured["metrics"]
    for sample, old in zip(scores, captured["responses"], strict=True):
        assert sample.score.metadata is not None
        for key in [
            "grading_status",
            "heuristic_refusal",
            "judge_label",
            "judge_success",
        ]:
            assert (sample.score.metadata or {})[key] == old[key]
        assert (sample.score.metadata or {})["raw_judge_text"] == old["judge_raw"]
    assert len(calls) == 5
    assert all(config.max_tokens == 128 for _, config in calls)
    assert aggregate_refusal([]) == dict.fromkeys(captured["metrics"], 0.0)


def test_native_eval_and_prompt_gap(tmp_path):
    path = tmp_path / "data"
    question = "a " * 1500
    Dataset.from_dict({"question": [question], "label": ["safe"]}).save_to_disk(
        str(path)
    )
    seen = []

    def answer(messages, *args, **kwargs):
        seen.append(messages)
        return ModelOutput.from_content("mockllm/model", "Sorry")

    logs = inspect_eval(
        xstest(dataset_path=str(path)),
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={
            "grader": get_model(
                "mockllm/grader",
                custom_outputs=[
                    ModelOutput.from_content("mockllm/grader", "2_full_refusal")
                ],
            )
        },
        log_dir=str(tmp_path / "logs"),
    )
    assert logs[0].status == "success"
    assert [m.role for m in seen[0]] == ["user"]
    assert (
        seen[0][0].text == question + "\n"
    )  # Explicit unresolved 1024-token truncation gap.
    assert logs[0].samples and logs[0].samples[0].scores
    score = next(iter(logs[0].samples[0].scores.values()))
    assert score.metadata is not None
    assert (
        score.metadata["question"] == question
    )  # Raw question, not decoded chat template.
    assert score.metadata["thinking"] is True
    assert score.metadata["raw_judge_text"] == "2_full_refusal"
    assert logs[0].results and logs[0].results.scores


def test_provider_controls():
    class FakeModel:
        config = SimpleNamespace(extra_body=None)
        api = SimpleNamespace(enable_thinking=True, do_sample=False)

        def __init__(self, provider):
            self.provider = provider

        def __str__(self):
            return self.provider + "/model"

    Model = cast("Any", FakeModel)
    assert refusal_config(Model("hf")).max_tokens == 2048
    with pytest.raises(ValueError):
        refusal_config(Model("hf"), judge=True)
    answer_body = refusal_config(Model("vllm")).extra_body
    judge_body = refusal_config(Model("vllm"), judge=True).extra_body
    assert answer_body and judge_body
    assert answer_body["chat_template_kwargs"]["enable_thinking"] is True
    assert judge_body["chat_template_kwargs"]["enable_thinking"] is False
    with pytest.raises(ValueError, match="unsupported"):
        refusal_config(cast("Any", Model("openai-api")))
    assert (
        refusal_config(
            cast("Any", Model("openai-api")), judge=True, externally_configured=True
        ).max_tokens
        == 128
    )
    assert strip_thinking("<think>unfinished") == "<think>unfinished"


def test_missing_grader(tmp_path):
    path = tmp_path / "data"
    Dataset.from_dict({"question": ["q"], "label": ["safe"]}).save_to_disk(str(path))
    logs = inspect_eval(
        xstest(dataset_path=str(path)),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
    )
    assert logs[0].status == "error"
    assert logs[0].error is not None
    assert "grader" in logs[0].error.message


def test_cli_registration(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "data"
    Dataset.from_dict({"prompt": ["q"], "label": ["safe"]}).save_to_disk(str(path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "llm_behavior_eval/xstest",
            "--model",
            "mockllm/model",
            "--model-role",
            "grader=mockllm/grader",
            "-T",
            f"dataset_path={path}",
            "--log-dir",
            str(tmp_path / "logs"),
            "--display",
            "none",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_independent_api_grader(tmp_path):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args):  # noqa: A002 - stdlib override
            pass

        def do_POST(self):  # noqa: N802 - HTTP handler API
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            payload = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "judge",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "3_partial_refusal",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        path = tmp_path / "data"
        Dataset.from_dict({"question": ["q"], "label": ["unsafe"]}).save_to_disk(
            str(path)
        )
        grader = get_model(
            "openai-api/airgap/judge",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local-test-placeholder",
        )
        logs = inspect_eval(
            xstest(dataset_path=str(path), judge_thinking_configured=True),
            model="mockllm/model",
            model_roles={"grader": grader},
            log_dir=str(tmp_path / "logs"),
            display="none",
        )
        assert logs[0].status == "success"
        assert logs[0].samples and logs[0].samples[0].scores
        score = next(iter(logs[0].samples[0].scores.values()))
        assert score.metadata is not None
        assert score.metadata["judge_label"] == "partial"
        assert len(requests) == 1
        assert requests[0]["model"] == "judge"
        assert requests[0]["max_tokens"] == 128
        assert [m["role"] for m in requests[0]["messages"]] == ["user"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

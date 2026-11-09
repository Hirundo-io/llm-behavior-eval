import json
import math
import subprocess
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import entry_points
from typing import cast

import pytest
from datasets import Dataset, DatasetDict
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Metric, SampleScore, Score

from llm_behavior_eval.common.dataset import (
    load_hf_dataset,
    sample_limit,
    to_inspect_dataset,
)
from llm_behavior_eval.common.generation import generation_config, thinking_args
from llm_behavior_eval.common.judge import strip_thinking
from llm_behavior_eval.common.metrics import field_rate, incomplete_response_rate
from llm_behavior_eval.inspect_template.dataset import record_to_sample
from llm_behavior_eval.inspect_template.task import inspect_template


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("plain", "plain"),
        ("<think>x</think> 2 ", "2"),
        ("x</think>a</think>b", "b"),
        ("<think>unfinished", "<think>unfinished"),
    ],
)
def test_thinking(answer, expected):
    assert strip_thinking(answer) == expected
    assert strip_thinking(answer, exclude_thinking_trace_for_judge=False) == answer
    assert strip_thinking(answer, thinking_start_token=None) == answer


def rate(metric: Metric) -> Callable[[list[SampleScore]], float]:
    return cast("Callable[[list[SampleScore]], float]", metric)


def test_rates():
    scores = [
        SampleScore(
            score=Score(value=1, metadata={"group": "safe", "stop_reason": "stop"})
        ),
        SampleScore(
            score=Score(
                value=0,
                metadata={
                    "group": "unsafe",
                    "stop_reason": "max_tokens",
                    "skipped": True,
                },
            )
        ),
        SampleScore(score=Score(value=0, metadata={"parse_error": "invalid"})),
    ]
    assert rate(field_rate())(scores) == 1
    assert rate(field_rate(metadata_key="group", metadata_value="safe"))(scores) == 1
    assert math.isnan(
        rate(field_rate(metadata_key="group", metadata_value="missing"))(scores)
    )
    assert rate(incomplete_response_rate())(scores) == pytest.approx(1 / 3)
    assert math.isnan(rate(incomplete_response_rate())([]))
    assert rate(field_rate("a"))([SampleScore(score=Score(value={"a": 0.5}))]) == 0.5


def test_dataset(tmp_path, monkeypatch):
    ds = Dataset.from_list(
        [{"id": str(i), "question": "q", "answer": "a"} for i in range(502)]
    )
    DatasetDict({"test": ds}).save_to_disk(tmp_path / "snapshot")
    loaded = load_hf_dataset("logical/id", dataset_path=str(tmp_path / "snapshot"))
    assert len(loaded) == 502
    assert (
        len(to_inspect_dataset(loaded, record_to_sample, dataset_id="logical/id"))
        == 500
    )
    assert (
        len(
            to_inspect_dataset(
                loaded, record_to_sample, dataset_id="logical/id", max_samples=0
            )
        )
        == 502
    )
    import datasets

    calls = []

    def load(source, **kwargs):
        calls.append((source, kwargs))
        return DatasetDict({"train": ds, "test": cast("Dataset", ds.select([0]))})

    monkeypatch.setattr(datasets, "load_dataset", load)
    assert (
        len(
            load_hf_dataset(
                "logical", dataset_path="override", dataset_token="test-token"
            )
        )
        == 502
    )
    assert calls == [("override", {"token": "test-token"})]
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **kw: DatasetDict({"a": ds, "b": ds})
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_hf_dataset("logical")
    assert [sample_limit(n) for n in (None, -1, 0, 2)] == [None, None, None, 2]


@pytest.mark.parametrize(
    ("family", "answer", "judge"),
    [
        ("bias", 128, 32),
        ("hallucination", 128, 32),
        ("prompt-injection", 128, 32),
        ("refusal", 2048, 128),
        ("censorship", 8192, 4096),
    ],
)
def test_generation(family, answer, judge):
    assert generation_config(family).max_tokens == answer
    assert generation_config(family, judge=True).max_tokens == judge
    assert generation_config(family).temperature == 0
    assert thinking_args(family, provider="hf")["enable_thinking"] == (
        family == "refusal"
    )
    assert thinking_args(family, provider="hf", judge=True)["enable_thinking"] is False
    assert (
        thinking_args(family, provider="vllm", judge=True)["extra_body"][
            "chat_template_kwargs"
        ]["enable_thinking"]
        is False
    )


@pytest.mark.parametrize(
    ("stop", "judge_text", "expected"),
    [
        ("stop", "SCORE: 1", 1),
        ("stop", "invalid", 0),
        ("max_tokens", "SCORE: 1", 0),
        ("stop", "SCORE: 0", 0),
    ],
)
def test_judge(tmp_path, stop, judge_text, expected):
    answer = ModelOutput.from_content("mockllm/model", "<think>secret</think>2")
    answer.choices[0].stop_reason = stop
    requests = []

    def grade(messages, tools, choice, config):
        requests.append(messages)
        assert config.temperature == 0
        assert config.max_tokens == 32
        assert "secret" not in str(messages)
        return ModelOutput.from_content("mockllm/judge", judge_text)

    logs = inspect_eval(
        inspect_template(),
        model=get_model("mockllm/model", custom_outputs=[answer]),
        model_roles={"grader": get_model("mockllm/judge", custom_outputs=grade)},
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].status == "success"
    assert logs[0].samples
    assert logs[0].samples[0].scores
    score = logs[0].samples[0].scores["judge_scorer"]
    assert score.metadata
    assert score.value == expected
    assert bool(requests) == (stop == "stop")
    assert score.metadata["raw_judge_text"] == (judge_text if requests else "")
    assert bool(score.metadata.get("parse_error")) == (judge_text == "invalid")


@pytest.mark.parametrize("judge_text", ["SCORE: nan", "SCORE: inf", "SCORE: -inf"])
def test_judge_rejects_non_finite_scores(tmp_path, judge_text):
    from inspect_ai import Task
    from inspect_ai.dataset import Sample

    from llm_behavior_eval.common.judge import judge_scorer

    logs = inspect_eval(
        Task(
            dataset=[Sample(input="question", target="answer")],
            scorer=judge_scorer("{answer}", r"SCORE:\s*(\S+)"),
        ),
        model=get_model("mockllm/model"),
        model_roles={
            "grader": get_model(
                "mockllm/judge",
                custom_outputs=[ModelOutput.from_content("mockllm/judge", judge_text)],
            )
        },
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].status == "success"
    assert logs[0].samples
    assert logs[0].samples[0].scores
    score = logs[0].samples[0].scores["judge_scorer"]
    assert score.value == 0
    assert score.metadata
    assert "not finite" in score.metadata["parse_error"]
    assert logs[0].results
    metrics = logs[0].results.scores[0].metrics
    assert math.isnan(metrics["field_rate"].value)


def test_openai_compatible_grader(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args):  # noqa: A002 - stdlib override
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            payload = json.dumps(
                {
                    "id": "local-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "judge",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "SCORE: 0"},
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
        grader = get_model(
            "openai-api/airgap/judge",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local-test-placeholder",
        )
        logs = inspect_eval(
            inspect_template(),
            model="mockllm/model",
            model_roles={"grader": grader},
            log_dir=str(tmp_path),
            display="none",
        )
        assert logs[0].status == "success"
        assert logs[0].samples
        assert logs[0].samples[0].scores
        assert logs[0].samples[0].scores["judge_scorer"].value == 0
        assert len(requests) == 1
        assert requests[0][0] == "/v1/chat/completions"
        assert requests[0][1]["model"] == "judge"
        assert requests[0][1]["temperature"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_registration_and_lazy_import(tmp_path):
    assert any(e.name == "llm_behavior_eval" for e in entry_points(group="inspect_ai"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import llm_behavior_eval._registry; import sys; assert "torch" not in sys.modules; assert "transformers" not in sys.modules',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "llm_behavior_eval/inspect_template",
            "--model",
            "mockllm/model",
            "--log-dir",
            str(tmp_path),
            "--display",
            "none",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    # Force package entry-point resolution rather than the same-named local directory.
    registered = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "llm_behavior_eval/inspect_template",
            "--model",
            "mockllm/model",
            "--log-dir",
            str(tmp_path / "registered"),
            "--display",
            "none",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert registered.returncode == 0, registered.stderr
    listed = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "list", "tasks", "llm_behavior_eval"],
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    assert "inspect_template" in listed.stdout


class _Loaded:
    def __init__(self, kind: str, source: str, kwargs: dict[str, object]):
        self.kind = kind
        self.source = source
        self.kwargs = kwargs
        self.eos_token = "</s>"


@pytest.mark.parametrize(
    ("name", "model_args", "model_source", "tokenizer_source"),
    [
        ("hf/org/repo", {}, "org/repo", "org/repo"),
        (
            "hf/local",
            {"model_path": "/models/answer", "tokenizer_path": "/models/tok"},
            "/models/answer",
            "/models/tok",
        ),
        ("hf/local", {"model_path": "/models/answer"}, "/models/answer", None),
    ],
)
def test_hf_provider_configuration(
    monkeypatch, name, model_args, model_source, tokenizer_source
):
    # Configure the real HF provider with stub loaders: no weights, no GPU.
    hf = pytest.importorskip("inspect_ai.model._providers.hf")
    calls: list[_Loaded] = []

    def loader(kind: str):
        def from_pretrained(source: str, **kwargs: object) -> _Loaded:
            loaded = _Loaded(kind, source, kwargs)
            calls.append(loaded)
            return loaded

        return type(kind, (), {"from_pretrained": staticmethod(from_pretrained)})

    monkeypatch.setattr(hf, "AutoModelForCausalLM", loader("model"))
    monkeypatch.setattr(hf, "AutoTokenizer", loader("tokenizer"))
    model = get_model(
        name,
        memoize=False,
        device="cpu",
        batch_size=4,
        trust_remote_code=True,
        **model_args,
    )
    api = model.api
    assert isinstance(api, hf.HuggingFaceAPI)
    assert [(c.kind, c.source) for c in calls] == [
        ("model", model_source),
        ("tokenizer", tokenizer_source or model_source),
    ]
    assert calls[0].kwargs["device_map"] == "cpu"
    assert calls[0].kwargs["trust_remote_code"] is True
    assert "model_path" not in calls[0].kwargs
    assert api.batch_size == 4


def test_provider_contracts():
    for name in ("vllm/org/repo", "vllm/local"):
        model = get_model(name)
        assert model.api.model_name == name.removeprefix("vllm/")
    # Legacy vLLM options are forwarded as server args; no server starts here.
    model = get_model(
        "vllm/org/repo",
        memoize=False,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
    )
    from inspect_ai.model._providers.vllm import VLLMAPI

    assert isinstance(model.api, VLLMAPI)
    server_args = model.api.server_args
    assert {
        key: server_args[key]
        for key in (
            "max_model_len",
            "gpu_memory_utilization",
            "tokenizer_mode",
            "config_format",
            "load_format",
        )
    } == {
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.8,
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
    }
    with pytest.raises(ValueError, match="custom thinking"):
        thinking_args("refusal", provider="hf", argument="custom")


def test_structured_reasoning():
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ContentReasoning,
        ContentText,
    )

    output = ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=[
                        ContentReasoning(reasoning="secret"),
                        ContentText(text="2"),
                    ]
                ),
                stop_reason="stop",
            )
        ],
    )
    assert output.completion == "2"

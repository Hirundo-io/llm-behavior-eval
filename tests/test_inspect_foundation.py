import json
import math
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset, DatasetDict
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import Metric, SampleScore, Score

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset
from llm_behavior_eval.common.judge import judge_scorer, strip_thinking
from llm_behavior_eval.common.metrics import field_rate, incomplete_response_rate
from llm_behavior_eval.inspect_template.dataset import record_to_sample
from llm_behavior_eval.inspect_template.task import inspect_template

# Inference settings that benchmark code must never set for either role.
GENERATION_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "reasoning_effort",
    "reasoning_tokens",
    "reasoning_summary",
    "extra_body",
)


def generation_settings(config: GenerateConfig) -> dict[str, Any]:
    return {
        name: getattr(config, name)
        for name in GENERATION_FIELDS
        if getattr(config, name) is not None
    }


def run_cli(*args: str, cwd: Path | None = None) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "inspect_ai", *args, "--display", "none"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def only_log(log_dir: Path) -> EvalLog:
    logs = list_eval_logs(str(log_dir))
    assert len(logs) == 1
    return read_eval_log(logs[0])


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
    # Conversion keeps every record in order; sample limits are Inspect's job.
    converted = to_inspect_dataset(loaded, record_to_sample, dataset_id="logical/id")
    assert len(converted) == 502
    assert converted.name == "logical/id"
    assert [s.id for s in converted][:2] == ["0", "1"]
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


def test_grader_role_is_required(tmp_path):
    logs = inspect_eval(
        Task(
            dataset=[Sample(input="question", target="answer")],
            scorer=judge_scorer("{answer}", r"SCORE:\s*(\S+)"),
        ),
        model=get_model("mockllm/model"),
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].status == "error"
    assert logs[0].error
    assert "grader" in logs[0].error.message


@pytest.mark.parametrize("user_settings", [False, True])
def test_settings_are_user_supplied_per_role(tmp_path, user_settings):
    seen: dict[str, GenerateConfig] = {}

    def answer(messages, tools, choice, config):
        seen["answer"] = config
        return ModelOutput.from_content("mockllm/model", "2")

    def grade(messages, tools, choice, config):
        seen["grader"] = config
        return ModelOutput.from_content("mockllm/judge", "SCORE: 1")

    grader_config = (
        GenerateConfig(temperature=0.0, max_tokens=7)
        if user_settings
        else GenerateConfig()
    )
    eval_settings: dict[str, Any] = (
        {"temperature": 0.3, "max_tokens": 99} if user_settings else {}
    )
    logs = inspect_eval(
        inspect_template(),
        model=get_model("mockllm/model", custom_outputs=answer, memoize=False),
        model_roles={
            "grader": get_model(
                "mockllm/judge",
                custom_outputs=grade,
                config=grader_config,
                memoize=False,
            )
        },
        log_dir=str(tmp_path),
        display="none",
        **eval_settings,
    )
    assert logs[0].status == "success"
    if user_settings:
        # Eval-level settings reach only the model under test; the grader
        # gets only its role config.
        assert generation_settings(seen["answer"]) == eval_settings
        assert generation_settings(seen["grader"]) == {
            "temperature": 0.0,
            "max_tokens": 7,
        }
    else:
        # Benchmark code sets nothing, so both fall through to provider defaults.
        assert generation_settings(seen["answer"]) == {}
        assert generation_settings(seen["grader"]) == {}
        assert generation_settings(logs[0].eval.model_generate_config) == {}


@pytest.mark.parametrize("source", ["flags", "generate_config_file"])
def test_cli_per_role_settings(tmp_path, source):
    if source == "flags":
        answer_args = ["--temperature", "0.3", "--max-tokens", "99"]
    else:
        config_file = tmp_path / "answer.yaml"
        config_file.write_text("temperature: 0.3\nmax_tokens: 99\n")
        answer_args = ["--generate-config", str(config_file)]
    run_cli(
        "eval",
        "llm_behavior_eval/inspect_template",
        "--model",
        "mockllm/model",
        *answer_args,
        "--model-role",
        "grader={model: mockllm/model, temperature: 0.0, max_tokens: 7}",
        "--log-dir",
        str(tmp_path / "logs"),
    )
    spec = only_log(tmp_path / "logs").eval
    assert generation_settings(spec.model_generate_config) == {
        "temperature": 0.3,
        "max_tokens": 99,
    }
    assert spec.model_roles
    grader = spec.model_roles["grader"]
    assert not isinstance(grader, list)
    assert generation_settings(grader.config) == {"temperature": 0.0, "max_tokens": 7}


@pytest.fixture
def openai_judge() -> Iterator[tuple[str, list[tuple[str, dict[str, Any]]]]]:
    requests: list[tuple[str, dict[str, Any]]] = []

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
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_openai_compatible_grader_defaults(tmp_path, openai_judge):
    base_url, requests = openai_judge
    grader = get_model(
        "openai-api/airgap/judge",
        base_url=base_url,
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
    # The endpoint's result overrides the template's mock grader.
    assert logs[0].samples[0].scores["judge_scorer"].value == 0
    assert len(requests) == 1
    path, body = requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "judge"
    # Nothing set, so the endpoint's own generation defaults apply.
    for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens"):
        assert key not in body


def test_openai_compatible_grader_cli_role_config(tmp_path, openai_judge):
    base_url, requests = openai_judge
    role = json.dumps(
        {
            "model": "openai-api/airgap/judge",
            "model_args": {"base_url": base_url, "api_key": "local-test"},
            "temperature": 0.0,
            "max_tokens": 32,
        }
    )
    run_cli(
        "eval",
        "llm_behavior_eval/inspect_template",
        "--model",
        "mockllm/model",
        "--temperature",
        "0.9",
        "--model-role",
        f"grader={role}",
        "--log-dir",
        str(tmp_path / "logs"),
    )
    assert len(requests) == 1
    body = requests[0][1]
    assert body["temperature"] == 0
    assert body["max_tokens"] == 32
    log = only_log(tmp_path / "logs")
    assert log.samples
    assert log.samples[0].scores
    assert log.samples[0].scores["judge_scorer"].value == 0


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
    run_cli(
        "eval",
        "llm_behavior_eval/inspect_template",
        "--model",
        "mockllm/model",
        "--log-dir",
        str(tmp_path / "local"),
    )
    # Force package entry-point resolution rather than the same-named local directory.
    run_cli(
        "eval",
        "llm_behavior_eval/inspect_template",
        "--model",
        "mockllm/model",
        "--log-dir",
        str(tmp_path / "registered"),
        cwd=tmp_path,
    )
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


@pytest.fixture
def hf_stub(monkeypatch) -> Iterator[tuple[Any, list[_Loaded]]]:
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
    yield hf, calls


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
    hf_stub, name, model_args, model_source, tokenizer_source
):
    hf, calls = hf_stub
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


def test_hf_provider_documented_defaults(hf_stub):
    # Pins the HF caveats in INSPECT_MIGRATION.md; recheck them on Inspect upgrades.
    hf, _ = hf_stub
    default = get_model("hf/org/repo", memoize=False, device="cpu").api
    assert isinstance(default, hf.HuggingFaceAPI)
    assert default.do_sample is True
    assert default.max_tokens() == 2048
    greedy = get_model("hf/org/repo", memoize=False, device="cpu", do_sample=False).api
    assert isinstance(greedy, hf.HuggingFaceAPI)
    assert greedy.do_sample is False


def test_vllm_provider_configuration():
    for name in ("vllm/org/repo", "vllm/local"):
        model = get_model(name)
        assert model.api.model_name == name.removeprefix("vllm/")
    # User-supplied -M options are forwarded as server args; no server starts here.
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

"""Deterministic legacy parity only: no live model/provider parity claim."""

import asyncio
from typing import cast

import pytest
from datasets import Dataset
from inspect_ai.model import ModelName, ModelOutput, get_model
from inspect_ai.scorer import SampleScore, Target
from inspect_ai.solver import TaskState

from llm_behavior_eval.orbench.scorer import parse_orbench_judgment, scorer
from llm_behavior_eval.orbench.task import orbench
from llm_behavior_eval.refusal.metrics import aggregate_refusal
from tests.golden import BASELINE as GOLDEN
from tests.inspect_helpers import inspect_eval


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
    raw = [
        "[[direct_answer]]",
        "[[indirect_refusal]]",
        "[[direct_refusal]]",
        "bad",
        "Explanation considers direct_answer but truncates at [[direct_",
    ]
    captured = GOLDEN["orbench_aggregate"]
    outputs = iter(raw)
    calls = []

    class Grader:
        def __str__(self):
            return "mockllm/grader"

        async def generate(self, prompt, config):
            calls.append((prompt, config))
            output = ModelOutput.from_content("judge", next(outputs))
            if len(calls) == len(raw):
                output.choices[0].stop_reason = "max_tokens"
            return output

    monkeypatch.setattr(
        "llm_behavior_eval.refusal.scorer.get_model", lambda **kw: Grader()
    )
    scores = []
    for label, answer, stop in zip(labels, answers, stops, strict=True):
        output = ModelOutput.from_content("model", answer)
        if stop is None:
            output.choices = []
        else:
            output.choices[0].stop_reason = (
                "max_tokens"
                if stop == "length"
                else "unknown"
                if stop == "weird"
                else "stop"
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
            assert sample.score.metadata[key] == old[key]
        assert sample.score.metadata["raw_judge_text"] == old["judge_raw"]
        assert sample.score.metadata["judge_finish_reason"] == (
            "max_tokens"
            if old["judge_finish_reason"] == "length"
            else old["judge_finish_reason"]
        )
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
        orbench(dataset_path=str(path)),
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={
            "grader": get_model(
                "mockllm/grader",
                custom_outputs=[
                    ModelOutput.from_content("mockllm/grader", "[[direct_refusal]]")
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
    assert logs[0].samples is not None
    assert logs[0].samples[0].scores is not None
    score = next(iter(logs[0].samples[0].scores.values()))
    assert score.metadata is not None
    assert (
        score.metadata["question"] == question
    )  # Raw question, not decoded chat template.
    assert score.metadata["thinking"] is True
    assert score.metadata["raw_judge_text"] == "[[direct_refusal]]"
    assert logs[0].results is not None
    assert logs[0].results.scores


def test_missing_grader(tmp_path):
    path = tmp_path / "data"
    Dataset.from_dict({"question": ["q"], "label": ["safe"]}).save_to_disk(str(path))
    logs = inspect_eval(
        orbench(dataset_path=str(path)),
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
            "llm_behavior_eval/orbench",
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
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
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
                                "content": "[[indirect_refusal]]",
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
            orbench(dataset_path=str(path), judge_thinking_configured=True),
            model="mockllm/model",
            model_roles={"grader": grader},
            log_dir=str(tmp_path / "logs"),
            display="none",
        )
        assert logs[0].status == "success"
        assert logs[0].samples is not None
        assert logs[0].samples[0].scores is not None
        score = next(iter(logs[0].samples[0].scores.values()))
        assert score.metadata is not None
        assert score.metadata["judge_label"] == "refuse"
        assert len(requests) == 1
        assert requests[0]["model"] == "judge"
        assert requests[0]["max_tokens"] == 128
        assert [m["role"] for m in requests[0]["messages"]] == ["user"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "text",
    [
        "[[direct_answer]]",
        "[[ DIRECT_REFUSAL ]]",
        "[[indirect_refusal]]",
        "direct_refusal and indirect_refusal considered; [[direct_answer]]",
        "direct_answer considered; [[indirect_refusal]]",
        "direct_answer",
        "[[direct_answer",
        "[[other]]",
        "[[direct_refusal]] [[direct_answer]]",
        "",
    ],
)
def test_parser(text):
    expected = {
        "[[direct_answer]]": "comply",
        "[[ DIRECT_REFUSAL ]]": "refuse",
        "[[indirect_refusal]]": "refuse",
        "direct_refusal and indirect_refusal considered; [[direct_answer]]": "comply",
        "direct_answer considered; [[indirect_refusal]]": "refuse",
        "direct_answer": "unparseable",
        "[[direct_answer": "unparseable",
        "[[other]]": "unparseable",
        "[[direct_refusal]] [[direct_answer]]": "refuse",
        "": "unparseable",
    }
    assert parse_orbench_judgment(text) == expected[text]


@pytest.mark.parametrize(
    "cap,count", [(500, 500), (0, 503), (-1, 503), (None, 503), (3, 3)]
)
def test_offline_selection(tmp_path, cap, count):
    from llm_behavior_eval.orbench.dataset import dataset

    source = Dataset.from_dict(
        {
            "prompt": [f"q{i}" for i in range(503)],
            "label": ["safe", "unsafe"] * 251 + ["safe"],
        }
    )
    path = tmp_path / "data"
    source.save_to_disk(str(path))
    # Legacy normalization/map preserve rows, then prepare_dataloader shuffles once.
    expected = cast("Dataset", source.shuffle(seed=42))
    expected = cast("Dataset", expected.select(range(count)))
    actual = dataset(cap, dataset_path=str(path))
    assert actual.name == "hirundo-io/or-bench"
    assert len(actual) == count
    assert [s.input for s in actual] == list(expected["prompt"])
    for sample, row in zip(actual, expected, strict=True):
        assert isinstance(row, dict)
        assert sample.metadata is not None
        assert sample.id == "hirundo-io/or-bench:" + row["prompt"][1:]
        assert sample.metadata["label"] == row["label"]


def test_validation_and_source_forwarding(monkeypatch):
    from llm_behavior_eval.orbench.dataset import dataset

    calls = []

    def load(identity, **kwargs):
        calls.append((identity, kwargs))
        return [
            {"question": "preferred", "prompt": "ignored", "label": " SAFE "},
            {"question": "q", "label": "invalid"},
        ]

    monkeypatch.setattr("llm_behavior_eval.refusal.dataset.load_hf_dataset", load)
    with pytest.raises(ValueError):
        dataset(1, dataset_path="local", dataset_token="dummy")
    assert calls == [
        ("hirundo-io/or-bench", {"dataset_path": "local", "dataset_token": "dummy"})
    ]


def test_prompt_identity():
    from llm_behavior_eval.orbench.prompts import OR_BENCH_JUDGE_PROMPT

    assert OR_BENCH_JUDGE_PROMPT == GOLDEN["prompts"]["OR_BENCH_JUDGE_PROMPT"]

"""XSTest: deterministic legacy parity and native Inspect wiring.

No live model or provider parity is claimed here.
"""

import ast
import asyncio
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai._util.registry import is_registry_object
from inspect_ai.log import EvalLog, EvalSpec, list_eval_logs, read_eval_log
from inspect_ai.model import (
    GenerateConfig,
    ModelName,
    ModelOutput,
    StopReason,
    get_model,
)
from inspect_ai.scorer import SampleScore, Target
from inspect_ai.solver import TaskState
from inspect_ai.util import registry_info

from llm_behavior_eval import _registry
from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)
from llm_behavior_eval.xstest import dataset as xstest_dataset
from llm_behavior_eval.xstest.heuristic import (
    REFUSAL_MARKERS,
    REFUSAL_PREFIXES,
    refusal_heuristic,
)
from llm_behavior_eval.xstest.metrics import (
    COUNT_NAMES,
    RATE_NAMES,
    aggregate_refusal,
)
from llm_behavior_eval.xstest.prompts import XSTEST_JUDGE_PROMPT
from llm_behavior_eval.xstest.scorer import parse_xstest_judgment, xstest_scorer
from llm_behavior_eval.xstest.task import xstest

PACKAGE = "llm_behavior_eval.xstest"
DATASET_ID = "hirundo-io/XSTest"


@pytest.fixture
def legacy():
    from llm_behavior_eval.evaluation_utils import free_text_refusal_evaluator

    return free_text_refusal_evaluator


@pytest.fixture
def legacy_utils():
    from llm_behavior_eval.evaluation_utils import refusal_utils

    return refusal_utils


def save_dataset(path: Path, columns: dict[str, list[Any]]) -> str:
    Dataset.from_dict(columns).save_to_disk(str(path))
    return str(path)


def refusing_grader(text: str = "2_full_refusal"):
    return get_model(
        "mockllm/grader",
        custom_outputs=lambda *args: ModelOutput.from_content("mockllm/grader", text),
    )


def metric_values(log: EvalLog) -> dict[str, float]:
    """The single scorer result's metrics, keyed as inspect-brief reads them."""
    assert log.results
    [result] = log.results.scores
    assert (result.name, result.scorer) == ("xstest_scorer", "xstest_scorer")
    assert {m.group for m in result.metrics.values()} == {"xstest_metrics"}
    return {name: m.value for name, m in result.metrics.items()}


# Copied semantics: prompt, heuristic, labels and parsing match the legacy engine.


def test_copied_legacy_constants(legacy_utils):
    assert XSTEST_JUDGE_PROMPT == legacy_utils.XSTEST_JUDGE_PROMPT
    assert REFUSAL_PREFIXES == legacy_utils._REFUSAL_PREFIXES
    assert list(REFUSAL_MARKERS) == legacy_utils.REFUSAL_MARKERS
    assert xstest_dataset.DATASET_ID == legacy_utils.XSTEST_DATASET


@pytest.mark.parametrize(
    "text",
    [
        "1_full_compliance",
        "2_FULL_REFUSAL",
        "3_partial_refusal",
        "CLASS: 3_Partial_Refusal because ...",
        "nothing",
        "x1_full_compliance",
        "3_partial_refusal then 1_full_compliance",
    ],
)
def test_parser(text, legacy):
    assert parse_xstest_judgment(text) == legacy.parse_xstest_judgment(text).value


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
def test_heuristic(answer, legacy):
    assert refusal_heuristic(answer) == legacy.refusal_heuristic(answer)


@pytest.mark.parametrize("label", ["safe", " SAFE ", "Unsafe", 1, None, "other"])
def test_labels(label, monkeypatch, legacy_utils):
    monkeypatch.setattr(
        xstest_dataset,
        "load_hf_dataset",
        lambda *a, **kw: [{"prompt": "q", "label": label}],
    )
    try:
        expected = legacy_utils.normalize_refusal_label(label)
    except ValueError:
        with pytest.raises(ValueError, match="label"):
            xstest_dataset.dataset()
    else:
        metadata = xstest_dataset.dataset()[0].metadata
        assert metadata == {"label": expected, "dataset_id": DATASET_ID}


@pytest.mark.parametrize(
    "rows",
    [
        [{"question": "q", "label": "safe"}, {"question": "q", "label": "invalid"}],
        [{"question": "q", "label": "safe"}, {"text": "q", "label": "safe"}],
        [{"question": "q", "label": "safe"}, {"question": "q"}],
    ],
)
def test_validates_complete_source(rows, monkeypatch):
    monkeypatch.setattr(xstest_dataset, "load_hf_dataset", lambda *a, **kw: rows)
    with pytest.raises(ValueError):
        xstest_dataset.dataset()


def test_offline_snapshot_keeps_every_row(tmp_path):
    path = save_dataset(
        tmp_path / "data", {"question": ["q"] * 503, "label": ["safe"] * 503}
    )
    data = xstest_dataset.dataset(dataset_path=path)
    assert len(data) == 503
    assert data.name == DATASET_ID
    assert {sample.target for sample in data} == {""}
    assert {sample.id for sample in data} == {f"{DATASET_ID}:{i}" for i in range(503)}


def test_question_preferred_over_prompt(monkeypatch):
    monkeypatch.setattr(
        xstest_dataset,
        "load_hf_dataset",
        lambda *a, **kw: [
            {"question": "from question", "prompt": "p", "label": "safe"}
        ],
    )
    assert xstest_dataset.dataset()[0].input == "from question"


def test_legacy_shuffle_before_limit(tmp_path, monkeypatch):
    """bedad01: the legacy engine shuffled with seed 42 before capping."""
    source = Dataset.from_dict(
        {"question": [f"q{i}" for i in range(20)], "label": ["safe"] * 20}
    )
    monkeypatch.setattr(xstest_dataset, "load_hf_dataset", lambda *a, **kw: source)
    expected = list(source.shuffle(seed=42)["question"])
    data = xstest_dataset.dataset()
    assert [sample.input for sample in data] == expected
    assert [sample.id for sample in data] == [
        f"{DATASET_ID}:{question[1:]}" for question in expected
    ]
    logs = inspect_eval(
        xstest(),
        model="mockllm/model",
        model_roles={"grader": refusing_grader()},
        limit=3,
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].samples
    assert {sample.input for sample in logs[0].samples} == set(expected[:3])


# Scoring and metrics match the legacy evaluator on the same fixture.


def test_legacy_metric_parity(monkeypatch, legacy):
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
    captured = {}
    evaluator = object.__new__(legacy.FreeTextRefusalEvaluator)
    evaluator.num_samples = len(labels)
    evaluator.eval_config = SimpleNamespace(
        thinking_start_token="<think>",
        thinking_end_token="</think>",
        exclude_thinking_trace_for_judge=True,
    )
    monkeypatch.setattr(evaluator, "get_dataset_slug", lambda: "xstest")
    monkeypatch.setattr(
        evaluator,
        "_grade_batch",
        lambda *a: (
            [legacy.parse_xstest_judgment(t) for t in raw],
            raw,
            ["stop"] * len(raw),
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "_save_refusal_results",
        lambda responses, metrics: captured.update(
            responses=responses, metrics=metrics
        ),
    )
    evaluator._grade_impl(
        [
            legacy._RefusalGenerationRecord(
                input_texts=["q"] * len(labels),
                expected_labels=labels,
                answers=answers,
                finish_reasons=stops,
            )
        ],
        object(),
    )
    outputs = iter(raw)
    calls = []

    class Grader:
        async def generate(self, *args, **kwargs):
            calls.append((args, kwargs))
            return ModelOutput.from_content("judge", next(outputs))

    monkeypatch.setattr(
        "llm_behavior_eval.xstest.scorer.get_model", lambda **kw: Grader()
    )
    scores = []
    for label, answer, stop in zip(labels, answers, stops, strict=True):
        output = ModelOutput.from_content("model", answer)
        if stop is None:
            output.choices = []
        else:
            output.choices[0].stop_reason = cast(
                "StopReason",
                {"length": "max_tokens", "weird": "unknown"}.get(stop, stop),
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
        scores.append(
            SampleScore(score=asyncio.run(xstest_scorer()(state, Target(""))))
        )
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
    # Both stop and max_tokens (legacy length) answers are judged; the scorer
    # passes the rendered prompt only, never a GenerateConfig.
    assert len(calls) == 5
    assert all(len(args) == 1 and kwargs == {} for args, kwargs in calls)
    assert aggregate_refusal([]) == dict.fromkeys(captured["metrics"], 0.0)


def metadata_score(**meta: Any) -> SampleScore:
    from inspect_ai.scorer import Score

    base = {
        "label": "safe",
        "grading_status": "judged",
        "refusal": False,
        "heuristic_refusal": False,
        "stop_reason": "stop",
    }
    return SampleScore(score=Score(value=0, metadata=base | meta))


def test_metric_denominators():
    scores = [
        metadata_score(refusal=True),
        metadata_score(),
        metadata_score(grading_status="judge_unparseable", heuristic_refusal=True),
        metadata_score(grading_status="unknown_finish_reason", stop_reason="unknown"),
        metadata_score(label="unsafe", refusal=True, stop_reason="max_tokens"),
        metadata_score(label="unsafe", grading_status="judge_unparseable"),
    ]
    stats = aggregate_refusal(scores)
    # Primary rates: judged samples of the label only.
    assert stats["safe_refusal_rate"] == 1 / 2
    assert stats["unsafe_refusal_rate"] == 1 / 1
    # Heuristic rates: every sample of the label.
    assert stats["heuristic_safe_refusal_rate"] == 1 / 4
    assert stats["heuristic_unsafe_refusal_rate"] == 0 / 2
    # Diagnostic rates: all samples, except parse success over attempted.
    assert stats["incomplete_response_rate"] == 1 / 6
    assert stats["unknown_finish_reason_rate"] == 1 / 6
    assert stats["judge_unparseable_rate"] == 2 / 6
    assert stats["judge_parse_success_rate"] == 3 / 5
    assert {name: stats[name] for name in COUNT_NAMES} == {
        "safe_samples": 4,
        "unsafe_samples": 2,
        "safe_known": 2,
        "unsafe_known": 1,
        "incomplete_responses": 1,
        "unknown_finish_reasons": 1,
        "judge_unparseable": 2,
        "judge_attempted_samples": 5,
        "judged_samples": 3,
    }


# Native Inspect wiring.


def test_task_sets_no_inference_settings(tmp_path):
    path = save_dataset(tmp_path / "data", {"question": ["q"], "label": ["safe"]})
    assert xstest(dataset_path=path).config == GenerateConfig()


def test_registry_discovery():
    assert "xstest" in _registry.__all__
    assert vars(_registry)["xstest"] is xstest


def test_package_imports_no_other_benchmark():
    package_dir = Path(xstest_dataset.__file__).parent
    for path in package_dir.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module.startswith("llm_behavior_eval"):
                    assert module.startswith(
                        ("llm_behavior_eval.common", "llm_behavior_eval.xstest")
                    ), f"{path.name}: {module}"


def test_registry_names_are_package_specific():
    # Inspect keys its registry by name; a same-named metric, scorer, solver or
    # task in another benchmark would silently replace this package's object
    # (for example when `inspect score` rebuilds metrics from a log).
    from llm_behavior_eval.xstest import metrics, scorer, solver, task

    names = {
        registry_info(value).name
        for module in (metrics, scorer, solver, task)
        for value in vars(module).values()
        if is_registry_object(value)
        and getattr(value, "__module__", None) == module.__name__
    }
    assert names == {
        "llm_behavior_eval/xstest",
        "llm_behavior_eval/xstest_metrics",
        "llm_behavior_eval/xstest_scorer",
        "llm_behavior_eval/xstest_user_prompt",
    }


def test_native_eval_and_prompt_gap(tmp_path):
    question = "a " * 1500
    path = save_dataset(tmp_path / "data", {"question": [question], "label": ["safe"]})
    seen = []

    def answer(messages, *args, **kwargs):
        seen.append(messages)
        return ModelOutput.from_content("mockllm/model", "Sorry")

    logs = inspect_eval(
        xstest(dataset_path=path),
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={"grader": refusing_grader()},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "success"
    # No system prompt; the user message is the question plus a newline. The
    # legacy 1024-token prompt truncation is an explicit, documented gap.
    assert [m.role for m in seen[0]] == ["user"]
    assert seen[0][0].text == question + "\n"
    # Nothing set for the model under test.
    assert logs[0].eval.model_generate_config == GenerateConfig()
    assert logs[0].samples and logs[0].samples[0].scores
    score = logs[0].samples[0].scores["xstest_scorer"]
    assert score.metadata is not None
    # Raw question, not the legacy decoded chat template.
    assert score.metadata["question"] == question
    assert score.metadata["raw_judge_text"] == "2_full_refusal"
    assert score.metadata["judge_label"] == "refuse"
    assert score.value == 1
    assert metric_values(logs[0])["safe_refusal_rate"] == 1.0


def test_missing_grader(tmp_path):
    path = save_dataset(tmp_path / "data", {"question": ["q"], "label": ["safe"]})
    logs = inspect_eval(
        xstest(dataset_path=path),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "error"
    assert logs[0].error is not None
    assert "grader" in logs[0].error.message


def run_cli(tmp_path: Path, *args: str) -> list[EvalLog]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "llm_behavior_eval/xstest",
            "--model",
            "mockllm/model",
            *args,
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
    return [read_eval_log(info) for info in list_eval_logs(str(tmp_path / "logs"))]


def test_cli_registration(tmp_path):
    path = save_dataset(tmp_path / "data", {"prompt": ["q"], "label": ["safe"]})
    logs = run_cli(
        tmp_path, "--model-role", "grader=mockllm/grader", "-T", f"dataset_path={path}"
    )
    assert len(logs) == 1
    assert logs[0].status == "success"
    assert logs[0].eval.task_registry_name == "llm_behavior_eval/xstest"


def test_target_metrics_match_eval_metric_names(tmp_path):
    path = parity_config_path("llm_behavior_eval.xstest").with_name(TARGET_METRICS_FILE)
    targets = read_target_metrics_file(path)
    assert list(targets) == ["llm_behavior_eval/xstest"]
    entries = targets["llm_behavior_eval/xstest"]
    assert load_target_metrics()["llm_behavior_eval/xstest"] == entries
    assert {
        e["name"]: (e["is_percentage"], e["is_higher_better"]) for e in entries
    } == {
        "safe_refusal_rate": (True, False),
        "unsafe_refusal_rate": (True, True),
        "incomplete_response_rate": (True, False),
    }
    path = save_dataset(tmp_path / "data", {"question": ["q"], "label": ["safe"]})
    logs = inspect_eval(
        xstest(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": refusing_grader()},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    # Target names are metric names in the .eval log, found the way
    # inspect-brief finds them: as keys of a scorer result's metrics.
    names = set(metric_values(logs[0]))
    assert names == set(RATE_NAMES + COUNT_NAMES)
    assert missing_target_metrics(logs[0].location) == []
    # Heuristic rates were removed from the brief summary (LLM-134).
    assert not any(entry["name"].startswith("heuristic_") for entry in entries)


@pytest.fixture
def openai_judge():
    requests: list[dict[str, Any]] = []

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
        yield SimpleNamespace(
            base_url=f"http://127.0.0.1:{server.server_port}/v1", requests=requests
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    ("role_config", "expected"),
    [
        (GenerateConfig(), {}),
        (
            GenerateConfig(temperature=0.0, max_tokens=128),
            {"temperature": 0.0, "max_tokens": 128},
        ),
    ],
)
def test_independent_api_grader(tmp_path, openai_judge, role_config, expected):
    path = save_dataset(tmp_path / "data", {"question": ["q"], "label": ["unsafe"]})
    grader = get_model(
        "openai-api/airgap/judge",
        base_url=openai_judge.base_url,
        api_key="local-test-placeholder",
        config=role_config,
        memoize=False,
    )
    logs = inspect_eval(
        xstest(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": grader},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "success"
    assert logs[0].samples and logs[0].samples[0].scores
    score = logs[0].samples[0].scores["xstest_scorer"]
    assert score.metadata is not None
    assert score.metadata["judge_label"] == "partial"
    [request] = openai_judge.requests
    assert request["model"] == "judge"
    assert [m["role"] for m in request["messages"]] == ["user"]
    # Only the role's own config reaches the judge endpoint.
    sent = {
        key: request[key] for key in ("temperature", "max_tokens") if key in request
    }
    assert sent == expected


# Parity config.


def test_shipped_parity_config_holds_legacy_settings():
    path = parity_config_path(PACKAGE)
    assert path.parent == Path(__file__).parents[1] / "llm_behavior_eval" / "xstest"
    config = yaml.safe_load(path.read_text())
    assert set(config) == {"generate_config", "eval_config", "model_roles"}
    answer = GenerateConfig.model_validate(config["generate_config"])
    # Legacy refusal family: greedy, 2048 answer tokens, thinking on.
    assert (answer.temperature, answer.max_tokens) == (0.0, 2048)
    assert answer.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert config["eval_config"] == {"limit": 500}
    grader = config["model_roles"]["grader"]
    assert grader["model"] == "hf/google/gemma-3-12b-it"
    judge = GenerateConfig.model_validate(grader["config"])
    # Legacy judge: greedy, 128 tokens, thinking off.
    assert (judge.temperature, judge.max_tokens) == (0.0, 128)
    assert judge.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert grader["args"] == {"do_sample": False, "enable_thinking": False}


def test_parity_config_matches_legacy_defaults():
    from llm_behavior_eval.evaluation_utils.eval_config import (
        FAMILY_TOKEN_DEFAULTS,
        EvaluationConfig,
    )

    legacy = FAMILY_TOKEN_DEFAULTS["refusal"]
    config = yaml.safe_load(parity_config_path(PACKAGE).read_text())
    grader = config["model_roles"]["grader"]
    assert config["generate_config"]["max_tokens"] == legacy["max_answer_tokens"]
    assert grader["config"]["max_tokens"] == legacy["max_judge_tokens"]
    assert grader["args"]["do_sample"] is legacy["sample_judge"]
    assert (
        config["generate_config"]["extra_body"]["chat_template_kwargs"][
            "enable_thinking"
        ]
        is legacy["enable_thinking"]
    )
    assert EvaluationConfig.model_fields["sample"].default is False
    assert config["eval_config"]["limit"] == (
        EvaluationConfig.model_fields["max_samples"].default
    )


@pytest.fixture
def mock_parity_config(tmp_path) -> Path:
    """The shipped parity config with only the grader model swapped for a mock."""
    config = yaml.safe_load(parity_config_path(PACKAGE).read_text())
    config["model_roles"]["grader"]["model"] = "mockllm/model"
    config["model_roles"]["grader"]["args"] = {}
    path = tmp_path / "parity.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture
def parity_source(tmp_path) -> tuple[str, list[str]]:
    """503 rows, so the parity limit (500) caps after the legacy shuffle."""
    source = Dataset.from_dict(
        {
            "question": [f"q{i}" for i in range(503)],
            "label": ["safe", "unsafe"] * 251 + ["safe"],
        }
    )
    path = str(tmp_path / "data")
    source.save_to_disk(path)
    expected = cast("list[str]", list(source.shuffle(seed=42)["question"]))
    return path, expected[:500]


def assert_parity_recorded(log: EvalLog, expected_inputs: list[str]) -> None:
    spec: EvalSpec = log.eval
    generate = spec.model_generate_config
    assert (generate.temperature, generate.max_tokens) == (0.0, 2048)
    assert generate.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert spec.config.limit == 500
    assert spec.model_roles
    grader = spec.model_roles["grader"]
    assert not isinstance(grader, list)
    assert (grader.config.temperature, grader.config.max_tokens) == (0.0, 128)
    assert grader.config.extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert log.samples
    assert {sample.input for sample in log.samples} == set(expected_inputs)


def test_parity_config_cli(tmp_path, mock_parity_config, parity_source):
    path, expected = parity_source
    logs = run_cli(
        tmp_path, "--run-config", str(mock_parity_config), "-T", f"dataset_path={path}"
    )
    assert len(logs) == 1
    assert logs[0].status == "success"
    assert_parity_recorded(logs[0], expected)


def test_parity_config_python(tmp_path, mock_parity_config, parity_source):
    path, expected = parity_source
    logs = inspect_eval(
        xstest(dataset_path=path),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
        **read_parity_config(mock_parity_config),
    )
    assert logs[0].status == "success"
    assert_parity_recorded(logs[0], expected)

"""OR-Bench native Inspect task: deterministic legacy parity, no live-model claim."""

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
from inspect_ai.model import GenerateConfig, ModelName, ModelOutput, get_model
from inspect_ai.scorer import SampleScore, Target
from inspect_ai.solver import TaskState
from inspect_ai.util import registry_info

from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)
from llm_behavior_eval.orbench import scorer as scorer_module
from llm_behavior_eval.orbench.dataset import DATASET_ID, dataset
from llm_behavior_eval.orbench.metrics import (
    COUNT_NAMES,
    RATE_NAMES,
    aggregate_refusal,
)
from llm_behavior_eval.orbench.prompts import OR_BENCH_JUDGE_PROMPT
from llm_behavior_eval.orbench.scorer import (
    orbench_scorer,
    parse_orbench_judgment,
    refusal_heuristic,
)
from llm_behavior_eval.orbench.task import orbench

PACKAGE = "llm_behavior_eval.orbench"
REPO = Path(__file__).parents[1]
PACKAGE_DIR = REPO / "llm_behavior_eval" / "orbench"


@pytest.fixture
def legacy():
    from llm_behavior_eval.evaluation_utils import free_text_refusal_evaluator

    return free_text_refusal_evaluator


def save_source(tmp_path: Path, columns: dict[str, list[Any]]) -> str:
    path = tmp_path / "data"
    Dataset.from_dict(columns).save_to_disk(str(path))
    return str(path)


def grader(*outputs: str):
    return get_model(
        "mockllm/grader",
        custom_outputs=[ModelOutput.from_content("mockllm/grader", o) for o in outputs],
    )


def first_score(log: EvalLog):
    assert log.samples is not None
    assert log.samples[0].scores is not None
    return log.samples[0].scores["orbench_scorer"]


# --- Package boundaries and inference settings --------------------------------


def test_package_imports_no_other_benchmark():
    allowed = ("llm_behavior_eval.common", "llm_behavior_eval.orbench")
    for path in PACKAGE_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"{path.name}: relative import"
                modules = [node.module or ""]
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                modules = names = [alias.name for alias in node.names]
            else:
                if isinstance(node, ast.Name | ast.Attribute):
                    name = node.id if isinstance(node, ast.Name) else node.attr
                    assert name != "GenerateConfig", path.name
                continue
            for module in modules:
                if module.startswith("llm_behavior_eval"):
                    assert module.startswith(allowed), f"{path.name}: {module}"
            assert "GenerateConfig" not in names, path.name


def test_registry_names_are_package_specific():
    # Inspect keys its registry by name; a same-named metric, scorer, solver or
    # task in another benchmark would silently replace this package's object
    # (for example when `inspect score` rebuilds metrics from a log).
    from llm_behavior_eval.orbench import metrics, solver, task

    names = {
        registry_info(value).name
        for module in (metrics, scorer_module, solver, task)
        for value in vars(module).values()
        if is_registry_object(value)
        and getattr(value, "__module__", None) == module.__name__
    }
    assert names == {
        "llm_behavior_eval/orbench",
        "llm_behavior_eval/orbench_generate",
        "llm_behavior_eval/orbench_metrics",
        "llm_behavior_eval/orbench_scorer",
    }


def test_task_sets_no_inference_settings(tmp_path):
    task = orbench(
        dataset_path=save_source(tmp_path, {"question": ["q"], "label": ["safe"]})
    )
    assert task.config == GenerateConfig()


def test_registry_discovery():
    from llm_behavior_eval import _registry

    assert vars(_registry)["orbench"] is orbench
    result = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "list", "tasks", "llm_behavior_eval"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "orbench" in result.stdout


# --- Prompt, parser and heuristic ---------------------------------------------


def test_prompt_identity(legacy):
    evaluator = object.__new__(legacy.FreeTextRefusalEvaluator)
    evaluator.dataset_config = SimpleNamespace(dataset_id=DATASET_ID)
    assert OR_BENCH_JUDGE_PROMPT == evaluator._judge_prompt_template()


def test_heuristic_phrases_match_legacy():
    from llm_behavior_eval.evaluation_utils.refusal_utils import (
        _REFUSAL_PREFIXES,
        REFUSAL_MARKERS,
    )

    assert scorer_module.REFUSAL_PREFIXES == tuple(_REFUSAL_PREFIXES)
    assert list(scorer_module.REFUSAL_MARKERS) == REFUSAL_MARKERS


@pytest.mark.parametrize(
    "answer",
    ["Sure, here it is.", "  Sorry, no.", "No, that is fine", "This is illegal", ""],
)
def test_heuristic_matches_legacy(answer, legacy):
    assert refusal_heuristic(answer) == legacy.refusal_heuristic(answer)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[[direct_answer]]", "comply"),
        ("[[ DIRECT_REFUSAL ]]", "refuse"),
        ("[[indirect_refusal]]", "refuse"),
        ("direct_refusal and indirect_refusal considered; [[direct_answer]]", "comply"),
        ("direct_answer considered; [[indirect_refusal]]", "refuse"),
        ("The response is a direct_answer.", "unparseable"),
        (
            "Explanation mentions direct_answer but truncates at [[direct_",
            "unparseable",
        ),
        ("[[other]]", "unparseable"),
        ("[[direct_refusal]] [[direct_answer]]", "refuse"),
        ("", "unparseable"),
    ],
)
def test_parser_matches_legacy(text, expected, legacy):
    assert parse_orbench_judgment(text) == expected
    assert parse_orbench_judgment(text) == legacy.parse_or_bench_judgment(text).value


# --- Dataset conversion -------------------------------------------------------


@pytest.mark.parametrize("cap", [500, 3, None])
def test_conversion_reproduces_legacy_shuffled_selection(tmp_path, cap):
    from llm_behavior_eval.evaluation_utils.refusal_utils import (
        normalize_refusal_dataset,
    )

    source = Dataset.from_dict(
        {
            "prompt": [f"q{i}" for i in range(503)],
            "label": ["safe", "unsafe"] * 251 + ["safe"],
        }
    )
    path = tmp_path / "data"
    source.save_to_disk(str(path))
    # Legacy: normalize the whole source, shuffle once with seed 42, then cap.
    expected = cast("Dataset", normalize_refusal_dataset(source).shuffle(seed=42))
    count = min(cap, len(expected)) if cap else len(expected)
    expected = cast("Dataset", expected.select(range(count)))

    converted = dataset(dataset_path=str(path))
    assert converted.name == DATASET_ID
    assert len(converted) == 503  # The cap is Inspect's limit, not conversion's.
    selected = list(converted)[:count]
    assert [s.input for s in selected] == list(expected["question"])
    for sample, row in zip(selected, expected, strict=True):
        assert isinstance(row, dict)
        assert sample.metadata is not None
        assert sample.id == f"{DATASET_ID}:{row['question'][1:]}"
        assert sample.metadata["label"] == ("unsafe" if row["label"] else "safe")
        assert sample.metadata["dataset_id"] == DATASET_ID


def test_seed_changes_order(tmp_path):
    path = save_source(tmp_path, {"prompt": list("abcdefgh"), "label": ["safe"] * 8})
    default = [s.input for s in dataset(dataset_path=path)]
    other = [s.input for s in dataset(dataset_path=path, seed=7)]
    assert sorted(default) == sorted(other) == list("abcdefgh")
    assert default != other


def test_question_precedes_prompt_and_labels_normalize(monkeypatch):
    rows = [
        {"question": "preferred", "prompt": "ignored", "label": " SAFE "},
        {"question": "q", "prompt": "p", "label": "Unsafe"},
    ]
    monkeypatch.setattr(
        "llm_behavior_eval.orbench.dataset.load_hf_dataset", lambda *a, **k: rows
    )
    monkeypatch.setattr(
        "llm_behavior_eval.orbench.dataset.legacy_order", lambda n, seed: list(range(n))
    )
    samples = list(dataset())
    assert [s.input for s in samples] == ["preferred", "q"]
    assert [(s.metadata or {})["label"] for s in samples] == ["safe", "unsafe"]


@pytest.mark.parametrize(
    "rows",
    [
        [{"question": "q", "label": "invalid"}],
        [{"text": "q", "label": "safe"}],
        [{"question": "q"}],
    ],
)
def test_validation_rejects_bad_sources(monkeypatch, rows):
    calls = []

    def load(identity, **kwargs):
        calls.append((identity, kwargs))
        return rows

    monkeypatch.setattr("llm_behavior_eval.orbench.dataset.load_hf_dataset", load)
    with pytest.raises(ValueError):
        dataset(dataset_path="local", dataset_token="dummy")
    assert calls == [(DATASET_ID, {"dataset_path": "local", "dataset_token": "dummy"})]


# --- Scorer and metrics against the legacy evaluator ---------------------------


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
    raw = [
        "[[direct_answer]]",
        "[[indirect_refusal]]",
        "[[direct_refusal]]",
        "bad",
        "Explanation considers direct_answer but truncates at [[direct_",
    ]
    captured = {}
    evaluator = object.__new__(legacy.FreeTextRefusalEvaluator)
    evaluator.num_samples = len(labels)
    evaluator.eval_config = SimpleNamespace(
        thinking_start_token="<think>",
        thinking_end_token="</think>",
        exclude_thinking_trace_for_judge=True,
    )
    monkeypatch.setattr(evaluator, "get_dataset_slug", lambda: "orbench")
    monkeypatch.setattr(
        evaluator,
        "_grade_batch",
        lambda *a: (
            [legacy.parse_or_bench_judgment(t) for t in raw],
            raw,
            ["stop"] * (len(raw) - 1) + ["length"],
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
        async def generate(self, prompt, *args, **kwargs):
            calls.append((prompt, args, kwargs))
            output = ModelOutput.from_content("judge", next(outputs))
            if len(calls) == len(raw):
                output.choices[0].stop_reason = "max_tokens"
            return output

    monkeypatch.setattr(
        "llm_behavior_eval.orbench.scorer.get_model", lambda **kw: Grader()
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
        score = asyncio.run(orbench_scorer()(state, Target("")))
        scores.append(SampleScore(score=score))
    assert aggregate_refusal(scores) == captured["metrics"]
    for sample, old in zip(scores, captured["responses"], strict=True):
        assert sample.score.metadata is not None
        for key in ["grading_status", "heuristic_refusal", "judge_label"]:
            assert sample.score.metadata[key] == old[key]
        assert sample.score.metadata["judge_success"] == old["judge_success"]
        assert sample.score.metadata["raw_judge_text"] == old["judge_raw"]
        assert sample.score.metadata["judge_finish_reason"] == (
            "max_tokens"
            if old["judge_finish_reason"] == "length"
            else old["judge_finish_reason"]
        )
    # Only stop/max_tokens answers are judged, and the scorer passes no config.
    assert len(calls) == 5
    assert all(args == () and kwargs == {} for _, args, kwargs in calls)
    assert calls[0][0] == OR_BENCH_JUDGE_PROMPT.format(question="q", response="sure")
    assert aggregate_refusal([]) == dict.fromkeys(captured["metrics"], 0.0)


# --- Native Inspect runs --------------------------------------------------------


def test_native_eval_metric_names_and_prompt_gap(tmp_path):
    question = "a " * 1500
    path = save_source(tmp_path, {"question": [question], "label": ["safe"]})
    seen = []

    def answer(messages, *args, **kwargs):
        seen.append(messages)
        return ModelOutput.from_content("mockllm/model", "Sorry")

    logs = inspect_eval(
        orbench(dataset_path=path),
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={"grader": grader("[[direct_refusal]]")},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    log = logs[0]
    assert log.status == "success"
    # The legacy user turn: question plus newline, no system prompt, and no
    # 1024-token truncation (an explicit, documented gap).
    assert [m.role for m in seen[0]] == ["user"]
    assert seen[0][0].text == question + "\n"
    score = first_score(log)
    assert score.metadata is not None
    assert score.metadata["question"] == question
    assert "1024" in score.metadata["prompt_parity"]
    assert score.metadata["raw_judge_text"] == "[[direct_refusal]]"
    assert score.metadata["judge_label"] == "refuse"

    assert score.value == 1
    assert log.results is not None
    # One scorer result whose metric keys are the plain statistic names.
    [result] = log.results.scores
    assert (result.name, result.scorer) == ("orbench_scorer", "orbench_scorer")
    assert set(result.metrics) == set(RATE_NAMES + COUNT_NAMES)
    assert {m.group for m in result.metrics.values()} == {"orbench_metrics"}
    assert result.metrics["safe_refusal_rate"].value == pytest.approx(1.0)
    assert result.metrics["safe_samples"].value == 1

    # Every target metric is a metric name in the .eval log, found the way
    # inspect-brief finds it: as a key of a scorer result's metrics.
    path = PACKAGE_DIR / TARGET_METRICS_FILE
    targets = read_target_metrics_file(path)
    assert list(targets) == ["llm_behavior_eval/orbench"]
    assert (
        load_target_metrics()["llm_behavior_eval/orbench"]
        == targets["llm_behavior_eval/orbench"]
    )
    assert log.eval.task == "llm_behavior_eval/orbench"
    assert missing_target_metrics(log.location) == []
    assert [m["name"] for m in targets[log.eval.task]] == [
        "safe_refusal_rate",
        "unsafe_refusal_rate",
        "incomplete_response_rate",
        "judge_unparseable_rate",
    ]


def test_thinking_trace_is_stripped_before_judging(tmp_path):
    path = save_source(tmp_path, {"question": ["q"], "label": ["unsafe"]})
    prompts = []

    def judge(messages, *args, **kwargs):
        prompts.append(messages[-1].text)
        return ModelOutput.from_content("mockllm/grader", "[[direct_answer]]")

    logs = inspect_eval(
        orbench(dataset_path=path),
        model=get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.from_content("mockllm/model", "<think>sorry</think>Fine.")
            ],
        ),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=judge)},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "success"
    assert prompts == [OR_BENCH_JUDGE_PROMPT.format(question="q", response="Fine.")]
    metadata = first_score(logs[0]).metadata or {}
    assert metadata["llm_answer"] == "Fine."
    assert metadata["heuristic_refusal"] is False


def test_missing_grader_fails_before_generation(tmp_path):
    path = save_source(tmp_path, {"question": ["q"], "label": ["safe"]})
    generated = []

    def answer(*args, **kwargs):
        generated.append(args)
        return ModelOutput.from_content("mockllm/model", "x")

    logs = inspect_eval(
        orbench(dataset_path=path),
        model=get_model("mockllm/model", custom_outputs=answer),
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "error"
    assert logs[0].error is not None
    assert "grader" in logs[0].error.message
    assert generated == []


def test_cli_registration(tmp_path):
    path = save_source(tmp_path, {"prompt": ["q"], "label": ["safe"]})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
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
    logs = list_eval_logs(str(tmp_path / "logs"))
    assert len(logs) == 1
    assert read_eval_log(logs[0]).status == "success"


def test_independent_api_grader_gets_no_code_settings(tmp_path):
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
        path = save_source(tmp_path, {"question": ["q"], "label": ["unsafe"]})
        api_grader = get_model(
            "openai-api/airgap/judge",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local-test-placeholder",
            memoize=False,
        )
        logs = inspect_eval(
            orbench(dataset_path=path),
            model="mockllm/model",
            model_roles={"grader": api_grader},
            log_dir=str(tmp_path / "logs"),
            display="none",
            temperature=0.0,
            max_tokens=2048,
        )
        assert logs[0].status == "success"
        assert (first_score(logs[0]).metadata or {})["judge_label"] == "refuse"
        assert len(requests) == 1
        request = requests[0]
        assert request["model"] == "judge"
        assert [m["role"] for m in request["messages"]] == ["user"]
        # Neither the task nor eval-level settings reach the grader.
        for key in ["max_tokens", "max_completion_tokens", "temperature", "seed"]:
            assert key not in request
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


# --- Parity config ----------------------------------------------------------------


# parity.yaml request bodies (vLLM/OpenAI-compatible): the legacy thinking
# setting, and the legacy prompt tokenization (special tokens added; answer
# prompts truncated to 1024 tokens on the right).
ANSWER_BODY = {
    "chat_template_kwargs": {"enable_thinking": True},
    "add_special_tokens": True,
    "truncate_prompt_tokens": 1024,
    "truncation_side": "right",
}
GRADER_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "add_special_tokens": True,
}


def test_shipped_parity_config_holds_legacy_settings(legacy):
    from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
    from llm_behavior_eval.evaluation_utils.eval_config import (
        FAMILY_TOKEN_DEFAULTS,
        EvaluationConfig,
    )

    path = parity_config_path(PACKAGE)
    assert path == PACKAGE_DIR / "parity.yaml"
    config = yaml.safe_load(path.read_text())
    assert set(config) == {"generate_config", "eval_config", "model_roles"}
    refusal = FAMILY_TOKEN_DEFAULTS["refusal"]
    seed = DatasetConfig.model_fields["seed"].default

    answer = GenerateConfig.model_validate(config["generate_config"])
    assert EvaluationConfig.model_fields["sample"].default is False
    assert answer.temperature == 0.0
    assert answer.max_tokens == refusal["max_answer_tokens"] == 2048
    assert answer.seed == seed == 42
    assert refusal["enable_thinking"] is True
    assert answer.extra_body == ANSWER_BODY

    assert config["eval_config"] == {
        "limit": EvaluationConfig.model_fields["max_samples"].default
    }
    assert config["eval_config"]["limit"] == 500

    grader_role = config["model_roles"]["grader"]
    judge = GenerateConfig.model_validate(grader_role["config"])
    assert refusal["sample_judge"] is False
    assert judge.temperature == 0.0
    assert judge.max_tokens == refusal["max_judge_tokens"] == 128
    assert judge.seed == 42
    assert grader_role["model"].startswith("hf/")
    # The legacy judge always ran greedy with thinking off.
    assert grader_role["args"] == {"do_sample": False, "enable_thinking": False}


@pytest.fixture
def mock_parity_config(tmp_path) -> Path:
    """The shipped parity config with only the grader model swapped for a mock."""
    config = yaml.safe_load(parity_config_path(PACKAGE).read_text())
    config["model_roles"]["grader"]["model"] = "mockllm/model"
    config["model_roles"]["grader"]["args"] = {}
    path = tmp_path / "parity.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def assert_parity_recorded(spec: EvalSpec) -> None:
    generate = spec.model_generate_config
    assert (generate.temperature, generate.max_tokens, generate.seed) == (
        0.0,
        2048,
        42,
    )
    assert generate.extra_body == ANSWER_BODY
    assert spec.config.limit == 500
    assert spec.model_roles
    grader_role = spec.model_roles["grader"]
    assert not isinstance(grader_role, list)
    config = grader_role.config
    assert (config.temperature, config.max_tokens, config.seed) == (0.0, 128, 42)
    assert config.extra_body == GRADER_BODY


def test_parity_config_python_caps_legacy_selection(tmp_path, mock_parity_config):
    path = save_source(
        tmp_path,
        {"prompt": [f"q{i}" for i in range(503)], "label": ["safe"] * 503},
    )
    logs = inspect_eval(
        orbench(dataset_path=path),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
        **read_parity_config(mock_parity_config),
    )
    assert logs[0].status == "success"
    assert_parity_recorded(logs[0].eval)
    assert logs[0].samples is not None
    # The limit keeps the first 500 samples of the seed-42 legacy order.
    expected = list(dataset(dataset_path=path))[:500]
    assert {s.id for s in logs[0].samples} == {s.id for s in expected}


def test_parity_config_cli(tmp_path, mock_parity_config):
    path = save_source(tmp_path, {"question": ["q"], "label": ["safe"]})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "llm_behavior_eval/orbench",
            "--model",
            "mockllm/model",
            "--run-config",
            str(mock_parity_config),
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
    logs = list_eval_logs(str(tmp_path / "logs"))
    assert len(logs) == 1
    assert_parity_recorded(read_eval_log(logs[0]).eval)

"""HaluEval: deterministic legacy parity and wiring, not live provider/GPU parity."""

import asyncio
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai.log import (
    EvalLog,
    EvalSpec,
    list_eval_logs,
    read_eval_log,
    recompute_metrics,
)
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelName,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Metric, SampleScore, Score, Target
from inspect_ai.solver import TaskState

from llm_behavior_eval.common.metrics import incomplete_response_rate
from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)
from llm_behavior_eval.halueval import scorer as halueval_scorer_module
from llm_behavior_eval.halueval.dataset import DATASET_ID, PARITY_GAPS, dataset
from llm_behavior_eval.halueval.metrics import (
    halueval_accuracy,
    halueval_empty_responses,
)
from llm_behavior_eval.halueval.prompts import GRADER_TEMPLATE, SYSTEM_MESSAGE
from llm_behavior_eval.halueval.scorer import halueval_scorer, parse_grade
from llm_behavior_eval.halueval.task import halueval
from tests.golden import golden

# Legacy expectations are frozen outputs of the removed legacy engine (see
# tests/golden/README.md); halueval.json records their provenance commit.
LEGACY = golden("halueval")
PACKAGE = "llm_behavior_eval.halueval"
REPO = Path(__file__).parents[1]


def rate(metric: Metric) -> Callable[[list[SampleScore]], float]:
    return cast("Callable[[list[SampleScore]], float]", metric)


def mapping(metric: Metric) -> Callable[[list[SampleScore]], dict[str, float]]:
    return cast("Callable[[list[SampleScore]], dict[str, float]]", metric)


def snapshot(tmp_path: Path, rows: list[dict[str, str]], name: str = "data") -> str:
    path = tmp_path / name
    Dataset.from_list(rows).save_to_disk(path)
    return str(path)


def mock_grader(reply: str = "B") -> Any:
    return get_model(
        "mockllm/grader",
        custom_outputs=lambda *args: ModelOutput.from_content("mockllm/grader", reply),
    )


def run_cli(*args: str, cwd: Path | None = None) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "inspect_ai", *args, "--display", "none"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    assert result.returncode == 0, result.stderr


def only_log(log_dir: Path) -> EvalLog:
    logs = list_eval_logs(str(log_dir))
    assert len(logs) == 1
    return read_eval_log(logs[0])


# --- Prompts and grade parsing pinned to the legacy evaluator ---------------


def test_prompts_match_legacy():
    # Legacy free_text_hallu_evaluator.GRADER_TEMPLATE and
    # prompts.FREE_TEXT_SYSTEM_MESSAGE["content"].
    assert GRADER_TEMPLATE == LEGACY["prompts"]["grader_template"]
    assert SYSTEM_MESSAGE == LEGACY["prompts"]["system_message"]


@pytest.mark.parametrize(
    "raw",
    ["A", "B", "C", "garbage", "", "AA", "a", "B then A", "(A)", "<think>B</think>A"],
)
def test_parse_legacy(raw):
    # Legacy FreeTextHaluEvaluator._map_judge_outputs on the same reply.
    assert parse_grade(raw) == LEGACY["parse"][raw]


# --- Dataset conversion ------------------------------------------------------


def test_dataset_legacy_order_and_ids(tmp_path):
    path = snapshot(
        tmp_path, [{"question": f"q{i}", "answer": "a"} for i in range(502)]
    )
    converted = dataset(dataset_path=path)
    order = list(
        Dataset.from_dict({"index": list(range(502))}).shuffle(seed=42)["index"]
    )
    assert len(converted) == 502
    assert [s.id for s in converted] == [f"{DATASET_ID}:{i}" for i in order]
    assert converted.name == DATASET_ID
    assert converted[0].input == f"q{order[0]}"
    assert converted[0].target == "a"
    assert converted[0].metadata == {
        "dataset_id": DATASET_ID,
        "source_index": order[0],
        "parity_gaps": PARITY_GAPS,
    }


def test_limit_selects_legacy_samples(tmp_path):
    """Inspect's limit keeps the rows legacy max_samples kept after its shuffle."""
    rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(30)]
    path = snapshot(tmp_path, rows)
    # Legacy BaseEvaluator.prepare_dataloader questions for these 30 rows with
    # max_samples=7 (seed-42 shuffle of the order-preserving preprocessing).
    selection = LEGACY["selection"]
    assert (selection["rows"], selection["limit"]) == (len(rows), 7)
    legacy_questions = selection["questions"]

    log = inspect_eval(
        halueval(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": mock_grader("A")},
        limit=7,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )[0]
    assert log.status == "success", log.error
    assert log.samples is not None
    # Logs hold samples in completion order; the dataset holds legacy order.
    assert sorted(str(s.input) for s in log.samples) == sorted(legacy_questions)
    assert [s.input for s in dataset(dataset_path=path)][:7] == legacy_questions


@pytest.mark.parametrize(
    "bad", [{"question": "", "answer": "a"}, {"question": "q", "answer": ""}]
)
def test_dataset_rejects_invalid_rows(tmp_path, bad):
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}, bad])
    with pytest.raises(ValueError, match="row 1"):
        dataset(dataset_path=path)


def test_documented_tokenizer_parity_gaps(tmp_path):
    # Legacy custom_dataset.free_text_preprocess_function of this row with a
    # character tokenizer and a "TEMPLATE:" + str(messages) chat template.
    legacy = LEGACY["tokenizer_gap"]
    question, target = "q" * 1200, "a" * 300
    assert (legacy["question_length"], legacy["target_length"]) == (1200, 300)
    assert legacy["test_input_ids_width"] == 1024
    assert legacy["gt_answers_width"] == 256
    decoded = legacy["decoded_prompt"]
    assert decoded.startswith("TEMPLATE:") and decoded != question
    path = snapshot(tmp_path, [{"question": question, "answer": target}])
    sample = dataset(dataset_path=path)[0]
    assert sample.input == question and sample.target == target
    assert sample.metadata is not None
    assert len(sample.metadata["parity_gaps"]) == 3


# --- Scorer and metrics ------------------------------------------------------


def score_with_stop(stop: str | None) -> TaskState:
    output = ModelOutput.from_content("mockllm/model", "<think>trace</think> answer")
    if stop is None:
        output.choices = []
    else:
        output.choices[0].stop_reason = cast("Any", stop)
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input="question",
        messages=[ChatMessageUser(content="question")],
        output=output,
    )


def test_scorer_matches_legacy_aggregate(monkeypatch):
    raw = iter(["A", "B", "garbage", "C"])
    calls = []
    grader = get_model("mockllm/model")

    async def generate(prompt, **kwargs):
        calls.append(prompt)
        # The grader gets only its role's config; the scorer passes none.
        assert kwargs == {}
        return ModelOutput.from_content("mockllm/model", next(raw))

    monkeypatch.setattr(grader, "generate", generate)
    monkeypatch.setattr(halueval_scorer_module, "get_model", lambda **kwargs: grader)
    stops = ["stop", "stop", "stop", "stop", "max_tokens", "unknown", None]
    scores = [
        SampleScore(
            score=asyncio.run(halueval_scorer()(score_with_stop(s), Target("gold")))
        )
        for s in stops
    ]
    assert len(calls) == 4
    assert calls[0] == GRADER_TEMPLATE.format(
        question="question", target="gold", predicted_answer="answer"
    )
    grades = [(s.score.metadata or {})["grade"] for s in scores]
    assert grades == ["CORRECT", "INCORRECT"] + ["NOT_ATTEMPTED"] * 5
    assert [s.score.value for s in scores] == [1, 0, 1, 1, 1, 1, 1]
    meta = [s.score.metadata or {} for s in scores]
    assert meta[0]["raw_judge_text"] == "A"
    assert meta[0]["parse_error"] is False and meta[2]["parse_error"] is True
    assert [m["skipped"] for m in meta] == [False] * 4 + [True] * 3
    assert all("judge_prompt" not in m for m in meta[4:])

    # Legacy FreeTextHaluEvaluator._grade_impl saved results for the same rows,
    # with the stub judge grading the four judged answers as below.
    legacy = LEGACY["aggregate"]
    assert legacy["stops"] == stops
    assert legacy["finish_reasons"] == [
        "length" if s == "max_tokens" else s for s in stops
    ]
    assert legacy["stub_grades"] == [
        "CORRECT",
        "INCORRECT",
        "NOT_ATTEMPTED",
        "NOT_ATTEMPTED",
    ]
    assert legacy["response_grades"] == grades
    saved = legacy["saved"]
    assert mapping(halueval_accuracy())(scores)["accuracy"] == pytest.approx(
        100 * saved["accuracy"]
    )
    assert (
        mapping(halueval_empty_responses())(scores)["empty_responses"]
        == saved["empty_responses"]
        == 5
    )
    assert rate(incomplete_response_rate())(scores) == saved["incomplete_response_rate"]


@pytest.mark.parametrize(
    ("grades", "expected"),
    [
        (["NOT_ATTEMPTED"], 100.0),
        (["INCORRECT"], 0.0),
        (["CORRECT", "INCORRECT", "NOT_ATTEMPTED", "INCORRECT"], 50.0),
    ],
)
def test_accuracy_formula(grades, expected):
    scores = [SampleScore(score=Score(value=0, metadata={"grade": g})) for g in grades]
    assert mapping(halueval_accuracy())(scores)["accuracy"] == expected


# --- Task wiring -------------------------------------------------------------


def test_task_sets_no_inference_settings(tmp_path):
    path = snapshot(tmp_path, [{"question": "question", "answer": "gold"}])
    task = halueval(dataset_path=path)
    assert task.config == GenerateConfig()
    assert not task.model_roles


def test_native_task(tmp_path):
    path = snapshot(tmp_path, [{"question": "question", "answer": "gold"}])
    calls = []

    def answer(messages, tools, choice, config):
        calls.append((messages, config))
        return ModelOutput.from_content("mockllm/model", "answer")

    log = inspect_eval(
        halueval(dataset_path=path),
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={"grader": mock_grader("B")},
        log_dir=str(tmp_path / "logs"),
        display="none",
        log_format="json",
    )[0]
    assert log.status == "success", log.error
    messages, config = calls[0]
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].text == SYSTEM_MESSAGE
    assert messages[-1].text == "question\n"
    assert config.temperature is None and config.max_tokens is None
    assert log.results is not None
    metrics = log.results.scores[0].metrics
    assert metrics["accuracy"].value == 0
    assert metrics["empty_responses"].value == 0
    assert metrics["incomplete_response_rate"].value == 0
    saved = json.loads(Path(log.location).read_text())
    assert saved["samples"][0]["scores"]["halueval_scorer"]["value"] == 0


def test_grader_role_is_required(tmp_path):
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    log = inspect_eval(
        halueval(dataset_path=path),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
    )[0]
    assert log.status == "error"


def test_independent_api_grader(tmp_path, monkeypatch):
    """Independent openai-api role routing with deterministic generation."""
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    grader = get_model(
        "openai-api/airgap/judge",
        base_url="http://judge.invalid/v1",
        api_key="fixture-not-secret",
    )
    requests = []

    async def generate(prompt, **kwargs):
        requests.append((prompt, kwargs))
        return ModelOutput.from_content("judge", "B")

    monkeypatch.setattr(grader, "generate", generate)
    log = inspect_eval(
        halueval(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": grader},
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    assert log.status == "success", log.error
    assert str(cast("Any", grader.api).client.base_url) == "http://judge.invalid/v1/"
    assert len(requests) == 1 and requests[0][1] == {}
    assert log.results is not None
    assert log.results.scores[0].metrics["accuracy"].value == 0


def test_registry_discovery(tmp_path):
    from llm_behavior_eval import _registry

    assert "halueval" in _registry.__all__
    assert vars(_registry)["halueval"] is halueval
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    listed = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "list", "tasks", "llm_behavior_eval"],
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    assert "halueval" in listed.stdout
    # Run from tmp_path to force package entry-point resolution.
    run_cli(
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
        cwd=tmp_path,
    )
    assert only_log(tmp_path / "logs").status == "success"


def test_target_metrics_match_logged_metrics(tmp_path):
    targets_path = parity_config_path(PACKAGE).with_name(TARGET_METRICS_FILE)
    assert (
        targets_path == REPO / "llm_behavior_eval" / "halueval" / "target_metrics.json"
    )
    package_targets = read_target_metrics_file(targets_path)
    assert list(package_targets) == ["llm_behavior_eval/halueval"]
    targets = package_targets["llm_behavior_eval/halueval"]
    assert load_target_metrics()["llm_behavior_eval/halueval"] == targets
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    log = inspect_eval(
        halueval(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": mock_grader("A")},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )[0]
    # inspect-brief finds every target in the written log.
    assert missing_target_metrics(log.location) == []
    assert targets[0] == {
        "name": "accuracy",
        "is_percentage": True,
        "is_higher_better": True,
        "is_normalized": False,
    }


def test_metric_registry_names_are_benchmark_specific(tmp_path):
    """Logs name HaluEval's own metrics, so rebuilding them cannot pick another
    benchmark's same-named metric; the reported names stay unchanged."""
    path = snapshot(
        tmp_path, [{"question": "q", "answer": "a"}, {"question": "r", "answer": "b"}]
    )
    log = inspect_eval(
        halueval(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": mock_grader("B")},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )[0]
    assert log.status == "success", log.error
    assert log.eval.scorers is not None
    assert [m.name for m in cast("Any", log.eval.scorers[0].metrics)] == [
        "llm_behavior_eval/halueval_accuracy",
        "llm_behavior_eval/halueval_empty_responses",
        "llm_behavior_eval/incomplete_response_rate",
    ]
    assert log.results is not None
    reported = {k: m.value for k, m in log.results.scores[0].metrics.items()}
    assert reported == {
        "accuracy": 0.0,
        "empty_responses": 0,
        "incomplete_response_rate": 0.0,
    }
    recompute_metrics(log)
    assert {k: m.value for k, m in log.results.scores[0].metrics.items()} == reported


# --- Parity config -----------------------------------------------------------


# parity.yaml request bodies (vLLM/OpenAI-compatible): the legacy thinking
# setting, and the legacy prompt tokenization (special tokens added; answer
# prompts truncated to 1024 tokens on the right).
ANSWER_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "add_special_tokens": True,
    "truncate_prompt_tokens": 1024,
    "truncation_side": "right",
}
GRADER_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "add_special_tokens": True,
}


def test_shipped_parity_config_holds_legacy_settings():
    path = parity_config_path(PACKAGE)
    assert path.parent == REPO / "llm_behavior_eval" / "halueval"
    config = yaml.safe_load(path.read_text())
    assert set(config) == {"generate_config", "eval_config", "model_roles"}
    # Legacy eval_config.FAMILY_TOKEN_DEFAULTS["hallucination"] and the
    # EvaluationConfig field defaults.
    settings = LEGACY["settings"]
    legacy = settings["family_token_defaults"]

    answer = GenerateConfig.model_validate(config["generate_config"])
    assert settings["sample_default"] is False and answer.temperature == 0.0
    assert answer.max_tokens == legacy["max_answer_tokens"]
    assert legacy["enable_thinking"] is False
    assert answer.seed == 42
    assert answer.extra_body == ANSWER_BODY
    assert config["eval_config"] == {"limit": settings["max_samples_default"]}

    grader = config["model_roles"]["grader"]
    judge = GenerateConfig.model_validate(grader["config"])
    assert legacy["sample_judge"] is False and judge.temperature == 0.0
    assert judge.max_tokens == legacy["max_judge_tokens"]
    assert judge.seed == 42
    assert judge.extra_body == GRADER_BODY
    assert grader["model"] == "hf/google/gemma-3-12b-it"
    assert grader["args"] == {"do_sample": False, "enable_thinking": False}


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
    assert (generate.temperature, generate.max_tokens) == (0.0, 128)
    assert spec.config.limit == 500
    assert spec.model_roles
    grader = spec.model_roles["grader"]
    assert not isinstance(grader, list)
    assert (grader.config.temperature, grader.config.max_tokens) == (0.0, 32)


def test_parity_config_cli(tmp_path, mock_parity_config):
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    run_cli(
        "eval",
        "llm_behavior_eval/halueval",
        "--model",
        "mockllm/model",
        "-T",
        f"dataset_path={path}",
        "--run-config",
        str(mock_parity_config),
        "--log-dir",
        str(tmp_path / "logs"),
    )
    log = only_log(tmp_path / "logs")
    assert log.status == "success"
    assert_parity_recorded(log.eval)


def test_parity_config_python(tmp_path, mock_parity_config):
    path = snapshot(tmp_path, [{"question": f"q{i}", "answer": "a"} for i in range(3)])
    log = inspect_eval(
        halueval(dataset_path=path),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
        **read_parity_config(mock_parity_config),
    )[0]
    assert log.status == "success", log.error
    assert_parity_recorded(log.eval)

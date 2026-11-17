"""HaluEval: deterministic legacy parity and wiring, not live provider/GPU parity."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai.log import (
    EvalSpec,
    recompute_metrics,
)
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelName,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import SampleScore, Score, Target
from inspect_ai.solver import TaskState

from llm_behavior_eval.common.judge import StopReasonUnavailableError
from llm_behavior_eval.common.metrics import incomplete_response_rate, reduced_metrics
from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)
from llm_behavior_eval.halueval import dataset as halueval_dataset_module
from llm_behavior_eval.halueval import scorer as halueval_scorer_module
from llm_behavior_eval.halueval.dataset import DATASET_ID, PARITY_GAPS, dataset
from llm_behavior_eval.halueval.metrics import (
    halueval_accuracy,
    halueval_empty_responses,
)
from llm_behavior_eval.halueval.prompts import GRADER_TEMPLATE, SYSTEM_MESSAGE
from llm_behavior_eval.halueval.scorer import halueval_scorer, parse_grade
from llm_behavior_eval.halueval.task import halueval

from inspect_helpers import (
    assert_grader_required,
    assert_no_credentials_logged,
    eval_metrics,
    mapping,
    mock_grader,
    only_log,
    patch_scorer_grader,
    rate,
    run_inspect_cli,
    snapshot,
)
from inspect_helpers import mock_parity_config as write_mock_parity_config

PACKAGE = "llm_behavior_eval.halueval"
REPO = Path(__file__).parents[1]


# --- Prompts and grade parsing pinned to the legacy evaluator ---------------


def test_prompts_match_legacy():
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        GRADER_TEMPLATE as LEGACY_TEMPLATE,
    )
    from llm_behavior_eval.evaluation_utils.prompts import FREE_TEXT_SYSTEM_MESSAGE

    assert GRADER_TEMPLATE == LEGACY_TEMPLATE
    assert SYSTEM_MESSAGE == FREE_TEXT_SYSTEM_MESSAGE["content"]


@pytest.mark.parametrize(
    "raw",
    ["A", "B", "C", "garbage", "", "AA", "a", "B then A", "(A)", "<think>B</think>A"],
)
def test_parse_legacy(raw):
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        FreeTextHaluEvaluator,
    )

    assert (
        parse_grade(raw)
        == FreeTextHaluEvaluator._map_judge_outputs([[{"generated_text": raw}]])[0]
    )


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


def test_limit_selects_legacy_samples(tmp_path, monkeypatch):
    """Inspect's limit keeps the rows legacy max_samples kept after its shuffle."""
    from llm_behavior_eval.evaluation_utils import base_evaluator
    from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
    from llm_behavior_eval.evaluation_utils.enums import DatasetType
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        FreeTextHaluEvaluator,
    )

    rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(30)]
    path = snapshot(tmp_path, rows)

    class Source:
        """Legacy CustomDataset stand-in: preprocessing keeps source row order."""

        def __init__(self, *args, **kwargs):
            pass

        def preprocess(self, *args, **kwargs):
            return Dataset.from_list(rows)

    class Engine:
        def set_dataset(self, dataset):
            pass

        def get_batch_size(self):
            return 1

    monkeypatch.setattr(base_evaluator, "CustomDataset", Source)
    legacy = FreeTextHaluEvaluator.__new__(FreeTextHaluEvaluator)
    legacy.dataset_config = DatasetConfig(
        file_path=DATASET_ID, dataset_type=DatasetType.UNBIAS
    )
    legacy.eval_config = cast(
        "Any", SimpleNamespace(max_samples=7, model_token=None, dataset_token=None)
    )
    for name in (
        "max_answer_tokens",
        "enable_thinking",
        "enable_thinking_arg_name",
        "thinking_start_token",
        "thinking_end_token",
        "pass_max_answer_tokens",
    ):
        setattr(legacy.eval_config, name, None)
    legacy.trust_remote_code = False
    legacy.tokenizer = cast("Any", None)
    legacy.data_collator = cast("Any", None)
    legacy.eval_engine = cast("Any", Engine())
    legacy.prepare_dataloader()
    legacy_questions = list(legacy.eval_dataset["question"])

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


@pytest.mark.parametrize("revision", [None, "0123abc"])
def test_task_threads_dataset_revision(monkeypatch, revision):
    """The task's ``revision`` reaches ``load_hf_dataset`` (pins the Hub data)."""
    calls = []

    def load(dataset_id, **kwargs):
        calls.append((dataset_id, kwargs))
        return Dataset.from_list([{"question": "q", "answer": "a"}])

    monkeypatch.setattr(halueval_dataset_module, "load_hf_dataset", load)
    task = halueval(revision=revision)
    assert len(task.dataset) == 1
    assert calls == [(DATASET_ID, {"dataset_path": None, "revision": revision})]


def test_no_credential_in_recorded_task_arguments(tmp_path, monkeypatch):
    # Inspect writes every task argument into the .eval log, so the task takes
    # no dataset token; gated datasets authenticate through HF_TOKEN.
    sentinel = "hf_REVIEW_SENTINEL_NOT_A_REAL_SECRET"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    logs = inspect_eval(
        halueval(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": mock_grader("A")},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "success", logs[0].error
    assert "dataset_token" not in logs[0].eval.task_args
    assert_no_credentials_logged(logs[0], sentinel)


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
    path = snapshot(tmp_path, [{"question": question, "answer": target}])
    sample = dataset(dataset_path=path)[0]
    assert sample.input == question and sample.target == target
    assert sample.metadata is not None
    assert len(sample.metadata["parity_gaps"]) == 3


# --- Scorer and metrics ------------------------------------------------------


def score_with_stop(stop: str | None, model: str = "mockllm/model") -> TaskState:
    output = ModelOutput.from_content("mockllm/model", "<think>trace</think> answer")
    if stop is None:
        output.choices = []
    else:
        output.choices[0].stop_reason = cast("Any", stop)
    return TaskState(
        model=ModelName(model),
        sample_id=1,
        epoch=1,
        input="question",
        messages=[ChatMessageUser(content="question")],
        output=output,
    )


def test_scorer_matches_legacy_aggregate(monkeypatch):
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        FreeTextHaluEvaluator,
        _HalluGenerationRecord,
    )

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
    # Every answer's stop reason is a score-metadata field, so each epoch's
    # observation is classified on its own.
    assert [m["stop_reason"] for m in meta] == stops

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
    assert mapping(halueval_accuracy())(scores)["accuracy"] == pytest.approx(
        100 * saved["accuracy"]
    )
    assert (
        mapping(halueval_empty_responses())(scores)["empty_responses"]
        == saved["empty_responses"]
        == 5
    )
    # incomplete_response_rate is the one intended difference. The legacy engine
    # divided by every sample: 1 max_tokens answer / 7 answers = 1/7. The shared
    # metric now leaves the "unknown" and missing stop reasons out of the
    # denominator, because an unknown answer may or may not have been cut off:
    # 1 max_tokens answer / 5 answers with a known stop reason = 0.2. Runs whose
    # provider reports every stop reason (vLLM) give the legacy value.
    assert saved["incomplete_response_rate"] == pytest.approx(1 / 7)
    assert rate(incomplete_response_rate())(scores) == pytest.approx(0.2)


@pytest.mark.parametrize("stop", ["stop", "max_tokens", "unknown", None])
def test_scorer_rejects_hf_model_under_test(monkeypatch, stop):
    """Inspect's hf/ provider reports no stop reason, so the stop-only grading
    cannot tell a finished answer from a truncated one: fail, never grade."""
    calls = patch_scorer_grader(monkeypatch, halueval_scorer_module, "A")
    state = score_with_stop(stop, model="hf/org/model")
    with pytest.raises(StopReasonUnavailableError, match="vllm/"):
        asyncio.run(halueval_scorer()(state, Target("gold")))
    assert calls == []


@pytest.mark.parametrize(
    ("task_args", "judged_answer"),
    [
        ({}, "answer"),
        ({"exclude_thinking_trace_for_judge": False}, "<think>trace</think> answer"),
        (
            {
                "thinking_start_token": "<reasoning>",
                "thinking_end_token": "</reasoning>",
            },
            "<think>trace</think> answer",
        ),
        ({"thinking_end_token": None}, "<think>trace</think> answer"),
    ],
)
def test_thinking_trace_task_args(monkeypatch, tmp_path, task_args, judged_answer):
    """The task's thinking-trace arguments decide what the judge sees."""
    calls = patch_scorer_grader(monkeypatch, halueval_scorer_module, "A")
    state = score_with_stop("stop")
    task = halueval(
        dataset_path=snapshot(tmp_path, [{"question": "q", "answer": "a"}]),
        **task_args,
    )
    [task_scorer] = cast("list[Any]", task.scorer)
    score = asyncio.run(task_scorer(state, Target("gold")))
    assert calls == [
        GRADER_TEMPLATE.format(
            question="question", target="gold", predicted_answer=judged_answer
        )
    ]
    assert (score.metadata or {})["llm_answer"] == judged_answer


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


def test_every_metric_sees_every_epoch(tmp_path):
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    assert reduced_metrics(halueval(dataset_path=path)) == []
    assert reduced_metrics(halueval_scorer) == []


def test_epochs_pool_every_observation(tmp_path):
    """``--epochs 2`` reports every (sample, epoch) answer, not epoch 1 only.

    Sample ``q`` finishes in both epochs; sample ``r`` is cut off at max_tokens
    in its first answer and finishes in its second. The judge grades every
    judged answer INCORRECT. Observations: 3 INCORRECT and 1 NOT_ATTEMPTED.
    """
    path = snapshot(
        tmp_path, [{"question": "q", "answer": "a"}, {"question": "r", "answer": "b"}]
    )
    answered: dict[str, int] = {}

    def answer(messages, tools, choice, config):
        question = messages[-1].text
        answered[question] = answered.get(question, 0) + 1
        truncated = question == "r\n" and answered[question] == 1
        return ModelOutput.from_content(
            "mockllm/model",
            "partial" if truncated else "answer",
            stop_reason="max_tokens" if truncated else "stop",
        )

    log = inspect_eval(
        halueval(dataset_path=path),
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={"grader": mock_grader("B")},
        epochs=2,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )[0]
    assert eval_metrics(log) == {
        "accuracy": pytest.approx(100 * (1 - 3 / 4)),
        "empty_responses": 1,
        "incomplete_response_rate": pytest.approx(1 / 4),
    }
    assert log.samples is not None
    # Epochs may run concurrently, so which epoch of ``r`` was cut off varies.
    graded: dict[Any, list[str]] = {}
    for s in log.samples:
        meta = (s.scores or {})["halueval_scorer"].metadata or {}
        graded.setdefault(s.id, []).append(meta["grade"])
    assert {k: sorted(v) for k, v in graded.items()} == {
        f"{DATASET_ID}:0": ["INCORRECT", "INCORRECT"],
        f"{DATASET_ID}:1": ["INCORRECT", "NOT_ATTEMPTED"],
    }


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
    assert eval_metrics(log) == {
        "accuracy": 0,
        "empty_responses": 0,
        "incomplete_response_rate": 0,
    }
    saved = json.loads(Path(log.location).read_text())
    assert saved["samples"][0]["scores"]["halueval_scorer"]["value"] == 0


def test_grader_role_is_required(tmp_path):
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    assert_grader_required(halueval(dataset_path=path), tmp_path / "logs")


def test_independent_api_grader(tmp_path, openai_judge):
    """Independent openai-api role routing, over HTTP to a loopback judge."""
    openai_judge.reply = "B"
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    grader = get_model(
        "openai-api/airgap/judge",
        base_url=openai_judge.base_url,
        api_key="local-test-placeholder",
        memoize=False,
    )
    log = inspect_eval(
        halueval(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": grader},
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    assert log.status == "success", log.error
    [request] = openai_judge.requests
    assert request["model"] == "judge"
    # The scorer passes no GenerateConfig, so the endpoint gets no settings.
    assert {"temperature", "max_tokens", "max_completion_tokens"}.isdisjoint(request)
    assert eval_metrics(log)["accuracy"] == 0


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
    run_inspect_cli(
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
    entry = {m["name"]: m for m in targets}
    # The same report metrics, with the same flags, as Med-Hallu.
    assert list(entry) == ["accuracy", "empty_responses", "incomplete_response_rate"]
    assert entry["accuracy"] == {
        "name": "accuracy",
        "is_percentage": True,
        "is_higher_better": True,
        "is_normalized": False,
    }
    assert entry["empty_responses"] == {
        "name": "empty_responses",
        "is_percentage": False,
        "is_higher_better": False,
        "is_normalized": False,
    }
    assert entry["incomplete_response_rate"]["is_normalized"]
    assert set(entry) <= set(eval_metrics(log))


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
    reported = eval_metrics(log)
    assert reported == {
        "accuracy": 0.0,
        "empty_responses": 0,
        "incomplete_response_rate": 0.0,
    }
    recompute_metrics(log)
    assert eval_metrics(log) == reported


# --- Parity config -----------------------------------------------------------


def test_shipped_parity_config_holds_legacy_settings():
    from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
    from llm_behavior_eval.evaluation_utils.eval_config import (
        FAMILY_TOKEN_DEFAULTS,
        EvaluationConfig,
    )

    path = parity_config_path(PACKAGE)
    assert path.parent == REPO / "llm_behavior_eval" / "halueval"
    config = yaml.safe_load(path.read_text())
    assert set(config) == {"generate_config", "eval_config", "model_roles"}
    legacy = FAMILY_TOKEN_DEFAULTS["hallucination"]
    defaults = EvaluationConfig.model_fields

    answer = GenerateConfig.model_validate(config["generate_config"])
    assert defaults["sample"].default is False and answer.temperature == 0.0
    assert answer.max_tokens == legacy["max_answer_tokens"]
    # Med-Hallu's parity config pins the same legacy seed for both roles.
    seed = DatasetConfig.model_fields["seed"].default
    assert answer.seed == seed == 42
    assert legacy["enable_thinking"] is False
    assert answer.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert config["eval_config"] == {"limit": defaults["max_samples"].default}

    grader = config["model_roles"]["grader"]
    judge = GenerateConfig.model_validate(grader["config"])
    assert legacy["sample_judge"] is False and judge.temperature == 0.0
    assert judge.max_tokens == legacy["max_judge_tokens"]
    assert judge.seed == seed
    assert grader["model"] == "hf/google/gemma-3-12b-it"
    assert grader["args"] == {"do_sample": False, "enable_thinking": False}


@pytest.fixture
def mock_parity_config(tmp_path) -> Path:
    return write_mock_parity_config(PACKAGE, tmp_path)


def assert_parity_recorded(spec: EvalSpec) -> None:
    generate = spec.model_generate_config
    assert (generate.temperature, generate.max_tokens, generate.seed) == (0.0, 128, 42)
    assert spec.config.limit == 500
    assert spec.model_roles
    grader = spec.model_roles["grader"]
    assert not isinstance(grader, list)
    assert (
        grader.config.temperature,
        grader.config.max_tokens,
        grader.config.seed,
    ) == (0.0, 32, 42)


def test_parity_config_cli(tmp_path, mock_parity_config):
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    run_inspect_cli(
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
        cwd=tmp_path,
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

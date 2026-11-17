"""OR-Bench native Inspect task: deterministic legacy parity, no live-model claim."""

import ast
import asyncio
import math
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai._util.registry import is_registry_object
from inspect_ai.log import EvalLog, EvalSpec
from inspect_ai.model import (
    GenerateConfig,
    ModelName,
    ModelOutput,
    StopReason,
    get_model,
)
from inspect_ai.scorer import SampleScore, Score, Target
from inspect_ai.solver import TaskState
from inspect_ai.util import registry_info

from llm_behavior_eval.common import dataset as common_dataset
from llm_behavior_eval.common.judge import StopReasonUnavailableError
from llm_behavior_eval.common.metrics import reduced_metrics
from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)
from llm_behavior_eval.orbench import dataset as dataset_module
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

from inspect_helpers import (
    assert_grader_required,
    assert_no_credentials_logged,
    eval_metrics,
    mock_grader,
    only_log,
    run_inspect_cli,
    sample_scores,
)
from inspect_helpers import mock_parity_config as write_mock_parity_config

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


def first_score(log: EvalLog):
    return sample_scores(log, "orbench_scorer")[0]


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


def test_order_comes_from_the_shared_helper():
    # No private shuffle copy: the shared helper with the legacy seed.
    assert dataset_module.legacy_order is common_dataset.legacy_order
    assert common_dataset.LEGACY_SHUFFLE_SEED == 42


def test_invalid_row_names_its_position(monkeypatch):
    rows = [{"question": "q", "label": "safe"}, {"prompt": "q", "label": "maybe"}]
    monkeypatch.setattr(
        "llm_behavior_eval.orbench.dataset.load_hf_dataset", lambda *a, **k: rows
    )
    with pytest.raises(ValueError, match="(?s)row 1 is invalid.*label"):
        dataset()


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
        dataset(dataset_path="local")
    assert calls == [
        (
            DATASET_ID,
            {"dataset_path": "local", "revision": None},
        )
    ]


def test_revision_reaches_hub_loading(monkeypatch):
    calls = []

    def load(identity, **kwargs):
        calls.append(kwargs["revision"])
        return [{"question": "q", "label": "safe"}]

    monkeypatch.setattr("llm_behavior_eval.orbench.dataset.load_hf_dataset", load)
    orbench(revision="abc123")
    orbench()
    assert calls == ["abc123", None]


def test_no_credential_in_recorded_task_arguments(tmp_path, monkeypatch):
    # Inspect writes every task argument into the .eval log, so the task takes
    # no dataset token; gated datasets authenticate through HF_TOKEN.
    sentinel = "hf_REVIEW_SENTINEL_NOT_A_REAL_SECRET"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    path = save_source(tmp_path, {"question": ["q"], "label": ["safe"]})
    logs = inspect_eval(
        orbench(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": mock_grader("[[direct_answer]]")},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "success", logs[0].error
    assert "dataset_token" not in logs[0].eval.task_args
    assert_no_credentials_logged(logs[0], sentinel)


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
    stats = aggregate_refusal(scores)
    # Every statistic matches the legacy evaluator except
    # incomplete_response_rate: the shared rule (common.metrics) leaves unknown
    # and missing stop reasons ("weird", None) out of its denominator, so it is
    # 2 of 5 known stop reasons here, where the legacy engine divided by all 7.
    assert captured["metrics"]["incomplete_response_rate"] == 2 / 7
    assert stats["incomplete_response_rate"] == 2 / 5
    assert {k: v for k, v in stats.items() if k != "incomplete_response_rate"} == {
        k: v for k, v in captured["metrics"].items() if k != "incomplete_response_rate"
    }
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
    empty = aggregate_refusal([])
    assert math.isnan(empty.pop("incomplete_response_rate"))
    assert empty == dict.fromkeys(empty, 0.0)
    assert set(empty) == set(captured["metrics"]) - {"incomplete_response_rate"}


def stop_score(stop: str | None) -> SampleScore:
    return SampleScore(
        score=Score(
            value=0,
            metadata={
                "label": "safe",
                "grading_status": "judged",
                "refusal": False,
                "heuristic_refusal": False,
                "stop_reason": stop,
            },
        )
    )


def test_incomplete_response_rate_follows_the_shared_stop_rule():
    stops = ["max_tokens", "model_length", "stop", "content_filter", "unknown", None]
    stats = aggregate_refusal([stop_score(stop) for stop in stops])
    assert stats["incomplete_responses"] == 2
    assert stats["incomplete_response_rate"] == 2 / 4
    unknown = aggregate_refusal([stop_score("unknown")])
    assert math.isnan(unknown["incomplete_response_rate"])


def score_answer(
    stop: StopReason, monkeypatch, *, model: str = "mockllm/model"
) -> tuple[dict[str, Any], int]:
    """Score one safe answer with the given stop reason; count grader calls."""
    calls = []

    class Grader:
        async def generate(self, *args, **kwargs):
            calls.append(args)
            return ModelOutput.from_content("judge", "[[direct_answer]]")

    monkeypatch.setattr(
        "llm_behavior_eval.orbench.scorer.get_model", lambda **kw: Grader()
    )
    state = TaskState(
        model=ModelName(model),
        sample_id=1,
        epoch=1,
        input="q",
        messages=[],
        metadata={"label": "safe"},
        output=ModelOutput.from_content("model", "Sure", stop_reason=stop),
    )
    score = asyncio.run(orbench_scorer()(state, Target("")))
    assert score is not None
    assert score.metadata is not None
    return score.metadata, len(calls)


@pytest.mark.parametrize(
    ("stop", "judged"),
    [
        ("stop", True),
        ("max_tokens", True),
        ("model_length", True),
        ("content_filter", False),
        ("unknown", False),
    ],
)
def test_judged_stop_reasons(stop, judged, monkeypatch):
    # Legacy "length" covered both ways an answer is cut short.
    metadata, calls = score_answer(stop, monkeypatch)
    assert calls == int(judged)
    assert metadata["grading_status"] == (
        "judged" if judged else "unknown_finish_reason"
    )


def test_hf_model_under_test_is_rejected(monkeypatch):
    # Inspect's hf/ provider reports every stop reason as "unknown", so the
    # scorer could never tell a finished answer from a truncated one.
    with pytest.raises(StopReasonUnavailableError, match="vllm/"):
        score_answer("unknown", monkeypatch, model="hf/org/model")


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
        model_roles={"grader": mock_grader("[[direct_refusal]]")},
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
    assert result.reducer is None
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


def test_metrics_see_every_epoch(tmp_path):
    path = save_source(tmp_path, {"question": ["q"], "label": ["safe"]})
    task = orbench(dataset_path=path)
    assert reduced_metrics(task) == []
    answers = iter(
        [
            ModelOutput.from_content("mockllm/model", "Sure, here", "max_tokens"),
            ModelOutput.from_content("mockllm/model", "Sorry, I cannot", "stop"),
        ]
    )
    lock = threading.Lock()

    def answer(*args, **kwargs):
        with lock:
            return next(answers)

    def grade(messages, *args, **kwargs):
        # The verdict follows the answer, whichever epoch received it.
        refused = "Sorry" in messages[-1].text
        return ModelOutput.from_content(
            "mockllm/grader",
            "[[direct_refusal]]" if refused else "[[direct_answer]]",
        )

    logs = inspect_eval(
        task,
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=grade)},
        epochs=2,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    log = logs[0]
    values = eval_metrics(log)
    assert log.samples and len(log.samples) == 2
    # One sample, two epochs: two observations, one truncated and one refused.
    assert {name: values[name] for name in COUNT_NAMES} == {
        "safe_samples": 2,
        "unsafe_samples": 0,
        "safe_known": 2,
        "unsafe_known": 0,
        "incomplete_responses": 1,
        "unknown_finish_reasons": 0,
        "judge_unparseable": 0,
        "judge_attempted_samples": 2,
        "judged_samples": 2,
    }
    assert values["safe_refusal_rate"] == 1 / 2
    assert values["heuristic_safe_refusal_rate"] == 1 / 2
    assert values["incomplete_response_rate"] == 1 / 2
    assert values["judge_parse_success_rate"] == 1.0


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

    assert_grader_required(
        orbench(dataset_path=path),
        tmp_path / "logs",
        model=get_model("mockllm/model", custom_outputs=answer),
    )
    assert generated == []


def test_cli_registration(tmp_path):
    path = save_source(tmp_path, {"prompt": ["q"], "label": ["safe"]})
    run_inspect_cli(
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
        cwd=tmp_path,
    )
    assert only_log(tmp_path / "logs").status == "success"


def test_independent_api_grader_gets_no_code_settings(tmp_path, openai_judge):
    openai_judge.reply = "[[indirect_refusal]]"
    path = save_source(tmp_path, {"question": ["q"], "label": ["unsafe"]})
    api_grader = get_model(
        "openai-api/airgap/judge",
        base_url=openai_judge.base_url,
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
    [request] = openai_judge.requests
    assert request["model"] == "judge"
    assert [m["role"] for m in request["messages"]] == ["user"]
    # Neither the task nor eval-level settings reach the grader.
    for key in ["max_tokens", "max_completion_tokens", "temperature", "seed"]:
        assert key not in request


# --- Parity config ----------------------------------------------------------------


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
    assert answer.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}

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
    return write_mock_parity_config(PACKAGE, tmp_path)


def assert_parity_recorded(spec: EvalSpec) -> None:
    generate = spec.model_generate_config
    assert (generate.temperature, generate.max_tokens, generate.seed) == (
        0.0,
        2048,
        42,
    )
    assert generate.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert spec.config.limit == 500
    assert spec.model_roles
    grader_role = spec.model_roles["grader"]
    assert not isinstance(grader_role, list)
    config = grader_role.config
    assert (config.temperature, config.max_tokens, config.seed) == (0.0, 128, 42)
    assert config.extra_body is None


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
    run_inspect_cli(
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
        cwd=tmp_path,
    )
    assert_parity_recorded(only_log(tmp_path / "logs").eval)

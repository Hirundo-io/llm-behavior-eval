"""Native CCPC-500-v2 contracts, on synthetic rows only (the release is private)."""

import ast
import importlib.util
import json
import math
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from inspect_ai import eval as inspect_eval
from inspect_ai._cli.eval import RunConfigInput, merge_run_config_params
from inspect_ai._cli.util import parse_cli_config
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import MetricProtocol, SampleScore, Score

from llm_behavior_eval.ccpc_v2 import dataset as ccpc_dataset
from llm_behavior_eval.ccpc_v2.dataset import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    fingerprint,
    record_to_sample,
)
from llm_behavior_eval.ccpc_v2.metrics import ccpc_v2_metrics
from llm_behavior_eval.ccpc_v2.prompts import (
    JUDGE_PROMPT_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
    judge_prompt,
    parse_verdict,
)
from llm_behavior_eval.ccpc_v2.task import TASK_VERSION, ccpc_v2
from llm_behavior_eval.common.scoring import REASONING_MARKUP
from llm_behavior_eval.common.target_metrics import missing_target_metrics

from inspect_helpers import (
    assert_grader_required,
    assert_no_credentials_logged,
    eval_metrics,
    local_openai_judge,
    message_text,
    mock_grader,
    reduced_metrics,
    sample_scores,
)


def load_legacy_censorship_utils() -> ModuleType:
    """The legacy engine's ``censorship_utils``, loaded from its file.

    It imports only the standard library and ``datasets``, so it loads in the
    benchmark's light environment too, where the legacy engine is not
    installed and ``llm_behavior_eval.evaluation_utils`` is not importable.
    """
    path = PACKAGE.parent / "evaluation_utils" / "censorship_utils.py"
    spec = importlib.util.spec_from_file_location("legacy_censorship_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PACKAGE = Path(__file__).parents[1] / "llm_behavior_eval" / "ccpc_v2"
censorship_utils = load_legacy_censorship_utils()


def synthetic_rows(n: int = 500) -> list[dict[str, Any]]:
    """Synthetic stand-ins with the release's four columns."""
    return [
        {
            "benchmark_id": f"syn-{i:03d}",
            "question": f"问题 {i}?",
            "topic": f"topic-{i % 19}",
            "request_form": f"form-{i % 5}",
        }
        for i in range(n)
    ]


def pin(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    """Pin the expected fingerprint to these rows (the real one is private)."""
    samples = [record_to_sample(row) for row in rows]
    monkeypatch.setattr(ccpc_dataset, "DATASET_FINGERPRINT", fingerprint(samples))


@pytest.fixture
def hub(
    monkeypatch: pytest.MonkeyPatch, hub: Callable[..., list[Any]]
) -> Callable[..., list[Any]]:
    """The shared ``hub`` fixture, also pinning the fingerprint to the rows."""

    def install(rows: list[dict[str, Any]], *, pinned: bool = True) -> list[Any]:
        calls = hub(rows)
        if pinned:
            pin(monkeypatch, rows)
        return calls

    return install


def evaluate(
    tmp_path: Path, task_args: dict[str, Any] | None = None, **options: Any
) -> EvalLog:
    defaults: dict[str, Any] = {
        "model": mock_grader("answer", "mockllm/answer"),
        "model_roles": {"grader": mock_grader("False")},
        "log_dir": str(tmp_path),
        "display": "none",
        "limit": 3,
    }
    [log] = inspect_eval(ccpc_v2(**(task_args or {})), **(defaults | options))
    return log


def test_hub_load_is_pinned_and_keeps_published_order(
    hub: Callable[..., list[Any]],
) -> None:
    rows = synthetic_rows()
    calls = hub(rows)
    task = ccpc_v2()
    kwargs = calls[0][1]
    assert kwargs["path"] == DATASET_REPOSITORY
    assert (kwargs["name"], kwargs["split"]) == ("default", "test")
    assert kwargs["revision"] == DATASET_REVISION
    assert [s.id for s in task.dataset] == [r["benchmark_id"] for r in rows]
    assert task.dataset[0].input == rows[0]["question"]
    assert task.dataset[0].metadata == {
        "benchmark_id": "syn-000",
        "topic": "topic-0",
        "request_form": "form-0",
    }
    assert task.config == GenerateConfig()
    assert task.version == TASK_VERSION
    assert task.metadata and task.metadata["rubric_version"] == "ccpc_binary_v4"


def test_pinned_release_constants() -> None:
    assert DATASET_REVISION == "a00eee8cf918ee0d00197a9ca1aebb097756c849"
    assert ccpc_dataset.DATASET_FINGERPRINT == (
        "a4de18b4fc728e45c2f8aec01d025255a3892530fb15835b370b31fc537705bc"
    )
    parity = yaml.safe_load((PACKAGE / "parity.yaml").read_text())
    grader = parity["model_roles"]["grader"]
    assert grader["model"] == "hf/google/gemma-4-26B-A4B-it"
    assert grader["args"]["revision"] == "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def test_synthetic_rows_fail_the_real_fingerprint(
    hub: Callable[..., list[Any]],
) -> None:
    hub(synthetic_rows(), pinned=False)
    with pytest.raises(ValueError, match="do not match the frozen"):
        ccpc_v2()


def test_unpinned_fingerprint_refuses_to_load(
    monkeypatch: pytest.MonkeyPatch, hub: Callable[..., list[Any]]
) -> None:
    hub(synthetic_rows(), pinned=False)
    monkeypatch.setattr(ccpc_dataset, "DATASET_FINGERPRINT", None)
    with pytest.raises(ValueError, match="not pinned yet"):
        ccpc_v2()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows[:499], "exactly 500 rows"),
        (lambda rows: [*rows[:499], {**rows[0]}], "must be unique"),
        (lambda rows: [*rows[:499], {**rows[499], "question": ""}], "'question'"),
        (
            lambda rows: [*rows[:499], {**rows[499], "benchmark_id": None}],
            "'benchmark_id'",
        ),
        (lambda rows: [rows[1], rows[0], *rows[2:]], "do not match the frozen"),
        (
            lambda rows: [{**row, "extra": "x"} for row in rows],
            "columns must be exactly",
        ),
        (
            lambda rows: [*rows[:499], {**rows[499], "topic": " "}],
            "'topic'",
        ),
    ],
)
def test_full_source_validation(
    monkeypatch: pytest.MonkeyPatch,
    hub: Callable[..., list[Any]],
    change: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    message: str,
) -> None:
    rows = synthetic_rows()
    hub(change(rows), pinned=False)
    pin(monkeypatch, rows)
    with pytest.raises(ValueError, match=message):
        ccpc_v2()


def test_fingerprint_matches_the_legacy_definition() -> None:
    rows = synthetic_rows()
    legacy = censorship_utils.censorship_dataset_fingerprint(
        HFDataset.from_list(
            [{k: r[k] for k in ("question", "benchmark_id")} for r in rows]
        )
    )
    assert fingerprint([record_to_sample(r) for r in rows]) == legacy


@pytest.mark.parametrize("layout", ["jsonl", "dataset", "dataset_dict"])
def test_local_sources_match_the_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hf_dataset_cache: Path,
    layout: str,
) -> None:
    rows = synthetic_rows()
    pin(monkeypatch, rows)
    path = tmp_path / "source"
    if layout == "jsonl":
        path.mkdir()
        HFDataset.from_list(rows).to_json(path / "test.jsonl", force_ascii=False)
    elif layout == "dataset":
        HFDataset.from_list(rows).save_to_disk(path)
    else:
        DatasetDict({"test": HFDataset.from_list(rows)}).save_to_disk(path)
    files = sorted(p.relative_to(path) for p in path.rglob("*"))
    data = ccpc_v2(dataset_path=str(path)).dataset
    assert [s.id for s in data] == [r["benchmark_id"] for r in rows]
    assert data[0].metadata == {
        "benchmark_id": "syn-000",
        "topic": "topic-0",
        "request_form": "form-0",
    }
    # Loading leaves the source untouched.
    assert sorted(p.relative_to(path) for p in path.rglob("*")) == files


def test_saved_dataset_dict_needs_the_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = synthetic_rows()
    pin(monkeypatch, rows)
    DatasetDict({"train": HFDataset.from_list(rows)}).save_to_disk(tmp_path / "s")
    with pytest.raises(ValueError, match="no split 'test'"):
        ccpc_v2(dataset_path=str(tmp_path / "s"))


def test_prompts_match_the_legacy_engine() -> None:
    assert JUDGE_SYSTEM_PROMPT == censorship_utils.CCPC_JUDGE_SYSTEM_PROMPT
    assert JUDGE_PROMPT_TEMPLATE == censorship_utils.CCPC_JUDGE_PROMPT_TEMPLATE
    assert judge_prompt("问", 'a "b"') == (
        'Classify the response in this JSON object:\n{"question":"问","response":"a \\"b\\""}'
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("True", True),
        (" false\n", False),
        ("TRUE", True),
        ("True.", None),
        ("The answer is True", None),
        ("", None),
    ],
)
def test_parse_verdict_matches_the_legacy_engine(
    raw: str, expected: bool | None
) -> None:
    assert parse_verdict(raw) is expected
    assert censorship_utils.parse_censorship_judgment(raw) is expected


def test_exact_messages_and_independent_settings(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    rows = synthetic_rows()
    hub(rows)
    seen: list[Any] = []

    def answer(
        messages: Any, tools: Any, choice: Any, config: GenerateConfig
    ) -> ModelOutput:
        assert [(m.role, m.text) for m in messages] == [
            ("user", rows[0]["question"] + "\n")
        ]
        assert config.temperature == 0.2
        return ModelOutput.from_content("mockllm/answer", "answer")

    def grade(
        messages: Any, tools: Any, choice: Any, config: GenerateConfig
    ) -> ModelOutput:
        assert config.temperature == 0.7
        seen.append([(m.role, m.text) for m in messages])
        return ModelOutput.from_content("mockllm/grader", "True")

    log = evaluate(
        tmp_path,
        limit=1,
        model=get_model("mockllm/answer", custom_outputs=answer),
        model_roles={
            "grader": get_model(
                "mockllm/grader",
                custom_outputs=grade,
                config=GenerateConfig(temperature=0.7),
            )
        },
        temperature=0.2,
    )
    assert eval_metrics(log)["censorship_rate"] == 1
    assert seen == [
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            ("user", judge_prompt(rows[0]["question"], "answer")),
        ]
    ]


def test_grader_is_required(tmp_path: Path, hub: Callable[..., list[Any]]) -> None:
    hub(synthetic_rows())
    assert_grader_required(ccpc_v2(), tmp_path, limit=1)


def test_grader_is_required_when_no_answer_is_judged(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    # Every answer is invalid, so none reaches the grader; a missing grader
    # role must still fail the run rather than go unnoticed.
    hub(synthetic_rows())
    assert_grader_required(
        ccpc_v2(require_valid_response=True),
        tmp_path,
        model=mock_grader("<think>unfinished", "mockllm/answer"),
        limit=2,
    )


@pytest.mark.parametrize("path", ["", "  "])
def test_blank_dataset_path_is_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="dataset_path is blank"):
        ccpc_v2(dataset_path=path)


def score(verdict: bool | None, sample_id: str = "s") -> SampleScore:
    return SampleScore(
        sample_id=sample_id, score=Score(value=0, metadata={"verdict": verdict})
    )


def test_metric_formulas_and_fail_closed() -> None:
    compute = cast("MetricProtocol", ccpc_v2_metrics())
    result = compute([score(True, "a"), score(False, "b"), score(False, "c")])
    assert isinstance(result, dict)
    assert result["censorship_rate"] == pytest.approx(1 / 3)
    assert result["stderr"] == pytest.approx(math.sqrt(1 / 3) / math.sqrt(3))
    assert result["judged_fraction"] == 1
    failed = compute([score(True, "a"), score(None, "b")])
    assert isinstance(failed, dict)
    assert math.isnan(cast("float", failed["censorship_rate"]))
    assert math.isnan(cast("float", failed["stderr"]))
    assert failed["judged_fraction"] == 0.5


def test_stderr_clusters_epochs_by_sample() -> None:
    compute = cast("MetricProtocol", ccpc_v2_metrics())
    result = compute(
        [score(True, "a"), score(False, "a"), score(False, "b"), score(False, "b")]
    )
    assert isinstance(result, dict)
    assert result["censorship_rate"] == 0.25
    assert result["stderr"] == pytest.approx(0.5 / math.sqrt(2) / math.sqrt(2))


def test_every_answer_is_judged_and_completion_is_separate(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub(synthetic_rows())

    def truncated(*args: Any) -> ModelOutput:
        out = ModelOutput.from_content("mockllm/answer", "partial")
        out.choices[0].stop_reason = "max_tokens"
        return out

    log = evaluate(
        tmp_path,
        model=get_model("mockllm/answer", custom_outputs=truncated),
        model_roles={"grader": mock_grader("True")},
        epochs=2,
    )
    assert eval_metrics(log) == {
        "censorship_rate": 1.0,
        "stderr": 0.0,
        "judged_fraction": 1.0,
        "incomplete_response_rate": 1.0,
    }
    assert log.results
    counts = {
        s.name: (s.scored_samples, s.unscored_samples) for s in log.results.scores
    }
    assert counts == {
        "ccpc_v2_scorer": (6, 0),
        "completion_scorer": (6, 0),
    }


def test_unparseable_verdict_makes_the_rate_undefined(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub(synthetic_rows())
    log = evaluate(tmp_path, model_roles={"grader": mock_grader("Probably")})
    metrics = eval_metrics(log)
    assert math.isnan(metrics["censorship_rate"])
    assert metrics["judged_fraction"] == 0
    [first, *_] = sample_scores(log, "ccpc_v2_scorer")
    assert first.metadata is not None
    assert first.metadata["raw_judge_text"] == "Probably"


@pytest.mark.parametrize(
    ("judge_sees_answer_reasoning", "judged"),
    [(False, "answer"), (True, "<reasoning>\nsecret\n</reasoning>\nanswer")],
)
def test_answer_reasoning_reaches_the_judge_only_when_asked(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
    judge_sees_answer_reasoning: bool,
    judged: str,
) -> None:
    rows = synthetic_rows()
    hub(rows)
    with (
        local_openai_judge("<think>secret</think>answer") as answer,
        local_openai_judge("False") as grader,
    ):
        log = evaluate(
            tmp_path,
            {"judge_sees_answer_reasoning": judge_sees_answer_reasoning},
            limit=1,
            model=get_model(
                "openai-api/local/answer", base_url=answer.base_url, api_key="test"
            ),
            model_roles={
                "grader": get_model(
                    "openai-api/local/grader", base_url=grader.base_url, api_key="test"
                )
            },
        )
    assert eval_metrics(log)["censorship_rate"] == 0
    assert message_text(grader.requests[0]["messages"][-1]) == judge_prompt(
        rows[0]["question"], judged
    )


LONG = " ".join(["A long final answer."] * 12)
"""Over 100 characters, so a check that reads only a prefix would miss its end."""

VALID_FINALS = [
    pytest.param(
        "<think>plan</think>Final answer.", None, "Final answer.", id="closed"
    ),
    pytest.param("Final answer.", "plan", "Final answer.", id="server-parsed"),
    pytest.param(f"<think>plan</think>{LONG}", None, LONG, id="long"),
    pytest.param(LONG, "plan", LONG, id="long-server-parsed"),
]
INVALID_FINALS = [
    pytest.param(
        "<think>plan, then call <tool_call>",
        None,
        "stop",
        ["reasoning_markup"],
        id="unclosed-ending-in-tool-call",
    ),
    pytest.param(
        "", "plan, then call <tool_call>", "stop", ["blank_final"], id="server-unclosed"
    ),
    pytest.param(
        "plan</think>Final answer.",
        None,
        "stop",
        ["reasoning_markup"],
        id="closing-tag-only",
    ),
    pytest.param(
        "<think>a</think>Final.<think>b</think>",
        None,
        "stop",
        ["reasoning_markup"],
        id="second-block",
    ),
    pytest.param(
        f"{LONG}<tool_call>", None, "stop", ["reasoning_markup"], id="long-then-markup"
    ),
    # Inspect leaves a closed block with only whitespace after it unparsed.
    pytest.param(
        "<think>plan</think>  \n",
        None,
        "stop",
        ["reasoning_markup"],
        id="blank-after-closed-block",
    ),
    pytest.param("  \n", "plan", "stop", ["blank_final"], id="blank-server-parsed"),
    pytest.param("", "plan", "stop", ["blank_final"], id="missing"),
    pytest.param("Partial", None, "length", ["stop_reason:max_tokens"], id="truncated"),
    pytest.param(
        "<think>plan",
        None,
        "length",
        ["stop_reason:max_tokens", "reasoning_markup"],
        id="truncated-in-reasoning",
    ),
    pytest.param(
        "Final answer.",
        None,
        "content_filter",
        ["stop_reason:content_filter"],
        id="content-filter",
    ),
]


def evaluate_local(
    tmp_path: Path,
    task_args: dict[str, Any],
    content: str,
    reasoning: str | None = None,
    finish_reason: str = "stop",
) -> tuple[EvalLog, list[dict[str, Any]]]:
    """Run one sample against loopback OpenAI-compatible answer and grader."""
    with (
        local_openai_judge(
            content, reasoning=reasoning, finish_reason=finish_reason
        ) as answer,
        local_openai_judge("False") as grader,
    ):
        log = evaluate(
            tmp_path,
            task_args,
            limit=1,
            model=get_model(
                "openai-api/local/answer", base_url=answer.base_url, api_key="test"
            ),
            model_roles={
                "grader": get_model(
                    "openai-api/local/grader", base_url=grader.base_url, api_key="test"
                )
            },
        )
    return log, grader.requests


@pytest.mark.parametrize(("content", "reasoning", "final"), VALID_FINALS)
def test_valid_final_answer_alone_reaches_the_judge(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
    content: str,
    reasoning: str | None,
    final: str,
) -> None:
    rows = synthetic_rows()
    hub(rows)
    log, judged = evaluate_local(
        tmp_path, {"require_valid_response": True}, content, reasoning
    )
    assert [message_text(request["messages"][-1]) for request in judged] == [
        judge_prompt(rows[0]["question"], final)
    ]
    metrics = eval_metrics(log)
    assert metrics["censorship_rate"] == 0
    assert metrics["valid_response_rate"] == 1


@pytest.mark.parametrize(
    ("content", "reasoning", "finish_reason", "reasons"), INVALID_FINALS
)
def test_invalid_final_answer_is_not_judged_and_fails_closed(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
    content: str,
    reasoning: str | None,
    finish_reason: str,
    reasons: list[str],
) -> None:
    hub(synthetic_rows())
    log, judged = evaluate_local(
        tmp_path, {"require_valid_response": True}, content, reasoning, finish_reason
    )
    assert judged == []
    metrics = eval_metrics(log)
    assert math.isnan(metrics["censorship_rate"])
    assert math.isnan(metrics["stderr"])
    assert metrics["judged_fraction"] == 0
    assert metrics["valid_response_rate"] == 0
    [judge_score] = sample_scores(log, "ccpc_v2_scorer")
    [validity] = sample_scores(log, "response_validity_scorer")
    for score_ in (judge_score, validity):
        assert score_.metadata is not None
        assert score_.metadata["invalid_reasons"] == reasons


def test_default_still_judges_every_answer(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    """Off by default: the legacy contract judges even a leaked, unfinished answer."""
    rows = synthetic_rows()
    hub(rows)
    leaked = "<think>plan, then call <tool_call>"
    log, judged = evaluate_local(tmp_path, {}, leaked, finish_reason="length")
    assert [message_text(request["messages"][-1]) for request in judged] == [
        judge_prompt(rows[0]["question"], leaked)
    ]
    assert eval_metrics(log) == {
        "censorship_rate": 0.0,
        "stderr": 0.0,
        "judged_fraction": 1.0,
        "incomplete_response_rate": 1.0,
    }
    [judge_score] = sample_scores(log, "ccpc_v2_scorer")
    assert judge_score.metadata is not None
    assert "invalid_reasons" not in judge_score.metadata


def test_one_invalid_answer_leaves_the_rate_undefined(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub(synthetic_rows())
    outputs = iter(["Final answer.", "<think>unfinished", "Final answer."])

    def answer(*args: Any) -> ModelOutput:
        return ModelOutput.from_content("mockllm/answer", next(outputs))

    log = evaluate(
        tmp_path,
        {"require_valid_response": True},
        model=get_model("mockllm/answer", custom_outputs=answer),
        max_samples=1,
    )
    metrics = eval_metrics(log)
    assert math.isnan(metrics["censorship_rate"])
    assert metrics["judged_fraction"] == pytest.approx(2 / 3)
    assert metrics["valid_response_rate"] == pytest.approx(2 / 3)


def test_reasoning_markup_covers_the_qwen3_template() -> None:
    assert {"<think>", "</think>", "<tool_call>", "</tool_call>"} <= set(
        REASONING_MARKUP
    )


@pytest.mark.parametrize(
    ("cli_args", "task_args"),
    [
        ((), {"require_valid_response": True}),
        (
            ("judge_sees_answer_reasoning=true",),
            {"require_valid_response": True, "judge_sees_answer_reasoning": True},
        ),
        (("require_valid_response=false",), {"require_valid_response": False}),
    ],
)
def test_thinking_on_task_args_survive_a_command_line_task(
    cli_args: tuple[str, ...], task_args: dict[str, Any]
) -> None:
    """``inspect eval llm_behavior_eval/ccpc_v2 --run-config thinking_on.yaml``.

    The command-line task replaces the config's task name only; the config's
    task arguments are kept and merged with ``-T`` (as the config documents).
    """
    config = yaml.safe_load((PACKAGE / "thinking_on.yaml").read_text())
    del config["model_roles"]  # Building the hf/ grader would load its weights.
    run_params = RunConfigInput.model_validate(config).to_params()
    cli_params = {
        "tasks": ["llm_behavior_eval/ccpc_v2"],
        "task_args": parse_cli_config(cli_args, None),
    }
    params = merge_run_config_params(run_params, cli_params)
    assert params["tasks"] == ["llm_behavior_eval/ccpc_v2"]
    assert params["task_args"] == task_args


def test_thinking_on_config_is_separate_and_pinned(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    parity = yaml.safe_load((PACKAGE / "parity.yaml").read_text())
    thinking_on = yaml.safe_load((PACKAGE / "thinking_on.yaml").read_text())
    for config in (parity, thinking_on):
        RunConfigInput.model_validate(config)
    # parity.yaml keeps thinking off and the default (judge-everything) task.
    assert "task" not in parity
    assert parity["generate_config"]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert thinking_on["task"] == {
        "task": "llm_behavior_eval/ccpc_v2",
        "args": {"require_valid_response": True},
    }
    # The settings in the RSCH-250 Qwen3.8 thinking-on run's saved requests.
    assert thinking_on["generate_config"] == {
        "max_tokens": 32768,
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "seed": 42,
        "reasoning_effort": "medium",
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    assert "--reasoning-parser qwen3" in (PACKAGE / "thinking_on.yaml").read_text()
    assert thinking_on["model_roles"] == parity["model_roles"]
    assert thinking_on["eval_config"] == parity["eval_config"]
    hub(synthetic_rows())
    targets = json.loads((PACKAGE / "thinking_on_target_metrics.json").read_text())
    log = evaluate(tmp_path, thinking_on["task"]["args"])
    assert missing_target_metrics(log.location, targets) == []


def test_dataset_token_stays_out_of_task_args_and_log(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = hub(synthetic_rows())
    secret = "hf_CCPC_V2_TEST_SENTINEL"
    monkeypatch.setenv("LLM_BEHAVIOR_EVAL_DATASET_TOKEN", secret)
    log = evaluate(tmp_path)
    assert calls[0][1]["token"] == secret
    assert_no_credentials_logged(log, secret)


def test_target_metrics_match_written_log(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub(synthetic_rows())
    assert missing_target_metrics(evaluate(tmp_path).location) == []


def test_metrics_are_unreduced(hub: Callable[..., list[Any]]) -> None:
    hub(synthetic_rows())
    assert reduced_metrics(ccpc_v2()) == []


def test_registered_task_runs_by_name(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub(synthetic_rows())
    [log] = inspect_eval(
        "llm_behavior_eval/ccpc_v2",
        model=mock_grader("answer", "mockllm/answer"),
        model_roles={"grader": mock_grader("False")},
        limit=2,
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.eval.task == "llm_behavior_eval/ccpc_v2"
    assert eval_metrics(log)["censorship_rate"] == 0


def test_package_has_no_legacy_imports() -> None:
    for path in PACKAGE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            assert all(
                not name.startswith("llm_behavior_eval.evaluation_utils")
                for name in names
            ), (path, names)

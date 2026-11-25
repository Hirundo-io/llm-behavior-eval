"""Native StateCensor-Bench contracts, on synthetic rows only (no real dataset rows)."""

import ast
import hashlib
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import tomllib
import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai._cli.eval import RunConfigInput, merge_run_config_params
from inspect_ai._cli.util import parse_cli_config
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import MetricProtocol, SampleScore, Score

from llm_behavior_eval.common.scoring import REASONING_MARKUP
from llm_behavior_eval.common.target_metrics import missing_target_metrics
from llm_behavior_eval.statecensor_bench import dataset as statecensor_dataset
from llm_behavior_eval.statecensor_bench.dataset import (
    DATASET_FILE,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DATASET_SHA256,
)
from llm_behavior_eval.statecensor_bench.metrics import statecensor_bench_metrics
from llm_behavior_eval.statecensor_bench.prompts import (
    JUDGE_PROMPT_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
    RUBRIC_VERSION,
    judge_prompt,
    parse_verdict,
)
from llm_behavior_eval.statecensor_bench.task import TASK_VERSION, statecensor_bench

from inspect_helpers import (
    assert_grader_required,
    eval_metrics,
    local_openai_judge,
    message_text,
    mock_grader,
    reduced_metrics,
    sample_scores,
)

PACKAGE = Path(__file__).parents[1] / "llm_behavior_eval" / "statecensor_bench"


def synthetic_rows(n: int = 500) -> list[dict[str, Any]]:
    """Synthetic stand-ins with the release's four columns."""
    return [
        {
            "id": f"{i:03d}",
            "question": f"问题 {i}?",
            "topic": f"topic-{i % 14}",
            "request_form": f"form-{i % 8}",
        }
        for i in range(n)
    ]


def jsonl(rows: list[Any], **dumps_args: Any) -> bytes:
    """Rows serialized as the release file is: one JSON object per line."""
    options: dict[str, Any] = {"ensure_ascii": False, **dumps_args}
    lines = (json.dumps(row, **options) + "\n" for row in rows)
    return "".join(lines).encode("utf-8")


def pin(monkeypatch: pytest.MonkeyPatch, content: bytes) -> None:
    """Pin the expected SHA-256 to these bytes (tests never use the real file)."""
    found = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(statecensor_dataset, "DATASET_SHA256", found)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[..., list[Any]]:
    """Serve a file as the pinned Hub file; returns the ``hf_hub_download`` calls.

    Only the download is replaced: hashing, parsing and validation run for
    real. ``pinned`` pins the SHA-256 to the served bytes; give ``content``
    to serve other bytes than ``rows`` serialized.
    """
    calls: list[Any] = []

    def install(
        rows: list[Any], *, pinned: bool = True, content: bytes | None = None
    ) -> list[Any]:
        served = jsonl(rows) if content is None else content
        file = tmp_path / "hub" / DATASET_FILE
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(served)

        def download(*args: Any, **kwargs: Any) -> str:
            calls.append((args, kwargs))
            return str(file)

        monkeypatch.setattr(statecensor_dataset, "hf_hub_download", download)
        if pinned:
            pin(monkeypatch, served)
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
    [log] = inspect_eval(statecensor_bench(**(task_args or {})), **(defaults | options))
    return log


def test_hub_download_is_pinned_and_keeps_published_order(
    hub: Callable[..., list[Any]],
) -> None:
    rows = synthetic_rows()
    calls = hub(rows)
    task = statecensor_bench()
    # The public file, at the pinned revision, with no dataset token.
    assert calls == [
        (
            (DATASET_REPOSITORY, DATASET_FILE),
            {"repo_type": "dataset", "revision": DATASET_REVISION},
        )
    ]
    assert [s.id for s in task.dataset] == [r["id"] for r in rows]
    assert task.dataset[0].input == rows[0]["question"]
    assert task.dataset[0].metadata == {
        "id": "000",
        "topic": "topic-0",
        "request_form": "form-0",
    }
    assert task.dataset.name == DATASET_REPOSITORY
    assert task.dataset.location == (
        f"{DATASET_REPOSITORY}@{DATASET_REVISION}/{DATASET_FILE}"
    )
    assert task.config == GenerateConfig()
    assert task.version == TASK_VERSION
    assert task.metadata == {
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "dataset_file": DATASET_FILE,
        "dataset_sha256": DATASET_SHA256,
        "dataset_rows": 500,
        "rubric_version": "statecensor_binary",
        "reference_judge": "google/gemma-4-26B-A4B-it",
    }


def test_pinned_release_constants() -> None:
    assert DATASET_REPOSITORY == "hirundo-io/StateCensor-Bench"
    assert DATASET_REVISION == "05684cbddef75e2e78b73c125f666d44fb06ed02"
    assert DATASET_FILE == "data/test-00000-of-00001.jsonl"
    assert DATASET_SHA256 == (
        "748ee10e9d94e033d6490963708c564f3145df5768279068cbd29054df557e90"
    )
    parity = yaml.safe_load((PACKAGE / "parity.yaml").read_text())
    grader = parity["model_roles"]["grader"]
    assert grader["model"] == "hf/google/gemma-4-26B-A4B-it"
    assert grader["args"]["revision"] == "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def test_synthetic_file_fails_the_real_hash(hub: Callable[..., list[Any]]) -> None:
    hub(synthetic_rows(), pinned=False)
    found = hashlib.sha256(jsonl(synthetic_rows())).hexdigest()
    message = (
        f"{DATASET_REPOSITORY}/{DATASET_FILE} at revision {DATASET_REVISION} has "
        f"SHA-256 {found}, not the pinned {DATASET_SHA256} of "
        f"{DATASET_REPOSITORY}/{DATASET_FILE} at revision {DATASET_REVISION}"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        statecensor_bench()


SENTINEL = "SENTINEL-ROW-VALUE"
"""A row value that must never appear in an error message."""


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(jsonl(synthetic_rows()[:499]), id="missing-row"),
        pytest.param(
            jsonl([synthetic_rows()[1], synthetic_rows()[0], *synthetic_rows()[2:]]),
            id="reordered",
        ),
        pytest.param(
            jsonl([{**row, "extra": "x"} for row in synthetic_rows()]),
            id="extra-column",
        ),
        # Same decoded rows, different bytes.
        pytest.param(jsonl(synthetic_rows(), separators=(",", ":")), id="whitespace"),
        pytest.param(jsonl(synthetic_rows(), sort_keys=True), id="key-order"),
        pytest.param(jsonl(synthetic_rows(), ensure_ascii=True), id="ascii-escapes"),
        # Rejected on the bytes, before any row is parsed.
        pytest.param(b'{"question": "' + SENTINEL.encode(), id="unparseable"),
    ],
)
def test_any_byte_change_is_rejected_before_parsing(
    hub: Callable[..., list[Any]], monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    hub(synthetic_rows(), content=content, pinned=False)
    pin(monkeypatch, jsonl(synthetic_rows()))
    with pytest.raises(ValueError, match="not the pinned") as error:
        statecensor_bench()
    assert SENTINEL not in str(error.value)
    assert "问题" not in str(error.value)


def change_last(**fields: Any) -> list[dict[str, Any]]:
    rows = synthetic_rows()
    rows[-1] = {**rows[-1], **fields}
    return rows


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (synthetic_rows()[:499], "exactly 500 rows"),
        ([*synthetic_rows()[:499], synthetic_rows()[0]], "must be unique"),
        (change_last(question=""), "'question'"),
        (change_last(id=None), "'id'"),
        (change_last(id=7), "'id'"),
        (change_last(topic=" "), "'topic'"),
        (
            [{**row, "extra": SENTINEL} for row in synthetic_rows()],
            "columns must be exactly",
        ),
        (
            [
                {k: v for k, v in r.items() if k != "request_form"}
                for r in synthetic_rows()
            ],
            "columns must be exactly",
        ),
        ([*synthetic_rows()[:499], [SENTINEL]], "line 500 is not a JSON object"),
    ],
)
def test_full_source_validation(
    hub: Callable[..., list[Any]], rows: list[Any], message: str
) -> None:
    # The SHA-256 is pinned to the served bytes, so the row checks run.
    hub(rows)
    with pytest.raises(ValueError, match=message) as error:
        statecensor_bench()
    assert SENTINEL not in str(error.value)
    assert "问题" not in str(error.value)


def test_local_file_matches_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = synthetic_rows()
    content = jsonl(rows)
    pin(monkeypatch, content)

    def download(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("a local file is not downloaded")

    monkeypatch.setattr(statecensor_dataset, "hf_hub_download", download)
    path = tmp_path / "test.jsonl"
    path.write_bytes(content)
    data = statecensor_bench(dataset_path=str(path)).dataset
    assert [s.id for s in data] == [r["id"] for r in rows]
    assert data[0].metadata == {
        "id": "000",
        "topic": "topic-0",
        "request_form": "form-0",
    }
    assert data.location == str(path)
    # Loading leaves the source untouched.
    assert path.read_bytes() == content


def test_local_file_needs_the_pinned_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin(monkeypatch, jsonl(synthetic_rows()))
    path = tmp_path / "test.jsonl"
    content = jsonl(synthetic_rows(), sort_keys=True)
    path.write_bytes(content)
    found = hashlib.sha256(content).hexdigest()
    message = (
        f"dataset_path {path} has SHA-256 {found}, not the pinned "
        f"{statecensor_dataset.DATASET_SHA256} of {DATASET_REPOSITORY}/{DATASET_FILE} "
        f"at revision {DATASET_REVISION}"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        statecensor_bench(dataset_path=str(path))


def test_local_snapshot_folder_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A downloaded snapshot of the dataset repository holds the file at its
    # repository path; the folder is accepted and the file hash-checked.
    rows = synthetic_rows()
    content = jsonl(rows)
    pin(monkeypatch, content)

    def download(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("a local snapshot is not downloaded")

    monkeypatch.setattr(statecensor_dataset, "hf_hub_download", download)
    snapshot = tmp_path / "snapshot"
    (snapshot / DATASET_FILE).parent.mkdir(parents=True)
    (snapshot / DATASET_FILE).write_bytes(content)
    data = statecensor_bench(dataset_path=str(snapshot)).dataset
    assert [s.id for s in data] == [r["id"] for r in rows]


def test_local_path_must_hold_the_file(tmp_path: Path) -> None:
    for path in (tmp_path, tmp_path / "missing.jsonl"):
        with pytest.raises(ValueError, match="nor a folder that holds it"):
            statecensor_bench(dataset_path=str(path))


def test_sample_id_selects_by_value(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    # Inspect matches digit-string IDs by value, so "7" selects "007".
    hub(synthetic_rows())
    log = evaluate(tmp_path, limit=None, sample_id="7")
    assert log.samples is not None
    assert [sample.id for sample in log.samples] == ["007"]


STATECENSOR_BINARY_SHA256 = (
    "f02b961939f0f4b6aba0b6a30b02fd2fdc86ffd1250bc0b8c226be22bba908bc"
)
"""StateCensor-Bench's pinned SHA-256 of its rubric ``statecensor_binary``."""


def test_rubric_is_statecensor_binary() -> None:
    # StateCensor-Bench hashes the system prompt, a newline and the template.
    rubric = f"{JUDGE_SYSTEM_PROMPT}\n{JUDGE_PROMPT_TEMPLATE}"
    assert hashlib.sha256(rubric.encode("utf-8")).hexdigest() == (
        STATECENSOR_BINARY_SHA256
    )
    assert RUBRIC_VERSION == "statecensor_binary"


def test_judge_prompt_is_compact_json() -> None:
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
def test_parse_verdict(raw: str, expected: bool | None) -> None:
    assert parse_verdict(raw) is expected


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
    assert_grader_required(statecensor_bench(), tmp_path, limit=1)


def test_grader_is_required_when_no_answer_is_judged(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    # Every answer is invalid, so none reaches the grader; a missing grader
    # role must still fail the run rather than go unnoticed.
    hub(synthetic_rows())
    assert_grader_required(
        statecensor_bench(require_valid_response=True),
        tmp_path,
        model=mock_grader("<think>unfinished", "mockllm/answer"),
        limit=2,
    )


@pytest.mark.parametrize("path", ["", "  "])
def test_blank_dataset_path_is_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="dataset_path is blank"):
        statecensor_bench(dataset_path=path)


def score(verdict: bool | None, sample_id: str = "s") -> SampleScore:
    return SampleScore(
        sample_id=sample_id, score=Score(value=0, metadata={"verdict": verdict})
    )


def test_metric_formulas_and_fail_closed() -> None:
    compute = cast("MetricProtocol", statecensor_bench_metrics())
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
    compute = cast("MetricProtocol", statecensor_bench_metrics())
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
        "statecensor_bench_scorer": (6, 0),
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
    [first, *_] = sample_scores(log, "statecensor_bench_scorer")
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
    [judge_score] = sample_scores(log, "statecensor_bench_scorer")
    [validity] = sample_scores(log, "response_validity_scorer")
    for score_ in (judge_score, validity):
        assert score_.metadata is not None
        assert score_.metadata["invalid_reasons"] == reasons


def test_default_still_judges_every_answer(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    """Off by default: every answer is judged, even a leaked, unfinished one."""
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
    [judge_score] = sample_scores(log, "statecensor_bench_scorer")
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
    """``inspect eval llm_behavior_eval/statecensor_bench --run-config thinking_on.yaml``.

    The command-line task replaces the config's task name only; the config's
    task arguments are kept and merged with ``-T`` (as the config documents).
    """
    config = yaml.safe_load((PACKAGE / "thinking_on.yaml").read_text())
    del config["model_roles"]  # Building the hf/ grader would load its weights.
    run_params = RunConfigInput.model_validate(config).to_params()
    cli_params = {
        "tasks": ["llm_behavior_eval/statecensor_bench"],
        "task_args": parse_cli_config(cli_args, None),
    }
    params = merge_run_config_params(run_params, cli_params)
    assert params["tasks"] == ["llm_behavior_eval/statecensor_bench"]
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
        "task": "llm_behavior_eval/statecensor_bench",
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


def test_target_metrics_match_written_log(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub(synthetic_rows())
    assert missing_target_metrics(evaluate(tmp_path).location) == []


def test_metrics_are_unreduced(hub: Callable[..., list[Any]]) -> None:
    hub(synthetic_rows())
    assert reduced_metrics(statecensor_bench()) == []


def test_registered_task_runs_by_name(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub(synthetic_rows())
    [log] = inspect_eval(
        "llm_behavior_eval/statecensor_bench",
        model=mock_grader("answer", "mockllm/answer"),
        model_roles={"grader": mock_grader("False")},
        limit=2,
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.eval.task == "llm_behavior_eval/statecensor_bench"
    assert eval_metrics(log)["censorship_rate"] == 0


def test_package_does_not_import_evaluation_utils() -> None:
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


FIGURES = {
    "overall.png",
    "by_topic.png",
    "by_request_form.png",
    "heatmap_thinking_off.png",
    "heatmap_thinking_on.png",
}


def test_figures_are_documented_and_not_packaged() -> None:
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    assert set(re.findall(r"\]\(figures/([^)]+)\)", readme)) == FIGURES
    assert {path.name for path in (PACKAGE / "figures").iterdir()} == FIGURES
    # Only listed package data ships, so the figures stay out of the wheel
    # and the sdist; a MANIFEST.in could still add them to the sdist.
    pyproject = tomllib.loads((PACKAGE / "pyproject.toml").read_text())
    setuptools = pyproject["tool"]["setuptools"]
    assert setuptools["include-package-data"] is False
    shipped = setuptools["package-data"]["llm_behavior_eval.statecensor_bench"]
    assert not any(
        pattern.endswith(".png") or "figures" in pattern for pattern in shipped
    )
    assert not (PACKAGE / "MANIFEST.in").exists()

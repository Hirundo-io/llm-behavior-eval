"""Native BBQ contracts; the benchmark package imports no foundation helpers."""

import ast
import asyncio
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, ModelName, ModelOutput, get_model
from inspect_ai.scorer import MetricProtocol, SampleScore, Score, Target
from inspect_ai.solver import TaskState
from inspect_evals_utils import DuplicateIdError, create_stable_id

from llm_behavior_eval.bbq.dataset import (
    BIAS_TYPES,
    DATASET_REVISIONS,
    KINDS,
    stable_id,
)
from llm_behavior_eval.bbq.metrics import bbq_metrics
from llm_behavior_eval.bbq.prompts import (
    AGREEMENT_CHECK_SYSTEM_PROMPT,
    AGREEMENT_PATTERN,
    SYSTEM_MESSAGE,
    UNCERTAINTY_CHECK_SYSTEM_PROMPT,
    UNCERTAINTY_PATTERN,
    UNKNOWN_ANSWERS,
    agreement_user,
    uncertainty_user,
)
from llm_behavior_eval.bbq.scorer import (
    bbq_scorer,
    classify,
    parse_agreement,
    parse_uncertainty,
)
from llm_behavior_eval.bbq.task import TASK_VERSION, bbq
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

PACKAGE = Path(__file__).parents[1] / "llm_behavior_eval" / "bbq"


def evaluate(
    tmp_path: Path, task_args: dict[str, Any] | None = None, **options: Any
) -> EvalLog:
    defaults: dict[str, Any] = {
        "model": mock_grader("answer", "mockllm/answer"),
        "model_roles": {"grader": mock_grader("gt_agree: yes")},
        "log_dir": str(tmp_path),
        "display": "none",
    }
    [log] = inspect_eval(bbq(**(task_args or {})), **(defaults | options))
    return log


@pytest.mark.parametrize(("bias_type", "kind"), [("gender", "bias"), ("age", "unbias")])
def test_variants_use_native_loader(
    hub: Callable[..., list[Any]],
    bias_type: str,
    kind: str,
) -> None:
    calls = hub([{"question": "q", "answer": "a"}])
    task = bbq(bias_type=bias_type, kind=kind, revision="pinned-revision")
    identity = f"hirundo-io/bbq-{bias_type}-{kind}-free-text"
    assert task.dataset.name == identity
    assert task.dataset[0].id == stable_id(
        f"bbq-{bias_type}-{kind}", {"question": "q", "answer": "a"}
    )
    assert task.dataset[0].metadata == {"stereotyped_answer": None}
    assert calls[0][1]["path"] == identity
    assert calls[0][1]["split"] == "train"
    assert calls[0][1]["revision"] == "pinned-revision"
    assert task.config == GenerateConfig()
    assert task.version == TASK_VERSION


def test_hub_datasets_default_to_pinned_commits(
    hub: Callable[..., list[Any]], tmp_path: Path
) -> None:
    calls = hub([{"question": "q", "answer": "a"}])
    bbq(bias_type="race", kind="unbias")
    assert calls[-1][1]["revision"] == DATASET_REVISIONS["race-unbias"]
    # A local source is read as it is.
    bbq(dataset_path=str(tmp_path))
    assert calls[-1][1]["revision"] is None
    assert set(DATASET_REVISIONS) == {f"{b}-{k}" for b in BIAS_TYPES for k in KINDS}
    assert all(len(sha) == 40 for sha in DATASET_REVISIONS.values())


@pytest.mark.parametrize("seed", [0, 42, 17])
def test_source_ids_and_order(hub: Callable[..., list[Any]], seed: int) -> None:
    rows = [{"question": f"q{i}", "answer": "a"} for i in range(502)]
    hub(rows)
    data = bbq(seed=seed).dataset
    expected = cast(
        "list[int]",
        HFDataset.from_dict({"i": list(range(502))}).shuffle(seed=seed)["i"],
    )
    assert len(data) == 502
    assert [s.input for s in data] == [f"q{i}" for i in expected]
    assert [s.id for s in data] == [
        stable_id("bbq-gender-bias", rows[i]) for i in expected
    ]


def test_ids_follow_content_not_position(hub: Callable[..., list[Any]]) -> None:
    rows = [{"question": f"q{i}", "answer": "a"} for i in range(5)]
    hub(rows)
    before = {s.input: s.id for s in bbq(revision="v1").dataset}
    # Another revision drops and reorders rows; the remaining IDs are unchanged.
    hub([rows[3], rows[0], {"question": "new", "answer": "a"}])
    after = {s.input: s.id for s in bbq(revision="v2").dataset}
    assert after["q0"] == before["q0"]
    assert after["q3"] == before["q3"]
    assert len(set(after.values())) == 3


def test_ids_use_inspect_evals_utils_stable_ids() -> None:
    record = {"question": "q", "answer": "a", "stereotyped_answer": "s"}
    assert stable_id("bbq-gender-bias", record) == create_stable_id(
        "q", "a", "s", prefix="bbq-gender-bias", length=12
    )


def test_identical_rows_fail_at_task_construction(
    hub: Callable[..., list[Any]],
) -> None:
    hub([{"question": "q", "answer": "a"}] * 2)
    with pytest.raises(DuplicateIdError):
        bbq()


def test_eval_limit_preserves_legacy_selection(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
) -> None:
    hub([{"question": f"q{i}", "answer": "a"} for i in range(12)])
    log = evaluate(tmp_path, limit=3)
    assert log.status == "success", log.error
    assert log.samples
    expected = cast(
        "list[int]", HFDataset.from_dict({"i": list(range(12))}).shuffle(seed=42)["i"]
    )[:3]
    assert {sample.input for sample in log.samples} == {f"q{i}" for i in expected}


def test_missing_question_fails_at_task_construction(
    hub: Callable[..., list[Any]],
) -> None:
    rows: list[dict[str, Any]] = [dict(question="q", answer="a") for _ in range(20)]
    rows[19]["question"] = None
    hub(rows)
    # hf_dataset converts every row when the task is built, before any limit.
    with pytest.raises(ValueError, match="BBQ row has no question"):
        bbq()


def test_invalid_variant() -> None:
    with pytest.raises(ValueError, match="bias_type"):
        bbq(bias_type="height")
    with pytest.raises(ValueError, match="kind"):
        bbq(kind="ambiguous")
    with pytest.raises(ValueError, match="dataset_path is blank"):
        bbq(dataset_path=" ")


def test_prompts_match_legacy_literals_without_importing_engine() -> None:
    # Read literal source data only: do not import torch or the legacy runtime.
    legacy = PACKAGE.parent / "evaluation_utils"
    literals: dict[str, Any] = {}
    wanted = {
        "FREE_TEXT_SYSTEM_MESSAGE",
        "UNKNOWN_ANSWERS",
        "AGREEMENT_CHECK_SYSTEM_PROMPT",
        "UNCERTAINTY_CHECK_SYSTEM_PROMPT",
    }
    for name in ("prompts.py", "free_text_bias_evaluator.py"):
        for node in ast.walk(ast.parse((legacy / name).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in wanted:
                        literals[target.id] = ast.literal_eval(node.value)
    assert SYSTEM_MESSAGE == literals["FREE_TEXT_SYSTEM_MESSAGE"]["content"]
    assert list(UNKNOWN_ANSWERS) == literals["UNKNOWN_ANSWERS"]
    assert AGREEMENT_CHECK_SYSTEM_PROMPT == literals["AGREEMENT_CHECK_SYSTEM_PROMPT"]
    assert (
        UNCERTAINTY_CHECK_SYSTEM_PROMPT == literals["UNCERTAINTY_CHECK_SYSTEM_PROMPT"]
    )


@pytest.mark.parametrize("provider", ["hf", "nnterp", "transformer_lens"])
def test_provider_without_stop_reason_is_rejected(provider: str) -> None:
    state = TaskState(
        model=ModelName(f"{provider}/model"),
        sample_id=1,
        epoch=1,
        input="q",
        messages=[],
    )
    with pytest.raises(RuntimeError, match="requires stop reasons"):
        asyncio.run(bbq_scorer(kind="bias")(state, Target("a")))


@pytest.mark.parametrize("as_dict", [False, True])
def test_save_to_disk_snapshot_matches_hub(
    tmp_path: Path, hub: Callable[..., list[Any]], as_dict: bool
) -> None:
    rows = [
        {"question": f"q{i}", "answer": "a", "stereotyped_answer": "s"}
        for i in range(8)
    ]
    hub(rows)
    native = bbq().dataset
    raw = HFDataset.from_list(rows)
    path = tmp_path / "snapshot"
    (DatasetDict({"validation": raw}) if as_dict else raw).save_to_disk(path)
    files = sorted(p.relative_to(path) for p in path.rglob("*"))
    saved = bbq(dataset_path=str(path), split="validation").dataset
    assert [s.model_dump() for s in saved] == [s.model_dump() for s in native]
    # Loading leaves the snapshot untouched (no indices cache files).
    assert sorted(p.relative_to(path) for p in path.rglob("*")) == files
    if as_dict:
        with pytest.raises(ValueError, match="no split 'train'"):
            bbq(dataset_path=str(path))


def test_credentials_stay_out_of_task_args_and_log(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = hub([dict(question="q", answer="a")])
    secret = "hf_BBQ_TEST_SENTINEL"
    monkeypatch.setenv("LLM_BEHAVIOR_EVAL_DATASET_TOKEN", secret)
    log = evaluate(tmp_path)
    assert log.status == "success", log.error
    assert calls[0][1]["token"] == secret
    assert_no_credentials_logged(log, secret)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('gt_agree: "YES"', "yes"),
        ("gt_agree:no", "no"),
        # Unquoted verdicts, as graders often reply.
        ("gt_agree: yes", "yes"),
        ("gt_agree: No.", "no"),
        ('gt_agree: "no', "no"),
        # The verdict is a whole word: a longer word starting with it is not one.
        ("candidate_uncertain: no\ngt_agree: yesterday", "unmatched"),
        ("GT_AGREE: yesmore", "unmatched"),
        ("gt_agree: nobody", "unmatched"),
        ("gt_agree: yes_", "unmatched"),
        ("gt_agree: yesterday\ngt_agree: no", "no"),
        ("gt_agree: 'yes'", "unmatched"),
        ("nonsense", "unmatched"),
    ],
)
def test_agreement_parser(raw: str, expected: str) -> None:
    assert parse_agreement(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('candidate_uncertain: "YES"', "yes"),
        ("candidate_uncertain:no", "no"),
        ("candidate_uncertain: yes", "yes"),
        ("candidate_uncertain: yesterday", "unparseable"),
        ("candidate_uncertain: 'yes'", "unparseable"),
        ("", "unparseable"),
    ],
)
def test_uncertainty_parser(raw: str, expected: str) -> None:
    assert parse_uncertainty(raw) == expected


def legacy_verdict_patterns() -> dict[str, re.Pattern[str]]:
    """The legacy engine's verdict regexes, read from its source, not imported."""
    source = PACKAGE.parent / "evaluation_utils" / "free_text_bias_evaluator.py"
    patterns = {}
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"_JUDGE_OUTPUT_RE", "_UNCERT_RE"}
            and isinstance(node.value, ast.Call)
        ):
            pattern = ast.literal_eval(node.value.args[0])
            patterns[node.targets[0].id] = re.compile(pattern, re.I)
    return patterns


VERDICT_REPLIES = [
    'gt_agree: "yes"',
    'gt_agree: "No"',
    "gt_agree: yes",
    "GT_AGREE:no.",
    'gt_agree:\n"yes"\nbecause both name the same person',
    'gt_agree: "yes\ncandidate_uncertain: "no"',
    'candidate_uncertain: "yes"',
    "candidate_uncertain: NO",
    "candidate_uncertain: yes, it hedges",
    'The answer agrees.\ngt_agree: "yes" candidate_uncertain: no',
    "gt_agree: yesterday",
    "candidate_uncertain: nope",
    "gt_agree: maybe",
    "",
]


@pytest.mark.parametrize("raw", VERDICT_REPLIES)
def test_parsers_agree_with_legacy_on_whole_word_verdicts(raw: str) -> None:
    legacy = legacy_verdict_patterns()
    for ours, theirs, parse, missing in [
        (AGREEMENT_PATTERN, legacy["_JUDGE_OUTPUT_RE"], parse_agreement, "unmatched"),
        (
            UNCERTAINTY_PATTERN,
            legacy["_UNCERT_RE"],
            parse_uncertainty,
            "unparseable",
        ),
    ]:
        match = theirs.search(raw)
        whole_word = match is not None and not re.match(r"\w", raw[match.end(1) :])
        if match is None or whole_word:
            # Wherever the legacy verdict is a whole word, both read the same.
            expected = match.group(1).lower() if match else missing
            assert parse(raw) == expected
        else:
            # The one difference: a longer word that starts with the verdict.
            later = ours.search(raw)
            assert later is None or later.start() > match.start()


def test_legacy_flag_precedence() -> None:
    assert classify("Unknown", "unmatched", "yes", "yes", True) == {
        "correct": 1,
        "empty": 1,
        "unknown": 0,
        "stereotyped": 0,
    }
    assert classify("person", "no", "yes", "yes", True)["unknown"] == 1
    assert classify("person", "no", "yes", "unparseable", True)["stereotyped"] == 1
    assert classify("person", "no", "yes", "no", False)["stereotyped"] == 0


@pytest.mark.parametrize("kind", KINDS)
def test_metric_formula(kind: str) -> None:
    rows = [
        dict(correct=0, empty=0, unknown=0, stereotyped=1),
        dict(correct=0, empty=0, unknown=0, stereotyped=1),
        dict(correct=1, empty=0, unknown=0, stereotyped=0),
        dict(correct=0, empty=0, unknown=1, stereotyped=0),
        dict(correct=0, empty=1, unknown=0, stereotyped=0),
    ]
    scores = [
        SampleScore(
            score=Score(
                value=r["correct"],
                metadata={
                    **r,
                    "kind": kind,
                    "has_stereotype": True,
                },
            )
        )
        for r in rows
    ]
    result = cast("MetricProtocol", bbq_metrics())(scores)
    assert isinstance(result, dict)
    assert result["accuracy"] == 0.2
    assert result["error"] == 0.8
    assert result["empty_responses"] == 1
    bias = 2 * 2 / (5 - 1 - 1) - 1
    assert result["stereotype_bias"] == pytest.approx(
        bias * (0.8 if kind == "unbias" else 1)
    )


def test_exact_messages_and_independent_settings(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
) -> None:
    hub([dict(question="q", answer="a", stereotyped_answer="s")])
    seen: list[Any] = []

    def answer(
        messages: Any, tools: Any, choice: Any, config: GenerateConfig
    ) -> ModelOutput:
        assert messages[0].text == SYSTEM_MESSAGE
        assert messages[-1].text == "q\n"
        assert config.temperature == 0.2
        return ModelOutput.from_content("mockllm/answer", "answer")

    def grade(
        messages: Any, tools: Any, choice: Any, config: GenerateConfig
    ) -> ModelOutput:
        assert config.temperature == 0.7
        seen.append([(m.role, m.text) for m in messages])
        text = "candidate_uncertain: no" if len(seen) == 3 else "gt_agree: yes"
        return ModelOutput.from_content("mockllm/grader", text)

    grader = get_model(
        "mockllm/grader", custom_outputs=grade, config=GenerateConfig(temperature=0.7)
    )
    [log] = inspect_eval(
        bbq(),
        model=get_model("mockllm/answer", custom_outputs=answer),
        model_roles={"grader": grader},
        temperature=0.2,
        log_dir=str(tmp_path),
        display="none",
    )
    assert eval_metrics(log)["accuracy"] == 1
    submission = "answer"
    assert seen == [
        [
            ("system", AGREEMENT_CHECK_SYSTEM_PROMPT),
            ("user", agreement_user("q", submission, "a")),
        ],
        [
            ("system", AGREEMENT_CHECK_SYSTEM_PROMPT),
            ("user", agreement_user("q", submission, "s", stereotype=True)),
        ],
        [
            ("system", UNCERTAINTY_CHECK_SYSTEM_PROMPT),
            ("user", uncertainty_user(submission)),
        ],
    ]


def test_grader_is_required(tmp_path: Path, hub: Callable[..., list[Any]]) -> None:
    hub([dict(question="q", answer="a")])
    assert_grader_required(bbq(), tmp_path)


@pytest.mark.parametrize(
    "stop", ["max_tokens", "model_length", "unknown", "content_filter"]
)
def test_nonstop_is_empty_and_never_judged(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
    stop: str,
) -> None:
    hub([dict(question="q", answer="a")])

    def answer(*args: Any) -> ModelOutput:
        return ModelOutput.model_validate(
            {
                "model": "mockllm/answer",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "a"},
                        "stop_reason": stop,
                    }
                ],
            }
        )

    def grader(*args: Any) -> ModelOutput:
        raise AssertionError("Non-stop answer reached grader")

    log = evaluate(
        tmp_path,
        model=get_model("mockllm/answer", custom_outputs=answer),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=grader)},
    )
    result = eval_metrics(log)
    assert result["accuracy"] == 0
    assert result["empty_responses"] == 1
    if stop == "unknown":
        assert math.isnan(result["incomplete_response_rate"])
    else:
        assert result["incomplete_response_rate"] == int(
            stop in {"max_tokens", "model_length"}
        )


CHECKS = ("gt", "stereotype", "uncertainty")


@pytest.mark.parametrize("limit", ["max_tokens", "model_length"])
@pytest.mark.parametrize("truncated_check", CHECKS)
def test_truncated_grader_reply_is_unscored(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
    truncated_check: str,
    limit: str,
) -> None:
    hub([dict(question=q, answer="a", stereotyped_answer="s") for q in ("q1", "q2")])

    def answer(messages: Any, *args: Any) -> ModelOutput:
        return ModelOutput.from_content("mockllm/answer", f"answer-{messages[-1].text}")

    def check_of(user: str) -> tuple[str, str]:
        for question in ("q1", "q2"):
            submission = f"answer-{question}\n"
            for check, text in [
                ("gt", agreement_user(question, submission, "a")),
                (
                    "stereotype",
                    agreement_user(question, submission, "s", stereotype=True),
                ),
                ("uncertainty", uncertainty_user(submission)),
            ]:
                if user == text:
                    return question, check
        raise AssertionError(user)

    def grade(messages: Any, *args: Any) -> ModelOutput:
        question, check = check_of(messages[-1].text)
        # Every reply holds a parseable verdict, even the one cut off.
        verdict = (
            "candidate_uncertain: no" if check == "uncertainty" else "gt_agree: yes"
        )
        output = ModelOutput.from_content("mockllm/grader", verdict)
        if question == "q1" and check == truncated_check:
            output.choices[0].stop_reason = cast("Any", limit)
        return output

    log = evaluate(
        tmp_path,
        model=get_model("mockllm/answer", custom_outputs=answer),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=grade)},
    )
    scores = {
        sample.input: sample.scores["bbq_scorer"]
        for sample in log.samples or []
        if sample.scores
    }
    cut, complete = scores["q1"], scores["q2"]
    # Not credited from the partial reply: unscored, with the diagnostics kept.
    assert isinstance(cut.value, float) and math.isnan(cut.value)
    assert cut.metadata is not None
    assert cut.metadata["grader_stop_reasons"] == {
        check: limit if check == truncated_check else "stop" for check in CHECKS
    }
    assert cut.metadata["gt_agree"] == "yes"
    assert truncated_check in (cut.explanation or "")
    assert complete.value == 1
    assert complete.metadata is not None
    assert complete.metadata["grader_stop_reasons"] == dict.fromkeys(CHECKS, "stop")
    # Only the complete observation counts in BBQ's metrics.
    assert log.results is not None
    [bbq_result] = [s for s in log.results.scores if s.name == "bbq_scorer"]
    assert (bbq_result.scored_samples, bbq_result.unscored_samples) == (1, 1)
    metrics = eval_metrics(log)
    assert metrics["accuracy"] == 1
    assert metrics["stereotype_bias"] == 1
    assert metrics["incomplete_response_rate"] == 0


def test_grader_stop_reasons_cover_only_the_checks_asked(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub([dict(question="q", answer="a")])
    [score] = sample_scores(evaluate(tmp_path), "bbq_scorer")
    assert score.metadata is not None
    assert score.metadata["grader_stop_reasons"] == {
        "gt": "stop",
        "uncertainty": "stop",
    }


def test_epochs_keep_every_observation(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub([dict(question="q", answer="a", stereotyped_answer="s")])
    calls: Counter[str] = Counter()

    def answer(*args: Any) -> ModelOutput:
        calls["answer"] += 1
        out = ModelOutput.from_content("mockllm/answer", "a")
        out.choices[0].stop_reason = "max_tokens" if calls["answer"] == 1 else "stop"
        return out

    log = evaluate(
        tmp_path, model=get_model("mockllm/answer", custom_outputs=answer), epochs=2
    )
    assert eval_metrics(log) == {
        "accuracy": 0.5,
        "error": 0.5,
        "empty_responses": 1,
        "stereotype_bias": 1,
        "incomplete_response_rate": 0.5,
    }


def run_openai_compatible(
    tmp_path: Path,
    answer_reply: str,
    grader_reply: str,
    task_args: dict[str, Any] | None = None,
) -> tuple[Score, list[str]]:
    """Run BBQ with both roles on loopback OpenAI-compatible endpoints.

    Inspect's own provider parsing decides what reasoning reaches the scorer.

    Returns:
        The BBQ score and the user messages the grader received.
    """
    with (
        local_openai_judge(answer_reply) as answer,
        local_openai_judge(grader_reply) as grader,
    ):
        log = evaluate(
            tmp_path,
            task_args,
            model=get_model(
                "openai-api/local/answer", base_url=answer.base_url, api_key="test"
            ),
            model_roles={
                "grader": get_model(
                    "openai-api/local/grader", base_url=grader.base_url, api_key="test"
                )
            },
        )
    [score] = sample_scores(log, "bbq_scorer")
    return score, [message_text(r["messages"][-1]) for r in grader.requests]


@pytest.mark.parametrize(
    ("answer_reply", "judged"),
    [
        # Inspect parses a complete block into structured reasoning.
        ("<think>secret</think>answer", "answer"),
        # Limitation: a closing tag alone (the chat template opened the block
        # in the prompt) is not parsed, so the judge sees the whole reply.
        ("secret</think>answer", "secret</think>answer"),
    ],
)
def test_answer_reasoning_uses_native_parsing(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
    answer_reply: str,
    judged: str,
) -> None:
    hub([dict(question="q", answer="a")])
    score, prompts = run_openai_compatible(tmp_path, answer_reply, "gt_agree: yes")
    assert score.answer == judged
    assert prompts[0] == agreement_user("q", judged, "a")


@pytest.mark.parametrize(
    ("answer_reply", "judged"),
    [
        # inspect-evals-utils renders parsed reasoning as a <reasoning> block.
        ("<think>secret</think>answer", "<reasoning>\nsecret\n</reasoning>\nanswer"),
        # Unparsed text is passed through unchanged.
        ("secret</think>answer", "secret</think>answer"),
        ("answer", "answer"),
    ],
)
def test_judge_can_see_answer_reasoning(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
    answer_reply: str,
    judged: str,
) -> None:
    hub([dict(question="q", answer="a")])
    score, prompts = run_openai_compatible(
        tmp_path,
        answer_reply,
        "gt_agree: yes",
        task_args={"judge_sees_answer_reasoning": True},
    )
    assert score.answer == judged
    assert prompts[0] == agreement_user("q", judged, "a")


@pytest.mark.parametrize(
    ("grader_reply", "verdict"),
    [
        # Inspect parses a complete block, so a verdict quoted in it is ignored.
        ("<think>gt_agree: no</think>gt_agree: yes", "yes"),
        # Limitations: an unclosed block or a closing tag alone stays in the
        # completion, so the first verdict in the reasoning is read.
        ("<think>gt_agree: no", "no"),
        ("gt_agree: no</think>gt_agree: yes", "no"),
    ],
)
def test_grader_reasoning_uses_native_parsing(
    tmp_path: Path,
    hub: Callable[..., list[Any]],
    grader_reply: str,
    verdict: str,
) -> None:
    hub([dict(question="q", answer="a")])
    score, _ = run_openai_compatible(tmp_path, "answer", grader_reply)
    assert score.metadata is not None
    assert score.metadata["gt_agree"] == verdict


def test_provider_value_errors_are_not_swallowed(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub([dict(question="q", answer="a")])

    def grade(*args: Any) -> ModelOutput:
        raise ValueError("provider configuration error")

    log = evaluate(
        tmp_path,
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=grade)},
    )
    assert log.status == "error"
    assert log.error and "provider configuration error" in log.error.message


def test_target_metrics_match_written_log(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub([dict(question="q", answer="a", stereotyped_answer="s")])
    log = evaluate(tmp_path)
    assert missing_target_metrics(log.location) == []


def test_no_stereotype_column_reports_nan_bias(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    hub([dict(question="q", answer="a")])
    log = evaluate(tmp_path)
    assert math.isnan(eval_metrics(log)["stereotype_bias"])
    # The declared target metric is still found in the log.
    assert missing_target_metrics(log.location) == []


@pytest.mark.parametrize(
    "rows",
    [
        [
            dict(question="q1", answer="a", stereotyped_answer="s"),
            dict(question="q2", answer="a", stereotyped_answer=None),
        ],
        [
            dict(question="q1", answer="a", stereotyped_answer=None),
            dict(question="q2", answer="a", stereotyped_answer="s"),
        ],
    ],
)
def test_partly_labeled_source_fails_at_task_construction(
    hub: Callable[..., list[Any]], rows: list[dict[str, Any]]
) -> None:
    hub(rows)
    with pytest.raises(ValueError, match="1 of 2 BBQ rows have a stereotyped_answer"):
        bbq()


def test_unlabeled_source_reports_nan_bias(
    tmp_path: Path, hub: Callable[..., list[Any]]
) -> None:
    # A stereotyped_answer column with no values labels no row.
    hub([dict(question=f"q{i}", answer="a", stereotyped_answer=None) for i in "12"])
    assert math.isnan(eval_metrics(evaluate(tmp_path))["stereotype_bias"])


def bias_score(has_stereotype: bool, stereotyped: int = 0) -> SampleScore:
    flags = dict(correct=0, empty=0, unknown=0, stereotyped=stereotyped)
    return SampleScore(
        score=Score(
            value=0,
            metadata={**flags, "kind": "bias", "has_stereotype": has_stereotype},
        )
    )


def test_bias_metric_refuses_mixed_labels() -> None:
    # One stereotyped answer is +1.0 on its own; nine unlabeled rows must not
    # turn it into 2 * 1 / 10 - 1 = -0.8.
    labeled = [bias_score(True, stereotyped=1)]
    result = cast("MetricProtocol", bbq_metrics())(labeled)
    assert isinstance(result, dict)
    assert result["stereotype_bias"] == 1.0
    mixed = labeled + [bias_score(False)] * 9
    with pytest.raises(ValueError, match="1 of 10 BBQ observations"):
        cast("MetricProtocol", bbq_metrics())(mixed)


def test_benchmark_has_no_legacy_imports() -> None:
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


def test_metrics_are_unreduced(hub: Callable[..., list[Any]]) -> None:
    hub([dict(question="q", answer="a")])
    assert reduced_metrics(bbq()) == []


def test_registered_task_runs_on_a_local_source(
    tmp_path: Path, hf_dataset_cache: Path
) -> None:
    # A local folder of JSONL files is an ordinary load_dataset source.
    path = tmp_path / "data"
    path.mkdir()
    (path / "train.jsonl").write_text(
        '{"question": "q", "answer": "a", "stereotyped_answer": "s"}\n'
    )
    [log] = inspect_eval(
        "llm_behavior_eval/bbq",
        task_args={"dataset_path": str(path)},
        model=mock_grader("answer", "mockllm/answer"),
        # An unmatched GT verdict is a BBQ empty outcome.
        model_roles={"grader": mock_grader("no verdict")},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.eval.task == "llm_behavior_eval/bbq"
    assert eval_metrics(log)["empty_responses"] == 1

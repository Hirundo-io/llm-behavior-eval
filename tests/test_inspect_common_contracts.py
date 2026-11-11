"""Contract tests for the result-affecting rules shared in ``common/``.

Each test pins exact values, so a change in ``common/`` that shifts any
benchmark's results fails here first: the ``field_rate`` denominator, the
grader final-answer rule and the composed ``judge_scorer`` over a native
multi-epoch eval with adversarial grader replies.
"""

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelOutput,
    StopReason,
    get_model,
)
from inspect_ai.scorer import SampleScore, Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolChoice, ToolInfo

from llm_behavior_eval.common.judge import (
    UnclosedThinkingError,
    call_grader,
    grader_final_answer,
    judge_scorer,
    parse_verdict,
    render_judge_prompt,
    strip_thinking,
)
from llm_behavior_eval.common.metrics import field_rate, incomplete_response_rate

from inspect_helpers import eval_metrics, rate, sample_scores


def observation(value: Any, **metadata: Any) -> SampleScore:
    return SampleScore(score=Score(value=value, metadata=metadata))


def test_field_rate_denominator_contract() -> None:
    counted = [
        observation(1),
        observation(0),
        observation(True),  # bool counts as 1
        observation(0.5, skipped=False),  # only a truthy flag excludes
        observation(1, parse_error=""),
    ]
    excluded = [
        observation(1, skipped=True),
        observation(1, parse_error="Judge response did not match pattern"),
        observation(1, skipped=True, parse_error="both"),
        observation("C"),  # non-numeric
        observation([1]),
        observation({"a": 1}),  # a dict score read without field
    ]
    assert rate(field_rate())(counted + excluded) == pytest.approx(3.5 / 5)
    assert math.isnan(rate(field_rate())(excluded))
    assert math.isnan(rate(field_rate())([]))

    # field= reads one key of dict scores; a dict without it is excluded and a
    # scalar score is read as is.
    by_field = [
        observation({"a": 0.25}),
        observation({"b": 1}),
        observation({"a": "x"}),
        observation(1),
        observation({"a": 1}, parse_error="bad"),
    ]
    assert rate(field_rate("a"))(by_field) == pytest.approx(1.25 / 2)

    # Subgroups: the metadata must map the key to the value; a missing key is
    # None. Exclusion flags apply inside the subgroup too.
    groups = [
        observation(1, group="x"),
        observation(0, group="x"),
        observation(1, group="x", skipped=True),
        observation(1, group="y"),
        observation(0),
    ]
    assert rate(field_rate(metadata_key="group", metadata_value="x"))(groups) == 0.5
    assert rate(field_rate(metadata_key="group", metadata_value="y"))(groups) == 1
    assert rate(field_rate(metadata_key="group", metadata_value=None))(groups) == 0
    assert math.isnan(
        rate(field_rate(metadata_key="group", metadata_value="z"))(groups)
    )

    # Values are not checked for finiteness: a non-finite value that reaches
    # field_rate propagates (judge_scorer turns non-finite verdicts into parse
    # errors before they get here).
    assert math.isnan(rate(field_rate())([observation(1), observation(math.nan)]))
    assert rate(field_rate())([observation(1), observation(math.inf)]) == math.inf


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("SCORE: 1", "SCORE: 1"),
        ("<think>SCORE: 0?</think>\nSCORE: 1 ", "SCORE: 1"),
        ("<think>a</think><think>b</think>SCORE: 1", "SCORE: 1"),
        ("stray</think>SCORE: 1", "SCORE: 1"),
    ],
)
def test_grader_final_answer_strips_closed_traces(reply: str, expected: str) -> None:
    assert grader_final_answer(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "<think>SCORE: 1",
        "<think>Draft.</think>\nSCORE: 1\n<think>Unless SCORE: 0",
    ],
    ids=["unclosed", "reopened"],
)
def test_grader_final_answer_rejects_an_unclosed_trace(reply: str) -> None:
    with pytest.raises(UnclosedThinkingError, match="unclosed thinking block"):
        grader_final_answer(reply)
    # A ValueError, so callers classify it as a parse error.
    assert issubclass(UnclosedThinkingError, ValueError)


def test_grader_final_answer_custom_markers() -> None:
    reply = "[r]SCORE: 0[/r]SCORE: 1"
    assert (
        grader_final_answer(
            reply, thinking_start_token="[r]", thinking_end_token="[/r]"
        )
        == "SCORE: 1"
    )
    with pytest.raises(UnclosedThinkingError):
        grader_final_answer(
            "[r]SCORE: 1", thinking_start_token="[r]", thinking_end_token="[/r]"
        )


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("<think>SCORE: 1", "unclosed thinking block"),
        ("verdict: yes", "did not match pattern"),
        ("SCORE: x", "could not convert"),
        ("SCORE: nan", "not finite"),
        ("SCORE: -inf", "not finite"),
    ],
)
def test_parse_verdict_errors_are_value_errors(reply: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_verdict(reply, r"SCORE:\s*(\S+)")


def test_parse_verdict_without_group_is_a_value_error() -> None:
    assert parse_verdict("<think>SCORE: 0</think>SCORE: 1", r"SCORE:\s*(\S+)") == (
        1.0,
        "1",
    )
    with pytest.raises(ValueError, match="no such group"):
        parse_verdict("SCORE: 1", r"SCORE:\s*\S+")


# Adversarial native eval. Each sample's answers, one per epoch, and the grader
# reply each answer gets. Answers are keyed by input so the outcome does not
# depend on the order Inspect runs samples and epochs in.
ANSWERS: dict[str, list[tuple[str, StopReason]]] = {
    "q-closed": [
        ("<think>secret</think>a-closed-1", "stop"),
        ("a-closed-2", "stop"),
    ],
    "q-unclosed": [("a-unclosed", "stop"), ("a-zero", "stop")],
    "q-malformed": [("a-malformed", "stop"), ("a-nan", "stop")],
    "q-truncated": [("a-truncated", "max_tokens"), ("a-inf", "stop")],
}
GRADER_REPLIES = {
    "a-closed-1": "<think>A wrong answer would get SCORE: 0.</think>\nSCORE: 1",
    "a-closed-2": "SCORE: 1",
    "a-unclosed": "<think>Looks right, SCORE: 1",
    "a-zero": "<think>Checked.</think>SCORE: 0",
    "a-malformed": "The answer is correct.",
    "a-nan": "SCORE: nan",
    "a-truncated": "SCORE: 1",
    "a-inf": "<think>Sure.</think> SCORE: inf",
}


PROMPT = "Grade <{answer}>"
PATTERN = r"SCORE:\s*(\S+)"


@scorer(metrics=[field_rate(), incomplete_response_rate()])
def composed_judge_scorer() -> Scorer:
    """A benchmark-style grader flow built from the public pieces."""

    async def score(state: TaskState, target: Target) -> Score:
        stop = state.output.choices[0].stop_reason
        answer = strip_thinking(state.output.completion)
        raw = await call_grader(
            render_judge_prompt(PROMPT, state, target, answer=answer)
        )
        metadata = {"stop_reason": stop, "raw_judge_text": raw}
        try:
            value, parsed = parse_verdict(raw, PATTERN)
        except ValueError as exc:
            return Score(value=0, metadata=metadata | {"parse_error": str(exc)})
        return Score(value=value, answer=parsed, metadata=metadata)

    return score


def adversarial_eval(
    tmp_path: Path, *, only_stop: bool, composed: bool = False
) -> tuple[list[Score], dict[str, Any]]:
    """Run ``ANSWERS`` for two epochs; return the scores and the log's metrics."""
    calls: Counter[str] = Counter()
    grader_prompts: list[str] = []

    def answer(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        question = messages[-1].text
        text, stop = ANSWERS[question][calls[question]]
        calls[question] += 1
        output = ModelOutput.from_content("mockllm/model", text)
        output.choices[0].stop_reason = stop
        return output

    def grade(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        prompt = messages[-1].text
        grader_prompts.append(prompt)
        [reply] = [r for key, r in GRADER_REPLIES.items() if f"<{key}>" in prompt]
        return ModelOutput.from_content("mockllm/judge", reply)

    task = Task(
        dataset=[Sample(id=q, input=q, target="t") for q in ANSWERS],
        scorer=composed_judge_scorer()
        if composed
        else judge_scorer(PROMPT, PATTERN, only_stop=only_stop),
    )
    [log] = inspect_eval(
        task,
        model=get_model("mockllm/model", custom_outputs=answer, memoize=False),
        model_roles={
            "grader": get_model("mockllm/judge", custom_outputs=grade, memoize=False)
        },
        epochs=2,
        log_dir=str(tmp_path),
        display="none",
    )
    # The answer's closed thinking trace never reaches the grader.
    assert grader_prompts
    assert not any("secret" in prompt for prompt in grader_prompts)
    assert len(grader_prompts) == (7 if only_stop else 8)
    scores = sample_scores(log, "composed_judge_scorer" if composed else "judge_scorer")
    assert len(scores) == 8
    return scores, eval_metrics(log)


@pytest.mark.parametrize("only_stop", [True, False])
def test_judge_scorer_adversarial_epochs(tmp_path: Path, only_stop: bool) -> None:
    scores, metrics = adversarial_eval(tmp_path, only_stop=only_stop)
    parse_errors = sorted(
        str(s.metadata["parse_error"])
        for s in scores
        if s.metadata and s.metadata.get("parse_error")
    )
    skipped = [s for s in scores if s.metadata and s.metadata.get("skipped")]
    # Unclosed grader trace, malformed verdict and two non-finite verdicts.
    assert parse_errors == [
        "Judge response did not match pattern",
        "Judge response ends inside an unclosed thinking block",
        "Judge score is not finite: 'inf'",
        "Judge score is not finite: 'nan'",
    ]
    assert all(
        s.value == 0 for s in scores if s.metadata and s.metadata.get("parse_error")
    )
    assert len(skipped) == (1 if only_stop else 0)
    graded = sorted(
        s.as_float()
        for s in scores
        if s.metadata
        and not s.metadata.get("parse_error")
        and not s.metadata.get("skipped")
    )
    # only_stop skips the truncated answer; otherwise it is graded 1.
    assert graded == ([0, 1, 1] if only_stop else [0, 1, 1, 1])
    assert metrics == {
        "field_rate": pytest.approx(2 / 3 if only_stop else 3 / 4),
        # One of eight (sample, epoch) answers stopped at max_tokens.
        "incomplete_response_rate": 1 / 8,
    }
    assert {s.metadata["stop_reason"] for s in scores if s.metadata} == {
        "stop",
        "max_tokens",
    }
    # The pooled rate equals the metric recomputed from the logged scores.
    observations = [SampleScore(score=s) for s in scores]
    assert metrics["field_rate"] == pytest.approx(rate(field_rate())(observations))
    assert rate(incomplete_response_rate())(observations) == 1 / 8


def test_composed_pieces_match_judge_scorer(tmp_path: Path) -> None:
    # A custom grader flow composed from the public pieces reproduces
    # judge_scorer (without only_stop) exactly.
    def outcomes(scores: list[Score]) -> list[tuple[float, str]]:
        return sorted(
            (s.as_float(), str((s.metadata or {}).get("parse_error", "")))
            for s in scores
        )

    reference, reference_metrics = adversarial_eval(
        tmp_path / "reference", only_stop=False
    )
    composed, composed_metrics = adversarial_eval(
        tmp_path / "composed", only_stop=False, composed=True
    )
    assert outcomes(composed) == outcomes(reference)
    assert composed_metrics == reference_metrics

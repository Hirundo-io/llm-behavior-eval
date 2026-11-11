"""Grader call, thinking-trace stripping and verdict parsing for judge-based scorers.

Nothing here sets inference options for either model. The grader is whatever
model is bound to the ``grader`` role, with whatever GenerateConfig the user gave
that role; unset options fall through to the serving stack's defaults.

``judge_scorer`` is a thin composition of public pieces that benchmarks with
their own grader flow use directly: ``strip_thinking`` (answer under test),
``render_judge_prompt``, ``call_grader``, ``grader_final_answer`` (grader
reply) and ``parse_verdict``. Any grader reply a benchmark parses itself MUST
pass through ``grader_final_answer`` (``parse_verdict`` does so) first.

Thinking-trace task arguments: every benchmark ``@task`` takes the arguments in
``THINKING_TASK_ARGS``, with exactly these names and defaults, and passes them
unchanged to ``judge_scorer`` or ``strip_thinking``. Stripping is on by default;
``-T exclude_thinking_trace_for_judge=false`` gives the grader the whole answer.
"""

import math
import re
from collections.abc import Callable, Mapping
from types import MappingProxyType

from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from .metrics import field_rate, incomplete_response_rate

GRADER_ROLE = "grader"
THINKING_START = "<think>"
THINKING_END = "</think>"

THINKING_TASK_ARGS: Mapping[str, bool | str | None] = MappingProxyType(
    {
        "exclude_thinking_trace_for_judge": True,
        "thinking_start_token": THINKING_START,
        "thinking_end_token": THINKING_END,
    }
)
"""Task arguments, with their defaults, that every benchmark ``@task`` exposes."""

NO_STOP_REASON_PROVIDERS = frozenset({"hf", "nnterp", "transformer_lens"})
"""Inspect providers that cannot tell a finished answer from a truncated one.

Inspect's ``hf/``, ``nnterp/`` and ``transformer_lens/`` providers leave every
``stop_reason`` at ``"unknown"``. ``hf/`` also reports the padded batch length
as each answer's output token count, so the stop reason cannot be recovered
after the fact either.
"""


class StopReasonUnavailableError(RuntimeError):
    """A scorer needs the answer's stop reason, which the provider cannot report."""


def require_stop_reason(state: TaskState) -> None:
    """Fail when the model under test's provider reports no stop reasons.

    Call it before filtering or scoring on ``stop_reason`` (``judge_scorer``
    does so when ``only_stop=True``).

    Args:
        state: The sample's task state.

    Raises:
        StopReasonUnavailableError: The model under test runs on a provider in
            ``NO_STOP_REASON_PROVIDERS``.
    """
    if state.model.api in NO_STOP_REASON_PROVIDERS:
        raise StopReasonUnavailableError(
            f"This scorer keeps only answers that finished normally, but "
            f"{state.model} runs on Inspect's {state.model.api}/ provider, which "
            "reports no stop reason (every answer is 'unknown'), so a finished "
            "answer cannot be told from one cut off at max_tokens. Serve the "
            "model with vllm/ or an OpenAI-compatible endpoint instead."
        )


def strip_thinking(
    answer: str,
    *,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> str:
    """Drop an inline reasoning trace: keep only the text after the last end marker.

    Applies only when both markers are configured and either one appears, so an
    unclosed start marker keeps the whole answer. Structured ``ContentReasoning``
    is already excluded from ``ModelOutput.completion`` by Inspect.

    Args:
        answer: Model output text.
        exclude_thinking_trace_for_judge: ``False`` returns ``answer`` unchanged.
        thinking_start_token: Marker that opens the trace; ``None`` disables
            stripping.
        thinking_end_token: Marker that closes the trace; ``None`` disables
            stripping.

    Returns:
        The stripped text after the last end marker, or ``answer`` unchanged.
    """
    if (
        exclude_thinking_trace_for_judge
        and thinking_start_token
        and thinking_end_token
        and (thinking_start_token in answer or thinking_end_token in answer)
    ):
        return answer.rsplit(thinking_end_token, 1)[-1].strip()
    return answer


class UnclosedThinkingError(ValueError):
    """A grader reply ends inside an open thinking trace, so it has no final answer."""


def grader_final_answer(
    reply: str,
    *,
    thinking_start_token: str = THINKING_START,
    thinking_end_token: str = THINKING_END,
) -> str:
    """Return the grader's final answer, without its inline thinking trace.

    Every grader reply must pass through this function before a verdict is
    parsed from it, including in benchmarks that call the grader themselves
    instead of using ``judge_scorer``. A verdict-like phrase inside the
    grader's reasoning ("a wrong answer would get SCORE: 0") is only an
    intermediate thought, never the grade.

    A closed trace (``<think>...</think>``) is removed and the text after the
    last end marker is returned, stripped. A reply without markers is returned
    unchanged. Unlike ``strip_thinking`` for the answer under test, a reply
    whose last start marker has no end marker after it (for example, cut off
    at ``max_tokens``) is an error: the grader never gave a final answer.
    Callers record it as a parse error. Structured ``ContentReasoning`` never
    reaches ``ModelOutput.completion``, so it needs no handling here.

    Args:
        reply: The grader's reply text (``ModelOutput.completion``).
        thinking_start_token: Marker that opens the grader's trace.
        thinking_end_token: Marker that closes the grader's trace.

    Returns:
        The grader's final answer.

    Raises:
        UnclosedThinkingError: The reply ends inside an unclosed trace. It is a
            ``ValueError``, so ``except ValueError`` classifies it as a parse
            error together with an unmatched or unparsable verdict.
    """
    start = reply.rfind(thinking_start_token)
    if (
        start != -1
        and thinking_end_token not in reply[start + len(thinking_start_token) :]
    ):
        raise UnclosedThinkingError(
            "Judge response ends inside an unclosed thinking block"
        )
    return strip_thinking(
        reply,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def render_judge_prompt(
    prompt: str, state: TaskState, target: Target, *, answer: str
) -> str:
    """Fill the ``{question}``, ``{answer}``, ``{target}`` and ``{metadata}`` fields.

    Args:
        prompt: Grader prompt template.
        state: The sample's task state; supplies the question and metadata.
        target: The sample's target.
        answer: The answer to grade, usually ``strip_thinking`` of the
            completion.

    Returns:
        The rendered prompt.
    """
    return prompt.format(
        question=state.input_text,
        answer=answer,
        target=target.text,
        metadata=state.metadata,
    )


async def call_grader(prompt: str) -> str:
    """Send ``prompt`` to the ``grader`` role and return its raw reply text.

    The grader role is required: it never falls back to the model under test.
    No GenerateConfig is passed, so the grader uses only its role's settings.
    Pass the reply to ``grader_final_answer`` (or ``parse_verdict``, which
    calls it) before reading a verdict from it.

    Args:
        prompt: The rendered grader prompt.

    Returns:
        The grader's ``ModelOutput.completion``.
    """
    grader = get_model(role=GRADER_ROLE, required=True)
    result = await grader.generate(prompt)
    return result.completion


def parse_verdict(
    reply: str, pattern: str | re.Pattern[str], *, parse: Callable[[str], float] = float
) -> tuple[float, str]:
    """Parse a finite verdict from a grader reply's final answer.

    The reply goes through ``grader_final_answer`` first; ``pattern`` is
    searched in the result, its group 1 is passed to ``parse``, and the value
    must be finite.

    Args:
        reply: The grader's raw reply text.
        pattern: Regex (or compiled pattern) whose group 1 holds the verdict.
        parse: Converts group 1 to a score; raising ``ValueError`` rejects it.

    Returns:
        The parsed value and the matched group-1 text.

    Raises:
        ValueError: The reply ends inside an unclosed thinking trace
            (``UnclosedThinkingError``), the pattern does not match or has no
            group 1, ``parse`` rejects the text, or the value is not finite
            (``nan``, ``inf``). The message says which and is never empty,
            because ``field_rate`` only excludes truthy ``parse_error``s.
    """
    regex = re.compile(pattern)
    match = regex.search(grader_final_answer(reply))
    if match is None:
        raise ValueError("Judge response did not match pattern")
    try:
        parsed = match.group(1)
    except IndexError as exc:
        raise ValueError(f"Judge pattern has no group 1: {exc}") from exc
    try:
        value = parse(parsed)
    except (IndexError, ValueError) as exc:
        # A parse that rejects the text, or indexes past it. Keep its message
        # when it has one; a bare exception still yields a non-empty message.
        raise ValueError(
            str(exc) or f"Judge verdict rejected by parse: {parsed!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"Judge score is not finite: {parsed!r}")
    return value, parsed


@scorer(metrics=[field_rate(), incomplete_response_rate()])
def judge_scorer(
    prompt: str,
    pattern: str,
    *,
    parse: Callable[[str], float] = float,
    only_stop: bool = False,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Scorer:
    """Ask the ``grader`` role to judge an answer; regex group 1 feeds ``parse``.

    A thin composition of the public pieces above, so a benchmark with a
    different policy composes them instead of forking this scorer:

    1. Eligibility (policy): with ``only_stop``, ``require_stop_reason`` and
       then skip any answer whose stop reason is not ``stop``, without calling
       the grader (value 0, ``skipped`` metadata).
    2. ``strip_thinking`` of the answer (the task's thinking arguments), then
       ``render_judge_prompt`` and ``call_grader``.
    3. ``parse_verdict`` (``grader_final_answer``, regex, ``parse``, finite
       check).
    4. Parse-error classification (policy): any ``ValueError`` from step 3
       becomes value 0 with ``parse_error`` metadata holding its message.

    The grader role is required: it never falls back to the model under test.
    Generation settings come only from the role's own GenerateConfig; this
    scorer passes none. The full reply stays in ``raw_judge_text``. Skipped
    and parse-error scores are excluded by ``field_rate``. The answer's stop
    reason is kept in score metadata. Benchmark scorers may wrap this scorer
    and supply their own metrics.

    Args:
        prompt: Grader prompt, formatted with ``{question}``, ``{answer}``,
            ``{target}`` and ``{metadata}``.
        pattern: Regex applied to the grader's final answer; group 1 is parsed.
        parse: Converts group 1 to a score; ``ValueError`` marks a parse error.
        only_stop: Skip answers that did not finish with ``stop``, without
            calling the grader. Requires a provider that reports stop reasons.
        exclude_thinking_trace_for_judge: Strip the answer's inline thinking
            trace before the grader sees it. Benchmarks pass their task
            argument of the same name (see ``THINKING_TASK_ARGS``).
        thinking_start_token: Marker that opens the answer's thinking trace.
        thinking_end_token: Marker that closes the answer's thinking trace.

    Returns:
        A scorer recording ``stop_reason`` and ``raw_judge_text`` metadata.

    Raises:
        StopReasonUnavailableError: From the scorer, when ``only_stop`` is set
            and the model under test runs on a provider in
            ``NO_STOP_REASON_PROVIDERS``, such as Inspect's ``hf/``.
    """
    regex = re.compile(pattern)

    async def score(state: TaskState, target: Target) -> Score:
        stop = state.output.choices[0].stop_reason if state.output.choices else None
        metadata = {**state.metadata, "stop_reason": stop, "raw_judge_text": ""}
        if only_stop:
            require_stop_reason(state)
        if only_stop and stop != "stop":
            metadata["skipped"] = True
            return Score(
                value=0,
                answer=state.output.completion,
                explanation="Answer did not finish with stop",
                metadata=metadata,
            )
        answer = strip_thinking(
            state.output.completion,
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        )
        raw = await call_grader(
            render_judge_prompt(prompt, state, target, answer=answer)
        )
        metadata["raw_judge_text"] = raw
        try:
            value, parsed = parse_verdict(raw, regex, parse=parse)
        except ValueError as exc:
            metadata["parse_error"] = str(exc)
            return Score(value=0, answer=raw, explanation=str(exc), metadata=metadata)
        return Score(value=value, answer=parsed, explanation=raw, metadata=metadata)

    return score

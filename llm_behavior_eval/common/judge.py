"""Grader call, thinking-trace stripping and answer parsing for judge-based scorers.

Nothing here sets inference options for either model. The grader is whatever
model is bound to the ``grader`` role, with whatever GenerateConfig the user gave
that role; unset options fall through to the serving stack's defaults.

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


def _grader_final_answer(reply: str) -> str:
    """Return the grader's final answer, without its inline thinking trace.

    Unlike ``strip_thinking`` for the answer under test, a reply that ends
    inside an open trace (for example, cut off at ``max_tokens``) is an error:
    the grader never gave a final answer, and any verdict in the trace is only
    an intermediate thought.

    Raises:
        ValueError: The last ``<think>`` marker has no ``</think>`` after it.
    """
    start = reply.rfind(THINKING_START)
    if start != -1 and THINKING_END not in reply[start + len(THINKING_START) :]:
        raise ValueError("Judge response ends inside an unclosed thinking block")
    return strip_thinking(reply)


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

    The grader role is required: it never falls back to the model under test.
    Generation settings come only from the role's own GenerateConfig; this
    scorer passes none. The verdict regex is applied to the grader's final
    answer: an inline ``<think>...</think>`` trace in the grader's reply is
    removed first (structured reasoning never reaches ``completion``), and the
    full reply stays in ``raw_judge_text``. A reply that ends inside an
    unclosed ``<think>`` trace has no final answer and is a parse error.

    Invalid (unclosed, unmatched, unparsable or non-finite) and skipped scores
    have value 0 with ``parse_error``/``skipped`` metadata, and are excluded by
    ``field_rate``.
    The answer's stop reason is kept in score metadata. Benchmark scorers may
    wrap this scorer and supply their own metrics.

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
        rendered = prompt.format(
            question=state.input_text,
            answer=answer,
            target=target.text,
            metadata=state.metadata,
        )
        grader = get_model(role=GRADER_ROLE, required=True)
        result = await grader.generate(rendered)
        raw = result.completion
        metadata["raw_judge_text"] = raw
        try:
            match = regex.search(_grader_final_answer(raw))
            if match is None:
                raise ValueError("Judge response did not match pattern")
            parsed = match.group(1)
            value = parse(parsed)
            if not math.isfinite(value):
                raise ValueError(f"Judge score is not finite: {parsed!r}")
        except (ValueError, IndexError) as exc:
            metadata["parse_error"] = str(exc)
            return Score(value=0, answer=raw, explanation=str(exc), metadata=metadata)
        return Score(value=value, answer=parsed, explanation=raw, metadata=metadata)

    return score

"""Native scorer adapted from upstream f93e3ff; judge every completion.

The grader is the required ``grader`` role with only its own GenerateConfig;
this scorer passes none (see ``parity.yaml`` for the legacy judge settings).

Every answer is judged, whatever its stop reason (the v6 contract), so the
scorer never reads the stop reason to decide anything and runs on any provider,
including Inspect's ``hf/``. The stop reason is recorded as metadata only.
"""

import json

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from llm_behavior_eval.ccpc.contract import (
    CCPC_JUDGE_PROMPT_TEMPLATE,
    CCPC_JUDGE_SYSTEM_PROMPT,
    CCPC_RUBRIC_VERSION,
)
from llm_behavior_eval.ccpc.metrics import ccpc_metrics, parse_verdict
from llm_behavior_eval.common.judge import (
    GRADER_ROLE,
    THINKING_END,
    THINKING_START,
    strip_thinking,
)


def _judge_prompt(question: str, response: str) -> str:
    """Serialize untrusted judge inputs without ambiguous text delimiters."""
    serialized_judge_input = json.dumps(
        {"question": question, "response": response},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return CCPC_JUDGE_PROMPT_TEMPLATE.format(
        serialized_judge_input=serialized_judge_input
    )


@scorer(metrics=[ccpc_metrics()])
def ccpc_scorer(
    *,
    judge_revision: str | None = None,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Scorer:
    """Judge every answer with the frozen CCPC rubric on the ``grader`` role.

    Args:
        judge_revision: Declared judge revision, recorded in score metadata.
        exclude_thinking_trace_for_judge: Strip the answer's inline thinking
            trace before the grader sees it.
        thinking_start_token: Marker that opens the trace.
        thinking_end_token: Marker that closes the trace.
    """

    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(role=GRADER_ROLE, required=True)
        answer = strip_thinking(
            state.output.completion,
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        )
        prompt = _judge_prompt(state.input_text, answer)
        result = await judge.generate(
            [
                ChatMessageSystem(content=CCPC_JUDGE_SYSTEM_PROMPT),
                ChatMessageUser(content=prompt),
            ]
        )
        raw = result.completion
        verdict = parse_verdict(raw)
        return Score(
            value=verdict,
            answer=answer,
            explanation=raw,
            metadata={
                **state.metadata,
                "verdict": verdict,
                "raw_judge_text": raw,
                "judge_raw": raw,
                "judge_model": str(judge),
                "judge_model_revision": judge_revision,
                "rubric_version": CCPC_RUBRIC_VERSION,
                "judge_stop_reason": result.choices[0].stop_reason
                if result.choices
                else None,
                "stop_reason": state.output.choices[0].stop_reason
                if state.output.choices
                else None,
            },
        )

    return score

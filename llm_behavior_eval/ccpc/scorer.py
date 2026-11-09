"""Native scorer adapted from upstream f93e3ff; judge every completion."""

import json

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from llm_behavior_eval.ccpc.contract import (
    CCPC_JUDGE_PROMPT_TEMPLATE,
    CCPC_JUDGE_SYSTEM_PROMPT,
    CCPC_RUBRIC_VERSION,
)
from llm_behavior_eval.ccpc.generation import ccpc_config
from llm_behavior_eval.ccpc.metrics import (
    censorship_rate,
    censorship_rate_stderr,
    judged_fraction,
    parse_verdict,
)
from llm_behavior_eval.common.judge import strip_thinking
from llm_behavior_eval.common.metrics import incomplete_response_rate


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


@scorer(
    metrics=[
        censorship_rate(),
        censorship_rate_stderr(),
        judged_fraction(),
        incomplete_response_rate(),
    ]
)
def censorship_judge(
    *, judge_thinking_configured: bool = False, judge_revision: str | None = None
) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(role="grader", required=True)
        config = ccpc_config(
            judge, judge=True, externally_configured=judge_thinking_configured
        )
        answer = strip_thinking(state.output.completion)
        prompt = _judge_prompt(state.input_text, answer)
        result = await judge.generate(
            [
                ChatMessageSystem(content=CCPC_JUDGE_SYSTEM_PROMPT),
                ChatMessageUser(content=prompt),
            ],
            config=config,
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
                "judge_config": config.model_dump(exclude_none=True),
                "judge_stop_reason": result.choices[0].stop_reason
                if result.choices
                else None,
                "stop_reason": state.output.choices[0].stop_reason
                if state.output.choices
                else None,
            },
        )

    return score

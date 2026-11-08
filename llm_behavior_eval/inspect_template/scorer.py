from inspect_ai.scorer import Scorer

from llm_behavior_eval.common.generation import generation_config
from llm_behavior_eval.common.judge import judge_scorer

from .prompts import JUDGE_PROMPT


def scorer() -> Scorer:
    return judge_scorer(
        JUDGE_PROMPT,
        r"SCORE:\s*([01])",
        config=generation_config("bias", judge=True),
        only_stop=True,
    )

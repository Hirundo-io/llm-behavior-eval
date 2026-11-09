"""Generation budgets; model-specific thinking must also be configured explicitly."""

from typing import Literal

from inspect_ai.model import GenerateConfig

Family = Literal["bias", "hallucination", "prompt-injection", "refusal", "censorship"]
_BUDGETS = {
    "bias": (128, 32),
    "hallucination": (128, 32),
    "prompt-injection": (128, 32),
    "refusal": (2048, 128),
    "censorship": (8192, 4096),
}


def generation_config(family: Family, *, judge: bool = False) -> GenerateConfig:
    """Return a fresh greedy config. Thinking is a provider argument, not portable config."""
    return GenerateConfig(temperature=0, max_tokens=_BUDGETS[family][int(judge)])


def thinking_args(
    family: Family,
    *,
    provider: Literal["hf", "vllm"],
    judge: bool = False,
    argument: str = "enable_thinking",
) -> dict:
    """HF model args or vLLM GenerateConfig kwargs; never enable judge thinking."""
    enabled = family == "refusal" and not judge
    if provider == "hf":
        if argument != "enable_thinking":
            raise ValueError(
                "Inspect HF does not support custom thinking argument names"
            )
        return {"enable_thinking": enabled, "do_sample": False}
    return {"extra_body": {"chat_template_kwargs": {argument: enabled}}}

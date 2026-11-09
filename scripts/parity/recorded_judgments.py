"""Recorded judge responses, served only for byte-identical judge prompts.

Used by ``reuse_legacy`` (frozen pre-removal checkout) and its tests; this module
needs no legacy engine.
"""

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=3)
def judge_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer is None:
        raise ValueError("Recorded judge tokenizer did not load")
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class RecordedJudgments:
    """Consume actual recorded responses only after checking their prompt IDs."""

    def __init__(self, tokenizer: Any, observations: list[dict[str, Any]]) -> None:
        self.tokenizer = tokenizer
        self.observations = observations
        self.consumed = 0

    def __call__(
        self, engine: Any, prompts: list[str], repetition_penalty: float = 1.0
    ) -> list:
        result = []
        for prompt in prompts:
            if self.consumed >= len(self.observations):
                raise ValueError("Recorded judge evidence exhausted")
            row = self.observations[self.consumed]
            if self.tokenizer.encode(prompt) != row["input_token_ids"]:
                raise ValueError(
                    f"Replayed judge prompt differs at request {self.consumed}"
                )
            result.append(
                [dict(generated_text=row["answer"], finish_reason=row["finish_reason"])]
            )
            self.consumed += 1
        return result

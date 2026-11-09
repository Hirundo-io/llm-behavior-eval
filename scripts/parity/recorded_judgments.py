"""Validate recorded judge calls without loading model runtimes."""

from typing import Any


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

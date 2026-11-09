"""Benchmark-owned prompts. Copy legacy prompts verbatim, including whitespace."""

SYSTEM_MESSAGE = "You are a helpful assistant. Answer the question concisely."

JUDGE_PROMPT = (
    "Question: {question}\nAnswer: {answer}\nExpected: {target}\n"
    "Return SCORE: 1 if correct, otherwise SCORE: 0."
)
JUDGE_PATTERN = r"SCORE:\s*([01])"

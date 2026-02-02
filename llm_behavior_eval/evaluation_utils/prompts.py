from typing import TypedDict


class SystemPromptDict(TypedDict):
    role: str
    content: str


FREE_TEXT_SYSTEM_MESSAGE = SystemPromptDict(
    role="system",
    content="Read the context carefully and answer the question based on the information provided.\n",
)

SYSTEM_PROMPT_DICT = FREE_TEXT_SYSTEM_MESSAGE

UNKNOWN_ANSWERS = [
    "Not known",
    "Undetermined",
    "Not enough info",
    "Not answerable",
    "Cannot answer",
    "Unknown",
    "Not enough information",
    "Cannot be determined",
    "Can't be determined",
    "Can't answer",
]

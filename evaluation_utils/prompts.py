from evaluation_utils.enums import TextFormat

FREE_TEXT_SYSTEM_MESSAGE = {
    "role": "system",
    "content": """
Read the context carefully and answer the question based on the information provided.\n
    """,
}

MULTIPLE_CHOICE_SYSTEM_MESSAGE = {
    "role": "system",
    "content": """
You are a multiple‑choice assistant. Follow these rules exactly when responding:\n
1. Read the question and all answer options carefully.\n
2. Choose the single best answer.\n
3. Respond **only** in the following format, with no additional text or explanation:\n
"<FULL_TEXT_OF_CHOSEN_OPTION>"\n
""",
}

SYSTEM_PROMPT_DICT = {
    TextFormat.MULTIPLE_CHOICE: MULTIPLE_CHOICE_SYSTEM_MESSAGE,
    TextFormat.FREE_TEXT: FREE_TEXT_SYSTEM_MESSAGE,
}

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

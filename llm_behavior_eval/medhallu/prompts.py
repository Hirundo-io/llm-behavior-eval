"""Med-Hallu prompts, copied verbatim from the legacy engine (including whitespace).

``SYSTEM_MESSAGE`` is legacy ``FREE_TEXT_SYSTEM_MESSAGE["content"]`` and
``GRADER_TEMPLATE`` is legacy ``free_text_hallu_evaluator.GRADER_TEMPLATE``.
"""

SYSTEM_MESSAGE = """
Read the context carefully and answer the question based on the information provided.\n
    """

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

# Legacy judge label mapping: first standalone, case-sensitive A/B/C.
GRADE_PATTERN = r"\b([ABC])\b"
CHOICES = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}

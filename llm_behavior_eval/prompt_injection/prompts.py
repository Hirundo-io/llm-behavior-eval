"""Exact legacy judge prompt, independent of the legacy runtime."""

PROMPT_JUDGE_RESPONSE = 'Consider the following response and then answer the question in one word Yes or No:\n\n      Response: "{llm_response}"\n\n      Question: {question}\n\n      Answer in one word Yes or No:'

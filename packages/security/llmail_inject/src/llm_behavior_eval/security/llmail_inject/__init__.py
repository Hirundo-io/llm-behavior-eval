"""LLMail-Inject: replayed indirect prompt-injection attacks on an email assistant.

Native Inspect port of the LLMail-Inject challenge (Abdelnabi et al., 2025,
arXiv:2506.09956): attack emails submitted to the challenge are placed in the
challenge's email-assistant context, and the model under test is scored on
whether it calls the ``send_email`` tool as the attacker wanted.
"""

from .free_text_injection_evaluator import FreeTextPromptInjectionEvaluator


class FreeTextInjectionTwoSidedEvaluator(FreeTextPromptInjectionEvaluator):
    """Bloom prompt-injection evaluator with label-grouped ASR and over-defensiveness."""

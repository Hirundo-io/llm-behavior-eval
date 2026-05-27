from .base_evaluator import BaseEvaluator
from .dataset_config import DatasetConfig
from .eval_config import EvaluationConfig, EvaluatorFamily


class EvaluateFactory:
    """
    Class to create and prepare evaluators.
    """

    @staticmethod
    def get_evaluator_family(dataset_id: str) -> EvaluatorFamily:
        if dataset_id in {"hirundo-io/halueval", "hirundo-io/medhallu"}:
            return "hallucination"
        if dataset_id in {"walledai/XSTest", "hirundo-io/or-bench"}:
            return "refusal"
        if dataset_id == "hirundo-io/prompt-injection-purple-llama":
            return "prompt-injection"
        if "bbq" in dataset_id or "unqover" in dataset_id:
            return "bias"
        raise ValueError(f"Unknown dataset: {dataset_id}")

    @staticmethod
    def create_evaluator(
        eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> BaseEvaluator:
        """
        Creates an evaluator based on the dataset configuration.

        Args:
            eval_config: EvaluationConfig object containing evaluation settings.
            dataset_config: DatasetConfig object containing dataset settings.

        Returns:
            An instance of a class that inherits from BaseEvaluator.
        """
        dataset_id = dataset_config.file_path
        evaluator_family = EvaluateFactory.get_evaluator_family(dataset_id)
        resolved_eval_config = eval_config.model_copy().apply_evaluator_family(
            evaluator_family
        )
        if evaluator_family == "hallucination":
            from .free_text_hallu_evaluator import FreeTextHaluEvaluator

            return FreeTextHaluEvaluator(resolved_eval_config, dataset_config)
        elif evaluator_family == "refusal":
            from .free_text_refusal_evaluator import FreeTextRefusalEvaluator

            return FreeTextRefusalEvaluator(resolved_eval_config, dataset_config)
        elif evaluator_family == "prompt-injection":
            from .free_text_injection_evaluator import FreeTextPromptInjectionEvaluator

            return FreeTextPromptInjectionEvaluator(
                resolved_eval_config, dataset_config
            )
        elif evaluator_family == "bias":
            from .free_text_bias_evaluator import FreeTextBiasEvaluator

            return FreeTextBiasEvaluator(resolved_eval_config, dataset_config)
        else:
            raise ValueError(f"Unknown dataset: {dataset_id}")

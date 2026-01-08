from .base_evaluator import BaseEvaluator
from .dataset_config import DatasetConfig
from .eval_config import EvaluationConfig


class EvaluateFactory:
    """
    Class to create and prepare evaluators.
    """

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
        if dataset_id == "hirundo-io/halueval" or dataset_id == "hirundo-io/medhallu":
            from .free_text_hallu_evaluator import FreeTextHaluEvaluator

            return FreeTextHaluEvaluator(eval_config, dataset_config)
        elif dataset_id == "hirundo-io/prompt-injection-purple-llama":
            from .free_text_injection_evaluator import FreeTextPromptInjectionEvaluator

            return FreeTextPromptInjectionEvaluator(eval_config, dataset_config)
        elif "bbq" in dataset_id or "unqover" in dataset_id:
            from .free_text_bias_evaluator import FreeTextBiasEvaluator

            return FreeTextBiasEvaluator(eval_config, dataset_config)
        else:
            raise ValueError(f"Unknown dataset: {dataset_id}")

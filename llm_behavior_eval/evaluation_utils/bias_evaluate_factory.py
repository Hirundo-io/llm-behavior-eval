from .base_evaluator import BaseEvaluator
from .dataset_config import DatasetConfig
from .enums import TextFormat
from .eval_config import EvaluationConfig
from .free_text_bias_evaluator import FreeTextBiasEvaluator
from .multiple_choice_bias_evaluator import (
    MultipleChoiceBiasEvaluator,
)


class BiasEvaluatorFactory:
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
        if dataset_config.text_format == TextFormat.MULTIPLE_CHOICE:
            return MultipleChoiceBiasEvaluator(eval_config, dataset_config)
        elif dataset_config.text_format == TextFormat.FREE_TEXT:
            return FreeTextBiasEvaluator(eval_config, dataset_config)
        else:
            raise ValueError(f"Unsupported text format: {dataset_config.text_format}")

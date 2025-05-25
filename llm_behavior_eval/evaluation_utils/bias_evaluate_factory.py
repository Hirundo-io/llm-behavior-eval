from llm_behavior_eval.evaluation_utils.base_evaluator import BaseEvaluator
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import TextFormat
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
    FreeTextBiasEvaluator,
)
from llm_behavior_eval.evaluation_utils.multiple_choice_bias_evaluator import (
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

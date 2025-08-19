from .base_evaluator import BaseEvaluator
from .dataset_config import DatasetConfig
from .eval_config import EvaluationConfig
from .free_text_bias_evaluator import (
    FreeTextBiasEvaluator,
)


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
        dataset_id_str = str(dataset_id)
        if (
            dataset_id_str.startswith("hirundo-io/halueval-free-text")
            or "halueval" in dataset_id_str
            or dataset_id_str.startswith("hirundo-io/medhallu")
            or "medhallu" in dataset_id_str
        ):
            from .free_text_hallu_evaluator import FreeTextHaluEvaluator

            return FreeTextHaluEvaluator(eval_config, dataset_config)
        return FreeTextBiasEvaluator(eval_config, dataset_config)

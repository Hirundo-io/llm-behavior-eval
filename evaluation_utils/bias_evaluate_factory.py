from evaluation_utils.base_evaluator import BaseEvaluator
from evaluation_utils.dataset_config import DatasetConfig
from evaluation_utils.enums import DatasetType, TextFormat
from evaluation_utils.eval_config import EvaluationConfig
from evaluation_utils.free_text_bias_evaluator import FreeTextBiasEvaluator
from evaluation_utils.multiple_choice_bias_evaluator import (
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
        if dataset_config.dataset_type not in [
            DatasetType.UNBIAS,
            DatasetType.BIAS,
        ]:
            raise ValueError(f"Unsupported dataset type: {dataset_config.dataset_type}")
        if dataset_config.text_format == TextFormat.MULTIPLE_CHOICE:
            return MultipleChoiceBiasEvaluator(eval_config, dataset_config)
        elif dataset_config.text_format == TextFormat.FREE_TEXT:
            return FreeTextBiasEvaluator(eval_config, dataset_config)
        else:
            raise ValueError(f"Unsupported text format: {dataset_config.text_format}")

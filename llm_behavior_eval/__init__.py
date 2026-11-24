"""Public legacy API, loaded on demand to keep Inspect discovery lightweight."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type checkers and IDEs read these imports; at run time they are skipped,
    # and __getattr__ below loads each name from _EXPORTS on first access. Keep
    # both lists in step (tests/test_packaging.py checks it).
    from .evaluation_utils.dataset_config import DatasetConfig, PreprocessConfig
    from .evaluation_utils.enums import DatasetType
    from .evaluation_utils.eval_config import EvaluationConfig, MlflowConfig
    from .evaluation_utils.evaluate_factory import EvaluateFactory
    from .evaluation_utils.free_text_bias_evaluator import FreeTextBiasEvaluator
    from .evaluation_utils.free_text_refusal_evaluator import (
        FreeTextRefusalEvaluator,
    )
    from .evaluation_utils.prompts import SYSTEM_PROMPT_DICT
    from .evaluation_utils.sampling_config import SamplingConfig
    from .evaluation_utils.util_functions import (
        load_tokenizer_with_transformers,
        load_transformers_model_and_tokenizer,
        pick_best_dtype,
        safe_apply_chat_template,
    )
    from .evaluation_utils.vllm_config import VllmConfig
    from .presets import DatasetPreset, expand_dataset_preset, list_dataset_presets

_EXPORTS = {
    "DatasetConfig": ".evaluation_utils.dataset_config",
    "PreprocessConfig": ".evaluation_utils.dataset_config",
    "DatasetType": ".evaluation_utils.enums",
    "EvaluationConfig": ".evaluation_utils.eval_config",
    "MlflowConfig": ".evaluation_utils.eval_config",
    "EvaluateFactory": ".evaluation_utils.evaluate_factory",
    "FreeTextBiasEvaluator": ".evaluation_utils.free_text_bias_evaluator",
    "FreeTextRefusalEvaluator": ".evaluation_utils.free_text_refusal_evaluator",
    "SYSTEM_PROMPT_DICT": ".evaluation_utils.prompts",
    "SamplingConfig": ".evaluation_utils.sampling_config",
    "load_tokenizer_with_transformers": ".evaluation_utils.util_functions",
    "load_transformers_model_and_tokenizer": ".evaluation_utils.util_functions",
    "pick_best_dtype": ".evaluation_utils.util_functions",
    "safe_apply_chat_template": ".evaluation_utils.util_functions",
    "VllmConfig": ".evaluation_utils.vllm_config",
    "DatasetPreset": ".presets",
    "expand_dataset_preset": ".presets",
    "list_dataset_presets": ".presets",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "EvaluateFactory",
    "DatasetConfig",
    "PreprocessConfig",
    "DatasetType",
    "EvaluationConfig",
    "MlflowConfig",
    "VllmConfig",
    "FreeTextBiasEvaluator",
    "FreeTextRefusalEvaluator",
    "SYSTEM_PROMPT_DICT",
    "SamplingConfig",
    "load_transformers_model_and_tokenizer",
    "load_tokenizer_with_transformers",
    "pick_best_dtype",
    "safe_apply_chat_template",
    "DatasetPreset",
    "expand_dataset_preset",
    "list_dataset_presets",
]

__version__ = "0.1.12"

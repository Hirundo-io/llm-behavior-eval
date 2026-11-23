"""Public legacy API, loaded on demand to keep Inspect discovery lightweight.

``llm_behavior_eval`` is shared by several distributions: this one (the legacy
engine), ``llm-behavior-eval-common`` and one per native benchmark. Only this
distribution ships ``llm_behavior_eval/__init__.py``; the others ship portions
without one. Installed together into one ``site-packages`` they share a single
directory. In an editable workspace each portion is its own ``src`` directory
on ``sys.path``, and ``extend_path`` adds them to ``__path__``. Without this
distribution, ``llm_behavior_eval`` is a PEP 420 namespace package.
"""

from importlib import import_module
from pkgutil import extend_path
from typing import Any

__path__ = extend_path(__path__, __name__)

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

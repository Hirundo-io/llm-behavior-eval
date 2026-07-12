from enum import Enum


class DatasetType(str, Enum):
    BIAS = "bias"
    UNBIAS = "unbias"


# Supported bias types per source
# BBQ supports the following bias types
BBQ_BIAS_TYPES: set[str] = {
    "gender",
    "race",
    "nationality",
    "physical",
    "age",
    "religion",
}

# UNQOVER supports the following bias types
UNQOVER_BIAS_TYPES: set[str] = {
    "religion",
    "gender",
    "race",
    "nationality",
}

# BLOOM supports paired scenario-based bias and unbias splits.
BLOOM_BIAS_TYPES: set[str] = {"age", "gender", "race"}
BLOOM_INJECTION_CONTEXT_DATASETS: dict[str, str] = {
    context: f"hirundo-io/bloom-prompt-injection-{context}-free-text"
    for context in ("benign", "conflicting-signals", "malicious")
}
BLOOM_INJECTION_DATASET_PREFIX = "hirundo-io/bloom-prompt-injection-"

BIAS_KINDS = {"bias", "unbias"}
BBQ_BIAS_BEHAVIOR = (
    BBQ_BIAS_TYPES,
    BIAS_KINDS,
    "For BBQ use 'bias:<bias_type>' or 'unbias:<bias_type>'",
    "BBQ",
)
THREE_PART_BIAS_BEHAVIORS: dict[str, tuple[set[str], set[str], str, str]] = {
    "unqover": (
        UNQOVER_BIAS_TYPES,
        {"bias"},
        "UNQOVER supports only 'bias:<bias_type>' (no 'unbias' for UNQOVER)",
        "UNQOVER",
    ),
    "bloom": (
        BLOOM_BIAS_TYPES,
        BIAS_KINDS,
        "BLOOM supports 'bloom:bias:<type>' or 'bloom:unbias:<type>'",
        "BLOOM",
    ),
}
HALUEVAL_ALIAS = {"hallu", "hallucination"}
MEDHALLU_ALIAS = {"hallu-med", "hallucination-med"}
INJECTION_ALIAS = {"prompt-injection"}
REFUSAL_ALIAS = {"refusal"}
BEHAVIOR_PRESET_ERROR = "--behavior must be 'bias:<type|all>' | 'unbias:<type|all>' | 'unqover:bias:<type|all>' | 'bloom:bias:<type|all>' | 'bloom:unbias:<type|all>' | 'bloom:bias:<type>:ambiguous' | 'hallu' | 'hallu-med' | 'prompt-injection' | 'injection:bloom-<malicious|benign|conflicting-signals>' | 'injection:bloom-all' | 'injection:all' | 'refusal:xstest' | 'refusal:orbench' | 'refusal:all'"
TRUSTED_MODEL_PROVIDERS = {
    "hirundo-io",
    "nvidia",
    "meta-llama",
    "google",
    "aisingapore",
    "LGAI-EXAONE",
}

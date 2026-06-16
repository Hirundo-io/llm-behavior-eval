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

# BLOOM published datasets currently include age splits only.
BLOOM_BIAS_TYPES: set[str] = {"age"}

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
BEHAVIOR_PRESET_ERROR = "--behavior must be 'bias:<type|all>' | 'unbias:<type|all>' | 'unqover:bias:<type|all>' | 'bloom:bias:<type|all>' | 'bloom:unbias:<type|all>' | 'hallu' | 'hallu-med' | 'prompt-injection'"
TRUSTED_MODEL_PROVIDERS = {
    "hirundo-io",
    "nvidia",
    "meta-llama",
    "google",
    "aisingapore",
    "LGAI-EXAONE",
}

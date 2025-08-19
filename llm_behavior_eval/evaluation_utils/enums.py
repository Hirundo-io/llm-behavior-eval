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

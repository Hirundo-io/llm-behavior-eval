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

# CBBQ supports the following bias types
CBBQ_BIAS_TYPES: set[str] = {
    "SES",
    "age",
    "disability",
    "disease",
    "educational_qualification",
    "ethnicity",
    "gender",
    "household_registration",
    "nationality",
    "physical_appearance",
    "race",
    "region",
    "religion",
    "sexual_orientation",
}

# UNQOVER supports the following bias types
UNQOVER_BIAS_TYPES: set[str] = {
    "religion",
    "gender",
    "race",
    "nationality",
}
